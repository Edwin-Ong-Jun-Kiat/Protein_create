"""
Backbone generation without RFdiffusion: parametric helical bundles.

Why this module exists
----------------------
Challenge 1 generated backbones with RFdiffusion on a V100. That is not an
option here and it is worth being precise about why, because it is not a matter
of patience: RFdiffusion's SE3Transformer depends on `dgl`, which has no working
Apple-Silicon build, and the stack is pinned to `torch 2.0.1+cu118`. There is no
MPS path and no CPU path. The model simply cannot be run on this laptop.

So this module generates backbones the way the field did before diffusion
models: parametrically. The topology it builds -- a 3-helix bundle of 50-80
residues -- is not a compromise chosen for convenience. It is the topology that
de novo minibinder campaigns actually use, because short helical bundles fold
robustly, express well in E. coli, and present a flat-to-convex face that can be
laid against a target groove. A helical bundle is a weaker *sampler* than
RFdiffusion, not a weaker *scaffold*.

How the geometry is built
-------------------------
Everything comes from internal coordinates via NeRF (natural extension reference
frame), so the chain is closed by construction and no loop-closure algorithm is
needed. Helix residues get ideal alpha-helical torsions; the only free
parameters are the loop torsions, which are optimised so the helices pack into a
compact bundle instead of splaying.

That inverts the usual difficulty. Building a bundle by placing rigid helices on
a supercoil and then *closing* the loops requires CCD or an analytic closure and
can fail; building it by torsions makes closure free and makes compactness the
thing you have to work for. Compactness is a smooth objective over ~10
parameters per loop, so random restarts plus a local refine handle it in
milliseconds.

Pure numpy and scipy. No GPU, no downloads, no model weights.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------- geometry

#: Ideal protein backbone bond lengths (A) and angles (degrees). Engh & Huber.
B_N_CA = 1.458
B_CA_C = 1.525
B_C_N = 1.329
B_C_O = 1.231

A_N_CA_C = 111.2
A_CA_C_N = 116.2
A_C_N_CA = 121.7
A_CA_C_O = 120.5

#: Ideal right-handed alpha-helix backbone torsions.
HELIX_PHI, HELIX_PSI = -57.0, -47.0
#: Trans peptide bond.
OMEGA_TRANS = 180.0

#: Rise per residue and radius of an ideal alpha helix, used for sanity checks
#: and for the bundle target geometry rather than for construction.
HELIX_RISE = 1.50
HELIX_RADIUS = 2.30

#: Target spacing between the axes of neighbouring helices in a bundle. Knobs
#: into holes packing puts this near 10 A for an antiparallel coiled coil.
BUNDLE_SPACING = 10.5

#: Any two CA atoms closer than this are counted as a clash by the optimiser.
CA_CLASH = 3.8
#: Below this, the backbone is physically impossible and the scaffold is binned.
#: The two thresholds are separate because tightly packed helices in real
#: structures do reach ~3.8 A CA-CA, so rejecting every bundle with one contact
#: at 3.8 throws away good scaffolds -- an early run discarded a bundle with
#: penalty 0.0 and ideal antiparallel geometry for exactly that reason.
SEVERE_CLASH = 3.5


def _place(a, b, c, bond, angle_deg, torsion_deg):
    """NeRF: position atom d given a-b-c and the internal coordinates c-d.

    The standard natural-extension-reference-frame construction: build the local
    frame at c from the b->c direction and the a-b-c plane, then place d in it.

    Written out in scalar arithmetic rather than numpy vector calls. This
    function is the hot spot of the whole scaffold stage -- the optimiser reaches
    it a few thousand times per bundle -- and on 3-vectors numpy's per-call
    overhead dominates the actual flops by an order of magnitude. The vectorised
    version of this routine was measured at ~20 us per call against ~2 us here,
    which was most of a 93-second `make_bundle`.
    """
    import math

    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    bx, by, bz = float(b[0]), float(b[1]), float(b[2])
    cx, cy, cz = float(c[0]), float(c[1]), float(c[2])

    angle = math.radians(angle_deg)
    tor = math.radians(torsion_deg)

    ux, uy, uz = cx - bx, cy - by, cz - bz
    un = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / un, uy / un, uz / un

    vx, vy, vz = bx - ax, by - ay, bz - az
    nx = vy * uz - vz * uy
    ny = vz * ux - vx * uz
    nz = vx * uy - vy * ux
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nn < 1e-8:                      # a, b, c collinear
        nx, ny, nz = uy, -ux, 0.0
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nn < 1e-8:
            nx, ny, nz, nn = 0.0, uz, -uy, math.sqrt(uz * uz + uy * uy)
    nx, ny, nz = nx / nn, ny / nn, nz / nn

    mx = ny * uz - nz * uy
    my = nz * ux - nx * uz
    mz = nx * uy - ny * ux

    sa = bond * math.sin(angle)
    d0 = -bond * math.cos(angle)
    d1 = sa * math.cos(tor)
    d2 = sa * math.sin(tor)

    return np.array([cx + d0 * ux + d1 * mx + d2 * nx,
                     cy + d0 * uy + d1 * my + d2 * ny,
                     cz + d0 * uz + d1 * mz + d2 * nz])


def build_backbone(phi, psi, omega=None):
    """Build N, CA, C, O coordinates from per-residue backbone torsions.

    `phi[i]` rotates about the N(i)-CA(i) bond and `psi[i]` about CA(i)-C(i);
    `omega[i]` is the C(i)-N(i+1) peptide torsion, trans unless told otherwise.
    phi[0] is unused (no preceding C), as is psi[-1]'s effect on a next residue.

    Returns an (L, 4, 3) array ordered N, CA, C, O -- the same layout
    ProteinMPNN's featuriser expects, which is why O is built here rather than
    left to the writer.
    """
    phi = np.asarray(phi, float)
    psi = np.asarray(psi, float)
    L = len(phi)
    if omega is None:
        omega = np.full(L, OMEGA_TRANS)
    omega = np.asarray(omega, float)

    N = np.zeros((L, 3))
    CA = np.zeros((L, 3))
    C = np.zeros((L, 3))

    # seed the first residue in a fixed frame
    N[0] = np.array([0.0, 0.0, 0.0])
    CA[0] = np.array([B_N_CA, 0.0, 0.0])
    ang = np.deg2rad(A_N_CA_C)
    C[0] = CA[0] + B_CA_C * np.array([-np.cos(ang), np.sin(ang), 0.0])

    for i in range(1, L):
        N[i] = _place(N[i - 1], CA[i - 1], C[i - 1], B_C_N, A_CA_C_N, psi[i - 1])
        CA[i] = _place(CA[i - 1], C[i - 1], N[i], B_N_CA, A_C_N_CA, omega[i - 1])
        C[i] = _place(C[i - 1], N[i], CA[i], B_CA_C, A_N_CA_C, phi[i])

    # carbonyl O: anti to the next N across the CA-C bond, i.e. psi + 180
    O = np.zeros((L, 3))
    for i in range(L):
        O[i] = _place(N[i], CA[i], C[i], B_C_O, A_CA_C_O, psi[i] + 180.0)

    return np.stack([N, CA, C, O], axis=1)


# ---------------------------------------------------------------- topology

_TEMPLATE_CACHE = {}


def helix_template(n):
    """An ideal `n`-residue alpha helix in a canonical frame, cached.

    Built by the same NeRF as everything else, just once per length.
    """
    if n not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[n] = build_backbone([HELIX_PHI] * n, [HELIX_PSI] * n)
    return _TEMPLATE_CACHE[n]


def _rigid_from_3(src, dst):
    """Rigid transform taking three source points onto three destination points.

    Kabsch. Exact here rather than least-squares, because the three atoms being
    matched (N, CA, C of a residue) have identical internal geometry in the
    template and in the target frame by construction.
    """
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cd - R @ cs


def build_bundle_coords(helix_lengths, loop_torsions, loop_lengths):
    """Build a helix-loop-...-helix backbone, rebuilding only the loops.

    The optimiser calls this tens of thousands of times, so the cost matters.
    Rebuilding the whole chain by NeRF on every objective evaluation is what the
    first version did, and for a 56-residue bundle with 16 free parameters it
    made a single `make_bundle` call take minutes: L-BFGS-B needs ~17 evaluations
    per iteration for its numerical gradient, each one walking 56 residues
    through three `_place` calls.

    Nothing about the helices changes between evaluations -- their torsions are
    fixed at ideal -- so the helices are built once by `helix_template` and then
    *placed* by a rigid transform. Only the 8-10 loop residues go through NeRF.
    Same coordinates, ~20x less work.

    Returns (L, 4, 3) in N, CA, C, O order.
    """
    loop_torsions = np.asarray(loop_torsions, float)
    blocks = [helix_template(helix_lengths[0]).copy()]
    k = 0

    for h, n_loop in enumerate(loop_lengths):
        prev = blocks[-1]
        a, b, c = prev[-1, 0], prev[-1, 1], prev[-1, 2]
        psi_cur = HELIX_PSI          # psi of the last helix residue
        loop = np.zeros((n_loop, 4, 3))

        for j in range(n_loop):
            phi_j, psi_j = loop_torsions[k], loop_torsions[k + 1]
            k += 2
            Ni = _place(a, b, c, B_C_N, A_CA_C_N, psi_cur)
            CAi = _place(b, c, Ni, B_N_CA, A_C_N_CA, OMEGA_TRANS)
            Ci = _place(c, Ni, CAi, B_CA_C, A_N_CA_C, phi_j)
            Oi = _place(Ni, CAi, Ci, B_C_O, A_CA_C_O, psi_j + 180.0)
            loop[j] = (Ni, CAi, Ci, Oi)
            a, b, c = Ni, CAi, Ci
            psi_cur = psi_j
        blocks.append(loop)

        # first residue of the next helix, then superpose its template onto it
        Nh = _place(a, b, c, B_C_N, A_CA_C_N, psi_cur)
        CAh = _place(b, c, Nh, B_N_CA, A_C_N_CA, OMEGA_TRANS)
        Ch = _place(c, Nh, CAh, B_CA_C, A_N_CA_C, HELIX_PHI)

        tmpl = helix_template(helix_lengths[h + 1])
        R, t = _rigid_from_3(tmpl[0, :3], np.array([Nh, CAh, Ch]))
        moved = (tmpl.reshape(-1, 3) @ R.T + t).reshape(tmpl.shape)
        blocks.append(moved)

    return np.concatenate(blocks, axis=0)


def bundle_torsions(helix_lengths, loop_torsions, loop_lengths):
    """Assemble full phi/psi arrays for a helix-loop-helix-...-helix chain.

    `loop_torsions` is a flat array of 2 * sum(loop_lengths) angles -- the only
    free parameters in the whole construction.
    """
    phi, psi, is_helix = [], [], []
    k = 0
    for h, hl in enumerate(helix_lengths):
        phi += [HELIX_PHI] * hl
        psi += [HELIX_PSI] * hl
        is_helix += [True] * hl
        if h < len(loop_lengths):
            n = loop_lengths[h]
            for _ in range(n):
                phi.append(loop_torsions[k]); psi.append(loop_torsions[k + 1])
                k += 2
            is_helix += [False] * n
    return np.array(phi), np.array(psi), np.array(is_helix)


def _helix_axes(ca, helix_lengths, loop_lengths, exact=False):
    """Centroid and unit axis of each helix segment.

    Two estimators. `exact=True` takes the first principal component of the
    segment's CAs, which is the right answer and is what the reported metrics
    use. The default is a cheap surrogate: the vector from the centroid of the
    first turn to the centroid of the last turn. Averaging over a turn cancels
    the helical wobble, so on an ideal helix the two agree to well under a
    degree -- and the surrogate costs no SVD, which matters because the
    optimiser evaluates this thousands of times per bundle.
    """
    axes, centres, spans = [], [], []
    i = 0
    for h, hl in enumerate(helix_lengths):
        seg = ca[i:i + hl]
        c = seg.mean(axis=0)
        if exact or hl < 8:
            _, _, vt = np.linalg.svd(seg - c)
            ax = vt[0]
        else:
            ax = seg[-4:].mean(axis=0) - seg[:4].mean(axis=0)
        n = np.linalg.norm(ax)
        ax = ax / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        # orient N->C so antiparallelism is measurable
        if np.dot(seg[-1] - seg[0], ax) < 0:
            ax = -ax
        axes.append(ax); centres.append(c); spans.append((i, i + hl))
        i += hl + (loop_lengths[h] if h < len(loop_lengths) else 0)
    return np.array(axes), np.array(centres), spans


def bundle_penalty(ca, helix_lengths, loop_lengths,
                   spacing=BUNDLE_SPACING, clash=CA_CLASH):
    """How far a conformation is from being a clean antiparallel bundle. Lower is better.

    Four terms, and the balance between them is the whole design of this module:

      * **spacing**     neighbouring helix axes should sit `spacing` apart.
                        Too close is a steric impossibility, too far is not a
                        bundle but two helices on a string.
      * **antiparallel** consecutive helices should run opposite ways, which is
                        what lets a short loop connect them without strain.
      * **clash**       hinge-loss on any CA pair under `clash`, excluding
                        near neighbours in sequence.
      * **compactness** radius of gyration against the empirical expectation
                        for a globular chain, 2.2 * L^0.38 A.

    The clash term is weighted hardest because a clashing backbone is not a
    scaffold at all, whereas imperfect spacing merely designs less well.
    """
    axes, centres, _ = _helix_axes(ca, helix_lengths, loop_lengths)
    n = len(axes)

    e_space = 0.0
    e_anti = 0.0
    for i in range(n - 1):
        d = np.linalg.norm(centres[i + 1] - centres[i])
        e_space += (d - spacing) ** 2
        # want dot = -1 for antiparallel
        e_anti += (np.dot(axes[i], axes[i + 1]) + 1.0) ** 2
    # non-adjacent helices in a 3-bundle should also be in contact
    for i in range(n):
        for j in range(i + 2, n):
            d = np.linalg.norm(centres[j] - centres[i])
            e_space += 0.5 * (d - spacing) ** 2

    D = cdist(ca, ca)
    L = len(ca)
    iu = np.triu_indices(L, k=4)        # ignore i..i+3, covalently constrained
    viol = np.clip(clash - D[iu], 0, None)
    e_clash = float(np.sum(viol ** 2))

    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    rg_target = 2.2 * L ** 0.38
    e_rg = (rg - rg_target) ** 2

    return 1.0 * e_space + 4.0 * e_anti + 20.0 * e_clash + 1.0 * e_rg


def make_bundle(helix_lengths=(16, 16, 16), loop_lengths=(4, 4), seed=0,
                n_screen=400, n_refine=3, maxiter=80, n_restarts=None):
    """Optimise loop torsions until the helices pack into a bundle.

    A cheap random screen followed by L-BFGS-B from the best few starts. Loop
    torsions are bounded to the broad allowed region of the Ramachandran plot
    rather than left free, so a converged solution is made of loop
    conformations that actually occur in proteins -- an unbounded optimiser will
    happily buy compactness with a left-handed bridge no sequence can encode.

    The screen-then-refine split is there to control the evaluation budget.
    L-BFGS-B has no gradient here, so each iteration costs ~2n objective calls
    for its finite differences; running it from a dozen random starts for 300
    iterations is ~50,000 calls and took 93 seconds per bundle. Screening
    `n_screen` random points costs one call each and locates the right basin for
    a fraction of that, after which only `n_refine` descents are needed --
    roughly 15x less work for the same final penalty.

    `n_restarts` is accepted as a deprecated alias for `n_refine`.

    Returns (coords (L,4,3), info dict) or (None, info) if nothing converged.
    """
    if n_restarts is not None:
        n_refine = n_restarts
    rng = np.random.default_rng(seed)
    n_loop_res = sum(loop_lengths)
    n_par = 2 * n_loop_res

    # (phi, psi) bounds per loop residue: generous beta/bridge/alpha region
    lo = np.tile([-170.0, -80.0], n_loop_res)
    hi = np.tile([-40.0, 170.0], n_loop_res)
    bounds = list(zip(lo, hi))

    def objective(x):
        ca = build_bundle_coords(helix_lengths, x, loop_lengths)[:, 1]
        return bundle_penalty(ca, helix_lengths, loop_lengths)

    screen = lo + rng.random((n_screen, n_par)) * (hi - lo)
    scores = np.array([objective(x) for x in screen])
    order = np.argsort(scores)[:max(1, n_refine)]

    best = (np.inf, None)
    for idx in order:
        try:
            res = minimize(objective, screen[idx], method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": maxiter})
        except Exception:
            continue
        if res.fun < best[0]:
            best = (float(res.fun), res.x)

    if best[1] is None:
        return None, {"converged": False}

    _, _, is_helix = bundle_torsions(helix_lengths, best[1], loop_lengths)
    coords = build_bundle_coords(helix_lengths, best[1], loop_lengths)
    ca = coords[:, 1]
    axes, centres, spans = _helix_axes(ca, helix_lengths, loop_lengths, exact=True)
    D = cdist(ca, ca)
    iu = np.triu_indices(len(ca), k=4)

    info = {
        "converged": True,
        "penalty": best[0],
        "length": int(len(ca)),
        "helix_lengths": list(helix_lengths),
        "loop_lengths": list(loop_lengths),
        "is_helix": is_helix.tolist(),
        "radius_of_gyration": round(float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean())), 2),
        "min_ca_ca": round(float(D[iu].min()), 2),
        "n_clashes": int(np.sum(D[iu] < CA_CLASH)),
        "n_severe_clashes": int(np.sum(D[iu] < SEVERE_CLASH)),
        "helix_axis_dots": [round(float(np.dot(axes[i], axes[i + 1])), 3)
                            for i in range(len(axes) - 1)],
        "helix_centre_dists": [round(float(np.linalg.norm(centres[i + 1] - centres[i])), 2)
                               for i in range(len(centres) - 1)],
        "helix_spans": [[int(a), int(b)] for a, b in spans],
    }
    return coords, info


#: The scaffold library the pipeline samples. Three-helix bundles dominate
#: because that is the workhorse minibinder topology; the two-helix hairpins are
#: there to give the docking stage something smaller for the narrow end of the
#: groove, and they fall in the <40 aa "microbinder" scoring category, which is
#: a separately-ranked prize in this challenge.
SCAFFOLD_SPECS = [
    # (helix_lengths, loop_lengths)
    ((14, 14, 14), (4, 4)),
    ((16, 16, 16), (4, 4)),
    ((18, 18, 18), (4, 4)),
    ((20, 20, 20), (5, 5)),
    ((22, 22, 22), (5, 5)),
    ((16, 20, 16), (4, 5)),
    ((20, 16, 20), (5, 4)),
    ((24, 24, 24), (5, 5)),
    ((16, 16), (4,)),
    ((20, 20), (4,)),
    ((13, 13), (3,)),
]


def generate_library(n_per_spec=3, specs=SCAFFOLD_SPECS, seed=0,
                     max_penalty=4000.0, n_refine=3):
    """Build a scaffold library: several independent bundles per topology.

    Returns a list of {coords, info}. Conformations above `max_penalty` are
    dropped -- they are the restarts that converged to a splayed or clashing
    local minimum, and passing them downstream wastes ProteinMPNN time on
    backbones no sequence can stabilise.
    """
    out = []
    k = 0
    for hl, ll in specs:
        for rep in range(n_per_spec):
            # n_refine, not n_restarts: this used to pass n_restarts=12, which
            # after the screen-then-refine rewrite aliased to twelve L-BFGS
            # descents instead of three and made the library build 4x slower
            # for no improvement in penalty.
            coords, info = make_bundle(hl, ll, seed=seed + k,
                                       n_refine=n_refine)
            k += 1
            if coords is None or info["penalty"] > max_penalty:
                continue
            if info["n_severe_clashes"] > 0:
                continue
            info["scaffold_id"] = f"s{len(out):03d}_h{len(hl)}_{info['length']}aa"
            info["seed"] = seed + k - 1
            out.append({"coords": coords, "info": info})
    return out


# ---------------------------------------------------------------- PDB output

_BB_NAMES = ("N", "CA", "C", "O")


def atom_line(serial, name, resname, chain, resseq, xyz, element=None,
              occupancy=1.0, bfactor=0.0):
    """One fixed-width PDB ATOM record, with the columns actually in the right place.

    Worth having in one place and getting right once. The PDB format is
    column-addressed, not whitespace-delimited:

        1-6  "ATOM  "      7-11 serial    13-16 atom name   17 altLoc
       18-20 resName         22 chainID   23-26 resSeq      27 iCode
       31-38 x            39-46 y         47-54 z
       55-60 occupancy    61-66 bFactor   77-78 element

    The first version of this writer ran the residue name straight on from the
    atom name, omitting the altLoc column at 17. Every field after it shifted
    left by one, which is still *readable* -- coordinates parse fine -- so the
    files looked correct and ProteinMPNN simply reported no chain Z. A one-column
    error that does not corrupt the geometry is exactly the kind that survives a
    visual check, so this is now the only place the format is written out.

    Atom names also follow the convention that a short name is indented one
    column, giving " CA " and " N  ", because the element symbol is
    right-justified in columns 13-14.
    """
    nm = name if len(name) >= 4 else f" {name:<3}"
    el = (element or name[0])[:2]
    x, y, z = xyz
    return (f"ATOM  {serial:>5} {nm}{'':1}{resname:>3} {chain}{resseq:>4}{'':1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{bfactor:>6.2f}"
            f"{'':10}{el:>2}")


def write_pdb(coords, dest, chain_id="A", sequence=None, start=1):
    """Write an (L, 4, 3) backbone as a PDB file.

    A poly-glycine backbone unless `sequence` is given. ProteinMPNN reads
    residue identity only to report the native sequence, so glycine is a
    deliberately honest placeholder for "no sequence yet" -- it has no side
    chain to imply one.
    """
    from pathlib import Path

    coords = np.asarray(coords, float)
    L = len(coords)
    seq3 = (["GLY"] * L if sequence is None
            else [_ONE_TO_THREE.get(a, "GLY") for a in sequence])

    lines, serial = [], 1
    for i in range(L):
        for j, name in enumerate(_BB_NAMES):
            lines.append(atom_line(serial, name, seq3[i], chain_id,
                                   start + i, coords[i, j]))
            serial += 1
    lines.append("TER")
    lines.append("END")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return dest


_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

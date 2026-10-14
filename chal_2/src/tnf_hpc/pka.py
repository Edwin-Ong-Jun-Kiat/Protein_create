"""
Interface histidine pKa estimation -- running the opposite way to challenge 1.

Challenge 1 wanted a histidine whose pKa goes **up** on binding, so it hunted
carboxylates and rewarded salt bridges. Challenge 2 needs the bound-state pKa to
go **down** (`linkage`), so every sign in the model below is flipped relative to
that project's `pka.py`:

    environment feature            chal_1 delta    chal_2 delta    why
    ---------------------------    ------------    ------------    -------------
    target carboxylate <= 4 A          +1.5           +1.5         stabilises His+
    target cation      <= 5 A          -0.7           -0.7         repels His+
    buried, no partner                 -1.0           -1.0         lone charge costs

The *deltas are the same physics* -- that is the point. What changed is which
sign is desirable. Challenge 1 maximised the delta; challenge 2 minimises it.
So the same function serves both, and the difference lives entirely in the
scoring, which is `his_score` below and `design.composite_score`.

Two consequences worth stating, because both cost design effort:

1. **A carboxylate near a binder histidine is now actively harmful.** In
   challenge 1 it was the prize. Here it pushes the switch backwards, and
   `his_score` returns a negative S for it rather than merely zero.

2. **The free state matters too.** `linkage.selectivity` takes a per-group
   pka_free, and raising it is a second lever worth as much as suppressing the
   bound state (ceiling 0.50 -> 0.90 going from pKa_free 6.4 to 7.0). The free
   state is the *unbound binder*, so this is a property of the binder's own
   surface, which `estimate_pka_free` measures on the binder alone.

Geometry on existing coordinates. CPU-only, instant. Not Poisson-Boltzmann --
these rank designs, and the absolute numbers are not predictions.
"""

import numpy as np

from . import linkage as L

# -------------------------------------------------- pKa environment model

PKA_MIN, PKA_MAX = 4.0, 9.0

HIS_N = ("ND1", "NE2")
HIS_RING = ("ND1", "NE2", "CE1", "CD2", "CG")

CARBOXYL_O = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
POLAR_O = {"SER": ("OG",), "THR": ("OG1",),
           "ASN": ("OD1",), "GLN": ("OE1",)}
CATION_ATOMS = {"ARG": ("NE", "NH1", "NH2"), "LYS": ("NZ",),
                "HIS": ("ND1", "NE2")}

#: Delta-pKa contributions. Magnitudes follow challenge 1's calibrated table so
#: the two projects stay comparable; only their desirability is reversed.
D_SALT_BRIDGE = 1.5       # carboxylate O within 4.0 A  -- BAD here
D_PROXIMAL = 0.8          # carboxylate O at 4.0-6.0 A  -- BAD here
D_HBOND = 0.3             # neutral H-bond acceptor within 3.5 A
D_CATION = -0.7           # Arg/Lys within 5.0 A        -- GOOD here
D_BURIED_NO_PARTNER = -1.0  # buried lone charge        -- GOOD here

R_SALT_BRIDGE = 4.0
R_PROXIMAL = 6.0
R_HBOND = 3.5
R_CATION = 5.0
BURIED_RSA = 0.15

#: The imidazole ring must be within this of any target atom for the histidine
#: to be treated as interfacial at all. As in challenge 1 this is looser than
#: the contact cutoff: a histidine can be pH-relevant from further out than it
#: counts as "in contact".
HIS_INTERFACE_CUTOFF = 8.0
BURIED_CONTACT = 4.5

#: Two histidines closer than this in space are not independent -- they compete
#: for the same partner and repel each other, so the linkage equation's
#: independent-sites assumption fails. See `independent_subset`.
HIS_MIN_SEPARATION_A = 7.0


def estimate_pka_bound(min_acid_d, min_base_d, min_polar_d, buried,
                       pka_free=L.HIS_PKA_FREE):
    """Four distances in, one bound-state pKa out. Same form as challenge 1's.

    Structure worth noting, because it is load-bearing:

      * the carboxylate term is if/elif -- a histidine gets credit for a salt
        bridge OR second-shell proximity, never both;
      * the burial penalty only applies with no acidic partner within 6 A,
        because burying a charge *next to* a carboxylate is fine -- it is
        burying a lone charge that costs;
      * clamped to [4.0, 9.0], so stacked bonuses cannot produce an absurd pKa.

    For challenge 2 the desirable output is a LOW number. A design where this
    returns 6.4 (no shift) has no pH switch; one where it returns 4.4 is at 98%
    of the achievable per-histidine selectivity.
    """
    pka = pka_free
    if min_acid_d <= R_SALT_BRIDGE:
        pka += D_SALT_BRIDGE
    elif min_acid_d <= R_PROXIMAL:
        pka += D_PROXIMAL
    if min_polar_d <= R_HBOND:
        pka += D_HBOND
    if min_base_d <= R_CATION:
        pka += D_CATION
    if buried and min_acid_d > R_PROXIMAL:
        pka += D_BURIED_NO_PARTNER
    return float(np.clip(pka, PKA_MIN, PKA_MAX))


def estimate_pka_free(his_res, binder_model, binder_rsa=None,
                      pka_free=L.HIS_PKA_FREE):
    """Bound-state model applied to the UNBOUND binder: the second lever.

    `linkage.ceiling_one_group` rises from 0.50 to 0.90 as pKa_free goes from
    6.4 to 7.0, so a histidine sitting next to one of the *binder's own*
    carboxylates while the binder is free starts from a higher pKa and has more
    to lose on binding. Challenge 1 never needed this -- it assumed a constant
    free-state 6.4 -- but here it is close to free selectivity, because the
    binder's own surface is entirely ours to design.

    The same geometric model is reused deliberately: it is the same physics on
    the same kind of neighbours, just measured against the binder instead of the
    target.
    """
    env = measure_his_environment(his_res, binder_model,
                                  exclude_self=True, rsa=binder_rsa)
    if env is None:
        return pka_free, {"resnum": his_res.id[1], "note": "no environment"}
    pka = estimate_pka_bound(env["min_acid_d"], env["min_base_d"],
                             env["min_polar_d"], env["buried"], pka_free)
    return pka, env


def measure_his_environment(his_res, context, exclude_self=False, rsa=None,
                            transform=None):
    """The four distances the pKa model needs, measured against `context`.

    `context` is a Bio.PDB Model or Chain. With `exclude_self=True` the
    histidine's own residue is skipped, which is what makes this usable on the
    binder alone for the free-state estimate.

    Returns None if the ring is further than `HIS_INTERFACE_CUTOFF` from the
    context, i.e. the histidine is not at the interface.
    """
    def xf(a):
        a = np.asarray(a, float)
        return a if transform is None else transform(a)

    ring = [his_res[a].coord for a in HIS_RING if a in his_res]
    if not ring:
        return None
    ring_t = xf(np.array(ring))
    n_atoms = [his_res[a].coord for a in HIS_N if a in his_res]
    charge_pts = xf(np.array(n_atoms)) if n_atoms else ring_t

    # gather context atoms, optionally skipping the histidine itself
    def residues_of(ctx):
        for item in ctx:
            if hasattr(item, "get_resname"):
                yield item
            else:
                for r in item:
                    yield r

    res_list = [r for r in residues_of(context)
                if not (exclude_self and r is his_res)]
    if not res_list:
        return None

    all_t = np.array([a.coord for r in res_list for a in r], float)

    def collect(table):
        pts = []
        for r in res_list:
            for n in table.get(r.get_resname().upper(), ()):
                if n in r:
                    pts.append(np.asarray(r[n].coord, float))
        return np.array(pts) if pts else np.array([[1e3, 1e3, 1e3]])

    acid = collect(CARBOXYL_O)
    base = collect(CATION_ATOMS)
    polar_pts = [p for p in collect(POLAR_O)]
    polar_pts += [np.asarray(r["O"].coord, float) for r in res_list if "O" in r]
    polar = np.array(polar_pts) if polar_pts else np.array([[1e3, 1e3, 1e3]])

    def mind(pts, arr):
        return float(np.min(np.linalg.norm(
            arr[None, :, :] - pts[:, None, :], axis=-1)))

    ring_to_ctx = mind(ring_t, all_t)
    if ring_to_ctx > HIS_INTERFACE_CUTOFF:
        return None

    r_sasa = None if rsa is None else rsa.get(his_res.id[1])
    buried = (mind(charge_pts[:1], all_t) < BURIED_CONTACT
              if r_sasa is None else bool(r_sasa < BURIED_RSA))

    return {
        "resnum": his_res.id[1],
        "min_acid_d": round(mind(charge_pts, acid), 2),
        "min_base_d": round(mind(charge_pts, base), 2),
        "min_polar_d": round(mind(charge_pts, polar), 2),
        "buried": bool(buried),
        "rel_sasa": r_sasa,
        "ring_to_context": round(ring_to_ctx, 2),
    }


def his_score(env_bound, env_free=None, pka_free=L.HIS_PKA_FREE):
    """Per-histidine selectivity contribution, with its reasoning attached.

    Returns a record carrying both pKa estimates, the resulting S_i, and a
    human-readable verdict. A *negative* S_i is possible and is not a bug: it is
    what a binder histidine salt-bridged to a target carboxylate earns, because
    that design binds *tighter* at pH 6.0 -- it is challenge 1's winning motif
    scoring as a defect here.
    """
    pka_b = estimate_pka_bound(env_bound["min_acid_d"], env_bound["min_base_d"],
                               env_bound["min_polar_d"], env_bound["buried"],
                               pka_free)
    pka_f = pka_free
    if env_free is not None:
        pka_f = estimate_pka_bound(env_free["min_acid_d"], env_free["min_base_d"],
                                   env_free["min_polar_d"], env_free["buried"],
                                   pka_free)

    s_i = float(L.selectivity([pka_b], [pka_f]))

    reasons = []
    if env_bound["min_acid_d"] <= R_SALT_BRIDGE:
        reasons.append(f"target carboxylate at {env_bound['min_acid_d']} A "
                       f"RAISES pKa (wrong direction)")
    elif env_bound["min_acid_d"] <= R_PROXIMAL:
        reasons.append(f"target carboxylate at {env_bound['min_acid_d']} A "
                       f"(second shell, wrong direction)")
    if env_bound["min_base_d"] <= R_CATION:
        reasons.append(f"target cation at {env_bound['min_base_d']} A suppresses pKa")
    if env_bound["buried"] and env_bound["min_acid_d"] > R_PROXIMAL:
        reasons.append("buried with no acidic partner, suppresses pKa")
    if not reasons:
        reasons.append("no pKa-shifting neighbour: contributes nothing")

    return {
        "resnum": env_bound["resnum"],
        "pka_free": round(pka_f, 2),
        "pka_bound": round(pka_b, 2),
        "delta_pka": round(pka_b - pka_f, 2),
        "S_i": round(s_i, 4),
        "fold_i": round(float(L.kd_ratio(s_i)), 2),
        "frac_of_ceiling": round(s_i / L.ceiling_one_group(pka_f), 3)
                           if L.ceiling_one_group(pka_f) > 0 else 0.0,
        "min_acid_d": env_bound["min_acid_d"],
        "min_base_d": env_bound["min_base_d"],
        "buried": env_bound["buried"],
        "reasons": reasons,
    }


def independent_subset(his_records, coords_by_resnum,
                       min_sep=HIS_MIN_SEPARATION_A):
    """Drop histidines too close together to be treated as independent sites.

    `linkage.selectivity` sums per-group contributions, which is only valid for
    non-interacting groups. The docking stage routinely produces three or four
    reachable positions clustered on one helix face around a single arginine --
    an early run had binder positions 19, 20, 21, 22 all within reach of the
    same Arg32. Summing those as four independent switches overstates S by
    roughly the number of duplicates.

    Greedy: keep the strongest, then any that is `min_sep` away from everything
    already kept. Returns (kept, dropped).
    """
    ranked = sorted(his_records, key=lambda r: -r["S_i"])
    kept, dropped = [], []
    for r in ranked:
        p = coords_by_resnum.get(r["resnum"])
        if p is None:
            dropped.append({**r, "dropped_because": "no coordinate"})
            continue
        clash = None
        for k in kept:
            q = coords_by_resnum.get(k["resnum"])
            if q is None:
                continue
            d = float(np.linalg.norm(np.asarray(p) - np.asarray(q)))
            if d < min_sep:
                clash = (k["resnum"], round(d, 2))
                break
        if clash is None:
            kept.append(r)
        else:
            dropped.append({**r, "dropped_because":
                            f"{clash[1]} A from His{clash[0]} (< {min_sep})"})
    return kept, dropped


# -------------------------------------------------- complex-level assessment

def score_complex(binder_chain, target_context, binder_rsa=None,
                  pka_free=L.HIS_PKA_FREE, enforce_independence=True):
    """Full reverse-polarity pH assessment of a binder/target complex.

    Returns per-histidine records plus the total S = log10(Kd(6.0)/Kd(7.4)).
    Only histidines that are interfacial by the ring test contribute; the free
    state is measured on the binder alone, so both of `linkage`'s levers are in
    play.
    """
    coords = {}
    his_records = []
    for res in binder_chain:
        if res.get_resname().upper() != "HIS":
            continue
        env_b = measure_his_environment(res, target_context, rsa=None)
        if env_b is None:
            continue
        _, env_f = estimate_pka_free(res, binder_chain, binder_rsa, pka_free)
        env_f = env_f if isinstance(env_f, dict) and "min_acid_d" in env_f else None
        rec = his_score(env_b, env_f, pka_free)
        his_records.append(rec)
        ref = "CB" if "CB" in res else "CA"
        if ref in res:
            coords[res.id[1]] = np.asarray(res[ref].coord, float)

    dropped = []
    used = his_records
    if enforce_independence:
        used, dropped = independent_subset(his_records, coords)

    S = float(np.sum([r["S_i"] for r in used])) if used else 0.0
    return {
        "n_interface_his": len(his_records),
        "n_independent_his": len(used),
        "his": his_records,
        "his_used": used,
        "his_dropped": dropped,
        "S": round(S, 4),
        "kd_fold": round(float(L.kd_ratio(S)), 2),
        "S_ceiling_for_n": round(len(used) * L.ceiling_one_group(pka_free), 3),
        "meets_target": S >= L.S_TARGET,
    }


def mouse_crossreactivity(binder_chain, target_context, identity, offset=0,
                          cutoff=5.0):
    """Fraction of contacted TARGET residues that are identical in mouse.

    Criterion 2. Measured on the actual docked pose rather than on the epitope
    definition, because a binder can sit on a well-conserved groove and still
    put most of its contacts on the few divergent positions in it -- the TNF
    receptor site has eleven (20, 30, 31, 53, 71, 73, 85, 89, 97, 111, 138).
    """
    b = np.array([a.coord for r in binder_chain for a in r
                  if a.element != "H"], float)
    contacted = []
    for item in target_context:
        residues = [item] if hasattr(item, "get_resname") else list(item)
        for res in residues:
            t = np.array([a.coord for a in res if a.element != "H"], float)
            if t.size and np.linalg.norm(
                    t[:, None, :] - b[None, :, :], axis=-1).min() <= cutoff:
                contacted.append(res.id[1] - offset)
    known = [identity[p][0] for p in contacted if p in identity]
    return {
        "n_contacted": len(contacted),
        "contacted_mature_pos": sorted(set(contacted)),
        "mouse_identity": round((sum(known) / len(known)) if known else 0.0, 4),
        "divergent_contacts": sorted({p for p in contacted
                                      if p in identity and not identity[p][0]}),
    }


# -------------------------------------------------- coordinate-based path
#
# ProteinMPNN returns a sequence on a backbone: there are no side chains to
# measure. The Bio.PDB functions above work on structures that have them (a
# crystal complex, or a repacked model), but the designs this pipeline produces
# do not, so the histidine imidazole has to be built before its environment can
# be measured at all.
#
# The alternative -- measuring everything from CB with thresholds widened by a
# side chain's reach -- was the first approach here and it is too blunt to
# distinguish the cases that matter: a cation 5 A from CB may be 2 A or 9 A from
# ND1 depending only on chi1, and those are a strong suppression and none.
# Building an explicit rotamer and then measuring is both more honest and no
# harder.

#: Histidine side-chain internal coordinates (bond, angle, torsion-source).
#: Approximate ideal geometry; good to ~0.1 A, which is far finer than the
#: distance thresholds it feeds.
_HIS_GEOM = {
    "CB":  (1.530, 110.5),   # from N, CA, C
    "CG":  (1.497, 113.8),   # chi1 about N-CA-CB
    "ND1": (1.378, 122.7),   # chi2 about CA-CB-CG
    "CD2": (1.354, 131.0),   # chi2 + 180
    "CE1": (1.321, 109.0),   # ring closure, planar
    "NE2": (1.374, 107.2),   # ring closure, planar
}

#: chi1 / chi2 pairs covering the populated histidine rotamers. Twelve
#: combinations is enough to answer "can the imidazole reach a cation" without
#: pretending to do a packing calculation.
HIS_ROTAMERS = [(c1, c2) for c1 in (-60.0, 180.0, 60.0)
                for c2 in (-90.0, -60.0, 60.0, 90.0)]


def build_his_sidechain(N, CA, C, chi1, chi2):
    """Build CB, CG, ND1, CD2, CE1, NE2 for one histidine rotamer.

    Returns a dict of coordinates. Uses the same NeRF placement as the scaffold
    builder, so the geometry is consistent with how the backbone was made.
    """
    from .scaffold import _place

    N = np.asarray(N, float); CA = np.asarray(CA, float); C = np.asarray(C, float)
    b, a = _HIS_GEOM["CB"]
    # CB is placed off the backbone frame; the -122.5 torsion from C puts it in
    # the standard L-amino-acid position
    CB = _place(C, N, CA, b, a, -122.5)
    b, a = _HIS_GEOM["CG"]
    CG = _place(N, CA, CB, b, a, chi1)
    b, a = _HIS_GEOM["ND1"]
    ND1 = _place(CA, CB, CG, b, a, chi2)
    b, a = _HIS_GEOM["CD2"]
    CD2 = _place(CA, CB, CG, b, a, chi2 + 180.0)
    b, a = _HIS_GEOM["CE1"]
    CE1 = _place(CB, CG, ND1, b, a, 180.0)
    b, a = _HIS_GEOM["NE2"]
    NE2 = _place(CB, CG, CD2, b, a, 180.0)
    return {"CB": CB, "CG": CG, "ND1": ND1, "CD2": CD2,
            "CE1": CE1, "NE2": NE2}


def _mind(pts, arr):
    if len(arr) == 0 or len(pts) == 0:
        return 1e3
    return float(np.min(np.linalg.norm(
        np.asarray(arr)[None, :, :] - np.asarray(pts)[:, None, :], axis=-1)))


def best_his_rotamer(backbone_i, target, binder_heavy=None,
                     clash_cut=2.8, pka_free=L.HIS_PKA_FREE):
    """Pick the histidine rotamer that best suppresses the bound-state pKa.

    `target` is a dict of (N,3) arrays with keys `all`, `acid`, `base`, `polar`.
    Rotamers that clash with the target or with the binder's own backbone are
    discarded; among the rest the one with the LOWEST estimated bound pKa wins,
    because that is the design objective (`linkage`).

    Choosing the most favourable reachable rotamer is an optimistic assumption
    and is stated as such: it answers "is there a rotamer that would do this",
    not "this is the rotamer the protein adopts". Challenge 1 made the same
    choice and its README said the same thing. Without side-chain repacking
    there is no better answer available, and the pessimistic alternative --
    taking a single arbitrary rotamer -- is not more honest, just noisier.

    Returns (env, record) or (None, reason).
    """
    N, CA, C = backbone_i[0], backbone_i[1], backbone_i[2]
    best = None
    n_clash = 0
    for chi1, chi2 in HIS_ROTAMERS:
        sc = build_his_sidechain(N, CA, C, chi1, chi2)
        ring = np.array([sc[k] for k in ("ND1", "NE2", "CE1", "CD2", "CG")])
        charge = np.array([sc["ND1"], sc["NE2"]])

        if _mind(ring, target["all"]) < clash_cut:
            n_clash += 1
            continue
        if binder_heavy is not None and len(binder_heavy):
            # exclude this residue's own backbone from the self-clash test
            if _mind(ring, binder_heavy) < clash_cut:
                n_clash += 1
                continue

        ring_to_target = _mind(ring, target["all"])
        if ring_to_target > HIS_INTERFACE_CUTOFF:
            continue

        env = {
            "chi1": chi1, "chi2": chi2,
            "min_acid_d": round(_mind(charge, target["acid"]), 2),
            "min_base_d": round(_mind(charge, target["base"]), 2),
            "min_polar_d": round(_mind(charge, target["polar"]), 2),
            "buried": bool(_mind(charge[:1], target["all"]) < BURIED_CONTACT),
            "ring_to_context": round(ring_to_target, 2),
        }
        pka_b = estimate_pka_bound(env["min_acid_d"], env["min_base_d"],
                                   env["min_polar_d"], env["buried"], pka_free)
        if best is None or pka_b < best[0]:
            best = (pka_b, env)

    if best is None:
        return None, f"no interfacial non-clashing rotamer ({n_clash} clashed)"
    return best[1], {"n_rotamers_clashing": n_clash}


#: A histidine's CB sits ~3.2 A from its imidazole nitrogens and an Asp/Glu CB
#: ~3.0 A from its carboxylate oxygens, so a CB-CB distance overstates the
#: functional-group separation by roughly this much.
REACH_CB_PAIR = 6.0


def binder_free_environment(binder_coords, sequence, i, pka_free=L.HIS_PKA_FREE):
    """The FREE-state environment of binder histidine `i`, from the binder alone.

    This is `linkage`'s second lever measured rather than assumed. The binder's
    own Asp/Glu are what raise pKa_free, and `mpnn.acid_pair_bias` puts them
    there deliberately, so reading the free state off a constant 6.4 would
    discard the effect the design stage worked for.

    CB-to-CB between the histidine and the binder's own charged residues, with
    the separation corrected by `REACH_CB_PAIR` for the two side chains'
    reach. Deliberately coarser than the bound-state treatment: it only has to
    separate "there is a carboxylate adjacent" from "there is not".

    **Two things this must NOT do**, both learned by getting them wrong:

    1. **No burial term.** The first version reused `best_his_rotamer` against
       the binder's own backbone, and `estimate_pka_bound` applies -1.0 for a
       buried histidine with no acidic partner. Every residue is within 4.5 A
       of its own backbone, so that penalty fired on essentially every
       histidine and drove pKa_free *down* to ~4.7 -- the exact opposite of the
       lever this function exists to measure. With the bound state unshifted at
       6.4, S_i came out at **-0.484** for a histidine that should simply have
       contributed zero, and 13 designs in a 1,056-design run were reported as
       binding *tighter* at pH 6.0 purely as an artefact. Burial cannot be
       assessed from a backbone without side chains or SASA, so it is not
       assessed.

    2. **No rotamer optimisation.** `best_his_rotamer` returns the *lowest*-pKa
       rotamer, which is the optimistic choice for the bound state and the
       pessimistic one for the free state. Applying it here biased pKa_free
       downward a second time. At CB-CB resolution the rotamer is not
       determined anyway, so this works from CB positions and no rotamer is
       chosen.

    The remaining asymmetry is intentional and in the safe direction: the bound
    state takes its most favourable reachable rotamer, while the free state is
    estimated without optimisation, so the lever is credited conservatively.
    """
    from .dock import pseudo_cb

    cb = pseudo_cb(binder_coords)
    here = cb[i]

    def nearest(letters):
        idx = [j for j, a in enumerate(sequence) if a in letters and j != i]
        if not idx:
            return 1e3
        d = float(np.min(np.linalg.norm(cb[idx] - here, axis=1)))
        return max(0.0, d - REACH_CB_PAIR)

    return {
        "resnum": i + 1,
        "min_acid_d": round(nearest(("D", "E")), 2),
        "min_base_d": round(nearest(("K", "R", "H")), 2),
        "min_polar_d": round(nearest(("S", "T", "N", "Q")), 2),
        "buried": False,          # see point 1 above
        "ring_to_context": None,
    }


def backbone_heavy(coords, exclude=None):
    """All backbone atoms as an (N,3) array, optionally dropping one residue."""
    coords = np.asarray(coords, float)
    keep = [i for i in range(len(coords)) if i != exclude]
    return coords[keep].reshape(-1, 3)


def target_atom_arrays(model, chain_ids):
    """Split a target model into the atom classes the pKa model needs.

    Returns {all, acid, base, polar} as (N,3) arrays.
    """
    allp, acid, base, polar = [], [], [], []
    for ch in model:
        if ch.id not in set(chain_ids):
            continue
        for res in ch:
            rn = res.get_resname().upper()
            for atom in res:
                if atom.element == "H":
                    continue
                allp.append(atom.coord)
            for n in CARBOXYL_O.get(rn, ()):
                if n in res:
                    acid.append(res[n].coord)
            for n in CATION_ATOMS.get(rn, ()):
                if n in res:
                    base.append(res[n].coord)
            for n in POLAR_O.get(rn, ()):
                if n in res:
                    polar.append(res[n].coord)
            if "O" in res:
                polar.append(res["O"].coord)
    f = lambda v: np.array(v, float) if v else np.zeros((0, 3))
    return {"all": f(allp), "acid": f(acid), "base": f(base), "polar": f(polar)}


def score_design(binder_coords, sequence, target, pka_free=L.HIS_PKA_FREE,
                 use_free_lever=True, enforce_independence=True):
    """Full reverse-polarity pH assessment of a designed (backbone, sequence) pair.

    The function `pipeline.stage_score` actually calls. Returns the same shape of
    result as `score_complex` so the two are interchangeable in reporting.
    """
    from .dock import pseudo_cb

    cb = pseudo_cb(binder_coords)
    records, coords = [], {}

    for i, aa in enumerate(sequence):
        if aa != "H":
            continue
        env_b, _ = best_his_rotamer(
            binder_coords[i], target,
            binder_heavy=backbone_heavy(binder_coords, exclude=i),
            pka_free=pka_free)
        if env_b is None:
            continue
        # his_score reports env["resnum"]; on this path the identifier is the
        # 1-based binder position, so set it before scoring rather than
        # patching the records afterwards.
        env_b["resnum"] = i + 1
        env_f = (binder_free_environment(binder_coords, sequence, i, pka_free)
                 if use_free_lever else None)
        rec = his_score(env_b, env_f, pka_free)
        rec["binder_pos"] = i + 1
        rec["chi"] = (env_b["chi1"], env_b["chi2"])
        records.append(rec)
        coords[i + 1] = cb[i]

    used, dropped = records, []
    if enforce_independence:
        used, dropped = independent_subset(records, coords)

    S = float(np.sum([r["S_i"] for r in used])) if used else 0.0
    return {
        "n_interface_his": len(records),
        "n_independent_his": len(used),
        "n_his_total": sequence.count("H"),
        "his": records,
        "his_used": used,
        "his_dropped": dropped,
        "S": round(S, 4),
        "kd_fold": round(float(L.kd_ratio(S)), 2),
        "S_ceiling_default": round(len(used) * L.ceiling_one_group(pka_free), 3),
        "meets_target": S >= L.S_TARGET,
    }

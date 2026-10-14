"""
Rigid-body docking of a scaffold into the inter-protomer groove.

This replaces RFdiffusion's hotspot conditioning. RFdiffusion grows a backbone
*around* a specified epitope, so placement and shape come out together; with a
fixed scaffold library the two have to be separated, and this module does the
placement half: find the rotation and translation that lay a given bundle into
the TNF-alpha A/B groove.

What is being optimised is not just contact. A pose is only useful here if it
simultaneously:

  * **straddles both protomers** -- otherwise it is not blocking the receptor,
    which is the whole point of the chosen epitope (`target` docstring, point 2);
  * **puts backbone positions within histidine reach of the conserved cations**
    R32 / R82 / K90 / K128, because those are the only places a bound-state pKa
    can be suppressed, and without them there is no pH switch at all
    (`linkage.ceiling_one_group`);
  * **does not clash**, and buries enough surface to have some affinity.

So `pose_score` is a weighted sum with a hard clash gate, and the cation term is
weighted comparably to the contact term rather than treated as a bonus. A pose
that buries 900 A^2 with no cation in reach is worth less here than a smaller
one with four, which is the opposite of how a normal docking score behaves and
is a direct consequence of the 4-histidine requirement.

Backbone-only, CB-level scoring. Side chains do not exist yet -- `mpnn` has not
run -- so anything finer than this would be false precision. scipy cKDTree,
no GPU, a few seconds per scaffold.
"""

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- geometry

#: Ideal virtual-CB offset from backbone N, CA, C. The standard coefficients
#: used across the design field; reproduces a real CB to ~0.1 A.
_CB_A, _CB_B, _CB_C = -0.58273431, 0.56802827, -0.54067466

#: Two residues' CB atoms closer than this are clashing.
CB_CLASH = 3.6
#: CB-CB within this counts as a residue-residue contact.
CB_CONTACT = 8.0
#: A histidine side chain reaches ~4 A from CB to NE2, so a binder CB this close
#: to a target cation's charged nitrogen could plausibly suppress its pKa after
#: repacking. Deliberately generous -- it answers "is it in reach", not "will
#: this rotamer form". Same logic as challenge 1's CB_SALT_BRIDGE, opposite sign.
CB_CATION_REACH = 9.0


def pseudo_cb(coords):
    """Virtual CB from (L, 4, 3) backbone N, CA, C, O coordinates."""
    coords = np.asarray(coords, float)
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    return _CB_A * a + _CB_B * b + _CB_C * c + CA


def random_rotations(n, seed=0):
    """`n` uniformly distributed rotation matrices (Shoemake's method).

    Uniform over SO(3), not over Euler angles -- sampling Euler angles
    independently clusters poses near the poles and would under-sample exactly
    the side-on approaches that lay a bundle flat into a groove.
    """
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q = np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=1)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------- the site

class DockSite:
    """The groove a binder is aimed at, and everything needed to score a pose.

    Holds the target's CB cloud, the hotspot centroid, an outward approach
    normal, and the coordinates of the conserved cations' charged nitrogens --
    which are the pH-switch anchors, so they get their own KD-tree.
    """

    #: Charged side-chain nitrogens that can suppress a histidine's pKa.
    CATION_ATOMS = {"ARG": ("NE", "NH1", "NH2"), "LYS": ("NZ",),
                    "HIS": ("ND1", "NE2")}
    #: Carboxylate oxygens, which RAISE a histidine pKa and so run the switch
    #: backwards. Tracked so `pose_score` can penalise putting the binder's
    #: switch face against them.
    ACID_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}

    def __init__(self, model, hotspots, chain_ids, identity=None, offset=0):
        self.chain_ids = list(chain_ids)
        self.hotspots = list(hotspots)

        cb, labels = [], []
        for ch in model:
            if ch.id not in set(chain_ids):
                continue
            for res in ch:
                atom = "CB" if "CB" in res else ("CA" if "CA" in res else None)
                if atom is None:
                    continue
                cb.append(np.asarray(res[atom].coord, float))
                labels.append((ch.id, res.id[1]))
        self.target_cb = np.array(cb)
        self.target_labels = labels
        self.cb_tree = cKDTree(self.target_cb)
        self.chain_of = np.array([l[0] for l in labels])
        # Per-target-residue mouse conservation, aligned to `target_cb` order,
        # so `score_pose` can read it by KD-tree index. None where the position
        # has no mouse counterpart in the alignment.
        if identity is None:
            self.conserved_mask = None
        else:
            self.conserved_mask = [
                (identity[p - offset][0] if (p - offset) in identity else None)
                for _, p in labels]

        # all heavy atoms, for a stricter clash check on the final poses
        heavy = [np.asarray(a.coord, float) for ch in model
                 if ch.id in set(chain_ids) for res in ch for a in res
                 if a.element != "H"]
        self.target_heavy = np.array(heavy)
        self.heavy_tree = cKDTree(self.target_heavy)

        # conserved cations = the only usable pKa suppressors
        cat, cat_lab, acid = [], [], []
        for ch in model:
            if ch.id not in set(chain_ids):
                continue
            for res in ch:
                rn = res.get_resname().upper()
                mature = res.id[1] - offset
                conserved = True
                if identity is not None:
                    same, _ = identity.get(mature, (None, None))
                    conserved = bool(same)
                for n in self.CATION_ATOMS.get(rn, ()):
                    if n in res and conserved:
                        cat.append(np.asarray(res[n].coord, float))
                        cat_lab.append((ch.id, mature, rn))
                for n in self.ACID_ATOMS.get(rn, ()):
                    if n in res and conserved:
                        acid.append(np.asarray(res[n].coord, float))
        self.cation_xyz = np.array(cat) if cat else np.zeros((0, 3))
        self.cation_labels = cat_lab
        self.acid_xyz = np.array(acid) if acid else np.zeros((0, 3))
        self.cation_tree = cKDTree(self.cation_xyz) if len(cat) else None
        self.acid_tree = cKDTree(self.acid_xyz) if len(acid) else None

        # hotspot centroid and an outward normal
        hs_idx = [i for i, l in enumerate(labels)
                  if (l[0], l[1] - offset) in set(hotspots)]
        if not hs_idx:
            raise ValueError("no hotspot residues found in the model")
        self.hotspot_xyz = self.target_cb[hs_idx]
        self.centre = self.hotspot_xyz.mean(axis=0)
        self.normal = self._outward_normal(model)

    def _outward_normal(self, model):
        """Unit vector pointing from the protein body out through the hotspots.

        Taken as hotspot centroid minus the centroid of the whole trimer, which
        for a groove on the outside of a compact trimer points into solvent. The
        docking search also samples rotations of the approach direction, so this
        only has to be approximately right -- it sets where the slide starts.
        """
        allxyz = np.array([a.coord for ch in model for r in ch for a in r],
                          dtype=float)
        v = self.centre - allxyz.mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])

    def summary(self):
        return {
            "chains": self.chain_ids,
            "n_target_residues": len(self.target_labels),
            "hotspots": self.hotspots,
            "centre": self.centre.round(2).tolist(),
            "normal": self.normal.round(3).tolist(),
            "n_conserved_cation_atoms": int(len(self.cation_xyz)),
            "conserved_cations": sorted({(c, p) for c, p, _ in self.cation_labels}),
            "n_conserved_acid_atoms": int(len(self.acid_xyz)),
        }


# ---------------------------------------------------------------- scoring

#: Two candidate histidine positions closer than this are not independent sites:
#: they compete for the same cation and repel one another, and ProteinMPNN will
#: not place histidines on adjacent helix positions anyway. Matches
#: `pka.HIS_MIN_SEPARATION_A`, which re-applies the same test after design.
HIS_SITE_SEPARATION = 7.0

POSE_WEIGHTS = {
    "contacts": 1.0,         # per contacting binder residue
    "both_chains": 25.0,     # flat: must straddle the protomer boundary
    "cation_reach": 2.0,     # per binder residue in His reach of a cons. cation
    "independent_cation_reach": 14.0,   # per *well-separated* such residue
    "hotspot_cover": 6.0,    # per hotspot covered
    "acid_face": -2.0,       # per binder residue pressed against a conserved acid
    "mouse_conserved": 20.0,  # x fraction of contacted target residues conserved
    "clash": -40.0,          # per clashing residue pair (also hard-gated)
}


def _count_independent(pts, min_sep=HIS_SITE_SEPARATION):
    """Greedy count of points no two of which are within `min_sep`.

    Why the docking score needs this rather than a raw count: the first version
    weighted `cation_reach` at 8.0 per residue and the best pose it found had
    five reachable positions -- 19, 20, 21, 22 and 24, all on one helix face
    around the same Arg32. ProteinMPNN then placed exactly **one** histidine
    there, because four consecutive histidines on a helix is not a sequence it
    will produce, and summing them as four independent switches would have
    overstated S fourfold in any case.

    What the pH switch actually needs is *separated* anchor positions, so the
    weight now sits on this count and the raw one is demoted to a tiebreak.
    """
    if len(pts) == 0:
        return 0
    kept = []
    for p in pts:
        if all(np.linalg.norm(p - q) >= min_sep for q in kept):
            kept.append(p)
    return len(kept)


def score_pose(site, binder_cb, weights=POSE_WEIGHTS):
    """Score one placed binder. Returns (score, metrics) or (-inf, metrics) if clashing.

    The cation term is the challenge-specific one. `cation_reach` counts binder
    residues whose CB is within `CB_CATION_REACH` of a *conserved* target cation
    nitrogen -- i.e. positions where ProteinMPNN could later place a histidine
    whose bound pKa would actually be suppressed. Four such positions is the
    minimum for 100-fold selectivity, so at 8.0 each this term is worth more
    than 30 points to a viable pose: comparable to `both_chains`, and
    deliberately so.
    """
    d_near, _ = site.cb_tree.query(binder_cb, k=1)
    n_clash = int(np.sum(d_near < CB_CLASH))

    counts = site.cb_tree.query_ball_point(binder_cb, r=CB_CONTACT,
                                           return_length=True)
    contacting = counts > 0
    n_contact_res = int(contacting.sum())
    n_contact_pairs = int(counts.sum())

    # which target chains and which target residues are contacted
    chains_hit = set()
    touched = set()
    if n_contact_res:
        idx = site.cb_tree.query_ball_point(binder_cb[contacting], r=CB_CONTACT)
        for lst in idx:
            # str() because site.chain_of is a numpy array and np.str_ leaks
            # into the metrics dict, which then fails json.dump downstream
            chains_hit.update(str(c) for c in site.chain_of[lst])
            touched.update(lst)
    both = len(chains_hit) >= 2

    # Criterion 2, scored at the pose stage rather than only after design.
    # The groove is chosen partly for conservation, but a pose inside a
    # well-conserved groove can still put most of its contacts on the eleven
    # divergent positions in it (20, 30, 31, 53, 71, 73, 85, 89, 97, 111, 138).
    # An early run docked without this term and the best poses came back at
    # mouse identity 0.60, well below the groove's own 0.73.
    frac_cons = 0.0
    if touched and site.conserved_mask is not None:
        vals = [site.conserved_mask[i] for i in touched
                if site.conserved_mask[i] is not None]
        frac_cons = (sum(vals) / len(vals)) if vals else 0.0

    # hotspot coverage
    hs_cover = 0
    if len(site.hotspot_xyz):
        dh = np.linalg.norm(
            site.hotspot_xyz[:, None, :] - binder_cb[None, :, :], axis=-1)
        hs_cover = int(np.sum(dh.min(axis=1) <= CB_CONTACT))

    # pH-switch anchor positions
    cation_reach = 0
    indep_reach = 0
    if site.cation_tree is not None:
        dc, _ = site.cation_tree.query(binder_cb, k=1)
        in_reach = np.where(dc <= CB_CATION_REACH)[0]
        cation_reach = int(len(in_reach))
        indep_reach = _count_independent(binder_cb[in_reach])
    acid_face = 0
    if site.acid_tree is not None:
        da, _ = site.acid_tree.query(binder_cb, k=1)
        acid_face = int(np.sum(da <= CB_CATION_REACH))

    metrics = {
        "n_clash": n_clash,
        "n_contact_res": n_contact_res,
        "n_contact_pairs": n_contact_pairs,
        "chains_contacted": sorted(chains_hit),
        "both_chains": both,
        "hotspot_cover": hs_cover,
        "cation_reach": cation_reach,
        "independent_cation_reach": indep_reach,
        "acid_face": acid_face,
        "frac_mouse_conserved": round(float(frac_cons), 4),
        "min_cb_dist": round(float(d_near.min()), 2),
    }

    if n_clash > 0:
        return -np.inf, metrics

    score = (weights["contacts"] * n_contact_res
             + weights["both_chains"] * float(both)
             + weights["cation_reach"] * cation_reach
             + weights["independent_cation_reach"] * indep_reach
             + weights["hotspot_cover"] * hs_cover
             + weights["acid_face"] * acid_face
             + weights["mouse_conserved"] * frac_cons)
    metrics["score"] = round(float(score), 3)
    return float(score), metrics


# ---------------------------------------------------------------- the search

def _slide_to_contact(site, cb_local, R, d_hi=28.0, d_lo=-4.0, step=0.5):
    """Slide a rotated binder along the approach normal to first non-clashing contact.

    Walks inward from far away and stops at the last position with no clash,
    which is the tightest placement for that orientation. Returns (cb, d) or
    (None, None) if the orientation cannot be seated without clashing.

    Searching distance this way rather than on a translation grid is what keeps
    the search affordable: one scan of ~64 steps per orientation instead of a
    three-dimensional grid, and it lands exactly where contact is maximal.
    """
    rot = cb_local @ R.T
    best = None
    for d in np.arange(d_hi, d_lo, -step):
        cb = rot + (site.centre + d * site.normal)
        dn, _ = site.cb_tree.query(cb, k=1)
        if dn.min() < CB_CLASH:
            break
        if dn.min() <= CB_CONTACT:
            best = (cb, float(d))
    return best if best is not None else (None, None)


def dock_scaffold(site, coords, n_rot=1500, seed=0, n_keep=8,
                  refine_steps=24, refine_sigma=0.12, weights=POSE_WEIGHTS):
    """Dock one scaffold into the site. Returns a list of scored poses, best first.

    Two stages, for the usual reason: a uniform scan over orientations is broad
    but coarse, so the best few are then perturbed locally. `refine_sigma` is in
    radians of rotation; the refinement also jitters the slide distance and
    lateral offset, because the coarse stage fixes the binder centroid on the
    hotspot axis and the true optimum is usually a little off it.
    """
    cb_local = pseudo_cb(coords)
    cb_local = cb_local - cb_local.mean(axis=0)
    Rs = random_rotations(n_rot, seed=seed)

    scored = []
    for i in range(n_rot):
        cb, d = _slide_to_contact(site, cb_local, Rs[i])
        if cb is None:
            continue
        s, m = score_pose(site, cb, weights)
        if np.isfinite(s):
            scored.append({"score": s, "R": Rs[i], "d": d, "offset": np.zeros(3),
                           "metrics": m})
    if not scored:
        return []
    scored.sort(key=lambda p: -p["score"])

    # local refinement of the best handful
    rng = np.random.default_rng(seed + 1)
    tangent = _tangent_basis(site.normal)
    refined = []
    for pose in scored[:n_keep]:
        best = pose
        for _ in range(refine_steps):
            R = _perturb_rotation(best["R"], rng, refine_sigma)
            off = best["offset"] + tangent.T @ rng.normal(0, 1.2, 2)
            cb0 = cb_local @ R.T
            placed = None
            for dd in np.arange(best["d"] + 3.0, best["d"] - 3.0, -0.4):
                cb = cb0 + (site.centre + dd * site.normal + off)
                dn, _ = site.cb_tree.query(cb, k=1)
                if dn.min() < CB_CLASH:
                    break
                if dn.min() <= CB_CONTACT:
                    placed = (cb, float(dd))
            if placed is None:
                continue
            s, m = score_pose(site, placed[0], weights)
            if np.isfinite(s) and s > best["score"]:
                best = {"score": s, "R": R, "d": placed[1], "offset": off,
                        "metrics": m}
        refined.append(best)

    refined.sort(key=lambda p: -p["score"])
    return refined


def _tangent_basis(n):
    """Two unit vectors spanning the plane perpendicular to `n`."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2])


def _perturb_rotation(R, rng, sigma):
    """Compose `R` with a small random rotation (axis-angle, Rodrigues)."""
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = rng.normal(0, sigma)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return dR @ R


def place_coords(site, coords, pose):
    """Apply a pose's transform to full (L, 4, 3) backbone coordinates.

    The search works on CB clouds centred at the origin, so the same centring
    has to be reproduced here or the backbone lands somewhere else entirely.
    """
    cb = pseudo_cb(coords)
    centre = cb.mean(axis=0)
    flat = (coords - centre).reshape(-1, 3) @ pose["R"].T
    flat = flat + (site.centre + pose["d"] * site.normal + pose["offset"])
    return flat.reshape(coords.shape)


def cation_adjacency(site, binder_cb, reach=CB_CATION_REACH):
    """Which binder positions are in histidine reach of which conserved cations.

    The handoff to `mpnn`: this is exactly the list of positions that should get
    a positive histidine bias, and `pka` later re-measures the same geometry on
    the designed complex. Returns one row per binder position in reach.
    """
    if site.cation_tree is None:
        return []
    rows = []
    for i, p in enumerate(binder_cb):
        idx = site.cation_tree.query_ball_point(p, r=reach)
        if not idx:
            continue
        d = np.linalg.norm(site.cation_xyz[idx] - p, axis=1)
        j = int(np.argmin(d))
        ch, pos, rn = site.cation_labels[idx[j]]
        rows.append({
            "binder_pos": i + 1,
            "nearest_cation": f"{rn}{pos}({ch})",
            "dist_A": round(float(d[j]), 2),
            "n_cations_in_reach": len(idx),
        })
    return rows

"""
Target preparation: the TNF-alpha trimer, its receptor site, and where a pH
switch can actually be built.

Stage 0 of the pipeline. Everything here is measurement on downloaded
coordinates -- no literature values are typed in, because the whole point is to
be able to check them. Runs in a few seconds on CPU.

Three things make TNF-alpha structurally different from challenge 1's EGFR, and
all three change the code rather than just the commentary:

1. **The target is a trimer and must stay one.** Soluble TNF-alpha is an
   obligate homotrimer; chains B and C are not crystallographic contaminants,
   they are the protein. Challenge 1's `stage_prepare` had to *strip* the
   cetuximab Fab out of 6ARU before computing SASA or it would bury the epitope.
   Here the mistake is the exact opposite: strip B and C and you expose a huge
   hydrophobic trimer core that is never solvent-accessible in reality, and the
   patch scorer will happily aim your binder straight into it.
   `relative_sasa_trimer` vs `relative_sasa_isolated` measures the damage.

2. **The receptor site straddles two protomers.** One TNFR chain binds in the
   groove between adjacent subunits, taking roughly half its contacts from each
   (measured in `receptor_footprint`: 16 + 16 residues for TNFR2 in 3ALQ). So
   the epitope is not a patch on a chain, it is a patch on a *pair* of chains,
   and a binder that only grips one protomer is not blocking the receptor.

3. **The pH switch wants cations, not carboxylates.** Challenge 1 hunted
   conserved Asp/Glu to raise a histidine's pKa. Challenge 2 needs the bound
   pKa *suppressed* (see `linkage`), so the valuable target residues are
   conserved Lys/Arg, and conserved carboxylates are now a liability at the
   histidine positions. `patch_score` is inverted accordingly.
"""

from pathlib import Path
import copy
import json
import urllib.request
import warnings

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from Bio import Align
from Bio.Align import substitution_matrices

# ---------------------------------------------------------------- constants

#: Human TNF-alpha homotrimer, 2.6 A, apo. The challenge names this structure.
PDB_TARGET = "1TNF"
#: Human TNF-alpha / human TNFR2 complex, 3.0 A. Two trimers (A-C, D-F) and six
#: receptor chains (R-W). Used to MEASURE the functional epitope.
PDB_TNFR2 = "3ALQ"
#: Mouse TNF-alpha / human TNFR1, 2.65 A. An independent footprint on the
#: *mouse* orthologue, which doubles as a cross-reactivity cross-check.
PDB_TNFR1_MOUSE = "7KP7"
#: Human TNF-alpha with a small de novo-selected antagonist, 2.1 A. Not used in
#: scoring; a worked example of something that already does what we want to do.
PDB_ANTAGONIST = "3L9J"

UNIPROT_HUMAN = "P01375"  # TNFA_HUMAN, 233 aa precursor
UNIPROT_MOUSE = "P06804"  # TNFA_MOUSE, 235 aa precursor

#: Soluble TNF-alpha is the precursor's C-terminal fragment. The challenge
#: specifies human residues 77-233 (157 aa), so mature position 1 = UniProt 77
#: and UNIPROT_OFFSET is what you add to a mature number to get a UniProt one.
#: 1TNF and 3ALQ both number by mature position already, which
#: `numbering_offset` verifies by alignment rather than trusting.
MATURE_RANGE_UNIPROT = (77, 233)
UNIPROT_OFFSET = 76

TRIMER_CHAINS = ("A", "B", "C")

ACIDIC = {"D", "E"}
BASIC = {"K", "R"}
CATIONIC = {"K", "R", "H"}
HYDROPHOBIC = set("AVILMFWY")

#: Maximum residue solvent accessibility, Tien et al. 2013 (theoretical).
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

_3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}


# ---------------------------------------------------------------- downloads

def fetch_pdb(pdb_id, inputs_dir="inputs"):
    """Download a PDB file once and cache it locally."""
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    dest = inputs_dir / f"{pdb_id.upper()}.pdb"
    if not dest.exists():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", dest)
    return dest


def fetch_uniprot(accession, inputs_dir="inputs"):
    """Download a UniProt FASTA once and return its sequence as a string."""
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    dest = inputs_dir / f"{accession}.fasta"
    if not dest.exists():
        urllib.request.urlretrieve(
            f"https://rest.uniprot.org/uniprotkb/{accession}.fasta", dest)
    return "".join(l.strip() for l in dest.read_text().splitlines()
                   if not l.startswith(">"))


# ---------------------------------------------------------------- parsing

def load_model(pdb_path, keep_chains=None, strip_hetero=True):
    """Parse the first model, optionally keeping only `keep_chains`.

    Unlike challenge 1 this returns the whole *model*, not a chain: for a trimer
    almost every useful quantity (SASA, the inter-protomer groove) is only
    defined with the neighbouring subunits present.
    """
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure("t", str(pdb_path))
    model = next(structure.get_models())

    if keep_chains is not None:
        for ch in [c for c in model if c.id not in set(keep_chains)]:
            model.detach_child(ch.id)
    if strip_hetero:
        for ch in model:
            drop = [r.id for r in ch
                    if r.id[0] != " " or not is_aa(r, standard=True)]
            for rid in drop:
                ch.detach_child(rid)
    # a chain emptied by stripping is not a chain
    for ch in [c for c in model if len(list(c)) == 0]:
        model.detach_child(ch.id)
    return model


def chain_sequence(chain):
    """Observed one-letter sequence plus the author residue numbers behind it."""
    seq, numbers = [], []
    for res in chain:
        letter = _3TO1.get(res.get_resname().upper())
        if letter is None:
            continue
        seq.append(letter)
        numbers.append(res.id[1])
    return "".join(seq), np.array(numbers, dtype=int)


# ---------------------------------------------------------------- alignment

def _aligner():
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score = -11
    al.extend_gap_score = -1
    al.mode = "global"
    return al


def align_index_map(query_seq, reference_seq):
    """Map each query index onto a reference index. {query_idx -> ref_idx}."""
    aln = _aligner().align(query_seq, reference_seq)[0]
    mapping = {}
    for (q0, q1), (r0, r1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(q1 - q0):
            mapping[q0 + k] = r0 + k
    return mapping


def mature_human(human_precursor):
    """The 157-residue soluble fragment the challenge targets (UniProt 77-233)."""
    lo, hi = MATURE_RANGE_UNIPROT
    return human_precursor[lo - 1:hi]


def numbering_offset(chain, human_mature_seq):
    """Resolve `author residue number - mature position` by alignment, not trust.

    Returns (offset, agreement, n_mapped). For 1TNF and 3ALQ the answer is 0 --
    they already number by mature position -- but a silent off-by-76 would put
    the entire epitope search outside the protein, so it is measured and logged.
    """
    seq, numbers = chain_sequence(chain)
    mapping = align_index_map(seq, human_mature_seq)
    if not mapping:
        return None, 0.0, 0
    offsets = [numbers[i] - (ref + 1) for i, ref in mapping.items()]
    vals, counts = np.unique(offsets, return_counts=True)
    return int(vals[np.argmax(counts)]), float(counts.max() / counts.sum()), len(mapping)


def mouse_identity_map(human_mature_seq, mouse_precursor):
    """Per-mature-position human/mouse identity: {mature_pos -> (identical, mouse_aa)}.

    The mouse mature start is found by aligning the two *precursors* and reading
    off whatever position corresponds to human 77, rather than assuming mouse
    has the same propeptide length (it does not -- 235 aa vs 233, and the mature
    forms are 156 vs 157).

    Reproducing the challenge brief's own figure -- 124 of 156 aligned residues,
    79% -- is the check that this function is right, and `check_env.py` asserts it.
    """
    # One global alignment of the human mature fragment against the whole mouse
    # precursor. A global aligner is free to place the 157-residue query
    # anywhere in the 235-residue subject, so this both locates the mouse mature
    # start and gives the residue correspondence in a single pass.
    mapping = align_index_map(human_mature_seq, mouse_precursor)
    identity = {}
    for h_idx, m_idx in mapping.items():
        identity[h_idx + 1] = (human_mature_seq[h_idx] == mouse_precursor[m_idx],
                               mouse_precursor[m_idx])
    mouse_start = min(mapping.values()) if mapping else 0
    return identity, mouse_precursor[mouse_start:]


# ---------------------------------------------------------------- surface

def relative_sasa_trimer(model, chain_ids=TRIMER_CHAINS):
    """Per-residue relative SASA computed WITH the whole trimer present.

    This is the correct one. Returns {(chain_id, resnum) -> rsa}.
    """
    work = copy.deepcopy(model)
    for ch in [c for c in work if c.id not in set(chain_ids)]:
        work.detach_child(ch.id)
    ShrakeRupley().compute(work, level="R")
    out = {}
    for ch in work:
        for res in ch:
            letter = _3TO1.get(res.get_resname().upper())
            if letter is None:
                continue
            out[(ch.id, res.id[1])] = float(res.sasa) / MAX_ASA[letter]
    return out


def relative_sasa_isolated(model, chain_id="A"):
    """Per-residue relative SASA of ONE protomer with its partners removed.

    The wrong one, kept so the error can be quantified instead of asserted.
    Challenge 1 had to delete chains to avoid burying its epitope; here deleting
    chains *un*-buries a trimer core that is never solvent-exposed in the real
    protein. Compare against `relative_sasa_trimer` and the residues that move
    are the ones a naive patch scorer would offer you as targets.
    """
    work = copy.deepcopy(model)
    for ch in [c for c in work if c.id != chain_id]:
        work.detach_child(ch.id)
    ShrakeRupley().compute(work, level="R")
    return {(chain_id, res.id[1]): float(res.sasa) / MAX_ASA[_3TO1[res.get_resname().upper()]]
            for res in work[chain_id] if res.get_resname().upper() in _3TO1}


def sasa_discrepancy(model, chain_id="A", threshold=0.25):
    """Residues the isolated-protomer SASA would wrongly call exposed.

    Returns rows sorted by how badly they move. These are exactly the trimer
    interior -- the positions where a binder cannot go.
    """
    tri = relative_sasa_trimer(model)
    iso = relative_sasa_isolated(model, chain_id)
    rows = []
    for key, iso_v in iso.items():
        tri_v = tri.get(key)
        if tri_v is None:
            continue
        if iso_v > threshold and tri_v <= threshold:
            rows.append({"chain": key[0], "resnum": key[1],
                         "rsa_isolated": round(iso_v, 3),
                         "rsa_trimer": round(tri_v, 3),
                         "delta": round(iso_v - tri_v, 3)})
    return sorted(rows, key=lambda r: -r["delta"])


def residue_centroids(model, atom_preference=("CB", "CA")):
    """One representative coordinate per residue: {(chain, resnum) -> xyz}."""
    coords = {}
    for ch in model:
        for res in ch:
            for name in atom_preference:
                if name in res:
                    coords[(ch.id, res.id[1])] = np.asarray(res[name].coord, float)
                    break
            else:
                coords[(ch.id, res.id[1])] = np.mean(
                    [a.coord for a in res], axis=0).astype(float)
    return coords


# ---------------------------------------------------------------- footprints

def _heavy(res):
    return np.array([a.coord for a in res if a.element != "H"], dtype=float)


def contact_residues(probe_residues, target_chain, cutoff=5.0):
    """Residue numbers of `target_chain` within `cutoff` of any probe atom."""
    blocks = [_heavy(r) for r in probe_residues]
    blocks = [b for b in blocks if b.size]
    if not blocks:
        return set()
    probe = np.vstack(blocks)
    hits = set()
    for res in target_chain:
        t = _heavy(res)
        if t.size and np.linalg.norm(
                t[:, None, :] - probe[None, :, :], axis=-1).min() <= cutoff:
            hits.add(res.id[1])
    return hits


def receptor_footprint(pdb_path, tnf_chains, receptor_chains, cutoff=5.0):
    """Measure which TNF residues each single receptor chain touches.

    The function that establishes point 2 in the module docstring. For every
    receptor chain it reports the contacted residues *per TNF protomer*, so the
    two-protomer straddle is visible as data rather than claimed: in 3ALQ every
    TNFR2 chain contacts two protomers with ~16 residues each, plus a handful
    on a third from the neighbouring trimer in the asymmetric unit.

    Returns {receptor_chain -> {tnf_chain -> sorted resnums}}.
    """
    model = load_model(pdb_path)
    out = {}
    for rk in receptor_chains:
        if rk not in model:
            continue
        per = {}
        for tk in tnf_chains:
            if tk not in model:
                continue
            hit = contact_residues(list(model[rk]), model[tk], cutoff)
            if hit:
                per[tk] = sorted(hit)
        out[rk] = per
    return out


def consensus_epitope(footprint, min_contacts=8):
    """Collapse a footprint into the two half-sites of one receptor-binding groove.

    A receptor chain's contacts split into a major and a second major protomer
    (the groove it sits in) plus incidental crystal contacts. Keeping only
    protomers with >= `min_contacts` residues drops the incidental ones, then
    the two half-sites are reported separately because a binder has to reach
    both.

    Returns a list of {receptor, primary, secondary, site_primary, site_secondary}.
    """
    grooves = []
    for rk, per in footprint.items():
        ranked = sorted(per.items(), key=lambda kv: -len(kv[1]))
        ranked = [(k, v) for k, v in ranked if len(v) >= min_contacts]
        if len(ranked) < 2:
            continue
        (p, sp), (s, ss) = ranked[0], ranked[1]
        grooves.append({"receptor": rk, "primary": p, "secondary": s,
                        "site_primary": sp, "site_secondary": ss,
                        "n_primary": len(sp), "n_secondary": len(ss)})
    return grooves


def pooled_epitope(grooves):
    """Union of half-sites across all receptor copies, as position frequencies.

    Pooling the six independent TNFR2 copies in 3ALQ turns one measurement into
    a consensus with a count attached, which is what lets `prepare` prefer
    positions seen in every copy over ones seen once.
    """
    prim, sec = {}, {}
    for g in grooves:
        for p in g["site_primary"]:
            prim[p] = prim.get(p, 0) + 1
        for p in g["site_secondary"]:
            sec[p] = sec.get(p, 0) + 1
    return prim, sec


def receptor_frequency(grooves):
    """How many receptor copies contact each mature position, in EITHER half-site.

    `pooled_epitope` keeps the two half-sites apart, which is what you want when
    describing the groove's two faces. For scoring you want the opposite: a
    position's total support. Which half-site a position falls in is an accident
    of which protomer you label primary -- receptor chain R calls protomer A
    primary and B secondary, chain T calls A primary and C secondary -- so
    taking a max over the two roles (the obvious move) silently halves the
    count and makes "seen in every copy" unreachable. Summing is correct.

    Returns {mature_pos -> count}, where `n_receptor_copies` is the ceiling.
    """
    prim, sec = pooled_epitope(grooves)
    return {p: prim.get(p, 0) + sec.get(p, 0)
            for p in set(prim) | set(sec)}


# ---------------------------------------------------------------- patches

#: Patch geometry. Radius is larger than challenge 1's 12 A because the groove
#: is a cleft spanning two chains rather than a patch on one face.
PATCH_RADIUS = 14.0
PATCH_MIN_MEMBERS = 8
PATCH_MIN_RSA = 0.25

#: Patch-score weights. **Inverted relative to challenge 1** -- see the module
#: docstring, point 3. Conserved cations are now the prize because they suppress
#: a bound histidine's pKa; conserved carboxylates are now a penalty because
#: they raise it and would run the switch backwards.
W_CONSERVED_CATION = 4.0
W_FRAC_CONSERVED = 6.0
W_HYDROPHOBIC = 0.4
W_CONSERVED_ACID = -1.5
W_SIZE = 0.15
W_SPANS_RECEPTOR_SITE = 6.0
W_FRAC_RECEPTOR = 6.0


def patch_score(n_cons_cation, frac_cons, n_hydrophobic, n_cons_acid,
                n_members, spans_receptor_site, frac_receptor):
    """The epitope patch score, inverted from challenge 1 for reverse polarity.

        score = 4.00 * conserved Lys/Arg       pKa SUPPRESSORS   (criterion 1)
              + 6.00 * fraction conserved      mouse cross-react (criterion 2)
              + 0.40 * hydrophobic             affinity anchor   (criterion 3)
              - 1.50 * conserved Asp/Glu       raise pKa = WRONG DIRECTION
              + 0.15 * patch size              bigger interface
              + 6.00 * spans receptor site     must block the receptor
              + 6.00 * fraction on receptor site   stay ON the functional site

    Two weights carry the challenge-specific reasoning. `W_CONSERVED_CATION` is
    positive where challenge 1's acid term was positive, and `W_CONSERVED_ACID`
    is now *negative* -- that single sign flip is the difference between a binder
    that releases at pH 6 and one that grips harder.

    The last two terms were not in the first version of this function and the
    patch table is the reason they exist. With a raw *count* of receptor-site
    members and a flat "touches two chains" bonus, the top-scoring patch was
    centred on A53 with three conserved cations, five receptor-site residues and
    hotspots at A11/A40/A52 -- nowhere near the groove. It won on cation count
    alone. Scoring the receptor overlap as a **fraction** with real weight, and
    requiring receptor-site residues on *both* chains rather than merely two
    chains somewhere in the patch, puts the measured groove back on top. A
    patch's cation content should only count once it is in the right place.
    """
    return (W_CONSERVED_CATION * n_cons_cation
            + W_FRAC_CONSERVED * frac_cons
            + W_HYDROPHOBIC * n_hydrophobic
            + W_CONSERVED_ACID * n_cons_acid
            + W_SIZE * n_members
            + W_SPANS_RECEPTOR_SITE * float(bool(spans_receptor_site))
            + W_FRAC_RECEPTOR * frac_receptor)


def build_residue_table(model, identity, rsa, coords, receptor_freq,
                        chain_ids=TRIMER_CHAINS, offset=0):
    """One row per observed trimer residue, carrying everything the scorer needs."""
    rows = []
    for ch in model:
        if ch.id not in set(chain_ids):
            continue
        for res in ch:
            letter = _3TO1.get(res.get_resname().upper())
            if letter is None:
                continue
            num = res.id[1]
            mature = num - offset
            same, maa = identity.get(mature, (None, None))
            rows.append({
                "chain": ch.id,
                "resnum": num,
                "mature_pos": mature,
                "aa": letter,
                "mouse_aa": maa,
                "mouse_identical": same,
                "rel_sasa": rsa.get((ch.id, num), np.nan),
                "receptor_freq": receptor_freq.get(mature, 0),
                "xyz": coords.get((ch.id, num)),
            })
    return rows


def _patch_stats(members, receptor_freq):
    cons_cation = [m for m in members
                   if m["aa"] in BASIC and m["mouse_identical"]]
    cons_acid = [m for m in members
                 if m["aa"] in ACIDIC and m["mouse_identical"]]
    hydrophobic = [m for m in members if m["aa"] in HYDROPHOBIC]
    chains = {m["chain"] for m in members}
    frac_cons = sum(bool(m["mouse_identical"]) for m in members) / len(members)

    on_site = [m for m in members if m["receptor_freq"] > 0]
    # The groove test: receptor-contacted residues on at least two protomers.
    # "Two chains anywhere in the patch" is not enough -- a patch straddling a
    # subunit boundary away from the functional site also satisfies that.
    site_chains = {m["chain"] for m in on_site}
    frac_receptor = len(on_site) / len(members)

    return {
        "n_residues": len(members),
        "n_chains": len(chains),
        "chains": sorted(chains),
        "n_on_receptor_site": len(on_site),
        "frac_receptor": round(frac_receptor, 4),
        "receptor_site_chains": sorted(site_chains),
        "spans_receptor_site": len(site_chains) >= 2,
        "n_conserved_cation": len(cons_cation),
        "n_conserved_acid": len(cons_acid),
        "n_hydrophobic": len(hydrophobic),
        "frac_conserved": round(frac_cons, 4),
        "score": round(patch_score(len(cons_cation), frac_cons, len(hydrophobic),
                                   len(cons_acid), len(members),
                                   len(site_chains) >= 2, frac_receptor), 3),
        "members": sorted((m["chain"], m["mature_pos"]) for m in members),
        "on_site_pos": sorted((m["chain"], m["mature_pos"]) for m in on_site),
        "conserved_cation_pos": sorted((m["chain"], m["mature_pos"]) for m in cons_cation),
        "conserved_acid_pos": sorted((m["chain"], m["mature_pos"]) for m in cons_acid),
    }


def score_patches(rows, receptor_freq, radius=PATCH_RADIUS,
                  min_rsa=PATCH_MIN_RSA, min_members=PATCH_MIN_MEMBERS,
                  require_receptor_site=True):
    """Score every exposed surface patch of the trimer, allowing cross-chain patches.

    The important difference from challenge 1's version: neighbours are gathered
    across *all* trimer chains, so a patch sitting in the A/B cleft contains
    residues from both and `spans_protomers` fires. Restricting the search to one
    chain at a time -- the obvious port of the challenge 1 code -- cannot
    represent the receptor site at all.
    """
    surf = [r for r in rows
            if r["xyz"] is not None and np.isfinite(r["rel_sasa"])
            and r["rel_sasa"] > min_rsa]
    if require_receptor_site:
        centres = [r for r in surf if r["receptor_freq"] > 0]
    else:
        centres = surf
    if not surf or not centres:
        return []
    xyz = np.array([r["xyz"] for r in surf])

    patches = []
    for centre in centres:
        d = np.linalg.norm(xyz - centre["xyz"], axis=1)
        members = [surf[j] for j in np.where(d <= radius)[0]]
        if len(members) < min_members:
            continue
        patches.append({
            "centre_chain": centre["chain"],
            "centre": centre["mature_pos"],
            "centre_xyz": centre["xyz"],
            **_patch_stats(members, receptor_freq),
        })
    return sorted(patches, key=lambda p: -p["score"])


def select_nonredundant(patches, n=4, min_separation=14.0):
    """Greedily take the best patches whose centres are >= `min_separation` apart."""
    chosen = []
    for p in patches:
        if all(float(np.linalg.norm(p["centre_xyz"] - q["centre_xyz"])) > min_separation
               for q in chosen):
            chosen.append(p)
        if len(chosen) >= n:
            break
    return chosen


def assign_hotspots(patches, max_hotspots=5):
    """Turn each patch into hotspot residues for the docking stage.

    The rule, and it follows directly from `linkage`: take up to 3 conserved
    cations first, because those are the positions that can suppress a bound
    histidine's pKa and so are the only ones that can deliver criterion 1. Top
    up to `max_hotspots` with the most exposed conserved residues, preferring
    ones the receptor itself contacts. If a patch has no conserved cation at all
    it keeps its hotspots but is flagged `switchable: False` -- it may still be a
    fine place to bind, it is just not a place to build a switch.
    """
    for i, p in enumerate(patches):
        on_site = set(p["on_site_pos"])
        # Only receptor-contacted positions may be hotspots. Without this the
        # selector happily returned A11/A40/A52 -- conserved cations that are
        # nowhere near the groove -- and aimed the whole docking stage at them.
        cations = [m for m in p["conserved_cation_pos"] if m in on_site]
        hs = cations[:3]
        p["n_switch_anchors"] = len(cations)
        p["switchable"] = len(cations) > 0
        # Top up, preferring chains not yet represented. The docking stage aims
        # the binder at the hotspot centroid, so a hotspot set that happens to
        # sit entirely on one protomer points it at a single subunit even when
        # the patch straddles the boundary -- which is the one thing a
        # receptor-blocking binder must not do.
        for m in sorted(p["on_site_pos"],
                        key=lambda m: (m[0] in {h[0] for h in hs}, m)):
            if len(hs) >= max_hotspots:
                break
            if m not in hs:
                hs.append(m)
        p["hotspots"] = sorted(hs)
        p["switch_anchors"] = sorted(cations)
        p["id"] = f"groove{i:02d}"
    return patches


# ---------------------------------------------------------------- output

def write_groove_pdb(model, chain_ids, dest):
    """Write just the protomer pair a binder will be docked against.

    Two chains, not three: ProteinMPNN and the docking search both run over the
    full context they are given, and the third protomer sits behind the groove
    where it cannot contact a binder. Dropping it is ~33% off every distance
    matrix for no loss of accuracy. SASA is still computed on the full trimer
    upstream, which is where the third chain actually matters.
    """
    from Bio.PDB import PDBIO, Select

    keep = set(chain_ids)

    class _Keep(Select):
        def accept_chain(self, chain):
            return chain.id in keep

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(str(dest), select=_Keep())
    return dest


def save_meta(meta, dest):
    """Write the stage-0 metadata as JSON, coercing numpy scalars and arrays.

    The default encoder exists because almost every value here arrives from
    numpy -- residue numbers as np.int64, fractions as np.float64 -- and
    json.dump rejects all of them.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        raise TypeError(type(o))

    dest.write_text(json.dumps(meta, indent=2, default=_enc))
    return dest

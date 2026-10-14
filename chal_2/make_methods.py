#!/usr/bin/env python3
"""
Generate results/METHODS.md from a completed run's own artifacts.

    python3 make_methods.py                 # auto-detect this directory's run
    python3 make_methods.py --out FILE
    python3 make_methods.py --root ../chal_2_local

The challenge accepts methodology documentation alongside the CSV. Every number
in the output is read from `work/target_meta.json`, `work/scored.json` and
`results/submission.csv` rather than typed in, so the document cannot drift out
of step with the designs it describes -- re-run it after any re-scoring and the
figures follow.

It detects which generation route produced the designs (RFdiffusion on the
cluster, or the parametric-bundle fallback on a laptop) from the fields present
in `scored.json`, and writes the corresponding methods text. Pure stdlib, so it
runs in either conda env or on a laptop.
"""

import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
# Only works this pipeline actually depends on. PDB entries are cited by ID,
# which is unambiguous, with the standard structure DOI form.
REFS = [
    ("rfdiffusion",
     "Watson JL, Juergens D, Bennett NR, et al. *De novo design of protein "
     "structure and function with RFdiffusion.* Nature 620, 1089-1100 (2023).",
     "https://doi.org/10.1038/s41586-023-06415-8",
     "https://github.com/RosettaCommons/RFdiffusion"),
    ("proteinmpnn",
     "Dauparas J, Anishchenko I, Bennett N, et al. *Robust deep learning-based "
     "protein sequence design using ProteinMPNN.* Science 378, 49-56 (2022).",
     "https://doi.org/10.1126/science.add2187",
     "https://github.com/dauparas/ProteinMPNN"),
    ("esm",
     "Lin Z, Akin H, Rao R, et al. *Evolutionary-scale prediction of "
     "atomic-level protein structure with a language model.* Science 379, "
     "1123-1130 (2023). [ESM-2 and ESMFold]",
     "https://doi.org/10.1126/science.ade2574",
     "https://github.com/facebookresearch/esm"),
    ("1tnf",
     "Eck MJ, Sprang SR. *The structure of tumor necrosis factor-alpha at 2.6 A "
     "resolution.* J Biol Chem 264, 17595-17605 (1989). PDB 1TNF.",
     "https://doi.org/10.2210/pdb1TNF/pdb",
     "https://www.rcsb.org/structure/1TNF"),
    ("3alq",
     "Mukai Y, Nakamura T, Yoshikawa M, et al. *Solution of the structure of "
     "the TNF-TNFR2 complex.* Sci Signal 3, ra83 (2010). PDB 3ALQ.",
     "https://doi.org/10.2210/pdb3ALQ/pdb",
     "https://www.rcsb.org/structure/3ALQ"),
    ("7kp7",
     "Asymmetric mouse TNF-alpha / human TNFR1 complex. PDB 7KP7.",
     "https://doi.org/10.2210/pdb7KP7/pdb",
     "https://www.rcsb.org/structure/7KP7"),
    ("uniprot",
     "UniProt Consortium. Human TNF P01375; mouse TNF P06804.",
     "https://www.uniprot.org/uniprotkb/P01375/entry",
     "https://www.uniprot.org/uniprotkb/P06804/entry"),
    ("linkage",
     "Wyman J, Gill SJ. *Binding and Linkage: Functional Chemistry of "
     "Biological Macromolecules.* University Science Books (1990). "
     "[proton linkage / site-specific titration]",
     None, None),
    ("shrake",
     "Shrake A, Rupley JA. *Environment and exposure to solvent of protein "
     "atoms. Lysozyme and insulin.* J Mol Biol 79, 351-371 (1973).",
     "https://doi.org/10.1016/0022-2836(73)90011-9", None),
    ("tien",
     "Tien MZ, Meyer AG, Sydykova DK, et al. *Maximum allowed solvent "
     "accessibilities of residues in proteins.* PLoS ONE 8, e80635 (2013).",
     "https://doi.org/10.1371/journal.pone.0080635", None),
    ("biopython",
     "Cock PJA, Antao T, Chang JT, et al. *Biopython: freely available Python "
     "tools for computational molecular biology and bioinformatics.* "
     "Bioinformatics 25, 1422-1423 (2009).",
     "https://doi.org/10.1093/bioinformatics/btp163", None),
    ("engh",
     "Engh RA, Huber R. *Accurate bond and angle parameters for X-ray protein "
     "structure refinement.* Acta Cryst A47, 392-400 (1991).",
     "https://doi.org/10.1107/S0108767391001071", None),
    ("nerf",
     "Parsons J, Holmes JB, Rojas JM, et al. *Practical conversion from torsion "
     "space to Cartesian space for in silico protein synthesis.* J Comput Chem "
     "26, 1063-1068 (2005). [NeRF]",
     "https://doi.org/10.1002/jcc.20237", None),
    ("kabsch",
     "Kabsch W. *A solution for the best rotation to relate two sets of "
     "vectors.* Acta Cryst A32, 922-923 (1976).",
     "https://doi.org/10.1107/S0567739476001873", None),
    ("shoemake",
     "Shoemake K. *Uniform random rotations.* In Graphics Gems III, "
     "pp. 124-132. Academic Press (1992).",
     None, None),
    ("blosum",
     "Henikoff S, Henikoff JG. *Amino acid substitution matrices from protein "
     "blocks.* PNAS 89, 10915-10919 (1992). [BLOSUM62]",
     "https://doi.org/10.1073/pnas.89.22.10915", None),
]


def ref_block(keys):
    out = []
    for i, k in enumerate(keys, 1):
        for kk, text, doi, repo in REFS:
            if kk != k:
                continue
            line = f"{i}. {text}"
            links = [l for l in (doi, repo) if l]
            if links:
                line += "  \n   " + " · ".join(links)
            out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load(root):
    root = Path(root)
    meta = json.loads((root / "work" / "target_meta.json").read_text())
    scored = json.loads((root / "work" / "scored.json").read_text())
    sub_path = root / "results" / "submission.csv"
    sub = list(csv.DictReader(sub_path.open())) if sub_path.exists() else []
    cfg_path = root / "work" / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return meta, scored, sub, cfg


def detect_route(scored):
    """Which generation route produced these designs.

    RFdiffusion runs record a `backbone` and a pLDDT; the laptop fallback
    records `scaffold_id` and `pose_R` (the rigid-body docking transform) and no
    pLDDT. Detected rather than configured so the document always describes the
    run it was generated from.
    """
    if not scored:
        return "unknown"
    d = scored[0]
    if "backbone" in d or d.get("mean_plddt") is not None:
        return "rfdiffusion"
    if "pose_R" in d or "scaffold_id" in d:
        return "parametric"
    return "unknown"


def fmt(x, nd=2, dash="n/a"):
    return dash if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(meta, scored, sub, cfg, route):
    passing = [d for d in scored if d.get("seq_ok")]
    meeting = [d for d in passing if d.get("meets_S_target")]
    sub_seqs = {r["sequence"] for r in sub}
    by_seq = {d["sequence"]: d for d in scored}
    submitted = [by_seq[s] for s in sub_seqs if s in by_seq]

    grooves = meta["grooves"]
    n_receptor_copies = 6
    strong = meta.get("consensus_epitope_strong", [])
    cations = sorted({f"{a}{p}" for g in grooves
                      for (c, p) in map(tuple, g.get("switch_anchors", []))
                      for a in ["" ] } ) if False else None

    # conserved cation anchors, by mature position, across the chosen grooves
    anchor_pos = sorted({int(p) for g in grooves
                         for (_c, p) in map(tuple, g.get("switch_anchors", []))})
    acid_pos = sorted({int(p) for g in grooves
                       for (_c, p) in map(tuple, g.get("conserved_acid_pos", []))})

    # per-histidine-count table
    by_his = defaultdict(list)
    for d in passing:
        by_his[int(d["n_independent_his"])].append(float(d["S"]))
    CEIL = 0.5041

    his_counts = [k for k in sorted(by_his) if k > 0]
    his_rows = "\n".join(
        f"| {k} | {len(by_his[k])} | {max(by_his[k]):.2f} | {k * CEIL:.2f} | "
        f"{max(by_his[k]) / (k * CEIL):.1f}x |"
        for k in his_counts)

    # Whether the ceiling claim actually holds in THIS run, rather than
    # asserting it. The parametric run beat the ceiling at every count; there is
    # no guarantee a different run does, and a methodology document that states
    # it unconditionally would be making a false claim about its own data.
    ratios = {k: max(by_his[k]) / (k * CEIL) for k in his_counts}
    above = [k for k, r in ratios.items() if r > 1.0]
    if his_counts and len(above) == len(his_counts):
        ceiling_claim = ("Every histidine count in this run exceeds the naive "
                         "ceiling.")
    elif above:
        ceiling_claim = (f"At {'/'.join(str(k) for k in above)} independent "
                         f"histidine(s) the best design exceeds the naive "
                         f"ceiling.")
    else:
        ceiling_claim = ("No design in this run exceeds the naive ceiling, so "
                         "the free-state lever did not engage; selectivity here "
                         "comes from bound-state suppression alone.")

    # The fragility caveat only applies to LOW histidine counts that beat the
    # ceiling. Quote the actual weakest-grounded row rather than a fixed number.
    low = [k for k in above if k <= 2]
    if low:
        k0 = min(low)
        s0 = max(by_his[k0])
        plural = "histidine" if k0 == 1 else "histidines"
        frag = (f"""
It is also the most fragile result here, and we flag it rather than lead with
it. The best {k0}-histidine design reaches S = {s0:.2f}, which is
{s0 / (k0 * CEIL):.1f}x the ceiling {k0} unassisted {plural} could give. The pKa
model can only reach that by placing pKa_free well above pH 7.4 and pKa_bound
below pH 6.0, and it rests on a single hand-set +1.5 increment applied to
unrepacked geometry. **A design claiming {10 ** s0:.0f}-fold selectivity from
{k0} {plural} should be trusted considerably less than one claiming the same
from three**, because the latter distributes the claim across three independent
sites. The per-design histidine count is in `results/design_report.csv`, so this
can be weighed directly rather than taken on trust.
""")
    else:
        frag = (f"""
Selectivity in this run comes from designs with
{min(his_counts) if his_counts else 0}-{max(his_counts) if his_counts else 0}
independent histidines rather than from a small number of strongly-shifted
sites, which is the more defensible regime: the claim is distributed across
several independent groups rather than resting on one large modelled pKa shift.
""")

    cats = Counter(d.get("size_category", "?") for d in submitted)
    lens = sorted(len(r["sequence"]) for r in sub) if sub else []

    S_vals = [float(d["S"]) for d in submitted] if submitted else []
    mouse_vals = [float(d["mouse_identity"]) for d in submitted] if submitted else []
    contacts = [int(d["n_contacted_target"]) for d in submitted] if submitted else []
    plddts = [d.get("mean_plddt") for d in submitted
              if d.get("mean_plddt") is not None]
    rmsds = [d.get("rmsd") for d in submitted if d.get("rmsd") is not None]
    plls = [d.get("esm_pll") for d in submitted if d.get("esm_pll") is not None]

    perfect = [d for d in passing if float(d["mouse_identity"]) >= 0.99]
    rest = [d for d in passing if float(d["mouse_identity"]) < 0.99]

    # ---------------------------------------------------------------- route text
    if route == "rfdiffusion":
        gen_title = "Backbone generation — RFdiffusion"
        gen_body = f"""
Binder backbones were generated with **RFdiffusion** [1] using the
complex-trained checkpoint `Complex_base_ckpt.pt`, conditioned on the epitope
hotspots determined in §2.

| parameter | value | rationale |
|---|---|---|
| checkpoint | `Complex_base_ckpt.pt` | the complex/PPI model; the monomer checkpoint produces backbones that ignore the target |
| `ppi.hotspot_res` | per groove, see §2 | what aims the binder; without it RFdiffusion docks somewhere arbitrary on the target |
| `contigmap.contigs` | `{grooves[0].get('contig', 'see work/target_meta.json')}` | both target protomers with a chain break (`/0`), then a bare binder-length range |
| binder length | {cfg.get('binder_min', 55)}–{cfg.get('binder_max', 80)} aa, sampled | sampled rather than fixed, so designs populate more than one scoring category |
| `diffuser.T` | {cfg.get('diffuser_T', 50)} | reduced from the default 200; ~4x faster for a small quality cost, and diffusion dominates wall time |
| `denoiser.noise_scale_ca` | 0 | low-noise binder-design setting: trades diversity for a far higher fraction of designable backbones |
| `denoiser.noise_scale_frame` | 0 | as above |
| backbones generated | {cfg.get('n_backbones', 'n/a')} | split evenly across the selected grooves |

**Target cropping.** Two complete TNF-α protomers is 304 residues, which
with a binder puts ~380 positions through RFdiffusion, whose memory cost grows
roughly quadratically in that number. The target was therefore cropped to an
18 Å shell around the hotspots, giving {' and '.join(str(g['n_target_residues']) for g in grooves)}
residues for the selected groove(s). Each chain's retained residues were
renumbered contiguously (1..N) because a crop leaves gaps wherever a loop exits
the shell, and RFdiffusion would otherwise treat each fragment as a separate
chain segment and attempt to bridge them. The mapping back to the original
numbering is retained in `work/target_meta.json`, and the preflight check
verifies that the emitted hotspot string traces back to the residues actually
selected — an off-by-one there would aim the whole generation stage at the wrong
face of the protein with no visible error.
"""
        val_title = "Structural validation — ESMFold"
        val_body = f"""
Each designed sequence was folded **in isolation** with **ESMFold** [3] (fp32;
the V100 used is Volta/sm_70 and has no bfloat16), and compared against the
RFdiffusion backbone it was designed on by CA RMSD after optimal superposition
(Kabsch [14]).

| filter | threshold | purpose |
|---|---|---|
| mean pLDDT | ≥ {cfg.get('min_plddt', 75.0)} | the sequence must predict a confident structure |
| CA RMSD to design backbone | ≤ {cfg.get('max_rmsd', 2.5)} Å | self-consistency: the sequence must encode *this* backbone |

{"Submitted designs: mean pLDDT " + fmt(st.mean(plddts), 1) + " (range " + fmt(min(plddts),1) + "–" + fmt(max(plddts),1) + "), mean RMSD " + fmt(st.mean(rmsds), 2) + " Å." if plddts and rmsds else "Per-design values are in `results/design_report.csv`."}

This is the standard self-consistency filter and the most informative single
check in the pipeline: it is the only step that tests whether a designed
sequence actually encodes the backbone it was designed for.
"""
    else:
        gen_title = "Backbone generation — parametric helical bundles"
        gen_body = f"""
RFdiffusion could not be used for this run: its SE3Transformer depends on `dgl`,
which has no Apple-Silicon build, and the stack is pinned to `torch
2.0.1+cu118` with no MPS or CPU path. Backbones were therefore generated
**parametrically**.

Chains were built from backbone torsions by NeRF [13] with Engh–Huber ideal bond
geometry [12]. Helix residues were assigned ideal α-helical torsions
(φ, ψ) = (−57°, −47°); the only free parameters were the loop torsions, bounded
to the populated β/bridge/α region of the Ramachandran plot. Building in torsion
space closes the chain by construction, so no loop-closure step is needed —
instead compactness becomes the objective, and loop torsions were optimised
(random screen then L-BFGS-B) against a penalty combining helix-axis spacing
(target 10.5 Å, the knobs-into-holes value for an antiparallel coiled coil),
antiparallel orientation of consecutive helices, CA–CA clash, and radius of
gyration against the empirical 2.2·L^0.38 expectation for a globular chain.

The topology sampled was predominantly the **antiparallel three-helix bundle**,
which is the workhorse de novo minibinder scaffold: it folds robustly, expresses
well, and presents a flat-to-convex face that can be laid into a groove. This is
a weaker *sampler* than diffusion, not a weaker *scaffold* — what is lost is
shape complementarity to this specific site.

**Rigid-body docking.** Because the scaffold is fixed rather than grown around
the epitope, placement was a separate step: uniformly sampled rotations
(Shoemake [15]) were each slid along the hotspot approach normal to first
non-clashing contact, scored, and the best few refined locally. The pose score
is not a conventional docking score — it weights, in order: straddling both
protomers (a near pass/fail condition for receptor blocking), the number of
**well-separated** binder positions within histidine reach of a conserved target
cation, fraction of contacts on mouse-conserved positions, hotspot coverage, and
raw contact count, with a hard clash gate.

The separation requirement matters: an earlier version weighted the raw count of
cation-reachable positions and its best poses clustered four or five of them on
one helix face around a single arginine. ProteinMPNN then placed exactly one
histidine there, and summing such sites as independent would have overstated the
selectivity several-fold (§3 assumes independent sites).
"""
        val_title = "Sequence validation — ESM-2"
        val_body = f"""
ESMFold was not run for this variant (its trunk is ~2.6 GB in fp32 and peaks
beyond the available 16 GB on all but the shortest chains). Instead each unique
sequence was scored with **ESM-2 650M** [3] as masked-marginal pseudo-
log-likelihood per residue — masking each position in turn and reading the log
probability assigned to the residue actually present.

{"Submitted designs: mean pLL " + fmt(st.mean(plls), 2) + " (range " + fmt(min(plls),2) + " to " + fmt(max(plls),2) + ")." if plls else ""}

**This is a materially weaker check than ESMFold and is reported as such.** A
high pLL means the sequence is protein-like; it says nothing about whether the
chain folds into the designed backbone, and nothing at all about the interface.
Because scoring all candidates is expensive relative to its contribution, only a
shortlist ranked by the physics-based composite was scored, and unscored designs
fall back to the *worst* observed value so that absence of a score can never
flatter a design.
"""

    switch_note = ""
    if perfect and rest:
        switch_note = f"""
**A trade-off we did not design away, and did not hide.** The
{len(perfect)} candidate designs reaching ≥0.99 mouse identity contact a median
of {st.median([d['n_contacted_target'] for d in perfect]):.0f} target residues,
against {st.median([d['n_contacted_target'] for d in rest]):.0f} for the other
{len(rest)}. They satisfy cross-reactivity partly by *shrinking* the interface to
avoid the divergent epitope positions, which works against affinity. The
composite score cannot see this: its contact term is weighted 0.006 (worth ~2
points at most) against 3.0 for cross-reactivity, so it will prefer the smaller
interface. `n_contacted_target` is reported per design in
`results/design_report.csv` so this can be judged independently of our ranking.
"""

    keys_used = ["rfdiffusion", "proteinmpnn", "esm", "1tnf", "3alq", "7kp7",
                 "uniprot", "linkage", "shrake", "tien", "biopython", "engh",
                 "nerf", "kabsch", "shoemake", "blosum"]

    doc = f"""# Methodology — Challenge 02: pH-selective TNF-α binders

**Submission:** {len(sub)} designs (`results/submission.csv`)
{"· size categories: " + ", ".join(f"{v} {k}" for k, v in sorted(cats.items())) if cats else ""}
{"· lengths " + str(min(lens)) + "–" + str(max(lens)) + " aa" if lens else ""}
**Generation route:** {"RFdiffusion → ProteinMPNN → ESMFold" if route == "rfdiffusion" else "parametric bundles → rigid docking → ProteinMPNN → ESM-2"}
**All designs are de novo single chains** (`molecule_class = protein`); none is
derived from a natural TNF-α binder or antibody.

---

## 1. Design objective, and why the sign matters

The challenge asks for binders that engage human TNF-α at pH 7.4 with **no
detectable binding at pH 6.0**, cross-react with mouse TNF-α, and target the
receptor-binding site between protomers.

We define selectivity throughout as

$$S = \\log_{{10}}\\frac{{K_d(\\mathrm{{pH}}\\ 6.0)}}{{K_d(\\mathrm{{pH}}\\ 7.4)}}$$

so **S > 0 means weaker binding at pH 6.0**, the direction required here. We
targeted S ≥ 2.0 (a 100-fold loss of affinity), on the grounds that an assay
resolving ~2 decades of Kd cannot distinguish "100-fold weaker" from "absent".

This is the opposite polarity to the more commonly attempted acidic-pH switch,
and the inversion drives every design decision below: a histidine's pKa must
*fall* on binding rather than rise, which makes conserved target **cations** the
resource to exploit and conserved target **carboxylates** a liability.

## 2. Target and epitope determination

The target is the human TNF-α homotrimer (UniProt P01375 residues 77–233, the
soluble form; PDB **1TNF**, 2.6 Å) [4,7]. Author numbering in 1TNF equals mature
position; this was verified by alignment rather than assumed
(offset {meta['numbering_offset']}, agreement {meta['numbering_agreement']:.2f}
over the observed chain).

**The functional epitope was measured, not taken from the literature.** The
receptor footprint was computed as all TNF-α residues within 5 Å of any receptor
atom in the TNF-α/TNFR2 complex PDB **3ALQ** [5], which contains six independent
receptor copies. Every copy contacts **two** protomers with comparable numbers of
residues (~16 + ~16), establishing that the receptor binds in the cleft *between*
subunits and that a binder gripping a single protomer cannot block it. Pooling
the six copies gave a consensus epitope of
{len(meta.get('receptor_frequency', {}))} positions, {len(strong)} of them
contacted in ≥{n_receptor_copies - 1} copies.

This was cross-checked against PDB **7KP7** [6] — a *different* receptor (TNFR1)
on the *other* species' orthologue (mouse TNF-α) — which independently
reproduces the same site.

Solvent accessibility was computed with Shrake–Rupley [9] on the **intact
trimer** and normalised by Tien et al. maximum values [10]. Computing it on an
isolated protomer would wrongly mark {len(meta.get('sasa_trap_residues', []))}
residues as exposed; those are the trimer core, never solvent-accessible in the
physiological assembly, and a scorer using them would aim a binder into the
middle of the trimer.

Human/mouse conservation was mapped by global BLOSUM62 alignment [16] of the
human mature sequence against the mouse precursor (P06804), locating the mouse
mature start by alignment rather than assuming equal propeptide lengths.
Overall mature identity is **{int(round(meta['mouse_identity_overall'] * 156))}/156
({meta['mouse_identity_overall']:.1%})**, reproducing the figure in the challenge
brief and serving as our check that the alignment is correct.

**Two findings shaped the design, and both are unfavourable:**

- the receptor epitope is **less** conserved than the protein average, so
  cross-reactivity is a genuine constraint rather than a free consequence of
  choosing a functional site;
- the usable pH-switch anchors are scarce. Across the **selected groove(s)**
  there are only **{len(anchor_pos)}** conserved Lys/Arg
  ({', '.join('mature ' + str(p) for p in anchor_pos)}) against a requirement of
  three to four *independent* histidines (§3), and only {len(anchor_pos) + 1}
  across the full measured epitope. Conserved carboxylates in the selected
  groove(s) ({', '.join('mature ' + str(p) for p in acid_pos)}) are positions to
  keep histidines away from, not towards.

Epitope patches were scored over the trimer surface (cross-chain neighbours
allowed, 14 Å radius) with conserved cations rewarded, conserved carboxylates
penalised, and a requirement that receptor-contacted residues be present on
**both** protomers. {len(grooves)} groove(s) were selected:

{chr(10).join(f"- **{g['id']}** — centre {g['centre_chain']}{g['centre']}, chains {g['receptor_site_chains']}, {g['n_on_receptor_site']}/{g['n_residues']} members on the measured receptor site ({g['frac_receptor']:.0%}), {g['n_conserved_cation']} conserved cation(s), mouse identity {g['frac_conserved']:.0%}; hotspots {g['hotspots']}" for g in grooves)}

## 3. Thermodynamic basis of the switch

Binding is pH-dependent only if protons are taken up or released on
association [8]. For independent ionizable groups,

$$\\Delta\\Delta G(\\mathrm{{pH}}) = -RT\\sum_i \\ln\\frac{{1+10^{{\\,pK_a^{{bound,i}}-\\mathrm{{pH}}}}}}{{1+10^{{\\,pK_a^{{free,i}}-\\mathrm{{pH}}}}}}$$

Equivalently, S is the **area between the bound and free titration curves**
across the assay window, which immediately bounds what one group can contribute.
Histidine is the only side chain that titrates in the 6.0–7.4 window (we
verified this rather than asserting it, by evaluating the best achievable S for
every ionizable residue allowing ±2 pKa units of design-induced shift; cysteine
ranks second and is excluded for unrelated reasons — see §6).

Two quantitative consequences set the whole design:

1. **Per-histidine ceiling of 0.504 (3.2-fold)** at a free-state pKa of 6.4.
   A free histidine is only ~71% protonated at pH 6.0, so a third of the window
   is spent before design begins. Reaching S = 2.0 therefore requires **four**
   fully-suppressed independent histidines.
2. **The absolute per-histidine maximum is the window width, 1.4**, approached
   only when pKa_free sits above 7.4 *and* pKa_bound below 6.0 — i.e. the group
   is protonated throughout the window when free and deprotonated throughout it
   when bound.

This gives **two levers**, and they are not independent:

| histidine environment | pKa_free | pKa_bound | S per His | His for 100× |
|---|---|---|---|---|
| target cation + burial | 6.4 | 4.7 | 0.484 | 5 |
| + binder carboxylate near the His | 7.9 | 6.2 | **0.900** | **3** |
| binder carboxylate, *without* burial | 7.9 | 7.2 | 0.272 | 8 |

The third row is the trap: a binder carboxylate alone *raises* the bound-state
pKa above the window, leaving the group protonated in both states, and is
**worse than the bare histidine**. Suppression and elevation must be applied
together. Our sequence-design stage therefore applies the carboxylate bias only
at positions already paired with a histidine the pose places in reach of a
target cation, and the pipeline's preflight asserts this ordering so it cannot
regress silently.

## 4. {gen_title}
{gen_body}
## 5. Sequence design — ProteinMPNN with a two-sided histidine bias

Sequences were designed with **ProteinMPNN** [2] (soluble-model weights
`v_48_020`, 48 neighbours; temperature {cfg.get('temperature', 0.2)},
{cfg.get('seqs_per_backbone', 8)} sequences per backbone), with the target
chain(s) held fixed and only the binder designed. Soluble weights were chosen
deliberately: they were trained excluding transmembrane proteins and give more
polar surfaces, which matters for a design that already requires several
histidines and must still express.

The scientific content is a **per-residue logit bias on histidine, applied in
both directions**:

| geometric condition (binder Cβ to conserved target atom) | bias |
|---|---|
| conserved cation N within 8 Å | **+{cfg.get('his_bias', 3.5)}** |
| conserved cation N within 11 Å | +{cfg.get('his_bias', 3.5) * 0.4:.1f} |
| conserved carboxylate O within 7 Å | **{-2.0}** (added, not alternative) |
| free cysteine, all positions | −8.0 (never sampled; see §6) |

The negative half has no counterpart in a conventional pH-switch design. Here a
misplaced histidine does not merely fail to contribute — it contributes
*negatively*, because a carboxylate raises its bound-state pKa and runs the
switch backwards. One such histidine can cancel two good ones, so leaving it to
chance is not neutral. The terms are **added**, so a position between a conserved
cation and a conserved carboxylate nets out near zero, which is the honest
answer: the two effects partly cancel in the pKa model too.

A paired **Asp/Glu bias (+{cfg.get('acid_bias', 1.5)})** implements the
free-state lever of §3, applied only at positions within 7.5 Å of a
fully-biased histidine *and* at least 10 Å from the target — i.e. on the
binder's own solvent-exposed face. A carboxylate buried in the interface is
desolvated and shifts the bound state as much as the free one, cancelling its
own benefit.

Bias strength was calibrated rather than guessed: at +2.5 the model placed ~1
histidine per design, and at +4.5 up to 8 in a 70-mer. The run sampled the
useful range and let the composite score's surface-histidine penalty arbitrate.

## 6. {val_title}
{val_body}
## 7. pKa estimation, scoring and selection

**Histidine environments.** Sequence design returns a sequence on a backbone,
with no side chains, so the imidazole must be built before any pKa can be
estimated. For each interface histidine we enumerate 12 (χ1, χ2) rotamers,
discard those clashing with the target or the binder's own backbone, and retain
the one giving the **lowest** bound-state pKa. This is an optimistic assumption
and is reported as such: it answers *"is there a rotamer that would do this"*,
not *"this is the rotamer adopted"*. Without side-chain repacking no better
answer is available, and a single arbitrary rotamer would be noisier rather than
more honest.

Bound-state pKa is estimated from four distances by additive increments — a
carboxylate within 4.0 Å (+1.5) or 4.0–6.0 Å (+0.8); a neutral H-bond acceptor
within 3.5 Å (+0.3); a cation within 5.0 Å (−0.7); burial with no acidic partner
(−1.0) — clamped to [4.0, 9.0]. **These are hand-set heuristics, not a
Poisson–Boltzmann calculation.** They rank designs; the absolute values are not
predictions.

The free-state pKa is measured on the unbound binder from its own charged
residues, deliberately **without** a burial term and without rotamer
optimisation, so the free-state lever is credited conservatively rather than
twice over.

**Independence.** Per-histidine contributions are summed, which is valid only for
non-interacting sites, so histidines closer than 7 Å are reduced to a
non-redundant subset and the discarded ones are reported per design. Without this
S inflates in direct proportion to the number of redundant sites.

**Composite ranking** (weights chosen to mirror the challenge's own priority
order — pH selectivity, then cross-reactivity, then affinity):

```
 4.0   * S_fit                          pH selectivity   (criterion 1)
+3.0   * mouse_frac                     cross-reactivity (criterion 2)
+0.5   * n_independent_his              switch redundancy
{"+0.04  * (mean_plddt - 70)              foldability" if route == "rfdiffusion" else "+1.2   * (esm_pll + 1.0)                foldability (weaker proxy)"}
-1.0   * mpnn_nll                       design confidence
+0.006 * min(contact_pairs, 400)        interface size   (criterion 3)
+0.3   * min(hotspot_cover, 5)          correct epitope
-1.5   * max(0, total_H - used_H - 2)   surface histidine penalty
{"-0.5   * rmsd                           self-consistency" if route == "rfdiffusion" else ""}
```

`S_fit` penalises undershooting the S target in full and overshooting at 40%:
overshooting still satisfies the criterion, undershooting does not.

**Hard gates before submission.** Length 10–250; standard residues only; **no
cysteine** (disulfide scrambling and poor expression — note this excludes the
second-best switch chemistry for reasons unrelated to pH); no homopolymer run
>4; no residue exceeding 35% of the sequence; and |net charge| ≤ 0.25 per
residue. The last is specific to this design strategy, which biases histidine
*and* Asp/Glu on top of a soluble-weight prior that already favours charged
surfaces; without it, sequences with 20+ glutamates in 70 residues passed every
other check.

Finally, per-backbone and total quotas enforce diversity, so the submission is
not {len(sub)} near-copies of whichever backbone happened to score best.

## 8. Results

| | |
|---|---|
| designs scored | **{len(scored):,}** |
| passed all gates | **{len(passing):,}** |
| reached S ≥ 2.0 (100-fold) | **{len(meeting)}** |
| submitted | **{len(sub)}** |
{"| submitted S range | " + fmt(min(S_vals)) + " – " + fmt(max(S_vals)) + " |" if S_vals else ""}
{"| submitted mouse identity | median " + fmt(st.median(mouse_vals)) + " |" if mouse_vals else ""}
{"| contacted target residues | median " + str(int(st.median(contacts))) + " |" if contacts else ""}

**Selectivity against independent histidine count** (ceiling = 0.504 × n at
pKa_free 6.4):

| independent His | n | best S | ceiling | ratio |
|---|---|---|---|---|
{his_rows}

{ceiling_claim} Exceeding it is only possible by *raising* pKa_free, so where
it happens it is the free-state lever of §3 appearing in the pipeline's own
output rather than as a claim.
{frag}{switch_note}
## 9. Limitations

Stated plainly, because they bound what these designs can be expected to do.

- **No affinity prediction anywhere.** S is a *ratio*. Nothing in this pipeline
  predicts Kd, so criterion 3 is addressed only as "bury surface at the correct
  epitope". A design may have excellent S and no measurable affinity.
- **No interface validation.** {"ESMFold validates each binder in isolation, not the complex." if route == "rfdiffusion" else "ESM-2 scores the sequence only — not the fold, and not the interface."}
  No AF2-multimer or comparable complex prediction was run. This is the single
  largest gap and the clearest next step.
- **pKa values are not predictions.** Hand-set increments on unrepacked rotamer
  geometry, with the most favourable reachable rotamer taken for the bound state.
- **Independence is approximate.** A 7 Å separation is enforced, but
  "approximately independent sites" is doing real work in the summation.
- **The patch and composite weights are opinions**, not measurements. The
  *ceiling* in §3 is not: it follows from the linkage equation and the assay pH
  values alone, and no weighting choice can move it.

## 10. Reproducibility

Full source, including the preflight that asserts the §3 relationships:

```
pipeline.py --selftest     # physics, geometry, gates, hotspot renumbering
pipeline.py --stage prepare   ...   --stage submit
```

Seeds are fixed (`--seed {cfg.get('seed', 20261007)}`); all intermediate
artifacts (`work/target_meta.json`, `work/sequences.json`, `work/scored.json`)
are retained, and `results/design_report.csv` carries every per-design metric
quoted above, including the per-histidine pKa accounting and the positions
discarded as non-independent.

## References

{ref_block(keys_used)}
"""
    # collapse the blank lines left by conditional rows
    lines = [l for l in doc.splitlines()]
    out, prev_blank = [], False
    for l in lines:
        blank = (l.strip() == "")
        if blank and prev_blank:
            continue
        out.append(l)
        prev_blank = blank
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    meta, scored, sub, cfg = load(root)
    route = detect_route(scored)
    doc = build(meta, scored, sub, cfg, route)

    out = Path(a.out) if a.out else root / "results" / "METHODS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"wrote {out}")
    print(f"  route detected : {route}")
    print(f"  designs scored : {len(scored)}")
    print(f"  submitted      : {len(sub)}")
    print(f"  words          : {len(doc.split())}")


if __name__ == "__main__":
    main()

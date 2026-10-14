"""
Ranking and submission gates: turning metrics into a decision.

Two functions carry the judgement in this project. `composite_score` decides
which designs reach `submission.csv`, and `seq_ok` decides which are even
allowed to. Both are deliberately readable as statements of priority rather than
as code, because they encode the challenge's own ranking of criteria:

    1. pH-selective binding -- bind at 7.4, nothing detectable at 6.0
    2. mouse cross-reactivity
    3. affinity against human TNF-alpha

and the weights below put roughly 6 points of composite on criterion 1, 3 on
criterion 2, and under 3 on everything else combined.
"""

import re
from collections import Counter

import numpy as np

from . import linkage as L

# -------------------------------------------------- the composite


def s_fit(S_total, target_S=L.S_TARGET):
    """Turn a raw selectivity into a reward, punishing undershoot in full.

        gap   = S_total - target_S
        S_fit = -|gap|        if gap < 0   (undershoot, punished in full)
              = -0.4 * gap    if gap >= 0  (overshoot, punished at 40%)

    Same asymmetry as challenge 1, same reason: overshooting still satisfies the
    criterion, undershooting does not. The difference is the target. Challenge 1
    aimed at S = 1.5 over a 0.9-wide window; here `L.S_TARGET` is 2.0 over a
    1.4-wide window, because "no *detectable* binding" is a stronger claim than
    "weaker binding" and an assay resolving two decades of Kd cannot tell
    100-fold from gone.
    """
    gap = np.asarray(S_total, dtype=float) - target_S
    return np.where(gap < 0, -np.abs(gap), -0.4 * gap)


COMPOSITE_WEIGHTS = {
    "S_fit": 4.0,            # criterion 1 -- deliberately dominant
    "mouse_frac": 3.0,       # criterion 2
    "independent_his": 0.5,  # redundancy in the switch
    "plddt_excess": 0.04,    # (mean_plddt - 70), foldability -- ESMFold
    "esm_pll_excess": 1.2,   # (pll + 1.0), the laptop's weaker stand-in
    "mpnn_nll": -1.0,        # ProteinMPNN's own confidence (lower is better)
    "contacts": 0.006,       # min(n_contact_pairs, 400), interface size
    "hotspot_cover": 0.3,    # min(cover, 5), hit the intended epitope
    "his_excess": -1.5,      # max(0, total_H - n_used_his - 2)
    "acid_face": -0.3,       # binder acids pressed into the target interface
    "rmsd": -0.5,            # self-consistency, only if ESMFold ran
}


def composite_score(S_total, mouse_frac, n_independent_his, esm_pll, mpnn_nll,
                    n_contact_pairs, hotspot_cover, total_his, acid_face=0,
                    rmsd=0.0, mean_plddt=None, target_S=L.S_TARGET,
                    breakdown=False):
    """The ranking function. One expression, readable as a priority list.

        4.0   * S_fit                          pH selectivity   (criterion 1)
      + 3.0   * mouse_frac                     cross-reactivity (criterion 2)
      + 0.5   * n_independent_his              switch redundancy
      + 0.04  * (mean_plddt - 70)              foldability     (ESMFold)
      + 1.2   * (esm_pll + 1.0)                foldability     (laptop only)
      - 1.0   * mpnn_nll                       design confidence
      + 0.006 * min(contact_pairs, 400)        interface size  (criterion 3)
      + 0.3   * min(hotspot_cover, 5)          right epitope
      - 1.5   * max(0, total_H - used_H - 2)   surface histidine penalty
      - 0.3   * acid_face                      misplaced binder carboxylates
      - 0.5   * rmsd                           self-consistency (if available)

    On the scales: a design that exactly meets S = 2.0 earns 0 from `S_fit` and
    one at S = 1.0 loses 4 points, which no amount of pLDDT or interface size
    recovers. That is intentional and matches challenge 1's structure -- a
    well-folded binder with no pH switch should not outrank a switchable one,
    because the challenge ranks criterion 1 first.

    `his_excess` is the counterweight to the histidine bias. Calibration runs
    found ProteinMPNN placing up to 8 histidines in a 70-residue binder at
    `his_bias = 4.5`; histidines beyond the ones actually doing thermodynamic
    work cost solubility and expression and buy nothing, so each one past
    (used + 2) costs 1.5 points.
    """
    # pLDDT when ESMFold ran (HPC), ESM-2 pLL when it did not (laptop). They
    # are not interchangeable -- pLDDT is a statement about the predicted fold,
    # pLL only about the sequence -- so whichever is available contributes and
    # the other is zeroed rather than being faked from the first.
    if mean_plddt is None:
        plddt_term = 0.0
        pll_term = COMPOSITE_WEIGHTS["esm_pll_excess"] * (np.asarray(esm_pll, float) + 1.0)
    else:
        plddt_term = COMPOSITE_WEIGHTS["plddt_excess"] * (np.asarray(mean_plddt, float) - 70.0)
        pll_term = 0.0

    terms = {
        "S_fit": COMPOSITE_WEIGHTS["S_fit"] * s_fit(S_total, target_S),
        "mouse_frac": COMPOSITE_WEIGHTS["mouse_frac"] * np.asarray(mouse_frac, float),
        "independent_his": COMPOSITE_WEIGHTS["independent_his"] * np.asarray(n_independent_his, float),
        "plddt_excess": plddt_term,
        "esm_pll_excess": pll_term,
        "mpnn_nll": COMPOSITE_WEIGHTS["mpnn_nll"] * np.asarray(mpnn_nll, float),
        "contacts": COMPOSITE_WEIGHTS["contacts"] * np.minimum(np.asarray(n_contact_pairs, float), 400),
        "hotspot_cover": COMPOSITE_WEIGHTS["hotspot_cover"] * np.minimum(np.asarray(hotspot_cover, float), 5),
        "his_excess": COMPOSITE_WEIGHTS["his_excess"] * np.maximum(
            0, np.asarray(total_his, float) - np.asarray(n_independent_his, float) - 2),
        "acid_face": COMPOSITE_WEIGHTS["acid_face"] * np.asarray(acid_face, float),
        "rmsd": COMPOSITE_WEIGHTS["rmsd"] * np.asarray(rmsd, float),
    }
    total = sum(terms.values())
    return (total, terms) if breakdown else total


# -------------------------------------------------- submission gates

#: The challenge's accepted length range.
MIN_LEN, MAX_LEN = 10, 250
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def seq_ok(seq):
    """The gate before submission. Returns (ok, reason).

    Every rule is here because some generator produced something that scored
    well and would have wasted a well in the lab:

      * length outside 10-250        -- outside the challenge's accepted range
      * non-standard residue         -- cannot be ordered
      * any cysteine                 -- disulfide scrambling, poor expression
      * a run of >4 identical residues -- low-complexity collapse
      * any residue over 35% of the sequence -- same, spread out
      * net charge magnitude > 0.25/residue -- a polyanion or polycation that
        will not behave in a binding assay

    The last rule is new relative to challenge 1 and this challenge needs it:
    the design strategy deliberately biases both histidine *and* Asp/Glu
    (`mpnn.acid_pair_bias`), and ProteinMPNN with a soluble-weight prior already
    favours charged surfaces. Early sequences from this pipeline ran to 20+ Glu
    in 70 residues. Nothing else in the scoring function notices.
    """
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        return False, f"length {len(seq)} outside {MIN_LEN}-{MAX_LEN}"
    if set(seq) - STANDARD_AA:
        bad = "".join(sorted(set(seq) - STANDARD_AA))
        return False, f"non-standard residue(s) {bad}"
    if "C" in seq:
        return False, "free cysteine"
    if re.search(r"(.)\1{4,}", seq):
        return False, "homopolymer run >4"
    counts = Counter(seq)
    top_aa, top_n = counts.most_common(1)[0]
    if top_n / len(seq) > 0.35:
        return False, f"{top_aa} is {top_n / len(seq):.0%} of the sequence"
    net = sum(counts[a] for a in "KR") - sum(counts[a] for a in "DE")
    if abs(net) / len(seq) > 0.25:
        return False, f"net charge {net:+d} over {len(seq)} residues"
    return True, "ok"


def molecule_class(seq, _unused=None):
    """The challenge's `molecule_class` column. Everything here is "protein".

    All designs from this pipeline are single-chain de novo bundles, so none is
    a nanobody, scFv or Fab. The separately-ranked size categories the challenge
    scores -- microbinder <40 aa, minibinder 40-100 aa, large >100 aa -- are a
    property of length rather than of this column, and `size_category` reports
    them for the design report.
    """
    return "protein"


def size_category(seq):
    """Which separately-ranked size bin a design falls in.

    The challenge scores microbinders (<40 aa), minibinders (40-100) and large
    binders (>100) as distinct categories, so a short design is not competing
    against a long one. Reported in the design report rather than the
    submission CSV, which only takes `molecule_class`.
    """
    n = len(seq)
    if n < 40:
        return "microbinder"
    if n <= 100:
        return "minibinder"
    return "large"


# -------------------------------------------------- reporting

def score_breakdown(designs):
    """The table relating selectivity to independent histidine count.

    The most instructive output of the run, for the same reason challenge 1's
    equivalent was: it shows the per-histidine ceiling appearing in your own
    data rather than as a claim in a docstring. The `ceiling` column is
    `n * L.ceiling_one_group()` at the default free-state pKa -- designs that
    beat it are ones where the free-state lever (`mpnn.acid_pair_bias`) raised
    pKa_free above 6.4, which is exactly what it is for.

    `designs` is an iterable of dicts with `n_independent_his` and `S`.
    """
    groups = {}
    for d in designs:
        groups.setdefault(int(d["n_independent_his"]), []).append(float(d["S"]))

    c1 = L.ceiling_one_group()
    lines = ["  reverse pH selectivity vs independent interface histidines",
             "  " + "-" * 74,
             f"  {'His':>3} {'n':>5} {'mean S':>8} {'best S':>8} {'best fold':>11} "
             f"{'ceiling*':>9}  {'bar (best S)':<14}",
             "  " + "-" * 74]
    for nh in sorted(groups):
        ss = groups[nh]
        best = max(ss)
        bar = "#" * min(14, int(round(best / 0.15)))
        lines.append(f"  {nh:>3} {len(ss):>5} {sum(ss) / len(ss):>8.2f} {best:>8.2f} "
                     f"{10 ** best:>10.0f}x {nh * c1:>8.2f}  {bar:<14}")
    lines += ["  " + "-" * 74,
              f"  * ceiling = {c1:.3f} x His at pKa_free 6.4, the hard limit over pH 6.0-7.4.",
              f"    Designs above it raised pKa_free instead; S_TARGET = {L.S_TARGET}.",
              "    Per-His maximum is the window width, 1.4, at pKa_free>7.4 and pKa_bound<6.0."]
    return "\n".join(lines)


def quota_filter(ranked, max_total=40, max_per_scaffold=4, max_per_groove=20):
    """Apply the submission quotas, keeping diversity.

    The challenge allows up to 40 designs in track 1 and forbids duplicates. A
    greedy top-40 would hand in forty near-copies off whichever scaffold
    happened to dock best, so this caps per-scaffold and per-groove counts
    first. Returns (kept, rejected_with_reasons).
    """
    kept, rejected = [], []
    seen_seq = set()
    per_scaffold, per_groove = Counter(), Counter()

    for d in ranked:
        seq = d["sequence"]
        if seq in seen_seq:
            rejected.append({**d, "rejected": "duplicate sequence"})
            continue
        sid = d.get("scaffold_id", "?")
        gid = d.get("groove_id", "?")
        if per_scaffold[sid] >= max_per_scaffold:
            rejected.append({**d, "rejected": f"scaffold quota ({max_per_scaffold})"})
            continue
        if per_groove[gid] >= max_per_groove:
            rejected.append({**d, "rejected": f"groove quota ({max_per_groove})"})
            continue
        if len(kept) >= max_total:
            rejected.append({**d, "rejected": f"total quota ({max_total})"})
            continue
        seen_seq.add(seq)
        per_scaffold[sid] += 1
        per_groove[gid] += 1
        kept.append(d)
    return kept, rejected

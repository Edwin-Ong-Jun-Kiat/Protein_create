"""
Thermodynamic linkage for challenge 2 -- and it runs the OPPOSITE WAY to challenge 1.

Challenge 1 (EGFR) wanted binding at acidic pH and none at blood pH. Challenge 2
wants the reverse: bind human TNF-alpha at pH 7.4, *no detectable binding* at
pH 6.0. Same equation, opposite sign, and the consequences are not symmetric.

The one equation
----------------
A binding equilibrium is pH-dependent only if the binder and target take up or
release protons when they associate. For an ionizable group with acid
dissociation constant pKa_free when the binder is free and pKa_bound when it is
bound:

    ddG(pH) = -RT * sum_i ln[ (1 + 10^(pKa_bound,i - pH)) / (1 + 10^(pKa_free,i - pH)) ]

If pKa_bound > pKa_free the group *prefers to be protonated when bound*, so
cheap protons (low pH) stabilise the complex -- that was challenge 1, and the
design move was a histidine salt-bridged to a target carboxylate.

Challenge 2 needs the complex to *fall apart* at low pH, so we need the group to
prefer being **deprotonated** when bound:

    pKa_bound < pKa_free        <-- the whole of challenge 2 in one line

Protonating it then costs binding energy. The design move is therefore the exact
inverse of challenge 1: place binder histidines against target **cations** and in
apolar burial (which suppress the bound-state pKa), and keep them **away** from
target carboxylates, which would raise it and push the switch the wrong way.

Sign convention used everywhere in this package
-----------------------------------------------
    S = log10( Kd(pH 6.0) / Kd(pH 7.4) )

S > 0 means weaker binding at pH 6.0, which is the direction challenge 2 asks
for. S = 2 means a 100-fold loss of affinity on going from blood to endosome.
Note this is the *reverse* of `chal_1_local`'s `S = log10(Kd(7.4)/Kd(6.5))`;
`selectivity_chal1_convention` is provided so the two can be compared directly.

Everything here is pure numpy. No structures, no models, microseconds.
"""

import numpy as np

#: numpy renamed `trapz` to `trapezoid` in 2.0 and the old name now warns or is
#: gone. This project runs on a laptop with numpy 2.x AND in a conda env pinned
#: around torch 2.0.1+cu118, which ships numpy 1.2x -- where `np.trapezoid` does
#: not exist at all. Resolving it once here keeps `selectivity_as_integral`
#: working on both; without it the HPC selftest dies with
#: `AttributeError: module 'numpy' has no attribute 'trapezoid'`.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

RT_KCAL = 0.5921  # kcal/mol at 298 K
LN10 = np.log(10.0)

PH_BIND = 7.4    # blood / physiological -- we WANT binding here
PH_RELEASE = 6.0 # acidic endosome -- we want NO detectable binding here

#: Free-histidine side-chain pKa. The only proteinogenic side chain with a pKa
#: inside the 6.0-7.4 window, which is why `why_histidine_only` exists below.
HIS_PKA_FREE = 6.4

#: What "no detectable binding" means numerically. A BLI/SPR assay that resolves
#: Kd over ~2 decades cannot distinguish "100x weaker" from "gone", so this is
#: the practical bar the challenge's criterion 1 sets.
S_TARGET = 2.0


def theta_protonated(ph, pka):
    """Fraction of a group that is protonated (Henderson-Hasselbalch).

    theta = 1 / (1 + 10^(pH - pKa))
    """
    ph = np.asarray(ph, dtype=float)
    pka = np.asarray(pka, dtype=float)
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def ddg_bind(ph, pka_bound, pka_free=HIS_PKA_FREE):
    """pH-dependent part of the binding free energy, kcal/mol. Negative = stabilising.

    `pka_bound` may be a scalar or an array of independent groups; the sum runs
    over the last axis. `pka_free` broadcasts against it, so each histidine can
    carry its own free-state pKa -- which matters here (see `selectivity`).
    """
    ph = np.asarray(ph, dtype=float)
    pka_bound = np.atleast_1d(np.asarray(pka_bound, dtype=float))
    pka_free = np.broadcast_to(
        np.atleast_1d(np.asarray(pka_free, dtype=float)), pka_bound.shape
    )
    ph_e = ph[..., None] if ph.ndim else ph
    num = np.log1p(10.0 ** (pka_bound - ph_e))
    den = np.log1p(10.0 ** (pka_free - ph_e))
    return -RT_KCAL * np.sum(num - den, axis=-1)


def selectivity(pka_bound, pka_free=HIS_PKA_FREE,
                ph_release=PH_RELEASE, ph_bind=PH_BIND):
    """S = log10( Kd(pH 6.0) / Kd(pH 7.4) ).  S > 0 is what challenge 2 wants.

    Both pKa arguments are per-group, because challenge 2 has **two** levers and
    challenge 1 effectively had one:

      * lower `pka_bound`  -- suppress the His pKa in the complex (target
        cations, apolar burial). This is the lever chal_1 used, inverted.
      * raise `pka_free`   -- *elevate* the His pKa in the free binder, e.g. by
        a binder-internal carboxylate next to it. A His that is more protonated
        when free has more to lose on binding at low pH.

    The second lever is easy to miss and it is worth more than it looks:
    `ceiling_one_group` is 0.50 at pKa_free 6.4 but 0.90 at pKa_free 7.0.
    """
    d_release = ddg_bind(ph_release, pka_bound, pka_free)
    d_bind = ddg_bind(ph_bind, pka_bound, pka_free)
    return (d_release - d_bind) / (RT_KCAL * LN10)


def selectivity_as_integral(pka_bound, pka_free=HIS_PKA_FREE,
                            ph_release=PH_RELEASE, ph_bind=PH_BIND, n=4001):
    """The same S, computed the way that explains where the ceiling comes from.

    The closed form above is algebraically identical to

        S = integral from pH_release to pH_bind of
                [ theta_free(pH) - theta_bound(pH) ] dpH

    i.e. S is the *area between the two titration curves* over the pH window --
    with free on top, where challenge 1 had bound on top. Because theta is a
    fraction in [0, 1], the width of the window (1.4 pH units) is a hard bound
    on what one group can contribute, and `ceiling_one_group` sharpens it.

    Checked against the closed form in `check_env.py` to 1e-9.
    """
    pka_bound = np.atleast_1d(np.asarray(pka_bound, dtype=float))
    pka_free = np.broadcast_to(
        np.atleast_1d(np.asarray(pka_free, dtype=float)), pka_bound.shape
    )
    grid = np.linspace(ph_release, ph_bind, n)
    gap = (theta_protonated(grid[:, None], pka_free)
           - theta_protonated(grid[:, None], pka_bound))
    return float(_TRAPZ(gap.sum(axis=-1), grid))


def ceiling_one_group(pka_free=HIS_PKA_FREE,
                      ph_release=PH_RELEASE, ph_bind=PH_BIND):
    """Best possible S from ONE group with this free-state pKa.

    Take pka_bound -> -infinity (never protonated when bound, theta_bound = 0):

        S_max = integral theta_free dpH
              = log10[ (1 + 10^(pKa_free - pH_release)) /
                       (1 + 10^(pKa_free - pH_bind)) ]

    For histidine at pKa_free = 6.4 this is **0.504**, i.e. 3.2-fold per
    histidine. That number is the single most important fact in this challenge,
    and it is worse than challenge 1's 0.69 for a concrete reason: at pH 6.0 a
    free histidine is only 71% protonated, so even a perfectly suppressed
    histidine cannot release a full proton's worth of binding energy.

    Combined with `S_TARGET = 2.0`, it forces **4 interface histidines minimum**
    (see `n_his_required`) -- where challenge 1 needed 2. The whole shape of the
    pipeline follows from this one line.
    """
    return float(np.log10(
        (1.0 + 10.0 ** (pka_free - ph_release)) /
        (1.0 + 10.0 ** (pka_free - ph_bind))
    ))


def n_his_required(s_target=S_TARGET, pka_free=HIS_PKA_FREE,
                   ph_release=PH_RELEASE, ph_bind=PH_BIND):
    """How many perfectly-suppressed histidines `s_target` needs. Ceil of a ratio.

    At the defaults: ceil(2.0 / 0.504) = 4. This is a *lower* bound -- it assumes
    every histidine is fully suppressed (pKa_bound << 6) and fully independent.
    Real designs land below it, so the pipeline asks for 4-6 and expects to pay
    for the extra ones in solubility and expression.
    """
    c = ceiling_one_group(pka_free, ph_release, ph_bind)
    if c <= 0:
        return np.inf
    return int(np.ceil(s_target / c))


def kd_ratio(S):
    """Fold-change in Kd between pH 6.0 and pH 7.4."""
    return 10.0 ** np.asarray(S, dtype=float)


def kd_profile(ph, pka_bound, kd_ref_nm=10.0, ph_ref=PH_BIND,
               pka_free=HIS_PKA_FREE):
    """Predicted Kd (nM) across a pH range, anchored to `kd_ref_nm` at `ph_ref`.

    Anchored at pH 7.4 by default, because that is where challenge 2 measures
    affinity (criterion 3) -- the curve then rises as pH falls.
    """
    d = ddg_bind(ph, pka_bound, pka_free)
    d_ref = ddg_bind(ph_ref, pka_bound, pka_free)
    return kd_ref_nm * np.exp((d - d_ref) / RT_KCAL)


def selectivity_chal1_convention(pka_bound, pka_free=HIS_PKA_FREE):
    """S in `chal_1_local`'s sign convention, log10(Kd(7.4)/Kd(6.5)), for comparison.

    Exists so the two projects' numbers can be put side by side without anyone
    having to remember which way round each one was. A design that is good for
    challenge 2 has a large positive `selectivity` here and a large *negative*
    value in this function.
    """
    return -float(selectivity(pka_bound, pka_free, ph_release=6.5, ph_bind=7.4))


# ---------------------------------------------------------------- why histidine

#: Side-chain pKa values in water, for the argument in `why_histidine_only`.
#: Standard reference values; treated as approximate and only the *ordering*
#: and distance from the 6.0-7.4 window matter to the conclusion.
SIDECHAIN_PKA_FREE = {
    "Asp": 3.9, "Glu": 4.3, "His": 6.4, "Cys": 8.4,
    "Tyr": 10.1, "Lys": 10.5, "Arg": 12.5,
}


def why_histidine_only(ph_release=PH_RELEASE, ph_bind=PH_BIND, max_shift=2.0):
    """Show that histidine is the only usable residue, rather than assuming it.

    For each ionizable side chain, ask the best S it could deliver if design
    could shift its pKa by up to `max_shift` units in the favourable direction
    (a generous allowance -- 2 units is about the most a protein interior does
    to a surface group).

    The point of computing it: challenge 2's window is 6.0-7.4, and a group only
    contributes if it actually *titrates* in that window. Carboxylates fail for a
    reason that is easy to get backwards -- to help here they would need
    pKa_bound < pKa_free, but they already sit near 4, so suppressing them
    further changes nothing, and *raising* them (what burial does) pushes the
    switch the wrong way entirely. Lys/Arg/Tyr are far above the window.
    """
    rows = []
    for name, pka_f in SIDECHAIN_PKA_FREE.items():
        # best case: free state shifted up, bound state shifted down
        best = float(selectivity([pka_f - max_shift], pka_f + max_shift,
                                 ph_release, ph_bind))
        rows.append({
            "residue": name,
            "pka_free_water": pka_f,
            "titrates_in_window": ph_release - 1.0 <= pka_f <= ph_bind + 1.0,
            "best_S_with_2_unit_shifts": round(best, 4),
            "best_fold": round(float(kd_ratio(best)), 2),
        })
    return sorted(rows, key=lambda r: -r["best_S_with_2_unit_shifts"])


def suppression_table(pka_free=HIS_PKA_FREE, deltas=(0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0)):
    """S per histidine as a function of how much its bound-state pKa is suppressed.

    The table that sets the pipeline's geometric thresholds. Its shape is the
    useful part: S saturates fast. A delta of -2.0 already captures 98% of the
    ceiling, so there is no benefit to chasing extreme suppression at one site --
    the way to a bigger switch is **more histidines**, not better ones. That is
    why `pka.py` scores a design on how many independent suppressed sites it has
    rather than on its single best one.
    """
    c = ceiling_one_group(pka_free)
    rows = []
    for d in deltas:
        s = float(selectivity([pka_free + d], pka_free))
        rows.append({
            "delta_pka": d,
            "pka_bound": round(pka_free + d, 2),
            "S_per_his": round(s, 4),
            "fold_per_his": round(float(kd_ratio(s)), 2),
            "frac_of_ceiling": round(s / c, 3) if c else 0.0,
            "n_his_for_100x": (int(np.ceil(2.0 / s)) if s > 0 else None),
        })
    return rows

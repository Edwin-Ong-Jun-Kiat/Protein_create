#!/usr/bin/env python3
"""AfDesign binder hallucination against the TfR1 apical domain, on a V100.

This is the AlphaFold-style half of challenge 04. Where the MI300A pipeline
builds a binder out of parts -- generate a backbone, dock it rigidly, let
ProteinMPNN pick a sequence, check it with ESMFold -- AfDesign does the whole
thing in one loop:

    start from a random sequence of length L
    fold target+binder with AF2
    read a loss off the prediction (is the binder folded? is it touching the
        hotspots? is the interface confident?)
    take the GRADIENT of that loss with respect to the sequence
    nudge the sequence downhill, and repeat

The sequence is the optimisation variable and the structure predictor is the
objective. That gradient is the thing ROCm could not provide: AF2's backward
pass needs an SVD, and jax-rocm7-plugin 0.6.0 registers no hipsolver_gesvd_ffi,
so `grad_fn` died on the MI300A no matter which flags were set. On CUDA the
solver is there.

    python af_design.py --probe              # 4 steps, timings, GPU proof
    python af_design.py --n 8                # 8 designs
    python af_design.py --n 24 --binder-len 65

Modes
-----
--probe is deliberately cheap and it exists because of a lesson from this
project: the first MI300A ESMFold job spent 3 h 20 min folding under 10 of 36
sequences before anyone looked at a GPU utilisation number. A probe reports
step-1 time (which is dominated by XLA compiling the AF2 graph) separately from
steady-state step time, because conflating them is how a 227 s/step CPU run can
be mistaken for a slow GPU.

The histidine bias
------------------
The whole competition metric is S = log10(Kd(6.0) / Kd(7.4)), and histidine is
the only residue whose pKa sits inside that window -- it is neutral at 7.4 and
protonated at 6.0. A binder can only be pH-selective if its interface carries
His in places where that charge change matters. AfDesign optimises AF2
confidence, which has no opinion about pH whatsoever, so left alone it will
not put His anywhere in particular.

--his-bias adds a constant to the histidine logit at every position, which
tilts the sequence prior without forcing any position. It is a weak,
interpretable knob, which is the point: a hard His constraint at hand-picked
positions would be me designing the switch and AF2 merely agreeing. The
resulting His count and placement is reported per design so the bias can be
calibrated against something.

Nothing here scores S. That is `pipeline.py --stage score` on the main track,
and running these sequences through it is the comparison this track exists for.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Mirrors the main pipeline's homopolymer gate, which rejected 4 of 12
# otherwise-passing designs in job 13459. Worth knowing before submission, not
# after.
MAX_HOMOPOLYMER = 4


def log(msg: str = "") -> None:
    print(msg, flush=True)


def longest_run(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def net_charge(seq: str) -> int:
    return sum(seq.count(c) for c in "KR") - sum(seq.count(c) for c in "DE")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path,
                    default=Path(os.environ.get("AF_TARGET",
                                                ROOT / "work" / "target_af_k382.pdb")))
    ap.add_argument("--data-dir", type=Path,
                    default=Path(os.environ.get("AF_PARAMS", ROOT / "store" / "af_params")))
    ap.add_argument("--chain", default="A")
    ap.add_argument("--hotspots", default="A382,A193,A189,A381,A158,A380",
                    help="target residues the binder is pushed towards")
    ap.add_argument("--binder-len", type=int, default=55)
    ap.add_argument("--n", type=int, default=8, help="number of designs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recycles", type=int, default=1,
                    help="AF2 recycles. 1 is BindCraft's design-time default; "
                         "3 is more accurate and ~2x the cost per step.")
    ap.add_argument("--soft-iters", type=int, default=100)
    ap.add_argument("--temp-iters", type=int, default=50)
    ap.add_argument("--hard-iters", type=int, default=10)
    ap.add_argument("--his-bias", type=float, default=1.5,
                    help="added to the histidine logit at every position; "
                         "0 disables it")
    ap.add_argument("--probe", action="store_true",
                    help="4 steps with timings, then exit; no designs written")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even if no GPU is visible (227 s/step -- only "
                         "useful for checking the code path)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    # ----------------------------------------------------------------- devices
    # Imported before colabdesign so the device report comes first: a CPU
    # fallback must be visible in the log's first ten lines, not inferred from
    # the runtime an hour later.
    import jax

    devs = jax.devices()
    log("=" * 62)
    log(" AfDesign -- TfR1 apical binder hallucination")
    log("=" * 62)
    log(f"jax {jax.__version__}  devices: "
        f"{[(d.platform, d.device_kind) for d in devs]}")
    on_gpu = any(d.platform == "gpu" for d in devs)
    if on_gpu:
        d0 = next(d for d in devs if d.platform == "gpu")
        log(f"GPU: {d0.device_kind}")
        # V100 is sm_70. Flagging the capability makes a future "why did this
        # stop working on the new nodes" one line instead of a bisect.
        cc = getattr(d0, "compute_capability", None)
        if cc:
            log(f"compute capability: {cc}")
    else:
        msg = ("no GPU visible to jax. On the MI300A this was a missing\n"
               "hipsolver_gesvd_ffi kernel; here it is more likely the CUDA\n"
               "wheel flavour vs the driver. Check the driver version printed\n"
               "by submit.sbatch and, if it is >= 525, rebuild with\n"
               "    AF_CUDA=12 bash install_af.sh --build")
        if not a.allow_cpu:
            log("\nERROR: " + msg)
            log("\nPass --allow-cpu to run anyway at ~227 s/step.")
            return 2
        log("WARNING: " + msg)

    if not a.target.exists():
        log(f"ERROR: target not found: {a.target}")
        log("Run: bash install_af.sh   (it calls build_af_target.py)")
        return 1
    params_dir = a.data_dir / "params"
    if not (params_dir / "params_model_1_ptm.npz").exists():
        log(f"ERROR: AF2 parameters not found under {params_dir}")
        log("Run: bash install_af.sh --build   on a LOGIN node")
        return 1

    cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    log(f"target : {a.target.name}")
    log(f"params : {params_dir}")
    log(f"cache  : {cache or 'NONE -- the AF2 graph recompiles every job (~250 s)'}")
    log(f"binder : {a.binder_len} aa, hotspots {a.hotspots}, "
        f"{a.recycles} recycle(s)")
    log("")

    # ------------------------------------------------------------- the model
    from colabdesign import clear_mem, mk_afdesign_model
    from colabdesign.af.alphafold.common import residue_constants

    t0 = time.time()
    af = mk_afdesign_model(protocol="binder", data_dir=str(a.data_dir),
                           use_multimer=False, num_recycles=a.recycles)
    af.prep_inputs(pdb_filename=str(a.target), chain=a.chain,
                   binder_len=a.binder_len, hotspot=a.hotspots)
    log(f"model built and inputs prepped in {time.time() - t0:.1f} s")
    log(f"target_len={af._target_len} binder_len={af._binder_len}")

    # --------------------------------------------------------- the His bias
    # Built once; shape (L, 20) broadcasts over the leading batch axis.
    bias = None
    if a.his_bias:
        import numpy as np
        h = residue_constants.restypes.index("H")
        bias = np.zeros((a.binder_len, 20), dtype=np.float32)
        bias[:, h] = a.his_bias
        log(f"histidine bias +{a.his_bias} at alphabet index {h} "
            f"('{residue_constants.restypes[h]}')")

    def restart(seed: int) -> str:
        """Reset the optimiser; return how the bias actually got applied.

        ColabDesign's restart()/set_seq() signature has moved between
        releases, and `rm_aa` and `bias` are the two arguments most likely to
        have. A TypeError here must not cost a queued job, so each is tried
        and the fallback is reported rather than assumed.
        """
        kw = {"seed": seed, "rm_aa": "C"}
        if bias is not None:
            try:
                af.restart(bias=bias, **kw)
                return "restart(bias=...)"
            except TypeError:
                pass
        try:
            af.restart(**kw)
        except TypeError:
            af.restart(seed=seed)
            log("  note: rm_aa='C' not accepted; cysteines are NOT excluded")
        if bias is not None:
            try:
                af.opt["bias"] = bias
                return "opt['bias']"
            except Exception as e:  # noqa: BLE001
                log(f"  WARNING: could not apply the histidine bias ({e}); "
                    f"running unbiased")
                return "none"
        return "none"

    # ----------------------------------------------------------------- probe
    if a.probe:
        how = restart(a.seed)
        log(f"bias applied via: {how}")
        log("")
        log("timing 4 single steps. Step 1 includes XLA compiling the whole")
        log("AF2 forward+backward graph, so it is NOT the step cost.")
        times = []
        for i in range(4):
            t = time.time()
            af.design(1)
            dt = time.time() - t
            times.append(dt)
            lg = af.aux.get("log", {})
            log(f"  step {i + 1}: {dt:7.1f} s   "
                f"loss {lg.get('loss', float('nan')):.2f}  "
                f"plddt {lg.get('plddt', float('nan')):.2f}  "
                f"i_ptm {lg.get('i_ptm', float('nan')):.2f}")
        steady = sum(times[1:]) / len(times[1:])
        log("")
        log(f"step 1 (compile + run) : {times[0]:.1f} s")
        log(f"steady state           : {steady:.1f} s/step")
        total = a.soft_iters + a.temp_iters + a.hard_iters
        log(f"=> one {total}-step design ~= "
            f"{(times[0] + steady * (total - 1)) / 60:.1f} min")
        log(f"=> {a.n} designs ~= "
            f"{(times[0] + steady * (total - 1) * a.n) / 3600:.1f} h "
            f"(step 1 amortised once if the XLA cache is warm)")
        log("")
        log("For reference, measured on the MI300A: 227.4 s/step on CPU,")
        log("and a GPU run that never produced a step at all.")
        if steady > 60 and on_gpu:
            log("")
            log("WARNING: over 60 s/step with a GPU present. That is CPU-like.")
            log("Check that the AF2 graph is not silently falling back -- look")
            log("for 'Unable to initialize backend' earlier in this log.")
        return 0

    # --------------------------------------------------------------- designs
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = a.out_dir / "af_pdb"
    pdb_dir.mkdir(exist_ok=True)
    csv_path = a.out_dir / "af_designs.csv"
    fields = ["name", "seed", "length", "sequence", "loss", "plddt", "ptm",
              "i_ptm", "i_con", "rmsd", "n_his", "net_charge",
              "longest_run", "homopolymer_ok", "seconds"]
    rows = []

    total_steps = a.soft_iters + a.temp_iters + a.hard_iters
    log(f"designing {a.n} binders, 3-stage "
        f"({a.soft_iters} soft / {a.temp_iters} temp / {a.hard_iters} hard) "
        f"= {total_steps} steps each")
    log("")
    run_t0 = time.time()

    for k in range(a.n):
        seed = a.seed + k
        name = f"af_k382_{a.binder_len}aa_s{seed:03d}"
        t = time.time()
        log(f"--- {k + 1}/{a.n}  {name}")
        how = restart(seed)
        if k == 0:
            log(f"    bias via {how}")
        # 3-stage is ColabDesign's standard schedule: optimise a continuous
        # (soft) sequence, anneal the temperature, then commit to one-hot.
        # Going straight to hard is what collapses a design into a
        # low-complexity, heavily charged sequence -- the main track's job
        # 13459 produced SAAEAKKEKERERFLEHAEKHGGDEGGHARAEHLHHHNEHAEEE, which
        # passes the homopolymer gate (longest run 3) yet carries net charge
        # -6 over 44 residues and raises obvious expressibility questions.
        # Those are what net_charge and the per-design report are here to
        # surface; the gate alone would not have caught it.
        af.design_3stage(a.soft_iters, a.temp_iters, a.hard_iters)

        seq = af.get_seq()[0]
        lg = dict(af.aux.get("log", {}))
        dt = time.time() - t
        run = longest_run(seq)
        row = {
            "name": name, "seed": seed, "length": len(seq), "sequence": seq,
            "loss": round(float(lg.get("loss", float("nan"))), 4),
            "plddt": round(float(lg.get("plddt", float("nan"))), 4),
            "ptm": round(float(lg.get("ptm", float("nan"))), 4),
            "i_ptm": round(float(lg.get("i_ptm", float("nan"))), 4),
            "i_con": round(float(lg.get("i_con", float("nan"))), 4),
            "rmsd": round(float(lg.get("rmsd", float("nan"))), 4),
            "n_his": seq.count("H"),
            "net_charge": net_charge(seq),
            "longest_run": run,
            "homopolymer_ok": int(run <= MAX_HOMOPOLYMER),
            "seconds": round(dt, 1),
        }
        rows.append(row)
        af.save_pdb(str(pdb_dir / f"{name}.pdb"))

        log(f"    {seq}")
        log(f"    plddt {row['plddt']:.2f}  i_ptm {row['i_ptm']:.2f}  "
            f"i_con {row['i_con']:.2f}  His {row['n_his']}  "
            f"charge {row['net_charge']:+d}  run {run}"
            f"{'' if row['homopolymer_ok'] else '  <-- FAILS the homopolymer gate'}")
        log(f"    {dt / 60:.1f} min")

        # Written after every design, not at the end. A 24 h wall clock that
        # expires mid-run must not take the finished designs with it.
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        clear_mem()

    elapsed = time.time() - run_t0
    log("")
    log("=" * 62)
    log(f"{len(rows)} designs in {elapsed / 3600:.2f} h "
        f"({elapsed / max(len(rows), 1) / 60:.1f} min each)")
    log(f"sequences : {csv_path}")
    log(f"structures: {pdb_dir}")

    if rows:
        best = max(rows, key=lambda r: r["i_ptm"])
        log("")
        log(f"best i_ptm: {best['name']}  i_ptm {best['i_ptm']:.2f}  "
            f"plddt {best['plddt']:.2f}  His {best['n_his']}")
        clean = [r for r in rows if r["homopolymer_ok"]]
        log(f"{len(clean)}/{len(rows)} pass the homopolymer gate "
            f"(longest run <= {MAX_HOMOPOLYMER})")
        with_his = [r for r in rows if r["n_his"] >= 3]
        log(f"{len(with_his)}/{len(rows)} carry >= 3 histidines")
        log("")
        log("NOTE: i_ptm and pLDDT say AF2 believes in the complex. Neither is")
        log("pH selectivity. Score these against S with the main track:")
        log("  cp results/af_designs.csv ../chal_4/work/af_candidates.csv")
        log("  cd ../chal_4 && \"$PHB_PY\" pipeline.py --stage score")

        (a.out_dir / "af_summary.json").write_text(json.dumps({
            "n": len(rows), "binder_len": a.binder_len,
            "hotspots": a.hotspots, "recycles": a.recycles,
            "steps_per_design": total_steps, "his_bias": a.his_bias,
            "hours": round(elapsed / 3600, 3),
            "best_i_ptm": best["i_ptm"], "best_name": best["name"],
            "n_homopolymer_ok": len(clean), "n_his_ge_3": len(with_his),
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

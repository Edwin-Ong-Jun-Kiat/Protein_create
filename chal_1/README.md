# EGFR pH-Selective Binder Design — Anthropic × Adaptyv 2026, Challenge 01

De novo design pipeline for pH-selective EGFR Domain III binders, built to run
unattended on a SLURM HPC with **NVIDIA V100** GPUs.

---

## 1. What the challenge actually rewards

From the challenge spec, criteria are ranked:

| Priority | Criterion | How this pipeline addresses it |
|---|---|---|
| **1** | **pH-selective binding** — binds at pH 6.5, no detectable binding at pH 7.4 | Interface histidine engineering, driven by a thermodynamic-linkage score (§3) |
| **2** | Mouse EGFR cross-reactivity | Epitope patches pre-filtered for human/mouse sequence identity |
| **3** | Affinity to human EGFR | RFdiffusion + ProteinMPNN + ESMFold self-consistency filtering |

Criterion 1 is the one almost nobody optimizes directly, and it is worth far more
than a marginal affinity gain. The pipeline therefore treats pH selectivity as the
**primary design objective**, not a post-hoc filter.

Target: PDB **6ARU** chain A (EGFR extracellular domain, 3.2 Å).
Recommended epitope: **Domain III** = mature residues **310–481**.

> Numbering note: UniProt P00533 includes a 24-residue signal peptide, so
> `P00533 position = mature position + 24`. The spec's "residues 25–645" is
> P00533 numbering; EGFR crystal structures use mature numbering. The pipeline
> resolves the offset by sequence alignment (never by assumption) and logs it.

---

## 2. Design strategy

### Stage 0 — Data-driven epitope selection
Rather than hardcoding the cetuximab epitope, the pipeline computes it:

1. Parse 6ARU chain A; strip glycans/waters/heteroatoms.
2. Align observed chain-A sequence to P00533 mature sequence → residue offset.
3. Per-residue relative SASA (Shrake–Rupley) → surface mask.
4. Align human vs **mouse** EGFR (UniProt Q01279) → conservation mask.
5. Score every surface Domain III patch on:
   - count of **conserved acidic residues** (Asp/Glu) — these are the H-bond/salt-bridge
     partners that make a binder histidine pH-responsive,
   - overall human/mouse identity across the patch (criterion 2),
   - penalty for excess basic residues (which would repel a protonated His⁺).
6. Emit the top non-redundant patches as RFdiffusion `hotspot_res` sets.

This makes criteria 1 and 2 *structural constraints on where we bind*, rather than
things we hope for later.

### Stage 1 — Backbone generation (RFdiffusion)
Binder scaffolding against truncated Domain III, hotspot-conditioned,
`noise_scale=0` (the Baker-lab binder protocol). Lengths sampled to populate the
challenge's stratification bins: microbinder (<40), minibinder (40–100), large (>100).

### Stage 2 — Sequence design (ProteinMPNN)
Target chain fixed, binder chain designed, soluble-model weights, cysteine omitted.
Crucially, a **per-position histidine bias** is injected via
`--bias_by_res_jsonl` at interface positions facing conserved target carboxylates.
Each backbone yields both a pH-biased and an unbiased sequence set, so the scorer
can choose rather than be forced.

### Stage 3 — Structure validation (ESMFold)
Fold each binder alone; keep designs with high pLDDT and low RMSD to the diffused
backbone (self-consistency). The ESMFold model is then **superimposed back onto the
diffused backbone**, which supplies real binder side-chain coordinates in the
complex frame — combined with the crystallographic target side chains from 6ARU,
this gives genuine His–carboxylate geometry for Stage 4.

### Stage 4 — pH + cross-reactivity scoring (§3)

### Stage 5 — Ranked `submission.csv`

---

## 3. The pH-selectivity score

Histidine is the only residue with a side-chain pKa (~6.0–6.5 free) inside the
6.5 / 7.4 window, so it is the only practical switch. The physics is standard
thermodynamic linkage: if an ionizable group's pKa is **higher in the bound state
than free**, binding gets stronger as pH drops.

For each ionizable group *i*:

```
ΔΔG_bind(pH) = -RT · Σ ln[ (1 + 10^(pKa_bound,i − pH)) / (1 + 10^(pKa_free,i − pH)) ]
```

The reported selectivity is the predicted log-fold Kd shift:

```
S = log10( Kd(7.4) / Kd(6.5) )
```

`S > 0` means tighter binding at pH 6.5 — the desired direction.

### Why you need at least two histidines (the binding constraint on this challenge)

Take the limit of the equation above as `pKa_bound → ∞`. The selectivity from a
single ionizable group converges to exactly:

```
S_max(one group) = pH_high − pH_low = 7.4 − 6.5 = 0.9   →  8× at most
```

That is a hard thermodynamic ceiling, not a modelling artifact. And a *realistic*
salt-bridged histidine (pKa 6.4 → 7.9) only reaches **S = 0.59, i.e. 4×**:

| Interface His (each salt-bridged) | Predicted S | Kd ratio 7.4 / 6.5 | Absolute ceiling |
|---|---|---|---|
| 1 | 0.59 | **4×** | 0.9 → 8× |
| 2 | 1.17 | 15× | 1.8 → 63× |
| 3 | 1.76 | 57× | 2.7 → 500× |
| 4 | 2.34 | 219× | 3.6 → 4000× |

**A single-histidine design cannot win criterion 1.** A 4× shift is not
"no detectable binding at pH 7.4" — it is the same binder, slightly weaker.
This is why `--min-interface-his` defaults to **2** (a physical floor, not a
preference) and `--target-S` to **1.5** (~30×, reached with 2–3 salt-bridged
His). The cap at 4 is the opposing constraint: surface histidine load degrades
expression and folding, so the usable window is narrow — 2 to 4 well-placed,
carboxylate-paired histidines.

This ceiling is also the reason the epitope search in Stage 0 optimizes for
*conserved acidic* patches specifically. Histidines that cannot reach a
carboxylate contribute almost nothing, so where you bind determines whether the
pH switch is achievable at all.

`pKa_bound` is estimated from the interface environment:

| Environment feature | Δ pKa |
|---|---|
| His N∈/Nδ within 4.0 Å of target Asp/Glu carboxylate O | **+1.5** |
| within 4.0–6.0 Å of carboxylate O | +0.8 |
| H-bond to backbone C=O or Ser/Thr/Asn/Gln oxygen | +0.3 |
| buried (rel. SASA < 0.15) with no acidic partner | **−1.0** |
| Arg/Lys/His within 5.0 Å | −0.7 |

Clamped to [4.0, 9.0]. **These are calibrated heuristics, not a Poisson–Boltzmann
calculation** — they rank designs sensibly but the absolute pKa values should not
be quoted as predictions. Interface His count is capped (default 4) because
excess surface histidine degrades expression and folding.

Cross-reactivity score = fraction of contacted target residues identical in mouse.

---

## 4. V100 compatibility (this drove several version pins)

V100 is `sm_70` (Volta), which means:

- **No bfloat16.** Anything defaulting to bf16 must be forced to fp32/fp16.
  ESMFold here runs **fp32**.
- **No FlashAttention-2** (requires sm_80+). The pipeline uses HuggingFace's
  pure-PyTorch ESMFold port, which needs no custom CUDA kernels — deliberately
  chosen over `facebook/esm`'s ESMFold, which wants OpenFold kernels that are
  painful to build.
- Torch is pinned to `2.0.1+cu118`, which still ships `sm_70` kernels and is
  compatible with the `dgl` build RFdiffusion's SE3Transformer needs.
- `numpy<2` throughout — this whole stack predates the NumPy 2 ABI break.

Two conda envs are created because RFdiffusion (dgl/e3nn, py3.9) and ESMFold
(transformers, py3.10) do not co-install cleanly:

- `egfr-rfdiff` — stages 0, 1, 2, 4, 5
- `egfr-fold` — stage 3

`pipeline.py` is a single script with **lazy imports**, so the same file runs
under either env; `submit.sbatch` activates the right one per stage.

Memory: Domain III truncation (~172 aa) + a ≤120 aa binder fits comfortably in
16 GB. Do **not** run RFdiffusion against the full 621-residue ECD on a V100.

---

## 5. Install (login node — needs internet)

```bash
cd /path/to/chal_1
bash setup_env.sh
```

Downloads ~8 GB (RFdiffusion weights ~3 GB, ESMFold ~2.6 GB, repos, FASTAs, 6ARU)
into `./third_party`, `./inputs`, and `./hf_cache`. Everything is local to this
directory, so **compute nodes need no internet**.

Verify before burning GPU hours:

```bash
conda activate egfr-rfdiff
python pipeline.py --selftest          # CPU-only: env + inputs + stage-0 logic
```

## 6. Run

### Learning mode (start here)

```bash
bash run_interactive.sh
```

This grabs an interactive V100 via `srun` (re-launching itself if you are on a
login node), then runs all six stages with `--learn` and **streams output live**
so you can watch RFdiffusion denoise, ProteinMPNN choose sequences, and ESMFold
accept or reject them.

`--learn` trades scale for turnaround — **~30–60 min instead of 8–12 h**:

| | competition | `--learn` |
|---|---|---|
| backbones | 150 | 12 |
| sequences per backbone | 8 | 4 |
| diffusion steps | 50 | 25 |
| epitope patches | 4 | 2 |
| length bins | micro + mini + large | micro + mini (large is slowest) |
| min pLDDT | 75 | 60 |
| max RMSD | 2.5 Å | 4.0 Å |
| min interface His | 2 | **0 — nothing filtered** |
| designs out | 40 | 10 |

The permissive gates are the point: with `--min-interface-his 0` nothing is
thrown away, so the score stage prints the full distribution and you can see
the 0.9-per-histidine ceiling in your own data:

```
  pH selectivity vs interface histidine count
  --------------------------------------------------------------------
  His     n   mean S   best S  best fold   ceiling  bar (best S)
  --------------------------------------------------------------------
    1     2     0.58     0.58         4x      0.9  ##
    2    16     1.17     1.17        15x      1.8  ####
    3     8     1.75     1.75        57x      2.7  ######
  --------------------------------------------------------------------
```

Anything you pass explicitly still wins, so you can scale one knob at a time:

```bash
bash run_interactive.sh --learn --n-backbones 40      # more diversity
bash run_interactive.sh --stage fold                  # redo one stage
bash run_interactive.sh --full                        # competition settings
```

### Things worth poking at while learning

| Where | What you learn |
|---|---|
| `work/target_meta.json` | Which epitope patches it picked and the score behind each — conserved acidic content, mouse identity, basic-residue penalty |
| `work/backbones/*.pdb` | RFdiffusion output. `pymol work/target_domIII.pdb work/backbones/<n>.pdb` shows the binder docked on Domain III |
| `work/mpnn/bias_by_res.jsonl` | The actual per-position histidine bias handed to ProteinMPNN |
| `work/esmfold/*.pdb` | ESMFold models. Overlay on the backbone to see what self-consistency RMSD means |
| `results/design_details.json` | Per-histidine estimated pKa and its distance to each target carboxylate |
| `results/design_report.csv` | Every metric per design, to sort and plot yourself |

Good experiments: set `--his-bias 0` and watch pH selectivity collapse; push
`--his-bias 6` and watch the sanity gate start rejecting histidine-saturated
sequences; compare the `biased` vs `neutral` ProteinMPNN passes in
`design_report.csv`.

### Batch runs

```bash
sbatch submit.sbatch                              # full, 40 designs
sbatch --export=ALL,LEARN=1 submit.sbatch         # learning mode in batch
sbatch --export=ALL,SKIP_SELFTEST=1 submit.sbatch # skip the CPU preflight
```

Or interactively, stage by stage:

```bash
conda activate egfr-rfdiff && python pipeline.py --stage prepare
conda activate egfr-rfdiff && python pipeline.py --stage diffuse
conda activate egfr-rfdiff && python pipeline.py --stage mpnn
conda activate egfr-fold   && python pipeline.py --stage fold
conda activate egfr-rfdiff && python pipeline.py --stage score
conda activate egfr-rfdiff && python pipeline.py --stage submit
```

Stages checkpoint to `work/.done_<stage>`; re-running skips completed stages
unless `--force`. Stage 1 supports `--shard i --nshards N` for SLURM array
parallelism across GPUs.

Runtime, single V100, default settings (150 backbones × 8 seqs):
diffuse ~6–10 h · mpnn ~10 min · fold ~25 min · score ~5 min.

## 7. Output

`results/submission.csv` — exactly the spec's columns:

```csv
name,sequence,molecule_class
egfr_ph_mini_001,SLEEKLK...,protein
```

plus `results/design_report.csv` with all metrics (pH score S, predicted pKa
shifts, interface His, mouse identity, pLDDT, RMSD, bin) for your own review.

Default is 40 designs (Track 1). Use `--n-submit 20` for Tracks 2/3.

## 8. Tuning knobs

| Flag | Default | Notes |
|---|---|---|
| `--learn` | off | Small/fast/permissive preset (§6); explicit flags still win |
| `--n-backbones` | 150 | Raise for more diversity if you have GPU budget |
| `--seqs-per-backbone` | 8 | ProteinMPNN sequences per backbone |
| `--his-bias` | 2.5 | Interface His logit bias; raise for stronger pH effect |
| `--min-interface-his` | 2 | Physical floor — one His caps at 4–8× (§3) |
| `--max-interface-his` | 4 | Expression/folding guardrail |
| `--target-S` | 1.5 | Desired log10 Kd(7.4)/Kd(6.5); 1.5 ≈ 30× |
| `--min-plddt` | 75 | Self-consistency gate |
| `--max-rmsd` | 2.5 | Å, ESMFold vs diffused backbone |
| `--n-submit` | 40 | Track 1 = 40; Tracks 2/3 = 20 |

## 9. Honest limitations

- Side chains at the interface are **not repacked**. His geometry comes from an
  ESMFold monomer model superimposed on the diffused backbone plus crystal target
  side chains. Good enough to rank; not a substitute for Rosetta/OpenMM refinement.
- The pKa model is heuristic (§3). If you have PyRosetta or PROPKA available,
  rescoring the top 40 with a real electrostatics calculation is the single
  highest-value improvement you could make.
- **No AlphaFold2-multimer interface validation, and this is the biggest gap.**
  AF2 initial-guess (Bennett et al. 2023) is the field-standard binder filter;
  ESMFold monomer self-consistency is a weaker proxy that says a design folds,
  not that it binds. It was left out to keep the V100 install tractable. There
  is no `--af2-validate` flag — adding AF2 rescoring of the top ~200 designs
  would be the second highest-value improvement after a real pKa calculation.
- ESMFold monomer pLDDT is a weaker signal for very short binders (<40 aa);
  microbinder designs are ranked with a relaxed gate and should be treated as
  higher-risk entries.

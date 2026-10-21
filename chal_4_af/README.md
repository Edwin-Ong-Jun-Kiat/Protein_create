# Challenge 04, AlphaFold track — Kelvin-2 / V100

The second of two tracks for TfR1 apical-domain pH-selective binders. The first
(`../chal_4`) runs on the AMD MI300A and builds binders out of parts. This one
runs AfDesign on an NVIDIA V100 and designs them by gradient descent through
AlphaFold2.

Both exist on purpose: the comparison is the point, and learning AlphaFold-style
design is the priority here regardless of which track scores better.

## Why this is on Kelvin-2 and not the MI300A

AF2 on the MI300A is blocked, not slow:

```
jaxlib._jax.XlaRuntimeError: UNIMPLEMENTED: No registered implementation for
custom call to hipsolver_gesvd_ffi for platform ROCM
```

AF2's combined forward+backward graph needs an SVD. On GPU, jax lowers that to
the vendor linear-algebra library — cuSOLVER on NVIDIA, hipSOLVER on AMD — and
`jax-rocm7-plugin 0.6.0` registers no handler for `gesvd`. The op is absent from
the ROCm build. No flag fixes a missing kernel. Disabling Triton GEMM fusion got
*past* an earlier `module != nullptr` abort, which is how this blocker became
visible at all.

CPU works there, at **227.4 s per optimisation step** — about 9.5 h for one
160-step design. cuSOLVER implements `gesvd`, so on a V100 the blocker simply is
not present, and the expected step cost is low single-digit seconds.

Two things not yet pinned down, stated as such:

- **Which loss term introduces the SVD.** If it were one removable term there
  would be a workaround, but that means patching AF2 internals, which silently
  alters the science. Not done.
- **The exact compute-capability floor** for newer jax on sm_70. This is why the
  install pins jax 0.4.20 — `dl_binder_design`'s own version, inside BindCraft's
  `jax>=0.4,<=0.6.0` range, and known-good on V100. Newer jax has raised that
  floor more than once.

## Files

| file | what it is |
|---|---|
| `push.sh` | upload from the Mac; chal_2's version, new remote dir, and one shared connection so a passphrase is asked for at most once |
| `install_af.sh` | **the installer.** conda env, jax 0.4.20 + CUDA, ColabDesign, 5.3 GB of AF2 params. Login node only. |
| `build_af_target.py` | makes the one sequence-contiguous target construct AF2 can use |
| `af_design.py` | the AfDesign loop, plus `--probe` timing mode |
| `submit.sbatch` | the V100 job (`k2-gpu-v100`) |
| `run.sh` | check → preflight → sbatch, in one command |

## Running it

On the Mac:

```bash
cd chal_4_af
bash push.sh --dry-run        # file list, contacts nothing, never prompts
bash push.sh
```

A push needs four round trips (probe for rsync, mkdir, transfer, fix
permissions), so without connection reuse a passphrase-protected key prompts
four times. `push.sh` opens one `ControlMaster` connection up front and runs
everything over it, and `ControlPersist=10m` means a `--shell` or `--pull`
shortly after does not ask again. To be asked zero times, load the key once per
login:

```bash
ssh-add --apple-use-keychain ~/.ssh/my-kelvin-key
```

The script says which of those three cases you are in before it connects.
`bash push.sh --stop` closes the shared connection early; `--no-mux` disables
reuse if it ever misbehaves.

On a Kelvin-2 **login** node:

```bash
cd /users/40423097/sharedscratch/Protein_create/chal_4_af
bash install_af.sh --build    # ONCE. 40-70 min, ~12 GB on scratch.
bash run.sh --probe           # 20 min. Do this before anything else.
bash run.sh                   # 8 designs
bash run.sh --full            # 24 designs, 24 h cap
```

`--probe` first is not ceremony. The MI300A's first ESMFold job spent 3 h 20 min
folding fewer than 10 of 36 sequences at 0% GPU utilisation before anyone looked
at a number. The probe separates step 1 (XLA compiling the whole AF2 graph,
~250 s) from the steady-state step cost, because conflating them is exactly how
a CPU fallback passes for a slow GPU.

## Lessons from the MI300A that are baked into these scripts

Each of these cost real time on the other track:

- **Every pip call goes through a constraints file.** Installing
  `chex`/`flax`/`optax` in a second unpinned call let the resolver move jax
  0.4 → 0.12 and numpy 1 → 2. The only symptom appeared three steps later as
  `AttributeError: module 'jax_rocm7_plugin._triton' has no attribute
  'registrations'`. A command-line pin does not constrain transitive
  dependencies; a constraints file does.
- **`--no-deps` for ColabDesign, then install its deps by hand** — including
  `matplotlib`, which is not obvious and only surfaces as an ImportError inside
  `colabdesign.shared.plot` on the first design call.
- **No `conda activate` anywhere.** In a non-interactive shell it can return 0
  without changing PATH, so pip installs into one interpreter while python is
  another. Every interpreter is named by absolute path.
- **Absolute interpreter paths generally.** On AAC, `module load rocm cray-python`
  on one line raised a conflict that aborted the whole line, leaving system
  python 3.9 on PATH — silently.
- **The target is checked by geometry, not numbering.**
  `../chal_4/work/target_cropped_k382.pdb` is renumbered 1..N, so it *reports*
  one contiguous segment while containing 21 chain breaks with a maximum CA-CA
  distance of 36.2 Å. Renumbering erased the evidence.
- **The hotspot list is checked against the construct before submission.** A
  150–280 construct cannot contain residue 382; keeping `A382` anyway aborted a
  job with `AssertionError: ERROR: positions 382 and chain A not found` after
  the queue wait. That shorter construct also necessarily drops K382, one of the
  three conserved cations the switch is built on — a cost, not a free speed-up.
- **The CSV is written after every design**, so a wall-clock kill loses the
  design in flight and nothing else.

## The histidine bias, and what this track does not do

S = log10(Kd(6.0)/Kd(7.4)), and histidine is the only residue with a pKa inside
that window. AfDesign optimises AF2 confidence, which is completely pH-blind, so
left alone it has no reason to place His anywhere useful.

`--his-bias` (default 1.5) adds a constant to the histidine logit at every
position. It is deliberately a weak, uniform prior rather than a hard constraint
at hand-picked positions — the latter would be me designing the switch and AF2
merely agreeing. His count and net charge are reported per design so the knob
can be calibrated.

**Nothing here computes S.** Scoring is the main track's `--stage score`, and
putting these sequences through it is the comparison this track exists for:

```bash
cp results/af_designs.csv ../chal_4/work/af_candidates.csv
cd ../chal_4 && source env.sh && "$PHB_PY" pipeline.py --stage score
```

Also not done yet, in rough priority order:

1. **AF2-initial-guess (`dl_binder_design`)** — the best-validated binder
   filter. It pins `jax 0.4.20` + CUDA, i.e. exactly this environment. It was
   impossible on ROCm; here it is the obvious next step.
2. **BindCraft** proper, whose *scoring* needs PyRosetta and therefore an
   academic licence. AfDesign itself does not.

## Target construct

Chain A, residues 150–390 of template **11QC** — 241 residues, 0 chain breaks,
max consecutive CA-CA 3.85 Å. Spans the whole apical domain (189–383) with ~40
residues of flank either side. `target_meta.json` records offset 0, so these are
UniProt P02786 numbers and hotspots keep their familiar labels.

Primary site `k382`: hotspots **382, 193, 189, 381, 158, 380**; switch cations
189/193/382; switch acids 156/194/383; 72.7% human–mouse conserved; 43.9 Å from
the transferrin site.

# Challenge 02 — TNF-α pH-selective binders (Kelvin-2 / V100)

The HPC pipeline for [challenge 2](https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/tnf-alpha):
binders that grip human TNF-α at pH 7.4, release at pH 6.0, and cross-react
with mouse. RFdiffusion → ProteinMPNN → ESMFold on one V100.

`../chal_2_local` is the laptop version of the same science. This one replaces
the parametric-bundle generator and the rigid-body docking search with real
diffusion, which is the whole reason to use the cluster.

**Two commands.** On your Mac:

```bash
cd ~/Downloads/Protein_create/chal_2 && bash push.sh
```

Then on Kelvin-2:

```bash
cd /users/40423097/sharedscratch/Protein_create/chal_2 && bash run.sh
```

`run.sh` does environment setup, the CPU preflight, and `sbatch` in order, and
stops at the first failure. Add `--full` for the real run instead of the ~1 h
test. Details in [Running it](#running-it).

---

## Read this first: why your last two jobs died, and what changed

Two runs have now been lost to bugs that a passing selftest did not catch. Both
are fixed. The second one is the more dangerous kind and is described after the
first, under [the ProteinMPNN chain bug](#the-second-failure-proteinmpnn-was-asked-to-redesign-the-target).

### The first failure: JSON encoding

Job `10073878` failed like this:

```
[ err] SELFTEST FAILED (1 problem(s)):
    - stage prepare raised: TypeError('Object of type int64 is not JSON serializable')
```

Everything before it was healthy — inputs found, tools found, torch 2.0.1 with
sm_70 kernels confirmed, numbering offset correctly detected as 24, four epitope
patches scored sensibly. It died on the **last line** of `stage_prepare`, writing
`target_meta.json`.

The cause: residue numbers come out of numpy arrays as `np.int64`, SASA
fractions as `np.float64`, conservation flags as `np.bool_`. None is
JSON-serialisable, and `chal_1/pipeline.py` had **nine** bare `json.dumps()`
calls with no encoder. The values print identically to Python `int`s in the log,
so nothing upstream looks wrong.

**Two things were done about it.**

1. **`../chal_1/pipeline.py` is fixed** (backup at `pipeline.py.bak`). A shared
   `_json_default` now handles numpy scalars, arrays, sets and `Path`, and all
   nine call sites pass it. Its selftest also now exercises the encoder
   *before* stage prepare, so a recurrence is reported as a one-line encoder
   fault rather than as an opaque "stage prepare raised". Your queued EGFR run
   should go through — resubmit it as-is.

2. **This pipeline cannot have that bug.** There is exactly one way it writes
   JSON — `save_json()`, which always passes the encoder — and the selftest
   asserts the encoder handles `int64`/`float64`/`bool_`/`ndarray`/`set`/`Path`.

### Two more preflight checks that exist because of that failure

The lesson was not "fix the encoder", it was **the selftest passed and the run
still died**. So the selftest now also:

- **traces the hotspot renumbering back to the hotspots that were chosen.** The
  two-chain target gets cropped and renumbered (below), so the string handed to
  RFdiffusion is in different coordinates from the residues selected. An
  off-by-one there aims the entire 12-hour diffusion stage at the wrong face of
  the protein, and *every design would look fine*. The check prints
  `A35=A86, A36=A87, A37=A88, ...` and fails if any emitted hotspot does not map
  back to a real one;
- **checks each contig span equals the number of residues actually written**,
  and that the cropped target is under 200 residues so RFdiffusion will not OOM.

### The second failure: ProteinMPNN was asked to redesign the target

The next run got through `prepare` and the full ~12 h of `diffuse`, then died at
the top of `mpnn`:

```
assign_fixed_chains.py: error: unrecognized arguments: --designed_chain_list C
```

`--designed_chain_list` does not exist. `assign_fixed_chains.py` takes exactly
one chain flag, `--chain_list`, and it means **the chains to design** —
everything else in the parsed complex is derived as fixed automatically. The
driver was passing both flags, so argparse rejected the unknown one.

**The crash was the lucky part.** The obvious repair is to delete the unknown
flag and keep the other, and that is the wrong fix:

```python
"--chain_list", " ".join([designed_chain] + list(fixed_chains)),   # "C A B"
```

Read as a *designed* list, that asks ProteinMPNN to design the binder **and both
TNF-α protomers**. It exits 0, writes plausible FASTAs, and hands back a target
whose sequence has been mutated. `score` then computes mouse cross-reactivity
and histidine pKa against a TNF-α that does not exist, and `submission.csv`
looks entirely normal. This is the same failure class as the hotspot
renumbering above: no error, every design looks fine, and the whole run is
aimed at the wrong thing.

`run_proteinmpnn` now passes only the binder chain, and `fixed_chains` is
repurposed from an argument into an assertion — it reads back what
`assign_fixed_chains.py` actually decided and refuses to continue unless the
binder alone is being designed:

```
mpnn: designing ['C'], holding ['A', 'B'] fixed
```

That line is printed once per groove. **If you do not see it in the log, the
sequences from that run are not usable** — discard them rather than scoring
them. `_assigned_chains()` additionally fails if the backbones disagree on
which chain is the binder, which would otherwise put the chain-keyed histidine
bias array onto the target for the odd ones out.

---

## The one thing to know about the science

**Challenge 2 is not challenge 1 with the numbers changed. It is the opposite
sign, and it is harder.**

Challenge 1 wanted binding in acid, which you build by putting a histidine next
to a target carboxylate so its pKa rises on binding. Challenge 2 wants the
reverse, so the pKa must *fall* on binding — target **cations**, and avoiding
target carboxylates.

| | challenge 1 (EGFR) | challenge 2 (TNF-α) |
|---|---|---|
| wants | bind pH 6.5, not 7.4 | bind pH 7.4, not **6.0** |
| pH window | 0.9 units | **1.4 units** |
| needs His pKa to | **rise** on binding | **fall** on binding |
| prize residues | conserved Asp/Glu | conserved **Lys/Arg** |
| Asp/Glu at the His are | the mechanism | a **liability** |
| max S per histidine | 0.69 (4.9×) | **0.50 (3.2×)** |
| histidines needed | 2 | **4** (3 with the second lever) |

The ceiling is *worse* despite the wider window, because at pH 6.0 a free
histidine is only 71% protonated — a third of the window is spent before design
starts. `src/tnf_hpc/linkage.py` derives it, and `--selftest` asserts it.

There is a second lever: raise `pKa_free` as well as suppressing `pKa_bound`,
which takes S from 0.48 to 0.90 per histidine and cuts the requirement to three.
**The levers only work together** — a binder carboxylate alone gives 0.27, worse
than the bare histidine, because it leaves the bound pKa above the window. The
selftest asserts that ordering (`bare < both`, `free-lever-alone < bare`) so it
cannot silently regress.

---

## What is different from challenge 1, structurally

**The target is a trimer and the binding site spans two protomers.** Soluble
TNF-α is an obligate homotrimer and its receptor binds in the cleft *between*
adjacent subunits — measured here, not assumed: in `3ALQ` all six TNFR2 copies
contact two protomers with ~16 residues each. So the binder is diffused against
chains A *and* B, and `contigmap.contigs` names both with a `/0` chain break.

**SASA is computed in trimer context.** Challenge 1 had to *strip* the cetuximab
Fab out of 6ARU before computing accessibility. Here the mistake is the exact
opposite: strip chains B and C and you expose a hydrophobic trimer core that is
never solvent-accessible, and the patch scorer offers it to you as a site. The
selftest reports how many residues that would mis-label (21).

**The target is cropped before diffusion.** Two full protomers is 304 residues;
with an 80-residue binder RFdiffusion would reason over ~384 positions and its
memory scales roughly quadratically. `crop_target` keeps an 18 Å shell around
the hotspots, giving 136 and 164 residues for the two grooves. The crop
renumbers each chain 1..N contiguously — a crop leaves gaps wherever a loop
exits the shell, and RFdiffusion would treat each fragment as a separate chain
segment and try to bridge them. The mapping back is kept in
`work/target_meta.json` and is what the selftest verifies.

---

## Stages

| stage | env | device | time | what |
|---|---|---|---|---|
| `prepare` | rfdiff | CPU | ~15 s | 1TNF trimer, TNFR2 footprint from 3ALQ, conservation, grooves, cropped target, contig |
| `diffuse` | rfdiff | GPU | **~10-12 h** | RFdiffusion backbones, hotspot-conditioned on the groove |
| `mpnn` | rfdiff | GPU | ~15 min | ProteinMPNN with the reverse-polarity His bias |
| `fold` | **fold** | GPU | ~45 min | ESMFold each binder: pLDDT + RMSD self-consistency |
| `score` | rfdiff | CPU | ~5 min | His rotamers, pKa both states, S, mouse identity, composite |
| `submit` | rfdiff | CPU | ~1 s | gates, quotas, CSV |

Every stage checkpoints to `work/.done_<stage>`; `sbatch submit.sbatch` again
resumes. `diffuse` additionally skips batches whose PDBs already exist, so a
wall-clock kill costs at most one batch.

`fold` is the filter `chal_2_local` could not run at all, and it is the most
valuable thing the cluster buys after diffusion itself: it is the only check
that a designed *sequence* actually encodes the *backbone* it was designed on.

---

## Environments are shared with challenge 1 — nothing is rebuilt

`setup_env.sh` **verifies and does not install**. Challenge 1 already built
everything this needs, and job 10073878 proved it works:

```
rfdiff  py3.9   torch 2.0.1+cu118  dgl  e3nn     -> stages 0,1,2,4,5
fold    py3.10  torch 2.0.1+cu118  transformers  -> stage 3
third_party/{RFdiffusion,ProteinMPNN}            -> shared
```

It finds the store via `../chal_1/env.sh`, then the usual scratch locations,
verifies both interpreters and all five tool paths, runs an import check inside
each env, and writes `env.sh`. If anything is missing it tells you to run
challenge 1's installer rather than carrying a second copy of that logic that
could drift. Override with `TNF_STORE=/path/to/store bash setup_env.sh`.

`env.sh` exports both `TNF_*` and `EGFR_*` names, and `tools.py` falls back from
one to the other, so an `env.sh` written by either project drives either
pipeline.

### Two compatibility details that would otherwise bite

- **The `rfdiff` env is Python 3.9 with numpy 1.2x.** These modules were written
  on 3.12 with numpy 2.x. Every file is checked against the 3.9 grammar, and
  `numpy.trapezoid` — which does not exist before numpy 2.0 — is resolved once
  in `linkage.py` as `getattr(np, "trapezoid", None) or np.trapz`. Without that
  the selftest dies on an `AttributeError` on the cluster and nowhere else.
- **`push.sh` uploads `src/` as a directory.** Challenge 1 was one
  self-contained file and its `push.sh` sent named files only; copying that list
  verbatim would upload a `pipeline.py` that dies on
  `from tnf_hpc import linkage` before it ever reaches a GPU.

---

## Layout

```
chal_2/
├── run.sh                   ONE COMMAND on the cluster: setup + selftest + sbatch
├── push.sh                  upload the whole directory from the Mac
├── pipeline.py              six stages + --selftest
├── setup_env.sh             finds the shared env, writes env.sh
├── install_env.sh           builds it from scratch if it is missing
├── submit.sbatch            the SLURM job
├── run_interactive.sh       srun a live GPU session (use this while debugging)
├── src/tnf_hpc/
│   ├── linkage.py   the physics: reverse polarity, the ceiling, two levers
│   ├── target.py    stage 0: trimer, footprint, conservation, grooves
│   ├── pka.py       stage 4: His rotamers, pKa both states
│   ├── design.py    composite score, gates, quotas
│   ├── tools.py     cropping, contigs, RFdiffusion/MPNN/ESMFold drivers
│   ├── scaffold.py  }  kept only for NeRF/atom_line/pseudo_cb helpers;
│   └── dock.py      }  their generative roles are replaced by RFdiffusion
├── inputs/     1TNF, 3ALQ, 7KP7, two UniProt FASTAs (2.1 MB, committed)
└── work/, results/, logs/
```

---

## Running it

### Upload

The whole directory goes up as one unit; `push.sh` sends the contents and
excludes everything the cluster generates (`work/`, `results/`, `logs/`,
`env.sh`, `store/`, `__pycache__`).

```bash
cd ~/Downloads/Protein_create/chal_2
bash push.sh                 # rsync, falling back to tar-over-ssh
bash push.sh --dry-run       # see exactly what would be sent first
```

22 files, 2.7 MB, including the `src/tnf_hpc/` package and the four input
structures. Override the destination with `KELVIN_HOST`, `KELVIN_REMOTE` or
`KELVIN_KEY` if any of them differ.

**If the upload fails with `Permission denied (publickey)`,** run the same
connection by hand with `-v` and look for this pair of lines:

```
debug1: Server accepts key: .../my-kelvin-key RSA SHA256:qAr2nf...
40423097@login.kelvin.alces.network: Permission denied (publickey).
```

"Server accepts key" followed by a denial means the key *is* in the cluster's
`authorized_keys` and was matched — the server asked you to prove it, then
rejected the proof. That is an account-state problem (expired allocation,
disabled shell, unmountable home), not a key problem, so regenerating keys will
not help; mail research computing with your username and quote those two lines.
It is specifically **not** the OpenSSH 8.8+ `ssh-rsa`/SHA-1 deprecation that
usually breaks RSA keys after a macOS upgrade: the server advertises
`rsa-sha2-512` in `server-sig-algs` and the client signs with it. Do not retry
in a loop — repeated publickey failures are what fail2ban counts, and a banned
IP costs more than the wait.

### Run

```bash
ssh -i ~/.ssh/my-kelvin-key 40423097@login.kelvin.alces.network
cd /users/40423097/sharedscratch/Protein_create/chal_2

bash run.sh                  # setup -> selftest -> queue the ~1 h test run
bash run.sh --full           # setup -> selftest -> queue the full run
bash run.sh --check          # setup -> selftest only, submit nothing
bash run.sh --interactive    # setup -> selftest -> live GPU session
```

`run.sh` runs on a **login node** and does four things in order, stopping at the
first failure:

1. `setup_env.sh` — finds the shared conda envs and tool checkouts, or builds
   them via `install_env.sh` if they are genuinely missing, and writes `env.sh`
2. `source env.sh` and verify both interpreters are executable
3. `pipeline.py --selftest` — ~60 s on CPU
4. `sbatch` — prints the job id and the commands to watch it

**A failed selftest never reaches `sbatch`.** That is the point: job `10073878`
spent a GPU allocation to discover a JSON encoding bug that a CPU preflight
catches in under a minute.

**Do the default test run first.** It uses 20 backbones and one groove with
permissive filters, exercises every stage and both conda envs, and writes a real
(bad) `submission.csv` in about an hour. Diffusion is ~90% of the full run's
wall time, so finding a stage-4 bug after 11 hours of diffusion is the expensive
way to learn it.

### Watch it, and collect

```bash
squeue -u $USER
tail -f logs/tnf_<jobid>.out
column -s, -t results/submission.csv | head -20
```

Then, back on the Mac:

```bash
bash push.sh --pull          # lands in results_from_kelvin/
```

If the job dies on the wall clock, just resubmit the same way — every stage
checkpoints to `work/.done_<stage>` and `diffuse` skips batches whose PDBs
already exist, so a kill costs at most one batch.

### Knobs

Pass these to `sbatch` as environment variables, or to `pipeline.py` directly
if you are driving stages by hand.

| flag / env | effect |
|---|---|
| `LEARN=1` | 20 backbones, 1 groove, `diffuser.T=25`, permissive filters (what `run.sh` uses by default) |
| `N_BACKBONES=400` | more sampling; shard it across GPUs — see the end of `submit.sbatch` |
| `N_GROOVES=1` | one epitope instead of two |
| `SKIP_ESMFOLD=1` | skip stage `fold`; much weaker, use only to isolate a bug |
| `SKIP_SELFTEST=1` | straight to the GPU (not recommended) |
| `--crop-radius 15` | smaller target, faster diffusion, less context |
| `--binder-min/--binder-max` | default 55–80; go below 40 to target the microbinder category |
| `--his-bias 4.5` | more histidines placed; watch `his_excess` and the gates |
| `--min-plddt 70 --max-rmsd 3.0` | loosen if the filters reject everything |
| `--force` | ignore the `work/.done_*` checkpoints |

### If the shared environment has gone

`setup_env.sh` normally finds challenge 1's store on scratch and builds nothing.
If it is missing it falls back to `install_env.sh`, which is challenge 1's
installer vendored here so this directory stands alone: python 3.9/3.10 envs,
`torch 2.0.1+cu118`, the `dgl 1.1.2+cu118` pin, `e3nn`, an editable
SE3Transformer install, and the RFdiffusion checkpoint with resume and size
validation. 40–70 minutes and ~20 GB of scratch.

It refuses to run anywhere without `sbatch` on `PATH`, so an accidental
`bash run.sh` on your Mac fails immediately instead of part-building a CUDA
environment. Point it elsewhere with `TNF_STORE=/path/to/store`, or override the
guard with `FORCE_INSTALL=1`.

---

## What this does *not* tell you

- **Whether any design binds.** `S` is a *ratio*; nothing here predicts Kd.
  Criterion 3 is addressed only as "bury surface in the right place".
- **Whether the complex is real.** ESMFold validates the binder *alone*. There
  is no AF2-multimer or Boltz check on the interface, which is the
  field-standard binder filter and the biggest remaining gap. RFdiffusion plus
  a pLDDT/RMSD filter is a much better proxy than the laptop version had, but
  it is still not an interface prediction.
- **Absolute pKa values.** Hand-set increments on rotamer geometry with no
  repacking, and `best_his_rotamer` optimistically takes the most favourable
  reachable rotamer for the bound state. It ranks; it does not predict.
- **Cooperativity.** Summing per-histidine contributions assumes independent
  sites; a 7 Å separation is enforced, but "roughly independent" is doing work.

One measured caveat carried over from the laptop run: designs with perfect mouse
cross-reactivity get there partly by *shrinking* the interface to avoid the nine
divergent epitope positions, which trades against affinity. The composite score
cannot see it (`contacts` weighted 0.006 against 3.0 for cross-reactivity).
Check `n_contacted_target` in `results/design_report.csv` before trusting a
design with `mouse_identity = 1.00`.

---

## Provenance

| fact used | source | status |
|---|---|---|
| TNF-α trimer | `files.rcsb.org/download/1TNF.pdb`, 2.6 Å | downloaded |
| TNFR2 footprint | **measured** from 3ALQ chains R–W, 5 Å contacts | computed, 6 copies |
| TNFR1 footprint on mouse TNF-α | **measured** from 7KP7 | independent cross-check, 86% agreement |
| human / mouse sequences | UniProt P01375, P06804 | downloaded |
| mature form = UniProt 77–233 | the challenge brief | stated there; offset verified by alignment (0 for 1TNF) |
| 79% species identity | **computed**, 124/156 | reproduces the brief's own figure; asserted in `--selftest` |
| RFdiffusion checkpoint | `Complex_base_ckpt.pt` | shared with challenge 1 |
| ProteinMPNN weights | soluble, `v_48_020.pt` | shared with challenge 1 |
| every Δ-pKa (+1.5, −0.7, −1.0, …) | hand-set in `pka.py` | **uncited heuristics**, no calibration set |
| patch-score and composite weights | hand-set in `target.py`, `design.py` | **opinions, not measurements** |
| all S / ceiling values | derived from the linkage equation | computed, cross-checked two ways |

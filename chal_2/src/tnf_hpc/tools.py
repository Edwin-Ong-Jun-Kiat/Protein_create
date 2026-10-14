"""
Drivers for the three GPU tools, plus the target preparation RFdiffusion needs.

This is the module that has no counterpart in `chal_2_local`. On a laptop the
backbone came from `scaffold.py` and the placement from `dock.py`, because
RFdiffusion's SE3Transformer depends on `dgl`, which has no Apple-Silicon
build. On Kelvin-2 with a V100 the real thing runs, so the parametric bundle
library and the rigid-body docking search are both retired -- RFdiffusion grows
a backbone around the epitope and does placement and shape complementarity
together, which is strictly better than choosing from a fixed library.

Three things in here are specific to this target and were not needed for
challenge 1's single-chain EGFR:

1. **The target is two chains.** TNF-alpha's receptor site is the groove
   between adjacent protomers (`target` docstring), so the binder has to be
   diffused against chains A *and* B. RFdiffusion handles multi-chain targets,
   but the contig string has to name both, with a chain break between them.

2. **The target must be cropped.** Two full protomers is 304 residues; add a
   70-residue binder and RFdiffusion is folding ~374 positions, and its memory
   scales roughly with the square of that. `crop_target` keeps only residues
   within a shell of the hotspots, which takes it to ~120 and makes diffusion
   several times faster and far less likely to OOM. RFdiffusion only needs the
   local epitope context, not the whole protein.

3. **The histidine bias is two-sided**, and ProteinMPNN's CLI takes it as a
   `--bias_by_res_jsonl` file rather than an array, so the bias built by
   `mpnn_bias_by_res` has to be keyed by chain and position in the exact
   format the helper scripts expect.

Everything here shells out. Nothing imports torch, so this module is safe to
import in either conda env.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess

import numpy as np

from .target import _3TO1

# --------------------------------------------------------------- environment

#: Environment variables `env.sh` is expected to export. Named TNF_* but they
#: fall back to the EGFR_* names challenge 1's setup_env.sh wrote, because the
#: conda envs and third-party checkouts are identical and there is no reason to
#: build them twice. See setup_env.sh.
ENV_FALLBACKS = {
    "TNF_STORE": "EGFR_STORE",
    "TNF_RFDIFF_PY": "EGFR_RFDIFF_PY",
    "TNF_FOLD_PY": "EGFR_FOLD_PY",
}


def env_path(name, default=None, must_exist=False):
    """Read a TNF_* variable, falling back to challenge 1's EGFR_* equivalent."""
    val = os.environ.get(name)
    if not val and name in ENV_FALLBACKS:
        val = os.environ.get(ENV_FALLBACKS[name])
    if not val:
        val = default
    if val is None:
        return None
    p = Path(val)
    if must_exist and not p.exists():
        raise FileNotFoundError(f"{name} points at {p}, which does not exist")
    return p


def store_root():
    """Where the shared envs and third-party checkouts live (on scratch)."""
    s = env_path("TNF_STORE")
    if s is not None:
        return s
    for cand in (Path(f"/mnt/scratch2/users/{os.environ.get('USER','')}"),
                 Path(f"/mnt/scratch/users/{os.environ.get('USER','')}")):
        if cand.is_dir():
            return cand / "egfr_binder_store"
    return Path.cwd() / "store"


def tool_paths():
    """Resolve the RFdiffusion and ProteinMPNN checkouts and the checkpoint."""
    tp = store_root() / "third_party"
    return {
        "rfdiffusion": tp / "RFdiffusion",
        "rfdiffusion_script": tp / "RFdiffusion" / "scripts" / "run_inference.py",
        "rfdiffusion_ckpt": tp / "RFdiffusion" / "models" / "Complex_base_ckpt.pt",
        "proteinmpnn": tp / "ProteinMPNN",
        "proteinmpnn_script": tp / "ProteinMPNN" / "protein_mpnn_run.py",
        "mpnn_parse": (tp / "ProteinMPNN" / "helper_scripts"
                       / "parse_multiple_chains.py"),
        "mpnn_assign": (tp / "ProteinMPNN" / "helper_scripts"
                        / "assign_fixed_chains.py"),
    }


def run(cmd, cwd=None, env=None, log=None, check=True, timeout=None):
    """Run a subprocess, streaming its output into our own log.

    Returns the CompletedProcess. On failure the tail of stderr is included in
    the raised message, because an RFdiffusion traceback 400 lines into a Hydra
    config dump is otherwise invisible in a SLURM log.
    """
    cmd = [str(c) for c in cmd]
    if log:
        log("  $ " + " ".join(cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=full_env,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and check:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        out_tail = "\n".join((proc.stdout or "").strip().splitlines()[-10:])
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"--- stderr tail ---\n{tail}\n--- stdout tail ---\n{out_tail}")
    return proc


# --------------------------------------------------------------- target prep

def crop_target(model, chain_ids, hotspots, offset=0, radius=18.0,
                min_residues=40):
    """Residues to keep per chain: everything within `radius` of a hotspot CB.

    Returns {chain_id: sorted list of author residue numbers}.

    Why crop at all: RFdiffusion's cost grows with the number of positions it
    has to reason about, roughly quadratically in memory. Two complete TNF-alpha
    protomers is 304 residues before the binder is added. Keeping an 18 A shell
    around the hotspots cuts that to roughly 120 while preserving every residue
    the binder could contact (a binder reaches ~10 A from its own surface, so
    18 A around the hotspot centroid is generous).

    Why not crop harder: a too-tight crop leaves RFdiffusion with a target that
    looks like a loose peptide rather than a folded surface, and it will happily
    diffuse a binder into a cleft that does not exist in the intact trimer. 18 A
    keeps the local fold intact. `min_residues` guards against a pathological
    crop on a chain that the hotspots barely touch.
    """
    hs = {(c, p) for c, p in hotspots}
    centres = []
    for ch in model:
        if ch.id not in set(chain_ids):
            continue
        for res in ch:
            if (ch.id, res.id[1] - offset) in hs:
                atom = "CB" if "CB" in res else "CA"
                if atom in res:
                    centres.append(np.asarray(res[atom].coord, float))
    if not centres:
        raise ValueError("no hotspot residues found while cropping the target")
    centres = np.array(centres)

    keep = {}
    for ch in model:
        if ch.id not in set(chain_ids):
            continue
        nums = []
        for res in ch:
            pts = np.array([a.coord for a in res if a.element != "H"], float)
            if pts.size == 0:
                continue
            d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=-1).min()
            if d <= radius:
                nums.append(int(res.id[1]))
        keep[ch.id] = sorted(nums)
    for c, v in keep.items():
        if len(v) < min_residues:
            raise ValueError(
                f"crop kept only {len(v)} residues of chain {c}; "
                f"increase --crop-radius")
    return keep


def write_cropped_pdb(model, keep, dest):
    """Write the cropped two-chain target, renumbered contiguously per chain.

    **Renumbering is deliberate and it is the subtle part.** RFdiffusion's
    contig parser expects each chain's kept range to be describable as
    contiguous spans, and a crop produces gaps wherever a loop wanders outside
    the shell. Rather than emit a contig with a dozen fragments per chain --
    which RFdiffusion treats as separate chain segments and will try to bridge
    -- the kept residues are renumbered 1..N contiguously and the mapping back
    to the original numbering is returned.

    Everything downstream (hotspot strings, scoring, mouse conservation) has to
    go through that mapping, so it is returned rather than stored implicitly.
    Returns {(chain, new_number): (chain, original_number)}.
    """
    from .scaffold import atom_line

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines, serial = [], 1
    mapping = {}
    for cid in sorted(keep):
        chain = model[cid]
        newnum = 0
        for res in chain:
            if res.id[1] not in set(keep[cid]):
                continue
            newnum += 1
            mapping[(cid, newnum)] = (cid, int(res.id[1]))
            for atom in res:
                if atom.element == "H":
                    continue
                lines.append(atom_line(serial, atom.get_name(),
                                       res.get_resname(), cid, newnum,
                                       atom.coord, element=atom.element))
                serial += 1
        lines.append("TER")
    lines.append("END")
    dest.write_text("\n".join(lines) + "\n")
    return mapping


def build_contig(keep, binder_min, binder_max):
    """RFdiffusion `contigmap.contigs` for a two-chain target plus a binder.

    Produces e.g. ``[A1-61/0 B1-58/0 60-80]``:

      * one span per target chain, in the renumbered coordinates;
      * ``/0`` after each target chain is RFdiffusion's chain break;
      * a bare length range for the binder, which is what makes it *de novo*
        rather than a motif to be rebuilt.

    The binder length is given as a range so RFdiffusion samples lengths rather
    than producing N copies of one size -- useful here because the challenge
    scores microbinders (<40), minibinders (40-100) and large binders (>100) as
    separate categories.
    """
    spans = []
    for cid in sorted(keep):
        n = len(keep[cid])
        spans.append(f"{cid}1-{n}/0")
    spans.append(f"{binder_min}-{binder_max}")
    return "[" + " ".join(spans) + "]"


def hotspot_string(hotspots, mapping):
    """RFdiffusion `ppi.hotspot_res`, translated into the cropped numbering.

    `hotspots` are (chain, mature_pos) in the original numbering; `mapping` is
    what `write_cropped_pdb` returned. A hotspot that fell outside the crop is
    dropped with a warning rather than silently renumbered to something wrong.
    """
    inverse = {v: k for k, v in mapping.items()}
    out, missing = [], []
    for c, p in hotspots:
        key = inverse.get((c, int(p)))
        if key is None:
            missing.append((c, p))
            continue
        out.append(f"{key[0]}{key[1]}")
    return "[" + ",".join(out) + "]", missing


# --------------------------------------------------------------- RFdiffusion

def run_rfdiffusion(py, out_prefix, target_pdb, contig, hotspots, n_designs,
                    ckpt=None, diffuser_T=50, extra=None, log=None,
                    timeout=None):
    """One RFdiffusion binder-design call.

    Flags chosen for a V100 (sm_70) and for binder design specifically:

      * ``inference.ckpt_override_path`` -> Complex_base_ckpt.pt, the
        complex-trained checkpoint. The default checkpoint is monomer-trained
        and produces backbones that ignore the target.
      * ``ppi.hotspot_res`` is what actually aims the binder. Without it
        RFdiffusion will dock somewhere on the target and usually not where you
        want.
      * ``denoiser.noise_scale_ca`` and ``noise_scale_frame`` at 0 -- the
        standard low-noise setting for binder design, which trades diversity
        for a much higher fraction of designable backbones. With noise at the
        default 1.0 most outputs fail the later pLDDT filter.
      * ``diffuser.T=50`` rather than the default 200: 4x faster for a small
        quality cost, and diffusion is ~90% of this pipeline's wall time.
    """
    tools = tool_paths()
    cmd = [
        py, str(tools["rfdiffusion_script"]),
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={target_pdb}",
        f"inference.num_designs={n_designs}",
        f"diffuser.T={diffuser_T}",
        f"contigmap.contigs={contig}",
        f"ppi.hotspot_res={hotspots}",
        "denoiser.noise_scale_ca=0",
        "denoiser.noise_scale_frame=0",
    ]
    if ckpt:
        cmd.append(f"inference.ckpt_override_path={ckpt}")
    if extra:
        cmd.extend(extra)
    return run(cmd, cwd=tools["rfdiffusion"], log=log, timeout=timeout)


# --------------------------------------------------------------- ProteinMPNN

#: ProteinMPNN's fixed alphabet; the order a bias vector must follow.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

HIS_BIAS_NEAR = 8.0
HIS_BIAS_FAR = 11.0
HIS_BIAS_FAR_SCALE = 0.4
HIS_PENALTY_NEAR = 7.0
ACID_PAIR_NEAR = 7.5
ACID_PAIR_MIN_TARGET_DIST = 10.0


def mpnn_bias_by_res(binder_cb, binder_chain, cation_xyz, acid_xyz, target_cb,
                     his_bias=3.5, his_penalty=-2.0, acid_bias=1.5,
                     pair_acids=True):
    """The two-sided histidine bias, in ProteinMPNN's `bias_by_res_jsonl` format.

    Identical geometry to `chal_2_local`'s `mpnn.his_bias_vector`; only the
    output shape differs, because the CLI wants
    ``{chain: [[21 floats] per position]}`` rather than an array over the whole
    complex.

    The rule, and it is the inverse of challenge 1's on both halves:

        conserved target CATION       <=  8 A  ->  +his_bias
        conserved target CATION       <= 11 A  ->  +his_bias * 0.4
        conserved target CARBOXYLATE  <=  7 A  ->  +his_penalty   (added)

    Challenge 1 biased histidine *towards* conserved carboxylates, because a
    carboxylate raises a bound histidine's pKa and that project wanted the pKa
    up. Here it must come down, so cations are the prize and carboxylates are
    penalised -- a single carboxylate-adjacent histidine can cancel two good
    ones, so leaving it to chance is not neutral.

    `pair_acids` adds the free-state lever: Asp/Glu biased at binder positions
    near a fully-biased histidine but >= 10 A from the target, i.e. on the
    binder's own solvent face. Worth 0.48 -> 0.90 S per histidine, but ONLY in
    combination with bound-state suppression -- alone it makes designs worse
    (see linkage.py and chal_2_local notebook 03 section 4).

    Returns (bias_dict, info_rows).
    """
    his_i = MPNN_ALPHABET.index("H")
    cys_i = MPNN_ALPHABET.index("C")
    asp_i = MPNN_ALPHABET.index("D")
    glu_i = MPNN_ALPHABET.index("E")

    cation_xyz = np.atleast_2d(cation_xyz) if len(cation_xyz) else np.zeros((0, 3))
    acid_xyz = np.atleast_2d(acid_xyz) if len(acid_xyz) else np.zeros((0, 3))
    target_cb = np.atleast_2d(target_cb) if len(target_cb) else np.zeros((0, 3))

    rows = np.zeros((len(binder_cb), len(MPNN_ALPHABET)), dtype=float)
    info = []
    anchors = []

    for i, ref in enumerate(binder_cb):
        d_cat = (float(np.min(np.linalg.norm(cation_xyz - ref, axis=1)))
                 if cation_xyz.size else np.inf)
        d_acid = (float(np.min(np.linalg.norm(acid_xyz - ref, axis=1)))
                  if acid_xyz.size else np.inf)
        b, parts = 0.0, []
        if d_cat <= HIS_BIAS_NEAR:
            b += his_bias
            parts.append(f"+{his_bias} cation {d_cat:.1f}A")
            anchors.append(i)
        elif d_cat <= HIS_BIAS_FAR:
            b += his_bias * HIS_BIAS_FAR_SCALE
            parts.append(f"+{his_bias * HIS_BIAS_FAR_SCALE:.2f} cation {d_cat:.1f}A")
        if d_acid <= HIS_PENALTY_NEAR:
            b += his_penalty
            parts.append(f"{his_penalty} carboxylate {d_acid:.1f}A")
        rows[i, his_i] = b
        # a free cysteine fails the submission gate anyway; cheaper never to
        # sample one than to discard the sequence afterwards
        rows[i, cys_i] = -8.0
        info.append({"pos": i + 1, "his_bias": round(b, 3),
                     "d_cation": None if not np.isfinite(d_cat) else round(d_cat, 2),
                     "d_acid": None if not np.isfinite(d_acid) else round(d_acid, 2),
                     "acid_bias": 0.0,
                     "reason": "; ".join(parts) or "no conserved charge in reach"})

    if pair_acids and anchors:
        anchor_cb = binder_cb[anchors]
        for i, ref in enumerate(binder_cb):
            if i in anchors:
                continue
            d_his = float(np.min(np.linalg.norm(anchor_cb - ref, axis=1)))
            if d_his > ACID_PAIR_NEAR:
                continue
            d_tgt = (float(np.min(np.linalg.norm(target_cb - ref, axis=1)))
                     if target_cb.size else np.inf)
            if d_tgt < ACID_PAIR_MIN_TARGET_DIST:
                continue
            rows[i, asp_i] += acid_bias
            rows[i, glu_i] += acid_bias
            info[i]["acid_bias"] = acid_bias
            info[i]["reason"] += (f"; +{acid_bias} acid ({d_his:.1f}A from a "
                                  f"biased His, {d_tgt:.1f}A from target)")

    return {binder_chain: rows.tolist()}, info


def _assigned_chains(assigned):
    """Read assign_fixed_chains.py's output: {name: [[designed], [fixed]]}.

    Every complex in the directory must agree, otherwise some backbones would
    be designed on a different chain from the rest and the bias array -- which
    is keyed by chain -- would land on the target in those.
    """
    data = json.loads(Path(assigned).read_text().strip())
    if not data:
        raise RuntimeError(f"{assigned} is empty; no complexes were parsed")
    seen = {(tuple(d), tuple(sorted(f))) for d, f in data.values()}
    if len(seen) != 1:
        raise RuntimeError(
            f"inconsistent chain assignment across {len(data)} complexes: "
            f"{sorted(seen)}")
    design, fixed = seen.pop()
    return list(design), list(fixed)


def run_proteinmpnn(py, pdb_dir, out_dir, designed_chain, fixed_chains,
                    n_seq=8, temperature=0.2, bias_jsonl=None, seed=0,
                    use_soluble=True, log=None, timeout=None):
    """Run ProteinMPNN over a directory of complexes.

    Uses the helper scripts to build the chain-id and fixed-chain JSONLs, which
    is the documented path and avoids hand-writing the parsed-chain format.
    `--use_soluble_model` is on by default: those weights were trained
    excluding transmembrane proteins and give more polar surfaces, which matters
    for a design that already needs three or four histidines and must still
    express in E. coli.
    """
    tools = tool_paths()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = out_dir / "parsed_chains.jsonl"
    assigned = out_dir / "assigned_chains.jsonl"

    run([py, str(tools["mpnn_parse"]), "--input_path", str(pdb_dir),
         "--output_path", str(parsed)], log=log, timeout=timeout)
    # assign_fixed_chains.py takes ONE chain flag, --chain_list, and it means
    # the chains to DESIGN; everything else in the parsed complex is fixed
    # automatically. There is no --designed_chain_list. Passing the target
    # chains here would redesign TNF-alpha itself, silently and with no error,
    # so fixed_chains is used only to check the caller agrees with the PDB.
    run([py, str(tools["mpnn_assign"]), "--input_path", str(parsed),
         "--output_path", str(assigned),
         "--chain_list", designed_chain], log=log, timeout=timeout)

    design, fixed = _assigned_chains(assigned)
    if design != [designed_chain] or set(fixed) != set(fixed_chains):
        raise RuntimeError(
            f"ProteinMPNN chain assignment is wrong: designing {design}, "
            f"fixing {fixed}; expected to design ['{designed_chain}'] and fix "
            f"{sorted(set(fixed_chains))}. Refusing to redesign the target.")
    if log:
        log(f"    mpnn: designing {design}, holding {sorted(fixed)} fixed")

    cmd = [py, str(tools["proteinmpnn_script"]),
           "--jsonl_path", str(parsed),
           "--chain_id_jsonl", str(assigned),
           "--out_folder", str(out_dir),
           "--num_seq_per_target", str(n_seq),
           "--sampling_temp", str(temperature),
           "--seed", str(seed),
           "--batch_size", "1"]
    if use_soluble:
        cmd.append("--use_soluble_model")
    if bias_jsonl:
        cmd += ["--bias_by_res_jsonl", str(bias_jsonl)]
    run(cmd, log=log, timeout=timeout)
    return out_dir / "seqs"


def parse_mpnn_fasta(path):
    """Read a ProteinMPNN output FASTA into records with its score fields.

    ProteinMPNN writes the native sequence first (``>name, score=...``) then the
    designs (``>T=0.2, sample=1, score=..., global_score=..., seq_recovery=...``).
    The native is skipped: it is the poly-glycine placeholder RFdiffusion
    emitted, not a design.
    """
    recs = []
    name, header = None, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
            continue
        if header is None:
            continue
        fields = {}
        for part in header.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    fields[k.strip()] = float(v)
                except ValueError:
                    fields[k.strip()] = v.strip()
        is_native = "sample" not in fields
        if not is_native:
            recs.append({"sequence": line.split("/")[0],
                         "mpnn_score": fields.get("score"),
                         "global_score": fields.get("global_score"),
                         "sample": int(fields.get("sample", 0))})
        header = None
    return recs


# --------------------------------------------------------------- ESMFold

ESMFOLD_SNIPPET = r'''
import json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np, torch
from transformers import AutoTokenizer, EsmForProteinFolding

seqs = json.load(open(sys.argv[1]))
out_path = sys.argv[2]
cache = sys.argv[3] if len(sys.argv) > 3 else None

tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache)
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                             cache_dir=cache,
                                             low_cpu_mem_usage=True)
model = model.cuda().eval()
# Volta (sm_70) has no bfloat16 and no FlashAttention-2, so this stays fp32
# throughout. set_chunk_size trades speed for memory on the attention maps and
# is what keeps a long sequence inside 32 GB.
model.trunk.set_chunk_size(64)

res = {}
for k, (name, s) in enumerate(seqs.items()):
    try:
        enc = tok([s], return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            o = model(**{q: v.cuda() for q, v in enc.items()})
        plddt = o["plddt"][0, :, 1].cpu().numpy()
        ca = o["positions"][-1, 0, :, 1].cpu().numpy()
        res[name] = {"mean_plddt": float(plddt.mean()),
                     "min_plddt": float(plddt.min()),
                     "plddt": [round(float(x), 2) for x in plddt],
                     "ca": [[round(float(c), 3) for c in p] for p in ca]}
    except Exception as exc:
        res[name] = {"error": f"{type(exc).__name__}: {exc}"}
    if (k + 1) % 10 == 0:
        print(f"  esmfold {k+1}/{len(seqs)}", flush=True)
    torch.cuda.empty_cache()

json.dump(res, open(out_path, "w"))
print(f"wrote {out_path}")
'''


def run_esmfold(py, sequences, out_json, work_dir, hf_cache=None, log=None,
                timeout=None):
    """Fold a dict of {name: sequence} with ESMFold, in the `fold` conda env.

    Runs as a generated script in a subprocess rather than being imported,
    because ESMFold needs `transformers` and the diffusion stages need
    `dgl` + torch 2.0.1, and those live in two different conda envs. Shelling
    out is what lets one pipeline.py drive both.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    seq_json = work_dir / "esmfold_input.json"
    script = work_dir / "_run_esmfold.py"
    seq_json.write_text(json.dumps(sequences))
    script.write_text(ESMFOLD_SNIPPET)
    cmd = [py, str(script), str(seq_json), str(out_json)]
    if hf_cache:
        cmd.append(str(hf_cache))
    run(cmd, log=log, timeout=timeout)
    return json.loads(Path(out_json).read_text())


def kabsch_rmsd(P, Q):
    """CA RMSD after optimal superposition, for the self-consistency filter.

    Compares the ESMFold model of a designed sequence against the RFdiffusion
    backbone it was designed on. A large RMSD means the sequence does not
    encode the backbone, which no amount of pH-switch cleverness rescues -- it
    is the single most informative filter in the pipeline and the one
    `chal_2_local` could not run at all.
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    n = min(len(P), len(Q))
    P, Q = P[:n], Q[:n]
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((Pc @ R.T) - Qc) ** 2).sum() / n))


# --------------------------------------------------------------- PDB helpers

def read_backbone(pdb_path, chain_id=None):
    """Read N, CA, C, O per residue from a PDB as an (L, 4, 3) array.

    Deliberately not Bio.PDB: RFdiffusion writes backbone-only files whose
    residue names are all GLY, and the strict parser warns on every one. This
    reads the columns directly, which is also faster over a few thousand files.
    Returns (coords, chain, resnums).
    """
    want = ("N", "CA", "C", "O")
    acc = {}
    order = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        if name not in want:
            continue
        cid = line[21]
        if chain_id is not None and cid != chain_id:
            continue
        rnum = int(line[22:26])
        key = (cid, rnum)
        if key not in acc:
            acc[key] = {}
            order.append(key)
        acc[key][name] = (float(line[30:38]), float(line[38:46]),
                          float(line[46:54]))
    coords, chains, nums = [], [], []
    for key in order:
        d = acc[key]
        if not all(a in d for a in want):
            continue
        coords.append([d[a] for a in want])
        chains.append(key[0])
        nums.append(key[1])
    return np.array(coords, float), chains, nums


def binder_chain_of(pdb_path, target_chains):
    """Which chain id RFdiffusion gave the binder.

    RFdiffusion relabels chains in its output, and which letter the binder gets
    depends on the contig order, so it is detected rather than assumed: the
    binder is the chain that is not one of the target chains. If the labels
    collide (it sometimes reuses A), the longest novel chain wins.
    """
    _, chains, _ = read_backbone(pdb_path)
    counts = {}
    for c in chains:
        counts[c] = counts.get(c, 0) + 1
    novel = {c: n for c, n in counts.items() if c not in set(target_chains)}
    if novel:
        return max(novel, key=novel.get)
    return max(counts, key=counts.get)

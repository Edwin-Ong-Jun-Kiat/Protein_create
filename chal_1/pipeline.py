#!/usr/bin/env python3
"""
=============================================================================
 EGFR pH-selective binder design pipeline
 Anthropic x Adaptyv 2026 Protein Design Competition -- Challenge 01 (EGFR)
=============================================================================

Target      : PDB 6ARU chain A, EGFR extracellular domain
Epitope     : Domain III (mature residues 310-481)
Objectives  : (1) pH-selective binding  -- binds pH 6.5, not pH 7.4  [PRIMARY]
              (2) mouse EGFR cross-reactivity
              (3) affinity to human EGFR

Stages
  prepare : parse target, find conserved acidic surface patches -> hotspots
  diffuse : RFdiffusion hotspot-conditioned binder backbones
  mpnn    : ProteinMPNN sequence design with interface histidine bias
  fold    : ESMFold self-consistency + side-chain recovery
  score   : thermodynamic-linkage pH score + cross-reactivity + composite
  submit  : ranked submission.csv

This is ONE script run under TWO conda envs (see README section 4), so every
heavy import is deliberately lazy and lives inside the stage that needs it.
Run `python pipeline.py --selftest` (CPU only) before spending GPU hours.
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ROOT = Path(os.environ.get("EGFR_ROOT", Path(__file__).resolve().parent))
THIRD_PARTY = Path(os.environ.get("EGFR_TP", ROOT / "third_party"))
INPUTS = Path(os.environ.get("EGFR_INPUTS", ROOT / "inputs"))
WORK = ROOT / "work"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

TARGET_PDB_IN = INPUTS / "6aru.pdb"
HUMAN_FASTA = INPUTS / "egfr_human_P00533.fasta"
MOUSE_FASTA = INPUTS / "egfr_mouse_Q01279.fasta"
TARGET_CHAIN = "A"

# EGFR ectodomain subdomain boundaries in MATURE numbering (signal peptide
# removed). Domain I 1-165, II 166-309, III 310-481, IV 482-618.
DOMAIN_III_MATURE = (310, 481)
# UniProt P00533 includes the 24-residue signal peptide, so
# p00533_position = mature_position + 24. We never assume the PDB uses one
# convention or the other -- the offset is derived by alignment in stage 0.
SIGNAL_PEPTIDE_LEN = 24

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C",
}

# Theoretical maximum solvent-accessible surface area, Tien et al. 2013 (A^2).
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# ProteinMPNN's fixed alphabet order -- bias vectors must match this exactly.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
HIS_IDX = MPNN_ALPHABET.index("H")

# Thermodynamics
RT_KCAL = 0.5921  # kcal/mol at 298 K
PH_LOW, PH_HIGH = 6.5, 7.4
PKA_HIS_FREE = 6.4  # model-compound histidine imidazole pKa

# Design length bins matching the challenge's stratification categories
BINS = {
    "micro": (20, 39),    # microbinder  < 40 aa
    "mini": (55, 95),     # minibinder   40-100 aa
    "large": (110, 150),  # large protein > 100 aa
}
STAGES = ["prepare", "diffuse", "mpnn", "fold", "score", "submit"]


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def log(msg: str, level: str = "info") -> None:
    colors = {"info": "\033[1;36m", "warn": "\033[1;33m", "err": "\033[1;31m",
              "ok": "\033[1;32m"}
    c = colors.get(level, "")
    print(f"{c}[{level:>4}]\033[0m {msg}", flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    log(msg, "err")
    sys.exit(1)


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None,
            env: Optional[dict] = None, log_file: Optional[Path] = None) -> None:
    """Run a subprocess, streaming to a log file. Raises on non-zero exit."""
    printable = " ".join(str(c) for c in cmd)
    log(f"$ {printable[:400]}{'...' if len(printable) > 400 else ''}")
    full_env = {**os.environ, **(env or {})}
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as fh:
            fh.write(f"\n\n===== {printable}\n\n")
            fh.flush()
            proc = subprocess.run(cmd, cwd=cwd, env=full_env,
                                  stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=full_env)
    if proc.returncode != 0:
        tail = ""
        if log_file and log_file.exists():
            tail = "\n".join(log_file.read_text(errors="replace").splitlines()[-40:])
        die(f"command failed (exit {proc.returncode}): {printable}\n\n--- log tail ---\n{tail}")


def read_fasta_one(path: Path) -> str:
    """Read a single-record FASTA and return the bare sequence."""
    if not path.exists():
        die(f"missing input FASTA: {path}. Run setup_env.sh first.")
    seq = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if seq:
                break
            continue
        seq.append(line.strip())
    s = "".join(seq).upper()
    if not s:
        die(f"empty FASTA: {path}")
    return s


def marker(stage: str) -> Path:
    return WORK / f".done_{stage}"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    n_backbones: int = 150
    seqs_per_backbone: int = 8
    diffuser_T: int = 50
    n_patches: int = 4
    his_bias: float = 2.5
    max_interface_his: int = 4
    # A single ionizable group cannot give more than (pH_HIGH - pH_LOW) = 0.9
    # log units of selectivity, and a realistic salt-bridged His reaches only
    # ~0.59 (4x). One histidine therefore CANNOT produce "binds at 6.5, no
    # binding at 7.4" -- two is the floor, three is comfortable. See README S3.
    min_interface_his: int = 2
    target_S: float = 1.5
    min_plddt: float = 75.0
    max_rmsd: float = 2.5
    n_submit: int = 40
    seed: int = 20261001
    shard: int = 0
    nshards: int = 1
    force: bool = False
    bin_weights: Dict[str, float] = field(
        default_factory=lambda: {"micro": 0.20, "mini": 0.55, "large": 0.25})
    name_prefix: str = "egfr_ph"

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=_json_default))


# --------------------------------------------------------------------------- #
# JSON encoding
# --------------------------------------------------------------------------- #
def _json_default(o):
    """Coerce the numpy scalars that leak into every metadata dict.

    Residue numbers arrive from numpy arrays as ``np.int64``, SASA fractions as
    ``np.float64`` and conservation flags as ``np.bool_``. None of the three is
    JSON-serialisable, and ``json.dumps`` raises

        TypeError: Object of type int64 is not JSON serializable

    which is exactly how this pipeline failed in job 10073878 -- after a clean
    selftest and four correctly scored epitope patches, at the very last line
    of ``stage_prepare``. The values print identically to Python ints in the
    log, so nothing upstream looks wrong.

    Every ``json.dumps`` in this file passes ``default=_json_default``. Adding
    one without it will reintroduce the same failure somewhere else.
    """
    import numpy as _np

    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable: {o!r}")


# --------------------------------------------------------------------------- #
# STAGE 0 -- prepare target and select epitope
# --------------------------------------------------------------------------- #

def _align_map(query: str, ref: str) -> Dict[int, int]:
    """Global-align `query` to `ref`; return {query_idx0 -> ref_idx0}."""
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    # Generous gap penalties: the PDB sequence is a contiguous subsequence of
    # the full-length UniProt entry, so we want one big terminal gap, not many
    # internal ones.
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    try:
        from Bio.Align import substitution_matrices
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        aligner.match_score = 2
        aligner.mismatch_score = -1

    aln = aligner.align(query, ref)[0]
    mapping: Dict[int, int] = {}
    # aligned -> [[(qs,qe),...],[(rs,re),...]] blocks of ungapped correspondence
    q_blocks, r_blocks = aln.aligned
    for (qs, qe), (rs, re) in zip(q_blocks, r_blocks):
        for k in range(qe - qs):
            mapping[qs + k] = rs + k
    return mapping


def _parse_target(pdb_path: Path, chain_id: str):
    """Return (structure, chain, residues, seq, resnums) for one chain."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    model = next(structure.get_models())
    if chain_id not in model:
        die(f"chain {chain_id} not found in {pdb_path}; "
            f"present: {[c.id for c in model]}")
    chain = model[chain_id]

    residues, seq, resnums = [], [], []
    for res in chain:
        het, num, icode = res.id
        # Skip waters, glycans, ligands and insertion codes.
        if het.strip() not in ("", "H_MSE"):
            continue
        if icode.strip():
            continue
        aa = AA3TO1.get(res.get_resname().upper())
        if aa is None or "CA" not in res:
            continue
        residues.append(res)
        seq.append(aa)
        resnums.append(num)
    if len(residues) < 100:
        die(f"only {len(residues)} standard residues parsed from chain "
            f"{chain_id} -- is {pdb_path} the right structure?")
    return structure, chain, residues, "".join(seq), resnums


def stage_prepare(cfg: Config) -> None:
    """Parse target, compute SASA + human/mouse conservation, pick hotspots."""
    import numpy as np
    from Bio.PDB import ShrakeRupley, PDBIO, Select

    WORK.mkdir(parents=True, exist_ok=True)
    log("stage prepare: parsing target structure")

    structure, chain, residues, pdb_seq, resnums = _parse_target(
        TARGET_PDB_IN, TARGET_CHAIN)
    log(f"  chain {TARGET_CHAIN}: {len(residues)} residues observed, "
        f"numbering {resnums[0]}-{resnums[-1]}")

    human = read_fasta_one(HUMAN_FASTA)
    mouse = read_fasta_one(MOUSE_FASTA)
    log(f"  human P00533 {len(human)} aa | mouse Q01279 {len(mouse)} aa")

    # --- Resolve the numbering convention by alignment, never by assumption --
    h_map = _align_map(pdb_seq, human)
    if len(h_map) < 0.8 * len(pdb_seq):
        die("PDB chain A does not align to P00533 -- wrong chain or structure?")
    ident = sum(1 for qi, ri in h_map.items() if pdb_seq[qi] == human[ri])
    log(f"  PDB<->P00533 identity: {ident}/{len(h_map)} "
        f"({100*ident/len(h_map):.1f}%)")

    # offset = P00533 residue number - PDB residue number  (expect 24)
    offsets = [(h_map[i] + 1) - resnums[i] for i in range(len(pdb_seq)) if i in h_map]
    offset = int(np.bincount(np.array(offsets) - min(offsets)).argmax()) + min(offsets)
    log(f"  detected numbering offset (P00533 - PDB) = {offset} "
        f"({'mature numbering, as expected' if offset == SIGNAL_PEPTIDE_LEN else 'UNEXPECTED'})")
    if offset != SIGNAL_PEPTIDE_LEN:
        log(f"  PDB does not use mature numbering; adapting Domain III range.",
            "warn")

    # Domain III in PDB numbering
    d3_p00533 = (DOMAIN_III_MATURE[0] + SIGNAL_PEPTIDE_LEN,
                 DOMAIN_III_MATURE[1] + SIGNAL_PEPTIDE_LEN)
    d3_pdb = (d3_p00533[0] - offset, d3_p00533[1] - offset)
    log(f"  Domain III: mature {DOMAIN_III_MATURE} = P00533 {d3_p00533} "
        f"= PDB {d3_pdb}")

    # --- Mouse conservation, mapped straight onto PDB residues --------------
    m_map = _align_map(pdb_seq, mouse)
    mouse_identical: Dict[int, bool] = {}
    for i, rn in enumerate(resnums):
        mi = m_map.get(i)
        mouse_identical[rn] = bool(mi is not None and mouse[mi] == pdb_seq[i])
    d3_res_nums = [rn for rn in resnums if d3_pdb[0] <= rn <= d3_pdb[1]]
    cons_frac = (sum(mouse_identical[rn] for rn in d3_res_nums) /
                 max(1, len(d3_res_nums)))
    log(f"  Domain III human/mouse identity: {100*cons_frac:.1f}%")

    # --- Relative SASA on the isolated target -------------------------------
    log("  computing SASA (Shrake-Rupley)")
    sr = ShrakeRupley()
    # Work on a chain-only copy so the Fab in 6ARU does not bury the epitope.
    import copy
    chain_only = copy.deepcopy(chain)
    # Detach from the model so it is unambiguous that SASA is computed on the
    # isolated receptor -- 6ARU is a complex, and leaving the cetuximab Fab in
    # scope would bury the very epitope we are trying to find.
    chain_only.detach_parent()
    for res in list(chain_only):
        if res.id[0].strip() not in ("", "H_MSE") or res.get_resname().upper() not in AA3TO1:
            chain_only.detach_child(res.id)
    sr.compute(chain_only, level="R")

    rel_sasa: Dict[int, float] = {}
    for res in chain_only:
        aa = AA3TO1.get(res.get_resname().upper())
        if aa is None:
            continue
        rel_sasa[res.id[1]] = res.sasa / MAX_ASA.get(aa, 200.0)

    # --- Write the truncated Domain III target ------------------------------
    class D3Select(Select):
        def accept_chain(self, c):
            return c.id == TARGET_CHAIN

        def accept_residue(self, res):
            het, num, icode = res.id
            return (het.strip() in ("", "H_MSE") and not icode.strip()
                    and res.get_resname().upper() in AA3TO1
                    and d3_pdb[0] <= num <= d3_pdb[1])

        def accept_atom(self, atom):
            # Drop hydrogens and alternate locations; keep crystallographic
            # side chains (needed later for real His-carboxylate geometry).
            return (atom.element != "H"
                    and (atom.get_altloc() in (" ", "A")))

    target_pdb = WORK / "target_domIII.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(target_pdb), select=D3Select())
    log(f"  wrote {target_pdb.name} ({len(d3_res_nums)} residues)")

    # --- Contig segments (handle unresolved gaps in Domain III) -------------
    segments: List[Tuple[int, int]] = []
    for rn in d3_res_nums:
        if segments and rn == segments[-1][1] + 1:
            segments[-1] = (segments[-1][0], rn)
        else:
            segments.append((rn, rn))
    segments = [s for s in segments if s[1] - s[0] + 1 >= 5]  # drop stubs
    contig_target = "/".join(f"{TARGET_CHAIN}{s}-{e}" for s, e in segments)
    log(f"  target contig: {contig_target}")

    # --- Epitope patch scoring ---------------------------------------------
    # A good patch for THIS challenge is: solvent exposed, in Domain III,
    # rich in human/mouse-conserved Asp/Glu (partners for a protonated binder
    # histidine -> criterion 1), broadly conserved (criterion 2), and not
    # dominated by Arg/Lys (which would repel His+).
    log("  scoring epitope patches")
    coords: Dict[int, "np.ndarray"] = {}
    aa_by_num: Dict[int, str] = {}
    for res in chain_only:
        num = res.id[1]
        if not (d3_pdb[0] <= num <= d3_pdb[1]):
            continue
        atom = res["CB"] if "CB" in res else res["CA"]
        coords[num] = atom.get_coord()
        aa_by_num[num] = AA3TO1[res.get_resname().upper()]

    surface = [n for n in coords if rel_sasa.get(n, 0.0) > 0.25]
    log(f"  {len(surface)} exposed Domain III residues (rel. SASA > 0.25)")

    patches = []
    for center in surface:
        nbrs = [n for n in surface
                if float(np.linalg.norm(coords[n] - coords[center])) <= 12.0]
        if len(nbrs) < 6:
            continue
        acidic_cons = [n for n in nbrs
                       if aa_by_num[n] in "DE" and mouse_identical.get(n, False)]
        basic = [n for n in nbrs if aa_by_num[n] in "KR"]
        n_cons = sum(1 for n in nbrs if mouse_identical.get(n, False))
        frac_cons = n_cons / len(nbrs)
        hydrophobic = [n for n in nbrs if aa_by_num[n] in "AVILMFWY"]

        score = (3.0 * len(acidic_cons)          # pH switch partners
                 + 6.0 * frac_cons               # mouse cross-reactivity
                 + 0.4 * len(hydrophobic)        # affinity / hot-spot anchor
                 - 1.0 * len(basic)              # His+ repulsion
                 + 0.15 * len(nbrs))             # patch size
        patches.append({
            "center": center, "score": round(score, 3),
            "residues": sorted(nbrs),
            "acidic_conserved": sorted(acidic_cons),
            "n_basic": len(basic),
            "frac_conserved": round(frac_cons, 3),
        })

    if not patches:
        die("no viable epitope patches found -- check SASA/domain boundaries")
    patches.sort(key=lambda p: -p["score"])

    # Greedy non-redundant selection: centers >= 14 A apart
    chosen: List[dict] = []
    for p in patches:
        if all(float(np.linalg.norm(coords[p["center"]] - coords[q["center"]])) > 14.0
               for q in chosen):
            chosen.append(p)
        if len(chosen) >= cfg.n_patches:
            break

    for i, p in enumerate(chosen):
        # Hotspots: prefer conserved acidic residues (the pH switch anchors),
        # topped up with the most exposed conserved neighbours. RFdiffusion
        # behaves best with 3-5 hotspots.
        hs = list(p["acidic_conserved"])[:3]
        extras = sorted(
            (n for n in p["residues"]
             if n not in hs and mouse_identical.get(n, False)),
            key=lambda n: -rel_sasa.get(n, 0.0))
        for n in extras:
            if len(hs) >= 4:
                break
            hs.append(n)
        if len(hs) < 3:
            hs = sorted(p["residues"], key=lambda n: -rel_sasa.get(n, 0.0))[:4]
        p["hotspots"] = sorted(hs)
        p["id"] = f"patch{i:02d}"
        p["hotspot_str"] = "[" + ",".join(f"{TARGET_CHAIN}{n}" for n in p["hotspots"]) + "]"
        log(f"    {p['id']}: score={p['score']:<7} "
            f"hotspots={p['hotspot_str']} "
            f"acidic_cons={p['acidic_conserved']} "
            f"cons={p['frac_conserved']}", "ok")

    meta = {
        "target_pdb": str(target_pdb),
        "target_chain": TARGET_CHAIN,
        "numbering_offset_p00533_minus_pdb": offset,
        "domain_iii_pdb": list(d3_pdb),
        "domain_iii_p00533": list(d3_p00533),
        "domain_iii_mature": list(DOMAIN_III_MATURE),
        "contig_target": contig_target,
        "domain_iii_mouse_identity": round(cons_frac, 4),
        "mouse_identical": {str(k): v for k, v in mouse_identical.items()},
        "rel_sasa": {str(k): round(v, 4) for k, v in rel_sasa.items()},
        "patches": chosen,
    }
    (WORK / "target_meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    log(f"stage prepare complete -> {WORK/'target_meta.json'}", "ok")


# --------------------------------------------------------------------------- #
# STAGE 1 -- RFdiffusion backbones
# --------------------------------------------------------------------------- #

def _check_gpu(require: bool = True) -> None:
    try:
        import torch
    except ImportError:
        die("torch not importable -- wrong conda env?")
    if not torch.cuda.is_available():
        if require:
            die("no CUDA device visible. Are you on a GPU node with --gres=gpu:1?")
        log("no CUDA device (expected on a login node)", "warn")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    log(f"GPU: {name} (sm_{cap[0]}{cap[1]}), "
        f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    if cap[0] < 7:
        die(f"GPU too old (sm_{cap[0]}{cap[1]}); this stack needs sm_70+")

    # Verify the torch build actually carries kernels for THIS device. This
    # check belongs here, not in setup: get_arch_list() short-circuits on
    # is_available() and returns [] on a login node, where it proves nothing.
    arch = torch.cuda.get_arch_list()
    want = f"sm_{cap[0]}{cap[1]}"
    if arch and not any(want in a for a in arch):
        die(f"this torch build has no {want} kernels for your {name}.\n"
            f"    compiled for: {arch}\n"
            f"    Reinstall torch 2.0.1+cu118 from "
            f"https://download.pytorch.org/whl/cu118")
    log(f"torch {torch.__version__} (cuda {torch.version.cuda}) "
        f"has {want} kernels", "ok")

    if cap[0] == 7 and cap[1] == 0:
        log("Volta detected: bfloat16 unavailable, running fp32 throughout", "warn")


def stage_diffuse(cfg: Config) -> None:
    """Hotspot-conditioned binder backbone generation with RFdiffusion."""
    import numpy as np

    meta = json.loads((WORK / "target_meta.json").read_text())
    _check_gpu()

    rfd = THIRD_PARTY / "RFdiffusion"
    runner = rfd / "scripts" / "run_inference.py"
    ckpt = rfd / "models" / "Complex_base_ckpt.pt"
    for p in (runner, ckpt):
        if not p.exists():
            die(f"missing {p}. Run setup_env.sh.")

    out_dir = WORK / "backbones"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the job list: (patch, length_bin, n_designs) spread over patches.
    rng = np.random.default_rng(cfg.seed)
    patches = meta["patches"]
    jobs: List[dict] = []
    for bin_name, weight in cfg.bin_weights.items():
        if weight <= 0:
            continue  # bin deliberately disabled (e.g. --learn skips 'large')
        n_bin = max(1, int(round(cfg.n_backbones * weight)))
        lo, hi = BINS[bin_name]
        for k in range(n_bin):
            patch = patches[k % len(patches)]
            jobs.append({
                "patch": patch["id"],
                "hotspot_str": patch["hotspot_str"],
                "bin": bin_name,
                "length": int(rng.integers(lo, hi + 1)),
            })
    rng.shuffle(jobs)

    # Group by (patch, bin, length) so one RFdiffusion call makes several
    # designs -- model load is ~30 s and we do not want to pay it per design.
    groups: Dict[Tuple[str, str, int], dict] = {}
    for j in jobs:
        key = (j["patch"], j["bin"], j["length"])
        if key not in groups:
            groups[key] = {**j, "num": 0}
        groups[key]["num"] += 1
    group_list = sorted(groups.values(), key=lambda g: (g["patch"], g["bin"], g["length"]))

    # SLURM array sharding
    my_groups = [g for i, g in enumerate(group_list) if i % cfg.nshards == cfg.shard]
    log(f"stage diffuse: {len(group_list)} groups total, "
        f"{len(my_groups)} on shard {cfg.shard}/{cfg.nshards} "
        f"({sum(g['num'] for g in my_groups)} designs)")

    env = {
        "HYDRA_FULL_ERROR": "1",
        # RFdiffusion's SE3Transformer is fp32; make sure TF32 is off so
        # numerics match the published protocol on Volta.
        "NVIDIA_TF32_OVERRIDE": "0",
    }

    for gi, g in enumerate(my_groups, 1):
        prefix = out_dir / f"{g['patch']}_{g['bin']}_L{g['length']}"
        # Resume support: skip groups that already produced all their designs.
        existing = sorted(out_dir.glob(f"{prefix.name}_*.pdb"))
        if len(existing) >= g["num"] and not cfg.force:
            log(f"  [{gi}/{len(my_groups)}] {prefix.name}: "
                f"{len(existing)} pdbs present, skipping")
            continue

        contig = f"[{meta['contig_target']}/0 {g['length']}-{g['length']}]"
        cmd = [
            sys.executable, str(runner),
            f"inference.input_pdb={meta['target_pdb']}",
            f"inference.output_prefix={prefix}",
            f"inference.ckpt_override_path={ckpt}",
            f"inference.num_designs={g['num']}",
            f"contigmap.contigs={contig}",
            f"ppi.hotspot_res={g['hotspot_str']}",
            f"diffuser.T={cfg.diffuser_T}",
            # noise_scale 0 is the Baker-lab binder protocol: it produces
            # more idealised, more designable backbones than the default 1.0.
            "denoiser.noise_scale_ca=0",
            "denoiser.noise_scale_frame=0",
        ]
        log(f"  [{gi}/{len(my_groups)}] {prefix.name} "
            f"x{g['num']} hotspots={g['hotspot_str']}")
        run_cmd(cmd, cwd=rfd, env=env,
                log_file=LOGS / f"rfdiffusion_shard{cfg.shard}.log")

    n = len(list(out_dir.glob("*.pdb")))
    log(f"stage diffuse complete: {n} backbones in {out_dir}", "ok")
    if n == 0:
        die("RFdiffusion produced no backbones -- inspect logs/rfdiffusion_*.log")


# --------------------------------------------------------------------------- #
# STAGE 2 -- ProteinMPNN with interface histidine bias
# --------------------------------------------------------------------------- #

def _read_pdb_chains(path: Path) -> Dict[str, List[dict]]:
    """Minimal PDB reader -> {chain: [{resnum, resname, atoms:{name:(x,y,z)}}]}."""
    chains: Dict[str, List[dict]] = {}
    cur: Dict[Tuple[str, int], dict] = {}
    order: Dict[str, List[Tuple[str, int]]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        altloc = line[16]
        resname = line[17:20].strip().upper()
        ch = line[21]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        if altloc not in (" ", "A"):
            continue
        if line[76:78].strip() == "H" or name.startswith("H"):
            continue
        key = (ch, resnum)
        if key not in cur:
            cur[key] = {"resnum": resnum, "resname": resname, "atoms": {}}
            order.setdefault(ch, []).append(key)
        cur[key]["atoms"][name] = (
            float(line[30:38]), float(line[38:46]), float(line[46:54]))
    for ch, keys in order.items():
        chains[ch] = [cur[k] for k in keys]
    return chains


def _identify_chains(pdb: Path, target_len_hint: int) -> Tuple[str, str]:
    """Return (target_chain, binder_chain) for an RFdiffusion output.

    RFdiffusion's chain lettering depends on contig order, and relying on that
    silently breaks if the contig changes. Identify by length instead: the
    target is the chain closest to the known Domain III length.
    """
    chains = _read_pdb_chains(pdb)
    if len(chains) < 2:
        die(f"{pdb} has {len(chains)} chain(s); expected target + binder")
    best = min(chains, key=lambda c: abs(len(chains[c]) - target_len_hint))
    others = [c for c in chains if c != best]
    if len(others) != 1:
        die(f"{pdb}: ambiguous chains {[(c, len(chains[c])) for c in chains]}")
    return best, others[0]


def _relabel(pdb_in: Path, pdb_out: Path, target_ch: str, binder_ch: str) -> int:
    """Rewrite so target=A, binder=B (ProteinMPNN wants stable lettering).

    Returns the binder length.
    """
    out_lines: List[str] = []
    seen: set = set()
    for line in pdb_in.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            ch = line[21]
            if ch == target_ch:
                new = "A"
            elif ch == binder_ch:
                new = "B"
                try:
                    seen.add(int(line[22:26]))
                except ValueError:
                    pass
            else:
                continue
            out_lines.append(line[:21] + new + line[22:])
        elif line.startswith("TER"):
            out_lines.append(line)
    out_lines.append("END")
    pdb_out.write_text("\n".join(out_lines) + "\n")
    return len(seen)


def stage_mpnn(cfg: Config) -> None:
    """Sequence design, twice: pH-biased and unbiased."""
    import numpy as np

    meta = json.loads((WORK / "target_meta.json").read_text())
    mpnn_dir = THIRD_PARTY / "ProteinMPNN"
    runner = mpnn_dir / "protein_mpnn_run.py"
    if not runner.exists():
        die(f"missing {runner}. Run setup_env.sh.")

    backbones = sorted((WORK / "backbones").glob("*.pdb"))
    if not backbones:
        die("no backbones found -- run --stage diffuse first")
    log(f"stage mpnn: {len(backbones)} backbones")

    # Count residues actually present in the truncated target pdb -- this is
    # the length hint used to tell target from binder in RFdiffusion output.
    tgt_chains = _read_pdb_chains(Path(meta["target_pdb"]))
    target_len = len(tgt_chains[TARGET_CHAIN])

    clean_dir = WORK / "mpnn" / "pdbs"
    clean_dir.mkdir(parents=True, exist_ok=True)

    hotspots = {h for p in meta["patches"] for h in p["hotspots"]}
    # Conserved target carboxylate oxygens: the anchors a binder His should
    # reach to become pH-responsive.
    acid_atoms: List[Tuple[int, "np.ndarray"]] = []
    for res in tgt_chains[TARGET_CHAIN]:
        if res["resname"] in ("ASP", "GLU"):
            if not meta["mouse_identical"].get(str(res["resnum"]), False):
                continue
            for an in ("OD1", "OD2", "OE1", "OE2"):
                if an in res["atoms"]:
                    acid_atoms.append((res["resnum"], np.array(res["atoms"][an])))

    log(f"  target: {target_len} residues, "
        f"{len(hotspots)} hotspot residues, "
        f"{len(acid_atoms)} conserved carboxylate oxygens")

    bias: Dict[str, dict] = {}
    manifest: Dict[str, dict] = {}
    n_biased_pos = 0

    for bb in backbones:
        tch, bch = _identify_chains(bb, target_len)
        name = bb.stem
        clean = clean_dir / f"{name}.pdb"
        blen = _relabel(bb, clean, tch, bch)
        chains = _read_pdb_chains(clean)
        binder = chains["B"]
        tgt = chains["A"]

        tgt_coords = np.array([r["atoms"]["CA"] for r in tgt if "CA" in r["atoms"]])
        tgt_nums = [r["resnum"] for r in tgt if "CA" in r["atoms"]]
        hs_coords = np.array([c for c, n in zip(tgt_coords, tgt_nums) if n in hotspots])

        vec = np.zeros((len(binder), len(MPNN_ALPHABET)), dtype=float)
        iface_positions: List[int] = []
        for i, res in enumerate(binder):
            # RFdiffusion emits backbone only; CB is present for non-Gly.
            ref = res["atoms"].get("CB") or res["atoms"].get("CA")
            if ref is None:
                continue
            ref = np.array(ref)
            # Interface = facing the hotspot patch.
            if hs_coords.size and np.min(np.linalg.norm(hs_coords - ref, axis=1)) <= 12.0:
                iface_positions.append(i)
                # Scale the His bias by proximity to a conserved carboxylate:
                # a His only becomes a pH switch if it can actually reach one.
                if acid_atoms:
                    d = min(float(np.linalg.norm(a - ref)) for _, a in acid_atoms)
                else:
                    d = 99.0
                if d <= 8.0:
                    vec[i, HIS_IDX] = cfg.his_bias
                    n_biased_pos += 1
                elif d <= 11.0:
                    vec[i, HIS_IDX] = cfg.his_bias * 0.4
                    n_biased_pos += 1
        bias[name] = {"B": vec.tolist()}
        manifest[name] = {
            "binder_len": blen,
            "n_interface": len(iface_positions),
            "interface_positions": iface_positions,
            "patch": name.split("_")[0],
            "bin": name.split("_")[1],
        }

    (WORK / "mpnn" / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    log(f"  applied His bias at {n_biased_pos} positions across "
        f"{len(backbones)} backbones")

    # ProteinMPNN reads the jsonl line-by-line and keeps the last object, so a
    # single line holding every design is the correct format here.
    bias_file = WORK / "mpnn" / "bias_by_res.jsonl"
    with open(bias_file, "w") as fh:
        fh.write(json.dumps(bias, default=_json_default) + "\n")

    helpers = mpnn_dir / "helper_scripts"
    pdbs_jsonl = WORK / "mpnn" / "pdbs.jsonl"
    chains_jsonl = WORK / "mpnn" / "assigned_chains.jsonl"

    run_cmd([sys.executable, str(helpers / "parse_multiple_chains.py"),
             "--input_path", str(clean_dir), "--output_path", str(pdbs_jsonl)],
            log_file=LOGS / "mpnn.log")
    run_cmd([sys.executable, str(helpers / "assign_fixed_chains.py"),
             "--input_path", str(pdbs_jsonl), "--output_path", str(chains_jsonl),
             "--chain_list", "B"],
            log_file=LOGS / "mpnn.log")

    for tag, use_bias in (("biased", True), ("neutral", False)):
        out = WORK / "mpnn" / tag
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(runner),
            "--jsonl_path", str(pdbs_jsonl),
            "--chain_id_jsonl", str(chains_jsonl),
            "--out_folder", str(out),
            "--num_seq_per_target", str(cfg.seqs_per_backbone),
            "--sampling_temp", "0.15",
            "--seed", str(cfg.seed + (1 if use_bias else 2)),
            "--batch_size", "1",
            # Free cysteines cause disulfide scrambling and hurt expression.
            "--omit_AAs", "CX",
            "--use_soluble_model",
        ]
        if use_bias:
            cmd += ["--bias_by_res_jsonl", str(bias_file)]
        log(f"  ProteinMPNN pass: {tag}")
        run_cmd(cmd, cwd=mpnn_dir, log_file=LOGS / "mpnn.log")

    # --- Collect sequences --------------------------------------------------
    records: List[dict] = []
    for tag in ("biased", "neutral"):
        seq_dir = WORK / "mpnn" / tag / "seqs"
        if not seq_dir.exists():
            log(f"  no seqs/ for pass {tag}", "warn")
            continue
        for fa in sorted(seq_dir.glob("*.fa")):
            name = fa.stem
            blen = manifest.get(name, {}).get("binder_len")
            entries = _parse_mpnn_fasta(fa)
            # entry 0 is the input (native/poly-X) sequence; skip it
            for si, ent in enumerate(entries[1:]):
                binder_seq = _pick_binder_chain(ent["seq"], blen)
                if binder_seq is None:
                    continue
                records.append({
                    "id": f"{name}__{tag}__{si}",
                    "backbone": name,
                    "pass": tag,
                    "sample": si,
                    "sequence": binder_seq,
                    "mpnn_score": ent.get("score"),
                    "mpnn_global_score": ent.get("global_score"),
                    **{k: v for k, v in manifest.get(name, {}).items()
                       if k in ("binder_len", "n_interface", "patch", "bin")},
                })

    if not records:
        die("ProteinMPNN produced no usable sequences -- see logs/mpnn.log")
    (WORK / "sequences.json").write_text(json.dumps(records, indent=2, default=_json_default))
    n_his = sum(r["sequence"].count("H") for r in records) / len(records)
    log(f"stage mpnn complete: {len(records)} sequences "
        f"(mean {n_his:.2f} His per design)", "ok")


def _parse_mpnn_fasta(path: Path) -> List[dict]:
    """Parse a ProteinMPNN output FASTA into [{seq, score, global_score}]."""
    entries: List[dict] = []
    header, seq_parts = None, []

    def flush():
        if header is None:
            return
        rec = {"seq": "".join(seq_parts)}
        for key in ("score", "global_score", "seq_recovery"):
            m = re.search(rf"{key}=([0-9.eE+-]+)", header)
            if m:
                try:
                    rec[key] = float(m.group(1))
                except ValueError:
                    pass
        entries.append(rec)

    for line in path.read_text().splitlines():
        if line.startswith(">"):
            flush()
            header, seq_parts = line[1:], []
        elif line.strip():
            seq_parts.append(line.strip())
    flush()
    return entries


def _pick_binder_chain(seq: str, binder_len: Optional[int]) -> Optional[str]:
    """ProteinMPNN joins chains with '/'; pick the binder by length."""
    parts = [p for p in seq.split("/") if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if binder_len:
        exact = [p for p in parts if len(p) == binder_len]
        if exact:
            return exact[0]
    # Chains are emitted alphabetically (A=target, B=binder) -> take the
    # shortest as a fallback, since the binder is always smaller than DomIII.
    return min(parts, key=len)


# --------------------------------------------------------------------------- #
# STAGE 3 -- ESMFold self-consistency  (runs in the egfr-fold env)
# --------------------------------------------------------------------------- #

def _kabsch(P, Q):
    """Superimpose P onto Q. Returns (R, centroid_P, centroid_Q, rmsd)."""
    import numpy as np

    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - cp, Q - cq
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    rmsd = float(np.sqrt(np.mean(np.sum((Pc @ R.T - Qc) ** 2, axis=1))))
    return R, cp, cq, rmsd


def stage_fold(cfg: Config) -> None:
    """Fold each unique binder sequence with ESMFold; keep self-consistent ones."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    records = json.loads((WORK / "sequences.json").read_text())
    _check_gpu()

    # Deduplicate: many backbones converge on the same sequence.
    by_seq: Dict[str, List[dict]] = {}
    for r in records:
        by_seq.setdefault(r["sequence"], []).append(r)
    uniq = sorted(by_seq)
    log(f"stage fold: {len(records)} sequences -> {len(uniq)} unique")

    out_dir = WORK / "esmfold"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = WORK / "fold_metrics.json"
    done: Dict[str, dict] = json.loads(cache.read_text()) if (cache.exists() and not cfg.force) else {}

    model_id = "facebook/esmfold_v1"
    log("  loading ESMFold (fp32 -- Volta has no bfloat16)")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = EsmForProteinFolding.from_pretrained(model_id, low_cpu_mem_usage=True)
    model = model.float().cuda().eval()
    # Chunked attention keeps peak memory inside a 16 GiB V100.
    model.trunk.set_chunk_size(64)

    bb_cache: Dict[str, "np.ndarray"] = {}

    def backbone_ca(bb_name: str):
        if bb_name not in bb_cache:
            p = WORK / "mpnn" / "pdbs" / f"{bb_name}.pdb"
            ch = _read_pdb_chains(p)
            bb_cache[bb_name] = np.array(
                [r["atoms"]["CA"] for r in ch["B"] if "CA" in r["atoms"]])
        return bb_cache[bb_name]

    for i, seq in enumerate(uniq, 1):
        key = seq
        if key in done:
            continue
        try:
            with torch.no_grad():
                enc = tok([seq], return_tensors="pt", add_special_tokens=False)
                enc = {k: v.cuda() for k, v in enc.items()}
                out = model(**enc)
            plddt = out["plddt"] if isinstance(out, dict) else out.plddt
            plddt = plddt.detach().float().cpu().numpy()[0]  # (L, 37)
            ca_plddt = plddt[:, 1]
            if float(ca_plddt.max()) <= 1.0:  # some versions return 0-1
                ca_plddt = ca_plddt * 100.0
            pdb_str = model.output_to_pdb(out)[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  [{i}/{len(uniq)}] OOM on L={len(seq)}, skipping", "warn")
            continue
        except Exception as exc:  # noqa: BLE001
            log(f"  [{i}/{len(uniq)}] ESMFold failed: {exc}", "warn")
            continue

        sid = f"seq{i:05d}"
        (out_dir / f"{sid}.pdb").write_text(pdb_str)

        # Self-consistency: does ESMFold reproduce the diffused backbone?
        rec0 = by_seq[seq][0]
        ref = backbone_ca(rec0["backbone"])
        pred_ch = _read_pdb_chains(out_dir / f"{sid}.pdb")
        pred_ca = np.array([r["atoms"]["CA"]
                            for r in next(iter(pred_ch.values()))
                            if "CA" in r["atoms"]])
        rmsd = float("nan")
        if len(pred_ca) == len(ref) and len(ref) >= 4:
            _, _, _, rmsd = _kabsch(pred_ca, ref)

        done[key] = {
            "sid": sid,
            "mean_plddt": round(float(ca_plddt.mean()), 2),
            "min_plddt": round(float(ca_plddt.min()), 2),
            "rmsd_to_backbone": None if rmsd != rmsd else round(rmsd, 3),
            "length": len(seq),
        }
        if i % 25 == 0 or i == len(uniq):
            cache.write_text(json.dumps(done, indent=2, default=_json_default))
            log(f"  [{i}/{len(uniq)}] "
                f"pLDDT={done[key]['mean_plddt']} "
                f"RMSD={done[key]['rmsd_to_backbone']}")

    cache.write_text(json.dumps(done, indent=2, default=_json_default))
    good = sum(1 for v in done.values()
               if v["mean_plddt"] >= cfg.min_plddt
               and (v["rmsd_to_backbone"] or 99) <= cfg.max_rmsd)
    log(f"stage fold complete: {len(done)} folded, {good} pass "
        f"pLDDT>={cfg.min_plddt} & RMSD<={cfg.max_rmsd}", "ok")


# --------------------------------------------------------------------------- #
# STAGE 4 -- pH-selectivity + cross-reactivity scoring
# --------------------------------------------------------------------------- #

def _ph_selectivity(pka_free: float, pka_bound: float) -> float:
    """log10(Kd(pH_HIGH) / Kd(pH_LOW)) for one ionizable group.

    Thermodynamic linkage: a group whose pKa rises on binding makes binding
    stronger at low pH. Positive return value = the direction we want.
    """
    import math

    def term(ph: float) -> float:
        return math.log10((1 + 10 ** (pka_bound - ph)) /
                          (1 + 10 ** (pka_free - ph)))

    return term(PH_LOW) - term(PH_HIGH)


def _estimate_pka_bound(min_acid_d: float, min_base_d: float,
                        min_polar_d: float, buried: bool) -> float:
    """Heuristic bound-state His pKa from interface environment (README S3).

    NOT a Poisson-Boltzmann calculation -- calibrated offsets that rank designs
    sensibly. Do not quote the absolute numbers as predictions.
    """
    pka = PKA_HIS_FREE
    if min_acid_d <= 4.0:
        pka += 1.5           # salt bridge to carboxylate stabilises His+
    elif min_acid_d <= 6.0:
        pka += 0.8           # water-mediated / second shell
    if min_polar_d <= 3.5:
        pka += 0.3           # neutral H-bond
    if min_base_d <= 5.0:
        pka -= 0.7           # like-charge repulsion from Arg/Lys
    if buried and min_acid_d > 6.0:
        pka -= 1.0           # burying a charge with no partner is costly
    return max(4.0, min(9.0, pka))


def stage_score(cfg: Config) -> None:
    """Combine pH selectivity, mouse conservation, and fold quality."""
    import numpy as np

    meta = json.loads((WORK / "target_meta.json").read_text())
    records = json.loads((WORK / "sequences.json").read_text())
    fold = json.loads((WORK / "fold_metrics.json").read_text())
    log(f"stage score: {len(records)} sequences, {len(fold)} folded")

    tgt_chains = _read_pdb_chains(Path(meta["target_pdb"]))
    tgt = tgt_chains[TARGET_CHAIN]
    mouse_ident = meta["mouse_identical"]
    rel_sasa = meta["rel_sasa"]

    # Pre-index target atoms by chemical role.
    tgt_all, tgt_all_res = [], []
    acid_o, base_n, polar_o = [], [], []
    for res in tgt:
        rn, rname = res["resnum"], res["resname"]
        for an, xyz in res["atoms"].items():
            tgt_all.append(xyz)
            tgt_all_res.append(rn)
            if rname == "ASP" and an in ("OD1", "OD2"):
                acid_o.append(xyz)
            elif rname == "GLU" and an in ("OE1", "OE2"):
                acid_o.append(xyz)
            elif rname == "ARG" and an in ("NE", "NH1", "NH2"):
                base_n.append(xyz)
            elif rname == "LYS" and an == "NZ":
                base_n.append(xyz)
            elif rname == "HIS" and an in ("ND1", "NE2"):
                base_n.append(xyz)
            elif an in ("OG", "OG1", "OH", "OD1", "OE1", "ND2", "NE2", "O"):
                # Neutral H-bond partners. The elif chain has already claimed
                # Asp/Glu carboxylate oxygens above, so there is no double
                # credit for a residue that is both acidic and polar.
                polar_o.append(xyz)
    tgt_all = np.array(tgt_all)
    tgt_all_res = np.array(tgt_all_res)
    acid_o = np.array(acid_o) if acid_o else np.zeros((0, 3))
    base_n = np.array(base_n) if base_n else np.zeros((0, 3))
    polar_o = np.array(polar_o) if polar_o else np.zeros((0, 3))

    # KD-trees built once over the fixed target. Without these, scoring a few
    # thousand designs means millions of per-atom distance calls.
    from scipy.spatial import cKDTree

    def _tree(arr):
        return cKDTree(arr) if arr.shape[0] else None

    tree_all, tree_acid = _tree(tgt_all), _tree(acid_o)
    tree_base, tree_polar = _tree(base_n), _tree(polar_o)

    def mind(pts, tree) -> float:
        """Min distance from any of `pts` (N,3) to the nearest atom in `tree`."""
        if tree is None:
            return 99.0
        d, _ = tree.query(np.atleast_2d(pts), k=1)
        return float(np.min(d))

    esm_dir = WORK / "esmfold"
    scored: List[dict] = []
    skipped = {"unfolded": 0, "plddt": 0, "rmsd": 0, "his": 0, "geom": 0}

    for rec in records:
        fm = fold.get(rec["sequence"])
        if fm is None:
            skipped["unfolded"] += 1
            continue

        micro = rec.get("bin") == "micro"
        plddt_gate = cfg.min_plddt - (10.0 if micro else 0.0)
        if fm["mean_plddt"] < plddt_gate:
            skipped["plddt"] += 1
            continue

        # --- Put the ESMFold model into the complex frame -------------------
        # This is what gives us real binder side-chain coordinates against the
        # crystallographic target side chains (README S2, stage 3).
        bb_path = WORK / "mpnn" / "pdbs" / f"{rec['backbone']}.pdb"
        esm_path = esm_dir / f"{fm['sid']}.pdb"
        if not (bb_path.exists() and esm_path.exists()):
            skipped["geom"] += 1
            continue
        bb_res = _read_pdb_chains(bb_path)["B"]
        esm_res = next(iter(_read_pdb_chains(esm_path).values()))
        ref_ca = np.array([r["atoms"]["CA"] for r in bb_res if "CA" in r["atoms"]])
        mob_ca = np.array([r["atoms"]["CA"] for r in esm_res if "CA" in r["atoms"]])
        if len(ref_ca) != len(mob_ca) or len(ref_ca) < 4:
            skipped["geom"] += 1
            continue
        R, cp, cq, rmsd = _kabsch(mob_ca, ref_ca)

        # Gate on the RMSD from THIS superposition, not the one cached in
        # stage fold. Identical sequences can arise from different backbones,
        # and stage fold only stored the RMSD against the first of them.
        if rmsd > cfg.max_rmsd:
            skipped["rmsd"] += 1
            continue
        rmsd = round(rmsd, 3)

        def xform(xyz):
            """Map ESMFold-frame coords (N,3) or (3,) into the complex frame."""
            return (np.asarray(xyz, dtype=float) - cp) @ R.T + cq

        # --- Interface contacts (vectorised) -------------------------------
        flat_xyz, flat_res = [], []
        for ri, r in enumerate(esm_res):
            for xyz in r["atoms"].values():
                flat_xyz.append(xyz)
                flat_res.append(ri)
        binder_xyz = xform(np.array(flat_xyz))

        neigh = tree_all.query_ball_point(binder_xyz, r=5.0)
        n_contacts = int(sum(len(ix) for ix in neigh))
        contacted: set = {int(tgt_all_res[j]) for ix in neigh for j in ix}
        if not contacted:
            skipped["geom"] += 1
            continue

        # --- Histidine pH switches -----------------------------------------
        his_details: List[dict] = []
        for idx, r in enumerate(esm_res):
            if r["resname"] != "HIS":
                continue
            ring = [r["atoms"][a] for a in ("ND1", "NE2", "CE1", "CD2", "CG")
                    if a in r["atoms"]]
            if not ring:
                continue
            ring_t = xform(np.array(ring))
            # The two ring nitrogens carry the charge; fall back to the whole
            # ring if ESMFold emitted an incomplete side chain.
            n_atoms = [r["atoms"][a] for a in ("ND1", "NE2") if a in r["atoms"]]
            charge_pts = xform(np.array(n_atoms)) if n_atoms else ring_t
            if mind(ring_t, tree_all) > 8.0:
                continue  # not an interface His; irrelevant to binding pH
            d_acid = mind(charge_pts, tree_acid)
            d_base = mind(charge_pts, tree_base)
            d_polar = mind(charge_pts, tree_polar)
            buried = mind(charge_pts[:1], tree_all) < 4.5
            pka_b = _estimate_pka_bound(d_acid, d_base, d_polar, buried)
            s_i = _ph_selectivity(PKA_HIS_FREE, pka_b)
            his_details.append({
                "pos": idx + 1, "d_acid": round(d_acid, 2),
                "d_base": round(d_base, 2), "pka_bound": round(pka_b, 2),
                "S_i": round(s_i, 3),
            })

        n_his = len(his_details)
        if not (cfg.min_interface_his <= n_his <= cfg.max_interface_his):
            skipped["his"] += 1
            continue

        S_total = sum(h["S_i"] for h in his_details)
        # Reward getting close to the target window; penalise overshoot less
        # harshly than undershoot (overshoot still reads as pH-selective).
        gap = S_total - cfg.target_S
        S_fit = -abs(gap) if gap < 0 else -0.4 * gap

        mouse_frac = (sum(1 for rn in contacted
                          if mouse_ident.get(str(rn), False)) / len(contacted))
        hotspot_overlap = len(contacted & {h for p in meta["patches"]
                                           for h in p["hotspots"]})

        seq = rec["sequence"]
        composite = (
            4.0 * S_fit                              # criterion 1 dominates
            + 3.0 * mouse_frac                       # criterion 2
            + 0.04 * (fm["mean_plddt"] - 70.0)       # foldability
            - 0.5 * rmsd                             # self-consistency
            + 0.006 * min(n_contacts, 400)           # interface size / affinity
            + 0.3 * min(hotspot_overlap, 4)
            - 2.0 * max(0, seq.count("H") - cfg.max_interface_his - 2)
        )

        scored.append({
            **rec,
            "sid": fm["sid"],
            "mean_plddt": fm["mean_plddt"],
            "min_plddt": fm["min_plddt"],
            "rmsd": rmsd,
            "n_interface_his": n_his,
            "S_pH": round(S_total, 3),
            "pred_fold_selectivity": round(10 ** S_total, 1),
            "his_detail": his_details,
            "n_contacts": n_contacts,
            "n_contacted_residues": len(contacted),
            "contacted_residues": sorted(contacted),
            "mouse_identity_at_interface": round(mouse_frac, 3),
            "hotspot_overlap": hotspot_overlap,
            "total_his": seq.count("H"),
            "composite": round(composite, 4),
        })

    scored.sort(key=lambda d: -d["composite"])
    (WORK / "scored.json").write_text(json.dumps(scored, indent=2, default=_json_default))
    log(f"  filtered out: {skipped}")
    log(f"stage score complete: {len(scored)} designs survived", "ok")
    if scored:
        top = scored[0]
        log(f"  best: S={top['S_pH']} ({top['pred_fold_selectivity']}x weaker "
            f"at pH 7.4), mouse={top['mouse_identity_at_interface']}, "
            f"pLDDT={top['mean_plddt']}, {top['n_interface_his']} iface His", "ok")
        _print_score_breakdown(scored)
    else:
        log("no designs survived scoring. Loosen --min-plddt / --max-rmsd, "
            "or raise --his-bias so more designs carry an interface His. "
            "Or run with --learn, which relaxes all three.", "warn")


def _print_score_breakdown(scored: List[dict]) -> None:
    """Show pH selectivity grouped by interface-histidine count.

    This is the single most instructive table the pipeline produces: it shows
    empirically that each histidine buys at most 0.9 log units (README S3), so
    selectivity is bought by His COUNT, not by one perfect histidine.
    """
    groups: Dict[int, List[dict]] = {}
    for d in scored:
        groups.setdefault(d["n_interface_his"], []).append(d)

    print("\n  pH selectivity vs interface histidine count")
    print("  " + "-" * 68)
    print(f"  {'His':>3} {'n':>5} {'mean S':>8} {'best S':>8} "
          f"{'best fold':>10} {'ceiling':>9}  {'bar (best S)':<12}")
    print("  " + "-" * 68)
    for nh in sorted(groups):
        g = groups[nh]
        ss = [d["S_pH"] for d in g]
        mean_s, best_s = sum(ss) / len(ss), max(ss)
        ceiling = 0.9 * nh
        bar = "#" * min(12, int(round(best_s / 0.3)))
        print(f"  {nh:>3} {len(g):>5} {mean_s:>8.2f} {best_s:>8.2f} "
              f"{10**best_s:>9.0f}x {ceiling:>8.1f}  {bar:<12}")
    print("  " + "-" * 68)
    print("  'ceiling' = 0.9 x His = the hard thermodynamic limit at pH 6.5 vs 7.4.")
    print("  A design needs S >~ 1.0 (10x) before 'binds at 6.5, not at 7.4'")
    print("  is a defensible claim -- which takes 2+ histidines. See README S3.")

    weak = [d for d in scored if d["n_interface_his"] <= 1]
    if weak:
        best_weak = max(d["S_pH"] for d in weak)
        print(f"\n  note: {len(weak)} design(s) have <=1 interface His; the best "
              f"manages only S={best_weak:.2f} ({10**best_weak:.0f}x).")
        print("  That is why --min-interface-his defaults to 2 outside --learn.")
    print()


# --------------------------------------------------------------------------- #
# STAGE 5 -- submission CSV
# --------------------------------------------------------------------------- #

def _seq_ok(seq: str) -> Tuple[bool, str]:
    if not (10 <= len(seq) <= 250):
        return False, f"length {len(seq)} outside 10-250"
    if set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
        return False, "non-standard residue"
    if "C" in seq:
        return False, "free cysteine"
    if re.search(r"(.)\1{4,}", seq):
        return False, "homopolymer run >4"
    if max(seq.count(a) for a in set(seq)) / len(seq) > 0.35:
        return False, "single residue >35% of sequence"
    return True, "ok"


def stage_submit(cfg: Config) -> None:
    scored = json.loads((WORK / "scored.json").read_text())
    if not scored:
        die("nothing to submit -- stage score produced no survivors")
    RESULTS.mkdir(parents=True, exist_ok=True)

    quota = {b: max(1, int(round(cfg.n_submit * w)))
             for b, w in cfg.bin_weights.items() if w > 0}
    # Fix rounding drift so the quotas sum to exactly n_submit. Only bins with
    # a non-zero weight take part, so a disabled bin stays empty.
    while sum(quota.values()) > cfg.n_submit:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < cfg.n_submit:
        quota[min(quota, key=quota.get)] += 1
    log(f"stage submit: quotas {quota} (total {sum(quota.values())})")

    chosen: List[dict] = []
    seen_seq: set = set()
    per_backbone: Dict[str, int] = {}
    per_patch: Dict[str, int] = {}
    rejected: Dict[str, str] = {}   # id -> reason, deduped across both passes

    def try_take(d: dict, bin_cap: Dict[str, int], bb_cap: int, patch_cap: int) -> bool:
        seq = d["sequence"]
        if seq in seen_seq:
            return False
        ok, why = _seq_ok(seq)
        if not ok:
            rejected[d["id"]] = why
            return False
        b = d.get("bin", "mini")
        patch = d.get("patch", "?")
        if bin_cap.get(b, 0) <= 0:
            return False
        if per_backbone.get(d["backbone"], 0) >= bb_cap:
            return False
        if per_patch.get(patch, 0) >= patch_cap:
            return False
        bin_cap[b] -= 1
        per_backbone[d["backbone"]] = per_backbone.get(d["backbone"], 0) + 1
        per_patch[patch] = per_patch.get(patch, 0) + 1
        seen_seq.add(seq)
        chosen.append(d)
        return True

    # Diversity-constrained greedy pass, then progressively relax.
    remaining = dict(quota)
    patch_cap = max(2, cfg.n_submit // max(1, cfg.n_patches) + 2)
    for d in scored:
        if len(chosen) >= cfg.n_submit:
            break
        try_take(d, remaining, bb_cap=2, patch_cap=patch_cap)

    if len(chosen) < cfg.n_submit:
        log(f"  only {len(chosen)}/{cfg.n_submit} after diversity pass; relaxing",
            "warn")
        loose = {b: cfg.n_submit for b in BINS}
        for d in scored:
            if len(chosen) >= cfg.n_submit:
                break
            try_take(d, loose, bb_cap=4, patch_cap=cfg.n_submit)

    if rejected:
        tally: Dict[str, int] = {}
        for why in rejected.values():
            tally[why] = tally.get(why, 0) + 1
        log(f"  {len(rejected)} designs rejected by sequence sanity checks: "
            f"{tally}", "warn")

    if not chosen:
        reasons = "\n".join(f"      {n:>5} x {w}" for w, n in sorted(
            ((w, sum(1 for r in rejected.values() if r == w))
             for w in set(rejected.values())), key=lambda t: -t[1]))
        die("no designs passed the sequence sanity checks, so there is nothing "
            "to submit.\n"
            f"    {len(scored)} design(s) reached this stage; all were rejected:\n"
            f"{reasons}\n"
            "    Most often this means ProteinMPNN collapsed to a low-complexity\n"
            "    sequence. Try raising --sampling temperature in stage mpnn, or\n"
            "    lowering --his-bias if histidine is dominating the composition.")

    # --- Write submission CSV (exact columns from the challenge spec) -------
    sub = RESULTS / "submission.csv"
    with open(sub, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence", "molecule_class"])
        for i, d in enumerate(chosen, 1):
            d["submit_name"] = f"{cfg.name_prefix}_{d.get('bin','mini')}_{i:03d}"
            w.writerow([d["submit_name"], d["sequence"], "protein"])

    # --- Full metrics for your own review ----------------------------------
    cols = ["submit_name", "id", "bin", "patch", "backbone", "pass",
            "binder_len", "S_pH", "pred_fold_selectivity", "n_interface_his",
            "total_his", "mouse_identity_at_interface", "mean_plddt",
            "min_plddt", "rmsd", "n_contacts", "n_contacted_residues",
            "hotspot_overlap", "mpnn_score", "composite", "sequence"]
    rep = RESULTS / "design_report.csv"
    with open(rep, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in chosen:
            w.writerow(d)

    (RESULTS / "design_details.json").write_text(json.dumps(chosen, indent=2, default=_json_default))

    bins_count: Dict[str, int] = {}
    for d in chosen:
        bins_count[d.get("bin", "?")] = bins_count.get(d.get("bin", "?"), 0) + 1
    mean_S = sum(d["S_pH"] for d in chosen) / len(chosen)
    mean_mouse = sum(d["mouse_identity_at_interface"] for d in chosen) / len(chosen)
    mean_plddt = sum(d["mean_plddt"] for d in chosen) / len(chosen)

    log(f"stage submit complete: {len(chosen)} designs -> {sub}", "ok")
    print(textwrap.dedent(f"""
        ------------------------------------------------------------------
         SUBMISSION SUMMARY
        ------------------------------------------------------------------
          designs              {len(chosen)}
          by bin               {bins_count}
          mean S (log10 Kd ratio)   {mean_S:.2f}   -> ~{10**mean_S:.0f}x weaker at pH 7.4
          mean interface mouse id   {mean_mouse:.1%}
          mean pLDDT                {mean_plddt:.1f}
          length range              {min(len(d['sequence']) for d in chosen)}-{max(len(d['sequence']) for d in chosen)} aa

          submission : {sub}
          metrics    : {rep}
        ------------------------------------------------------------------
    """))


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #

def selftest(cfg: Config) -> None:
    """CPU-only preflight: inputs, tools, imports, and the stage-0 logic."""
    log("=== SELFTEST ===")
    problems: List[str] = []

    log("checking inputs")
    for p in (TARGET_PDB_IN, HUMAN_FASTA, MOUSE_FASTA):
        if p.exists() and p.stat().st_size > 0:
            log(f"  OK   {p.name} ({p.stat().st_size/1024:.0f} KB)", "ok")
        else:
            problems.append(f"missing input: {p}")

    log("checking third-party tools")
    for p in (THIRD_PARTY / "RFdiffusion" / "scripts" / "run_inference.py",
              THIRD_PARTY / "RFdiffusion" / "models" / "Complex_base_ckpt.pt",
              THIRD_PARTY / "ProteinMPNN" / "protein_mpnn_run.py",
              THIRD_PARTY / "ProteinMPNN" / "helper_scripts" / "parse_multiple_chains.py"):
        if p.exists():
            log(f"  OK   {p.relative_to(ROOT) if ROOT in p.parents else p}", "ok")
        else:
            problems.append(f"missing tool: {p}")

    log("checking imports")
    for mod, why in (("numpy", "everywhere"), ("Bio", "PDB/alignment")):
        try:
            __import__(mod)
            log(f"  OK   {mod}", "ok")
        except ImportError:
            problems.append(f"cannot import {mod} ({why})")
    try:
        import torch
        log(f"  OK   torch {torch.__version__}", "ok")
        _check_gpu(require=False)
    except ImportError:
        problems.append("cannot import torch")

    try:
        __import__("transformers")
        log("  OK   transformers (this looks like the egfr-fold env)", "ok")
    except ImportError:
        log("  --   transformers absent (fine in egfr-rfdiff; stage fold "
            "runs in egfr-fold)", "warn")

    # Guard the failure that killed job 10073878: numpy scalars reaching
    # json.dumps. Checked before stage prepare so a broken encoder is reported
    # as a one-line encoder problem rather than as "stage prepare raised".
    try:
        import numpy as _np
        json.dumps({"i": _np.int64(24), "f": _np.float64(0.5),
                    "b": _np.bool_(True), "a": _np.array([1, 2]),
                    "s": {"A"}, "p": WORK},
                   default=_json_default)
        log("  OK   JSON encoder handles numpy int64/float64/bool_/ndarray", "ok")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"_json_default cannot encode numpy scalars: {exc!r}")

    log("exercising stage prepare (CPU)")
    try:
        stage_prepare(cfg)
        meta = json.loads((WORK / "target_meta.json").read_text())
        off = meta["numbering_offset_p00533_minus_pdb"]
        log(f"  OK   {len(meta['patches'])} patches, offset={off}, "
            f"contig={meta['contig_target'][:60]}", "ok")
        if off != SIGNAL_PEPTIDE_LEN:
            problems.append(
                f"numbering offset is {off}, expected {SIGNAL_PEPTIDE_LEN}; "
                "check the Domain III mapping before running diffuse")
        if not meta["patches"][0]["acidic_conserved"]:
            problems.append("top patch has no conserved acidic residues -- "
                            "pH switch design will be weak; inspect patches")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        problems.append(f"stage prepare raised: {exc!r}")

    print()
    if problems:
        log(f"SELFTEST FAILED ({len(problems)} problem(s)):", "err")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)
    log("SELFTEST PASSED -- safe to submit the GPU job", "ok")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="EGFR pH-selective binder design pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only preflight check; no GPU needed")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even if already completed")
    ap.add_argument("--learn", action="store_true",
                    help="LEARNING MODE: small, fast, permissive. ~30-60 min "
                         "end-to-end on one V100 instead of 8-12 h. Keeps "
                         "every intermediate and filters almost nothing, so "
                         "you can see what each stage actually produces. "
                         "Individual flags given after --learn still win.")
    ap.add_argument("--n-backbones", type=int, default=150)
    ap.add_argument("--seqs-per-backbone", type=int, default=8)
    ap.add_argument("--diffuser-T", type=int, default=50)
    ap.add_argument("--n-patches", type=int, default=4)
    ap.add_argument("--his-bias", type=float, default=2.5)
    ap.add_argument("--max-interface-his", type=int, default=4)
    ap.add_argument("--min-interface-his", type=int, default=2,
                    help="2 is a physical floor: one His caps out at 0.9 "
                         "log units of pH selectivity")
    ap.add_argument("--target-S", type=float, default=1.5,
                    help="desired log10(Kd(7.4)/Kd(6.5)); 1.5 ~= 30x")
    ap.add_argument("--min-plddt", type=float, default=75.0)
    ap.add_argument("--max-rmsd", type=float, default=2.5)
    ap.add_argument("--n-submit", type=int, default=40,
                    help="40 for Track 1; 20 for Tracks 2/3")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args(argv)

    # --- Learning mode ------------------------------------------------------
    # Small, fast, permissive. Anything the user typed explicitly still wins,
    # so `--learn --n-backbones 40` does what it looks like.
    raw = list(argv) if argv is not None else sys.argv[1:]
    given = {a.split("=")[0] for a in raw if a.startswith("--")}
    learn_bins = {"micro": 0.35, "mini": 0.65, "large": 0.0}
    if args.learn:
        preset = {
            "--n-backbones": ("n_backbones", 12),
            "--seqs-per-backbone": ("seqs_per_backbone", 4),
            "--diffuser-T": ("diffuser_T", 25),
            "--n-patches": ("n_patches", 2),
            # Relaxed so designs actually reach the report and you can see the
            # full distribution rather than an empty table.
            "--min-plddt": ("min_plddt", 60.0),
            "--max-rmsd": ("max_rmsd", 4.0),
            # 0 = do not filter on histidine count at all. This is the point:
            # you get to SEE that 1-His designs cap out near S=0.6 (4x).
            "--min-interface-his": ("min_interface_his", 0),
            "--n-submit": ("n_submit", 10),
        }
        applied, kept = [], []
        for flag, (dest, val) in preset.items():
            if flag in given:
                kept.append(f"{flag}={getattr(args, dest)}")
            else:
                setattr(args, dest, val)
                applied.append(f"{flag.lstrip('-')}={val}")
        log("LEARNING MODE: small/fast/permissive run", "warn")
        log(f"  preset: {', '.join(applied)}")
        log("  bins: large proteins skipped (slowest to diffuse)")
        if kept:
            log(f"  your explicit flags kept: {', '.join(kept)}")

    for d in (WORK, RESULTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        n_backbones=args.n_backbones,
        seqs_per_backbone=args.seqs_per_backbone,
        diffuser_T=args.diffuser_T,
        n_patches=args.n_patches,
        his_bias=args.his_bias,
        max_interface_his=args.max_interface_his,
        min_interface_his=args.min_interface_his,
        target_S=args.target_S,
        min_plddt=args.min_plddt,
        max_rmsd=args.max_rmsd,
        n_submit=args.n_submit,
        seed=args.seed,
        shard=args.shard,
        nshards=args.nshards,
        force=args.force,
    )
    if args.learn:
        cfg.bin_weights = learn_bins
        cfg.name_prefix = "egfr_learn"
    cfg.save(WORK / "config.json")

    if args.selftest:
        selftest(cfg)
        return

    fns = {
        "prepare": stage_prepare, "diffuse": stage_diffuse,
        "mpnn": stage_mpnn, "fold": stage_fold,
        "score": stage_score, "submit": stage_submit,
    }
    todo = STAGES if args.stage == "all" else [args.stage]

    for st in todo:
        m = marker(st)
        # Sharded diffuse must not be short-circuited by another shard's marker.
        if m.exists() and not cfg.force and not (st == "diffuse" and cfg.nshards > 1):
            log(f"skipping {st} (already done; --force to redo)")
            continue
        fns[st](cfg)
        m.write_text("")

    log("pipeline finished", "ok")


if __name__ == "__main__":
    main()

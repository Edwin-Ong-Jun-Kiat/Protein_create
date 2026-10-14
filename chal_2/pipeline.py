#!/usr/bin/env python3
"""
Challenge 02 -- TNF-alpha pH-selective binders. HPC pipeline (single V100).

    sbatch submit.sbatch                       the whole thing
    python pipeline.py --selftest              ~60 s CPU preflight, no GPU
    python pipeline.py --stage prepare         one stage
    sbatch --export=ALL,LEARN=1 submit.sbatch  small fast run (~1 h)

Six stages. Stage results checkpoint to work/.done_<stage>, so resubmitting
after a wall-clock kill resumes rather than restarts.

    prepare   CPU   1TNF trimer, TNFR2 footprint from 3ALQ, human/mouse
                    conservation, epitope grooves, cropped two-chain target
                    and the RFdiffusion contig              [rfdiff env]
    diffuse   GPU   RFdiffusion binder backbones, hotspot-conditioned on the
                    inter-protomer groove                   [rfdiff env]
    mpnn      GPU   ProteinMPNN with the REVERSE-polarity histidine bias
                                                            [rfdiff env]
    fold      GPU   ESMFold each binder alone: pLDDT + RMSD self-consistency
                                                            [fold env]
    score     CPU   histidine rotamers, pKa both states, S, mouse identity,
                    composite rank                          [rfdiff env]
    submit    CPU   sequence gates, quotas, write the CSV    [rfdiff env]

The physics is in src/tnf_hpc/linkage.py. Short version: each interface
histidine contributes at most 0.504 to S = log10(Kd(6.0)/Kd(7.4)) at a
free-state pKa of 6.4, so "no detectable binding at pH 6.0" needs four of them
-- or three if the free-state pKa is raised too. That is the opposite sign to
challenge 1 and a worse ceiling despite the wider pH window.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tnf_hpc import linkage as L          # noqa: E402
from tnf_hpc import target as T           # noqa: E402
from tnf_hpc import pka as PK             # noqa: E402
from tnf_hpc import design as DS          # noqa: E402
from tnf_hpc import tools as TL           # noqa: E402

INPUTS = ROOT / "inputs"
WORK = ROOT / "work"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

STAGES = ["prepare", "diffuse", "mpnn", "fold", "score", "submit"]

_C = {"info": "\033[1;36m[info]\033[0m", "ok": "\033[1;32m[  ok]\033[0m",
      "warn": "\033[1;33m[warn]\033[0m", "err": "\033[1;31m[ err]\033[0m"}
_PLAIN = {"info": "[info]", "ok": "[  ok]", "warn": "[warn]", "err": "[ err]"}


def log(msg, kind="info"):
    tags = _C if sys.stdout.isatty() else _PLAIN
    print(f"{tags[kind]} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# JSON encoding
# --------------------------------------------------------------------------- #
def _json_default(o):
    """Coerce the numpy scalars that leak into every metadata dict.

    Residue numbers come out of numpy arrays as ``np.int64``, SASA fractions as
    ``np.float64`` and conservation flags as ``np.bool_``. None is
    JSON-serialisable, and ``json.dumps`` raises

        TypeError: Object of type int64 is not JSON serializable

    Challenge 1's pipeline died exactly there in job 10073878 -- after a clean
    selftest and four correctly scored epitope patches, on the last line of
    stage prepare. The values print identically to Python ints in the log, so
    nothing upstream looks wrong and the run burns a GPU allocation to find out.

    Every write in this file goes through `save_json`, which always passes this.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable: {o!r}")


def save_json(obj, path):
    """The ONLY way this pipeline writes JSON. Never call json.dumps directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def load_json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    stage: str = "all"
    # target / epitope
    n_grooves: int = 2
    crop_radius: float = 18.0
    # diffusion
    n_backbones: int = 150
    binder_min: int = 55
    binder_max: int = 80
    diffuser_T: int = 50
    batch_size: int = 10
    # sequence design
    seqs_per_backbone: int = 8
    temperature: float = 0.2
    his_bias: float = 3.5
    acid_bias: float = 1.5
    # filters
    min_plddt: float = 75.0
    max_rmsd: float = 2.5
    target_S: float = 2.0
    min_independent_his: int = 2
    # submission
    n_submit: int = 40
    max_per_backbone: int = 3
    name_prefix: str = "tnf_ph"
    # plumbing
    seed: int = 20261007
    shard: int = 0
    nshards: int = 1
    force: bool = False
    learn: bool = False
    skip_esmfold: bool = False

    def apply_learn(self):
        """Small, fast, permissive. ~1 h instead of ~14 h.

        Exists for the same reason challenge 1's did: to find out that the
        plumbing works before committing a long GPU allocation. The filters are
        loosened too, because a 20-backbone run will not produce anything that
        passes production thresholds and a submission CSV with zero rows is a
        useless smoke test.
        """
        self.n_backbones = 20
        self.seqs_per_backbone = 4
        self.n_grooves = 1
        self.diffuser_T = 25
        self.min_plddt = 60.0
        self.max_rmsd = 4.0
        self.min_independent_his = 1
        self.n_submit = 10
        return self

    def save(self, path):
        save_json(asdict(self), path)


def done_marker(stage):
    return WORK / f".done_{stage}"


#: The config fields whose value changes a stage's OUTPUT. A checkpoint is only
#: reusable if every one of these still matches, which is what makes it safe to
#: resume a killed job while refusing to reuse results produced under different
#: settings.
STAGE_KEYS = {
    "prepare":  ("n_grooves", "crop_radius", "binder_min", "binder_max"),
    "diffuse":  ("n_backbones", "binder_min", "binder_max", "diffuser_T",
                 "n_grooves", "seed", "crop_radius"),
    "mpnn":     ("seqs_per_backbone", "temperature", "his_bias", "acid_bias",
                 "seed", "binder_min"),
    "fold":     ("skip_esmfold",),
    "score":    ("target_S", "min_plddt", "max_rmsd", "min_independent_his"),
    "submit":   ("n_submit", "max_per_backbone", "name_prefix"),
}


def stage_fingerprint(stage, cfg):
    """A stage's own settings PLUS every upstream stage's settings.

    Stages consume each other's files, so a stage's output depends on the whole
    chain in front of it, not only on its own flags. `fold`'s own settings are
    just `skip_esmfold`; without the upstream terms a `--learn` run followed by
    a full run would consider the fold checkpoint valid, skip folding, and leave
    `fold_metrics.json` describing 80 sequences while `sequences.json` holds
    1200. Designs with no entry get `plddt = None`, and the gate in
    `stage_score` only applies when pLDDT is present -- so they would pass
    without the single most valuable filter in the pipeline, silently.
    """
    own = {k: getattr(cfg, k) for k in STAGE_KEYS.get(stage, ())}
    idx = STAGES.index(stage) if stage in STAGES else 0
    for up in STAGES[:idx]:
        for k in STAGE_KEYS.get(up, ()):
            own.setdefault(k, getattr(cfg, k))
    return own


def already_done(stage, cfg):
    """Is there a reusable checkpoint for this stage?

    Not merely "does the marker exist". A `--learn` run marks all six stages
    done with 20 backbones and one groove; without this check a subsequent full
    run would skip every stage and present those results as the full run, with
    no error and no indication in the output. That is the worst kind of failure
    this pipeline could have -- a confidently wrong submission CSV -- so the
    marker records the settings that produced it and is rejected if they have
    changed.

    Resuming an identical run after a wall-clock kill still works, which is the
    case checkpoints exist for.
    """
    if cfg.force:
        return False
    marker = done_marker(stage)
    if not marker.exists():
        return False
    try:
        rec = load_json(marker)
    except Exception:
        return False
    want = stage_fingerprint(stage, cfg)
    got = rec.get("config", {})
    if got != want:
        diff = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        log(f"stage {stage}: checkpoint was made with different settings "
            f"{ {k: f'{a} -> {b}' for k, (a, b) in diff.items()} }; re-running",
            "warn")
        return False
    return True


def mark_done(stage, cfg, extra=None):
    save_json({"stage": stage,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "config": stage_fingerprint(stage, cfg),
               **(extra or {})}, done_marker(stage))


# --------------------------------------------------------------------------- #
# STAGE 0 -- prepare
# --------------------------------------------------------------------------- #
def stage_prepare(cfg):
    log("stage prepare: measuring the target")

    human = T.fetch_uniprot(T.UNIPROT_HUMAN, INPUTS)
    mouse = T.fetch_uniprot(T.UNIPROT_MOUSE, INPUTS)
    mature = T.mature_human(human)
    identity, mouse_mature = T.mouse_identity_map(mature, mouse)
    n_id = sum(v[0] for v in identity.values())
    log(f"  human mature {len(mature)} aa | mouse {len(mouse_mature)} aa")
    log(f"  human/mouse identity {n_id}/{len(identity)} "
        f"({n_id / len(identity):.1%}) -- brief says 124/156, 79%")

    model = T.load_model(T.fetch_pdb(T.PDB_TARGET, INPUTS),
                         keep_chains=T.TRIMER_CHAINS)
    offset, agree, nmap = T.numbering_offset(model["A"], mature)
    log(f"  1TNF chains {[c.id for c in model]}; numbering offset {offset} "
        f"(agreement {agree:.2f} over {nmap} residues)")
    if agree < 0.95:
        raise RuntimeError(f"numbering offset unreliable (agreement {agree:.2f})")

    log("  measuring the TNFR2 footprint from 3ALQ")
    fp = T.receptor_footprint(T.fetch_pdb(T.PDB_TNFR2, INPUTS),
                              list("ABCDEF"), list("RSTUVW"))
    grooves_measured = T.consensus_epitope(fp)
    freq = T.receptor_frequency(grooves_measured)
    for g in grooves_measured:
        log(f"    receptor {g['receptor']}: {g['primary']}({g['n_primary']}) "
            f"+ {g['secondary']}({g['n_secondary']})")
    strong = sorted(p for p, c in freq.items() if c >= len(grooves_measured) - 1)
    log(f"  consensus epitope: {len(freq)} positions, {len(strong)} seen in "
        f">= {len(grooves_measured) - 1} of {len(grooves_measured)} copies", "ok")

    rsa = T.relative_sasa_trimer(model)
    disc = T.sasa_discrepancy(model, "A")
    log(f"  SASA computed in TRIMER context: {len(disc)} residues would be "
        f"wrongly called exposed if B/C were stripped")

    rows = T.build_residue_table(model, identity, rsa,
                                 T.residue_centroids(model), freq, offset=offset)
    patches = T.score_patches(rows, freq)
    chosen = T.assign_hotspots(
        T.select_nonredundant(patches, n=cfg.n_grooves), max_hotspots=5)
    log(f"  {len(patches)} candidate patches -> {len(chosen)} grooves")

    meta_grooves = []
    for p in chosen:
        pair = p["receptor_site_chains"][:2]
        if len(pair) < 2:
            log(f"    {p['id']} does not span two protomers; skipping", "warn")
            continue
        hotspots = [tuple(h) for h in p["hotspots"]]

        keep = TL.crop_target(model, pair, hotspots, offset=offset,
                              radius=cfg.crop_radius)
        tgt_pdb = WORK / f"target_{p['id']}.pdb"
        mapping = TL.write_cropped_pdb(model, keep, tgt_pdb)
        contig = TL.build_contig(keep, cfg.binder_min, cfg.binder_max)
        hs_str, missing = TL.hotspot_string(hotspots, mapping)
        if missing:
            log(f"    {p['id']}: hotspots outside the crop: {missing}", "warn")

        n_kept = {c: len(v) for c, v in keep.items()}
        log(f"    {p['id']} score={p['score']:.2f} chains={pair} "
            f"anchors={p['switch_anchors']}", "ok")
        log(f"      cropped target {n_kept} = {sum(n_kept.values())} residues "
            f"(from {sum(1 for c in model if c.id in set(pair) for _ in c)})")
        log(f"      contig  {contig}")
        log(f"      hotspot {hs_str}")

        meta_grooves.append({
            **{k: v for k, v in p.items() if k != "centre_xyz"},
            "target_chains": pair,
            "target_pdb": str(tgt_pdb.relative_to(ROOT)),
            "crop_keep": keep,
            "crop_map": {f"{c}{n}": list(v) for (c, n), v in mapping.items()},
            "contig": contig,
            "hotspot_str": hs_str,
            "n_target_residues": sum(n_kept.values()),
        })

    if not meta_grooves:
        raise RuntimeError("no groove spans two protomers; cannot proceed")

    meta = {
        "target_pdb": T.PDB_TARGET,
        "footprint_pdb": T.PDB_TNFR2,
        "numbering_offset": offset,
        "numbering_agreement": agree,
        "human_mature_len": len(mature),
        "mouse_mature_len": len(mouse_mature),
        "mouse_identity_overall": round(n_id / len(identity), 4),
        "receptor_frequency": {str(k): v for k, v in sorted(freq.items())},
        "consensus_epitope_strong": strong,
        "sasa_trap_residues": disc[:25],
        "identity": {str(k): [bool(v[0]), v[1]] for k, v in identity.items()},
        "grooves": meta_grooves,
        "binder_length_range": [cfg.binder_min, cfg.binder_max],
    }
    save_json(meta, WORK / "target_meta.json")
    log(f"stage prepare complete -> {WORK / 'target_meta.json'}", "ok")
    return meta


# --------------------------------------------------------------------------- #
# STAGE 1 -- diffuse
# --------------------------------------------------------------------------- #
def stage_diffuse(cfg):
    log("stage diffuse: RFdiffusion binder backbones")
    meta = load_json(WORK / "target_meta.json")
    py = TL.env_path("TNF_RFDIFF_PY", must_exist=True)
    tools = TL.tool_paths()
    for k in ("rfdiffusion_script", "rfdiffusion_ckpt"):
        if not tools[k].exists():
            raise FileNotFoundError(f"{k} not found at {tools[k]}")

    bb_dir = WORK / "backbones"
    bb_dir.mkdir(parents=True, exist_ok=True)

    per_groove = max(1, cfg.n_backbones // len(meta["grooves"]))
    log(f"  {cfg.n_backbones} backbones over {len(meta['grooves'])} grooves "
        f"= {per_groove} each, in batches of {cfg.batch_size}")
    if cfg.nshards > 1:
        log(f"  shard {cfg.shard} of {cfg.nshards}")

    t0 = time.time()
    n_made = 0
    for g in meta["grooves"]:
        gid = g["id"]
        batches = [(i, min(cfg.batch_size, per_groove - i))
                   for i in range(0, per_groove, cfg.batch_size)]
        for bi, (start, count) in enumerate(batches):
            if bi % cfg.nshards != cfg.shard:
                continue
            prefix = bb_dir / f"{gid}_b{bi:03d}"
            expect = [Path(f"{prefix}_{j}.pdb") for j in range(count)]
            if all(p.exists() for p in expect) and not cfg.force:
                log(f"  {gid} batch {bi}: already present, skipping")
                n_made += count
                continue
            log(f"  {gid} batch {bi}: {count} designs "
                f"({time.time() - t0:.0f}s elapsed)")
            TL.run_rfdiffusion(
                py=py,
                out_prefix=str(prefix),
                target_pdb=ROOT / g["target_pdb"],
                contig=g["contig"],
                hotspots=g["hotspot_str"],
                n_designs=count,
                ckpt=tools["rfdiffusion_ckpt"],
                diffuser_T=cfg.diffuser_T,
                log=lambda m: log(m),
            )
            n_made += sum(1 for p in expect if p.exists())

    made = sorted(bb_dir.glob("*.pdb"))
    log(f"stage diffuse complete: {len(made)} backbone PDBs in "
        f"{time.time() - t0:.0f}s", "ok")
    if not made:
        raise RuntimeError("RFdiffusion produced no PDBs")
    return made


# --------------------------------------------------------------------------- #
# STAGE 2 -- mpnn
# --------------------------------------------------------------------------- #
def stage_mpnn(cfg):
    log("stage mpnn: ProteinMPNN with the reverse-polarity histidine bias")
    meta = load_json(WORK / "target_meta.json")
    identity = {int(k): (v[0], v[1]) for k, v in meta["identity"].items()}
    py = TL.env_path("TNF_RFDIFF_PY", must_exist=True)
    offset = meta["numbering_offset"]

    full = T.load_model(T.fetch_pdb(T.PDB_TARGET, INPUTS),
                        keep_chains=T.TRIMER_CHAINS)

    mpnn_root = WORK / "mpnn"
    mpnn_root.mkdir(parents=True, exist_ok=True)
    backbones = sorted((WORK / "backbones").glob("*.pdb"))
    log(f"  {len(backbones)} backbones to design")

    records = []
    bias_info_path = mpnn_root / "bias_by_res.jsonl"
    n_anchor_hist = {}

    with bias_info_path.open("w") as bias_log:
        for gi, g in enumerate(meta["grooves"]):
            gid = g["id"]
            pair = g["target_chains"]
            group = [b for b in backbones if b.name.startswith(gid)]
            if not group:
                continue
            # the conserved cation / carboxylate clouds this groove offers
            site_src = PK.target_atom_arrays(full, pair)
            cations, acids = _conserved_charge_clouds(full, pair, identity, offset)
            target_cb = np.array([r["CB"].coord if "CB" in r else r["CA"].coord
                                  for ch in full if ch.id in set(pair)
                                  for r in ch])
            log(f"  {gid}: {len(group)} backbones, {len(cations)} conserved "
                f"cation atoms, {len(acids)} conserved carboxylate atoms")

            stage_dir = mpnn_root / gid
            pdb_dir = stage_dir / "pdbs"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            bias_by_res = {}
            binder_chains = set()

            from tnf_hpc.dock import pseudo_cb
            for bb in group:
                binder_chain = TL.binder_chain_of(bb, pair)
                coords, chains, nums = TL.read_backbone(bb, chain_id=binder_chain)
                if len(coords) < cfg.binder_min - 5:
                    log(f"    {bb.name}: binder chain {binder_chain} has only "
                        f"{len(coords)} residues; skipping", "warn")
                    continue
                cb = pseudo_cb(coords)
                bias, info = TL.mpnn_bias_by_res(
                    cb, binder_chain, cations, acids, target_cb,
                    his_bias=cfg.his_bias, acid_bias=cfg.acid_bias)
                shutil.copy(bb, pdb_dir / bb.name)
                # keyed by the pdb NAME ProteinMPNN will use, which is the file
                # stem -- tied_featurize does bias_by_res_dict[name][chain]
                # and raises KeyError if either level is missing.
                bias_by_res[bb.stem] = bias
                binder_chains.add(binder_chain)
                n_full = sum(1 for q in info if q["his_bias"] >= cfg.his_bias)
                n_anchor_hist[n_full] = n_anchor_hist.get(n_full, 0) + 1
                bias_log.write(json.dumps(
                    {"backbone": bb.stem, "groove": gid,
                     "binder_chain": binder_chain, "positions": info},
                    default=_json_default) + "\n")
                records.append({"backbone": bb.stem, "groove_id": gid,
                                "target_chains": pair,
                                "binder_chain": binder_chain,
                                "length": int(len(coords)),
                                "n_full_bias_positions": n_full,
                                "n_acid_paired": sum(1 for q in info
                                                     if q["acid_bias"] > 0)})

            if not bias_by_res:
                log(f"  {gid}: no usable backbone, skipping", "warn")
                continue
            if len(binder_chains) != 1:
                # RFdiffusion labels the binder by contig order, so this should
                # be constant within a groove. If it is not, one bias file
                # cannot describe the batch and the designed-chain argument
                # below would be wrong for some of them.
                raise RuntimeError(
                    f"{gid}: RFdiffusion used more than one binder chain id "
                    f"{sorted(binder_chains)}; cannot design them in one batch")
            binder_chain = binder_chains.pop()

            # ONE JSON object on ONE line: {name: {chain: [[21]*len]}}.
            # protein_mpnn_run.py reads `list(json_file)` and keeps the LAST
            # line, so one dict per line would silently design only the final
            # backbone with a bias and leave every other one unbiased.
            bias_file = stage_dir / "bias_by_res.json"
            bias_file.write_text(
                json.dumps(bias_by_res, default=_json_default) + "\n")

            seq_dir = TL.run_proteinmpnn(
                py=py, pdb_dir=pdb_dir, out_dir=stage_dir,
                designed_chain=binder_chain,
                fixed_chains=pair, n_seq=cfg.seqs_per_backbone,
                temperature=cfg.temperature, bias_jsonl=bias_file,
                seed=cfg.seed + gi, log=lambda m: log(m))
            log(f"  {gid}: binder chain {binder_chain}, "
                f"{len(bias_by_res)} backbones -> {seq_dir}", "ok")

    log("  interface anchor positions per backbone (positions at full His bias):")
    for k in sorted(n_anchor_hist):
        log(f"    {k} anchors: {n_anchor_hist[k]} backbones"
            + ("   <- cannot reach S=2 (needs >=3)" if k < 3 else ""))

    # collect sequences
    out = []
    for g in meta["grooves"]:
        seq_dir = mpnn_root / g["id"] / "seqs"
        if not seq_dir.exists():
            continue
        by_bb = {r["backbone"]: r for r in records if r["groove_id"] == g["id"]}
        for fa in sorted(seq_dir.glob("*.fa")):
            base = fa.stem
            meta_bb = by_bb.get(base)
            if meta_bb is None:
                continue
            for rec in TL.parse_mpnn_fasta(fa):
                out.append({**meta_bb, **rec})
    save_json(out, WORK / "sequences.json")
    log(f"stage mpnn complete: {len(out)} sequences -> work/sequences.json", "ok")
    return out


def _conserved_charge_clouds(model, chain_ids, identity, offset):
    """Conserved cation nitrogens and carboxylate oxygens of the target.

    Conserved means identical in mouse. A non-conserved cation is useless here:
    suppressing a histidine's pKa against mouse-divergent Arg31 would satisfy
    criterion 1 and forfeit criterion 2, which the challenge ranks second.
    """
    cat, acid = [], []
    for ch in model:
        if ch.id not in set(chain_ids):
            continue
        for res in ch:
            rn = res.get_resname().upper()
            same, _ = identity.get(res.id[1] - offset, (None, None))
            if not same:
                continue
            for n in PK.CATION_ATOMS.get(rn, ()):
                if n in res:
                    cat.append(np.asarray(res[n].coord, float))
            for n in PK.CARBOXYL_O.get(rn, ()):
                if n in res:
                    acid.append(np.asarray(res[n].coord, float))
    f = lambda v: np.array(v, float) if v else np.zeros((0, 3))
    return f(cat), f(acid)


# --------------------------------------------------------------------------- #
# STAGE 3 -- fold
# --------------------------------------------------------------------------- #
def stage_fold(cfg):
    log("stage fold: ESMFold self-consistency")
    seqs = load_json(WORK / "sequences.json")
    out_json = WORK / "fold_metrics.json"

    if cfg.skip_esmfold:
        log("  --skip-esmfold: writing neutral metrics", "warn")
        save_json({}, out_json)
        return {}

    py = TL.env_path("TNF_FOLD_PY", must_exist=True)
    hf = TL.store_root() / "hf_cache"
    uniq = {}
    for d in seqs:
        uniq.setdefault(d["sequence"], f"{d['backbone']}_s{d.get('sample', 0)}")
    folded_names = {v: k for k, v in uniq.items()}
    log(f"  folding {len(uniq)} unique sequences of {len(seqs)} designs "
        f"(ESMFold v1, fp32 -- Volta has no bf16)")

    res = TL.run_esmfold(py=py, sequences=folded_names, out_json=out_json,
                         work_dir=WORK / "esmfold", hf_cache=hf,
                         log=lambda m: log(m))

    ok = [v for v in res.values() if "error" not in v]
    if ok:
        pl = np.array([v["mean_plddt"] for v in ok])
        log(f"  mean pLDDT: median {np.median(pl):.1f}, "
            f"range {pl.min():.1f}-{pl.max():.1f}", "ok")
        log(f"  {int((pl >= cfg.min_plddt).sum())}/{len(pl)} at or above "
            f"--min-plddt {cfg.min_plddt}")
    n_err = sum(1 for v in res.values() if "error" in v)
    if n_err:
        log(f"  {n_err} sequences failed to fold", "warn")
    log("stage fold complete -> work/fold_metrics.json", "ok")
    return res


# --------------------------------------------------------------------------- #
# STAGE 4 -- score
# --------------------------------------------------------------------------- #
def stage_score(cfg):
    log("stage score: histidine pKa, selectivity, cross-reactivity, rank")
    meta = load_json(WORK / "target_meta.json")
    identity = {int(k): (v[0], v[1]) for k, v in meta["identity"].items()}
    offset = meta["numbering_offset"]
    seqs = load_json(WORK / "sequences.json")
    fold = (load_json(WORK / "fold_metrics.json")
            if (WORK / "fold_metrics.json").exists() else {})

    # The pKa environment is measured against the FULL trimer in its original
    # coordinates, not the cropped target. RFdiffusion holds the target fixed,
    # and the crop only dropped residues and renumbered -- it never moved an
    # atom -- so the binder coordinates are already in 1TNF's frame. Using the
    # full trimer means a histidine near the third protomer is scored correctly
    # and every side chain is present.
    full = T.load_model(T.fetch_pdb(T.PDB_TARGET, INPUTS),
                        keep_chains=T.TRIMER_CHAINS)
    tgt_atoms = PK.target_atom_arrays(full, T.TRIMER_CHAINS)

    by_name = {}
    for d in seqs:
        by_name.setdefault(d["backbone"], []).append(d)

    scored = []
    for bi, (bb_name, group) in enumerate(sorted(by_name.items())):
        bb = WORK / "backbones" / f"{bb_name}.pdb"
        if not bb.exists():
            continue
        pair = group[0]["target_chains"]
        binder_chain = group[0]["binder_chain"]
        coords, _, _ = TL.read_backbone(bb, chain_id=binder_chain)
        if len(coords) == 0:
            continue

        mouse = _mouse_from_coords(coords, full, identity, offset)

        for d in group:
            seq = d["sequence"]
            if len(seq) != len(coords):
                # ProteinMPNN designs exactly the designable chain, so a
                # mismatch means the backbone/chain pairing is wrong rather
                # than that the sequence is short. Skip loudly, do not pad.
                continue
            ph = PK.score_design(coords, seq, tgt_atoms, use_free_lever=True)

            key = f"{bb_name}_s{d.get('sample', 0)}"
            fm = fold.get(key, {})
            plddt = fm.get("mean_plddt")
            rmsd = None
            if "ca" in fm:
                rmsd = TL.kabsch_rmsd(np.array(fm["ca"]), coords[:, 1])

            comp, terms = DS.composite_score(
                ph["S"], mouse["mouse_identity"], ph["n_independent_his"],
                esm_pll=0.0,
                mpnn_nll=d.get("mpnn_score") or 0.0,
                n_contact_pairs=mouse["n_contact_pairs"],
                hotspot_cover=mouse["hotspot_cover"],
                total_his=seq.count("H"),
                acid_face=0,
                rmsd=rmsd or 0.0,
                mean_plddt=plddt,
                target_S=cfg.target_S, breakdown=True)

            ok, reason = DS.seq_ok(seq)
            if ok and plddt is not None and plddt < cfg.min_plddt:
                ok, reason = False, f"pLDDT {plddt:.1f} < {cfg.min_plddt}"
            if ok and rmsd is not None and rmsd > cfg.max_rmsd:
                ok, reason = False, f"RMSD {rmsd:.2f} > {cfg.max_rmsd}"
            if ok and ph["n_independent_his"] < cfg.min_independent_his:
                ok, reason = False, (f"{ph['n_independent_his']} independent His "
                                     f"< {cfg.min_independent_his}")

            scored.append({
                **d,
                "S": ph["S"], "kd_fold": ph["kd_fold"],
                "n_interface_his": ph["n_interface_his"],
                "n_independent_his": ph["n_independent_his"],
                "his": ph["his_used"], "his_dropped": ph["his_dropped"],
                "meets_S_target": ph["S"] >= cfg.target_S,
                "mouse_identity": mouse["mouse_identity"],
                "divergent_contacts": mouse["divergent_contacts"],
                "n_contacted_target": mouse["n_contacted"],
                "n_contact_pairs": mouse["n_contact_pairs"],
                "hotspot_cover": mouse["hotspot_cover"],
                "mean_plddt": plddt, "rmsd": rmsd,
                "composite": float(comp),
                "composite_terms": {k: float(v) for k, v in terms.items()},
                "seq_ok": ok, "seq_reason": reason,
                "size_category": DS.size_category(seq),
            })
        if (bi + 1) % 25 == 0:
            log(f"  {bi + 1}/{len(by_name)} backbones scored")

    scored.sort(key=lambda d: -d["composite"])
    save_json(scored, WORK / "scored.json")

    passing = [d for d in scored if d["seq_ok"]]
    log(f"  {len(passing)}/{len(scored)} pass every gate", "ok")
    log(f"  {sum(d['meets_S_target'] for d in passing)} reach "
        f"S >= {cfg.target_S} ({10 ** cfg.target_S:.0f}-fold)")
    if passing:
        b = passing[0]
        pl = "n/a" if b["mean_plddt"] is None else f"{b['mean_plddt']:.1f}"
        log(f"  best composite {b['composite']:.2f}: S={b['S']:.2f} "
            f"({b['kd_fold']:.0f}x) His={b['n_independent_his']} "
            f"mouse={b['mouse_identity']:.2f} pLDDT={pl} len={b['length']}")
    print()
    print(DS.score_breakdown(passing or scored))
    log("stage score complete -> work/scored.json", "ok")
    return scored


def _mouse_from_coords(coords, full_model, identity, offset, cutoff=5.0):
    """Contacted target residues, their mouse conservation, and hotspot cover.

    Backbone + pseudo-CB on the binder side, because an RFdiffusion backbone has
    no side chains. That makes the contact list slightly conservative: it misses
    contacts only a long side chain would make.
    """
    from tnf_hpc.dock import pseudo_cb

    b = np.vstack([coords.reshape(-1, 3), pseudo_cb(coords)])
    contacted, pairs = [], 0
    for ch in full_model:
        for res in ch:
            t = np.array([a.coord for a in res if a.element != "H"], float)
            if t.size == 0:
                continue
            d = np.linalg.norm(t[:, None, :] - b[None, :, :], axis=-1)
            if d.min() <= cutoff:
                contacted.append(res.id[1] - offset)
                pairs += int((d <= cutoff).sum())
    known = [identity[p][0] for p in contacted if p in identity]
    return {
        "n_contacted": len(contacted),
        "n_contact_pairs": pairs,
        "hotspot_cover": len({p for p in contacted}),
        "mouse_identity": round((sum(known) / len(known)) if known else 0.0, 4),
        "divergent_contacts": sorted({p for p in contacted
                                      if p in identity and not identity[p][0]}),
    }


# --------------------------------------------------------------------------- #
# STAGE 5 -- submit
# --------------------------------------------------------------------------- #
def stage_submit(cfg):
    log("stage submit: gates, quotas, CSV")
    scored = load_json(WORK / "scored.json")
    passing = [d for d in scored if d["seq_ok"]]

    for d in passing:
        d["scaffold_id"] = d["backbone"]
    kept, rejected = DS.quota_filter(passing, max_total=cfg.n_submit,
                                     max_per_scaffold=cfg.max_per_backbone)
    log(f"  {len(kept)} kept of {len(passing)} passing "
        f"({len(scored) - len(passing)} failed a gate)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    sub = RESULTS / "submission.csv"
    with sub.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence", "molecule_class"])
        for i, d in enumerate(kept):
            w.writerow([f"{cfg.name_prefix}_{i:03d}", d["sequence"],
                        DS.molecule_class(d["sequence"])])

    rep = RESULTS / "design_report.csv"
    cols = ["name", "sequence", "length", "size_category", "molecule_class",
            "composite", "S", "kd_fold", "meets_S_target", "n_interface_his",
            "n_independent_his", "n_his_total", "mouse_identity",
            "n_contacted_target", "divergent_contacts", "mean_plddt", "rmsd",
            "mpnn_score", "groove_id", "backbone", "n_full_bias_positions",
            "n_acid_paired", "his_positions"]
    with rep.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, d in enumerate(kept):
            w.writerow([
                f"{cfg.name_prefix}_{i:03d}", d["sequence"], d["length"],
                d["size_category"], DS.molecule_class(d["sequence"]),
                round(d["composite"], 3), d["S"], d["kd_fold"],
                d["meets_S_target"], d["n_interface_his"],
                d["n_independent_his"], d["sequence"].count("H"),
                d["mouse_identity"], d["n_contacted_target"],
                ";".join(map(str, d["divergent_contacts"])),
                None if d["mean_plddt"] is None else round(d["mean_plddt"], 2),
                None if d["rmsd"] is None else round(d["rmsd"], 3),
                None if d.get("mpnn_score") is None else round(d["mpnn_score"], 4),
                d["groove_id"], d["backbone"], d["n_full_bias_positions"],
                d["n_acid_paired"],
                ";".join(str(h["binder_pos"]) for h in d["his"]),
            ])
    save_json(kept, RESULTS / "design_details.json")

    cats = {}
    for d in kept:
        cats[d["size_category"]] = cats.get(d["size_category"], 0) + 1
    log(f"  size categories: {cats}")

    # The challenge accepts methodology documentation alongside the CSV, and it
    # is generated from these artifacts rather than maintained by hand so the
    # figures in it cannot drift away from the designs it describes.
    try:
        import subprocess as _sp
        _sp.run([sys.executable, str(ROOT / "make_methods.py"),
                 "--root", str(ROOT)], check=True, capture_output=True, text=True)
        log(f"  wrote {(RESULTS / 'METHODS.md').relative_to(ROOT)} "
            f"(methodology, for submission)", "ok")
    except Exception as exc:  # noqa: BLE001
        log(f"  could not generate METHODS.md ({type(exc).__name__}); "
            f"run 'python make_methods.py' by hand", "warn")
    log(f"stage submit complete -> {sub.relative_to(ROOT)}", "ok")
    if kept:
        print()
        print(f"    {'name':<18}{'len':>4}{'S':>7}{'fold':>8}{'His':>5}"
              f"{'mouse':>7}{'pLDDT':>7}{'RMSD':>7}")
        for i, d in enumerate(kept[:8]):
            pl = "  n/a" if d["mean_plddt"] is None else f"{d['mean_plddt']:5.1f}"
            rm = "  n/a" if d["rmsd"] is None else f"{d['rmsd']:5.2f}"
            name = f"{cfg.name_prefix}_{i:03d}"
            print(f"    {name:<18}{d['length']:>4}"
                  f"{d['S']:>7.2f}{d['kd_fold']:>6.0f}x"
                  f"{d['n_independent_his']:>5}{d['mouse_identity']:>7.2f}"
                  f"{pl:>7}{rm:>7}")
    return kept


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest(cfg):
    log("=== SELFTEST ===")
    problems = []

    log("checking inputs")
    for name in ("1TNF.pdb", "3ALQ.pdb", "P01375.fasta", "P06804.fasta"):
        p = INPUTS / name
        if p.exists():
            log(f"  OK   {name} ({p.stat().st_size // 1024} KB)", "ok")
        else:
            log(f"  --   {name} absent; will be downloaded on first use", "warn")

    log("checking third-party tools (shared with challenge 1)")
    tools = TL.tool_paths()
    for k in ("rfdiffusion_script", "rfdiffusion_ckpt", "proteinmpnn_script",
              "mpnn_parse", "mpnn_assign"):
        if tools[k].exists():
            log(f"  OK   {tools[k]}", "ok")
        else:
            problems.append(f"missing tool: {tools[k]}")

    log("checking interpreters")
    for var in ("TNF_RFDIFF_PY", "TNF_FOLD_PY"):
        p = TL.env_path(var)
        if p and p.exists() and os.access(p, os.X_OK):
            log(f"  OK   {var} = {p}", "ok")
        else:
            problems.append(f"{var} is not an executable interpreter ({p})")

    log("checking imports")
    for mod in ("numpy", "scipy", "Bio"):
        try:
            __import__(mod)
            log(f"  OK   {mod}", "ok")
        except ImportError as e:
            problems.append(f"cannot import {mod} ({e})")
    try:
        import torch
        log(f"  OK   torch {torch.__version__}", "ok")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            log(f"  GPU: {name} (sm_{cap[0]}{cap[1]})")
            if cap[0] < 8:
                log("  Volta/Turing: no bfloat16, running fp32 throughout", "warn")
        else:
            log("  no CUDA visible (expected on a login node)", "warn")
    except ImportError:
        log("  --   torch absent (fine on a login node)", "warn")

    # The failure that killed challenge 1's job 10073878.
    log("checking the JSON encoder")
    try:
        json.dumps({"i": np.int64(24), "f": np.float64(0.5), "b": np.bool_(True),
                    "a": np.array([1, 2]), "s": {"A"}, "p": WORK},
                   default=_json_default)
        log("  OK   numpy int64/float64/bool_/ndarray, set and Path all encode",
            "ok")
    except Exception as e:  # noqa: BLE001
        problems.append(f"_json_default cannot encode numpy scalars: {e!r}")

    log("checking the physics")
    c = L.ceiling_one_group()
    if abs(c - 0.5041) > 1e-3:
        problems.append(f"per-His ceiling is {c:.4f}, expected 0.5041")
    else:
        log(f"  OK   ceiling {c:.4f}/His -> {L.n_his_required()} histidines "
            f"needed for S={L.S_TARGET}", "ok")
    s1, s2 = float(L.selectivity([4.0])), L.selectivity_as_integral([4.0])
    if abs(s1 - s2) > 1e-6:
        problems.append("closed-form S disagrees with the titration integral")
    else:
        log(f"  OK   closed form matches the integral ({abs(s1 - s2):.1e})", "ok")
    bare = float(L.selectivity([6.4 - 0.7 - 1.0], [6.4]))
    both = float(L.selectivity([7.9 - 0.7 - 1.0], [7.9]))
    acid_only = float(L.selectivity([7.9 - 0.7], [7.9]))
    if not (both > bare > acid_only):
        problems.append("the two pH levers no longer order correctly")
    else:
        log(f"  OK   levers: bare {bare:.3f} < both {both:.3f}; "
            f"free-lever-alone {acid_only:.3f} is worse, as expected", "ok")

    log("checking the sequence gates")
    good = "TEWEELVRRALEAARRGGHSGEGLEAAREYVARARELAPGERAEAFARLVGEALAA"
    for seq, want in (("A" * 5, False), (good.replace("GGH", "GGC"), False),
                      ("E" * 30, False), (good, True)):
        if DS.seq_ok(seq)[0] is not want:
            problems.append(f"seq_ok misjudged {seq[:16]}...")
    log("  OK   rejects short / Cys / polyanion, accepts a real design", "ok")

    log("exercising stage prepare (CPU)")
    try:
        meta = stage_prepare(cfg)
        n = sum(1 for v in meta["identity"].values() if v[0])
        if (n, len(meta["identity"])) != (124, 156):
            problems.append(f"human/mouse identity is {n}/{len(meta['identity'])}, "
                            "expected 124/156 -- check the alignment")
        else:
            log(f"  OK   human/mouse identity {n}/156 matches the brief", "ok")
        if meta["numbering_offset"] != 0:
            problems.append(f"1TNF numbering offset is "
                            f"{meta['numbering_offset']}, expected 0")
        for g in meta["grooves"]:
            if len(g["target_chains"]) < 2:
                problems.append(f"{g['id']} does not span two protomers")
            if not g["switchable"]:
                problems.append(f"{g['id']} has no conserved cation anchor; "
                                "no pH switch is possible there")
            if g["n_target_residues"] > 200:
                problems.append(f"{g['id']} cropped target is "
                                f"{g['n_target_residues']} residues -- "
                                "RFdiffusion may OOM; lower --crop-radius")

            # The crop renumbers every target residue to 1..N per chain, so the
            # hotspot string handed to RFdiffusion is in different coordinates
            # from the hotspots that were chosen. An off-by-one here aims the
            # whole diffusion stage at the wrong residues and nothing downstream
            # notices -- the designs look fine and bind the wrong face. So the
            # emitted string is traced back through the mapping and must land on
            # the hotspots that were actually selected.
            cmap = {k: tuple(v) for k, v in g["crop_map"].items()}
            want = {tuple(h) for h in g["hotspots"]}
            emitted = [x for x in g["hotspot_str"].strip("[]").split(",") if x]
            bad = [e for e in emitted if cmap.get(e) not in want]
            if bad:
                problems.append(f"{g['id']} hotspot string {bad} does not map "
                                f"back to the selected hotspots {sorted(want)}")
            elif emitted:
                shown = ", ".join(f"{e}={cmap[e][0]}{cmap[e][1]}"
                                  for e in emitted[:3])
                log(f"  OK   {g['id']} hotspots renumber correctly "
                    f"({shown}, ...)", "ok")
            # and each contig span must equal the number of residues kept
            import re as _re
            for cid, nums in g["crop_keep"].items():
                m = _re.search(rf"{cid}1-(\d+)", g["contig"])
                if not m or int(m.group(1)) != len(nums):
                    problems.append(
                        f"{g['id']} contig span for chain {cid} disagrees with "
                        f"the {len(nums)} residues actually written")
        log(f"  OK   {len(meta['grooves'])} grooves, contig "
            f"{meta['grooves'][0]['contig']}", "ok")
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        problems.append(f"stage prepare raised: {exc!r}")

    print()
    if problems:
        log(f"SELFTEST FAILED ({len(problems)} problem(s)):", "err")
        for p in problems:
            print(f"    - {p}")
        return 1
    log("SELFTEST PASSED -- safe to run the GPU stages", "ok")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--learn", action="store_true",
                    help="small, fast, permissive run (~1 h)")
    ap.add_argument("--force", action="store_true",
                    help="ignore work/.done_* checkpoints")

    ap.add_argument("--n-grooves", type=int, default=2)
    ap.add_argument("--crop-radius", type=float, default=18.0)
    ap.add_argument("--n-backbones", type=int, default=150)
    ap.add_argument("--binder-min", type=int, default=55)
    ap.add_argument("--binder-max", type=int, default=80)
    ap.add_argument("--diffuser-T", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seqs-per-backbone", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--his-bias", type=float, default=3.5)
    ap.add_argument("--acid-bias", type=float, default=1.5)
    ap.add_argument("--min-plddt", type=float, default=75.0)
    ap.add_argument("--max-rmsd", type=float, default=2.5)
    ap.add_argument("--target-S", type=float, default=2.0)
    ap.add_argument("--min-independent-his", type=int, default=2)
    ap.add_argument("--n-submit", type=int, default=40)
    ap.add_argument("--max-per-backbone", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--skip-esmfold", action="store_true")

    a = ap.parse_args(argv)
    cfg = Config(
        stage=a.stage, n_grooves=a.n_grooves, crop_radius=a.crop_radius,
        n_backbones=a.n_backbones, binder_min=a.binder_min,
        binder_max=a.binder_max, diffuser_T=a.diffuser_T,
        batch_size=a.batch_size, seqs_per_backbone=a.seqs_per_backbone,
        temperature=a.temperature, his_bias=a.his_bias, acid_bias=a.acid_bias,
        min_plddt=a.min_plddt, max_rmsd=a.max_rmsd, target_S=a.target_S,
        min_independent_his=a.min_independent_his, n_submit=a.n_submit,
        max_per_backbone=a.max_per_backbone, seed=a.seed, shard=a.shard,
        nshards=a.nshards, force=a.force, learn=a.learn,
        skip_esmfold=a.skip_esmfold)
    if a.learn:
        cfg.apply_learn()

    for d in (WORK, RESULTS, INPUTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    cfg.save(WORK / "config.json")

    if a.selftest:
        return selftest(cfg)

    fns = {"prepare": stage_prepare, "diffuse": stage_diffuse,
           "mpnn": stage_mpnn, "fold": stage_fold, "score": stage_score,
           "submit": stage_submit}
    names = STAGES if cfg.stage == "all" else [cfg.stage]

    for name in names:
        if already_done(name, cfg):
            log(f"stage {name}: checkpoint present, skipping "
                f"(--force to redo)", "warn")
            continue
        t0 = time.time()
        fns[name](cfg)
        mark_done(name, cfg, {"seconds": round(time.time() - t0, 1)})
        log(f"stage {name} took {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

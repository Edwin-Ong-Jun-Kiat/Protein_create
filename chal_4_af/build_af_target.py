#!/usr/bin/env python3
"""Build the ONE sequence-contiguous TfR1 construct that AfDesign can use.

AF2 models a chain as a polypeptide. Hand it a structure with gaps and it will
invent loops to close them, and those invented loops are near the binding site.
So the construct must be a single unbroken stretch of residues.

The main pipeline's cropped targets are not that -- and they actively hide it.
`work/target_cropped_k382.pdb` is renumbered 1..N, so a naive "how many
segments?" check reports ONE contiguous segment while the file in fact has 21
chain breaks with a maximum CA-CA distance of 36.2 A. Renumbering erased the
evidence. Everything here is therefore checked on *geometry*, not numbering.

Default construct: chain A, residues 150-390 of the 11QC template, which is
241 residues spanning the whole apical domain (189-383) with ~40 residues of
flank on each side. target_meta.json records offset 0, so these numbers are
UniProt P02786 numbers and the k382 hotspots keep their familiar labels.

    python build_af_target.py --out work/target_af_k382.pdb
    python build_af_target.py --first 150 --last 280 --out work/target_af_small.pdb

The second form is the length-scaling experiment, and it is NOT a free lunch:
150-280 cannot contain residue 382, so it drops K382 itself -- one of the three
conserved cations the pH switch is built on. The script says so rather than
letting it pass silently; it bit me once already as
    AssertionError: ERROR: positions 382 and chain A not found
from an AfDesign run still carrying the full hotspot list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

ROOT = Path(__file__).resolve().parent
DEFAULT_CIF = ROOT / "inputs" / "11QC.cif"
DEFAULT_META = ROOT / "inputs" / "target_meta.json"

# A consecutive pair of residues in one chain is ~3.8 A CA-CA. Anything beyond
# this is a break however the residues are numbered.
CA_CA_MAX = 4.5


class _Keep(Select):
    """Standard amino acids of one chain in a residue-number window, altloc A."""

    def __init__(self, chain: str, first: int, last: int) -> None:
        self.chain, self.first, self.last = chain, first, last

    def accept_chain(self, c):  # noqa: D102
        return c.id == self.chain

    def accept_residue(self, r):  # noqa: D102
        het, num, _icode = r.id
        return het == " " and is_aa(r, standard=True) and self.first <= num <= self.last

    def accept_atom(self, a):  # noqa: D102
        # Keep one altloc, and drop hydrogens: AF2 ignores them and they only
        # inflate the file.
        return a.element != "H" and (not a.is_disordered() or a.get_altloc() in (" ", "A"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif", type=Path, default=DEFAULT_CIF)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--first", type=int, default=150)
    ap.add_argument("--last", type=int, default=390)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-breaks", action="store_true",
                    help="write the file even if it has chain breaks (do not)")
    a = ap.parse_args()

    if not a.cif.exists():
        print(f"ERROR: template not found: {a.cif}", file=sys.stderr)
        return 1

    print(f"template : {a.cif.name}")
    print(f"construct: chain {a.chain}, residues {a.first}-{a.last} "
          f"({a.last - a.first + 1} requested)")

    # QUIET=1: the parser warns about every discontinuity and negative B-factor
    # in a 639-residue dimer, which buries the output that matters.
    structure = MMCIFParser(QUIET=True).get_structure("tfr1", str(a.cif))
    model = next(structure.get_models())
    if a.chain not in {c.id for c in model}:
        print(f"ERROR: no chain {a.chain}; have {sorted(c.id for c in model)}",
              file=sys.stderr)
        return 1

    sel = _Keep(a.chain, a.first, a.last)
    kept = [r for r in model[a.chain] if sel.accept_residue(r)]
    if not kept:
        print("ERROR: the window selected zero residues", file=sys.stderr)
        return 1

    nums = [r.id[1] for r in kept]
    print(f"modelled : {len(kept)} residues, {nums[0]}-{nums[-1]}")
    absent = sorted(set(range(a.first, a.last + 1)) - set(nums))
    if absent:
        print(f"  not modelled in the template: {len(absent)} "
              f"({absent[:12]}{' ...' if len(absent) > 12 else ''})")

    # --- geometry, not numbering -------------------------------------------
    missing_ca = [n for r, n in zip(kept, nums) if "CA" not in r]
    if missing_ca:
        print(f"ERROR: {len(missing_ca)} residues have no CA: {missing_ca[:10]}",
              file=sys.stderr)
        return 1
    ca = np.array([r["CA"].get_coord() for r in kept], dtype=float)
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    breaks = [(nums[i], nums[i + 1], float(d[i])) for i in np.where(d > CA_CA_MAX)[0]]
    print(f"geometry : max consecutive CA-CA {d.max():.2f} A, "
          f"{len(breaks)} break(s) over {CA_CA_MAX} A")
    for lo, hi, dist in breaks[:10]:
        print(f"           break {lo} -> {hi}: {dist:.1f} A")
    if breaks and not a.allow_breaks:
        print("\nERROR: this construct is not contiguous, so AF2 would invent\n"
              "loops to close the gaps above. Pick a window without them, or\n"
              "pass --allow-breaks if you have decided that is acceptable.",
              file=sys.stderr)
        return 1

    # --- which hotspots survive the crop? ----------------------------------
    hotspots: list[int] = []
    if a.meta.exists():
        meta = json.loads(a.meta.read_text())
        site = meta["sites"][meta["primary_site"]]
        hotspots = list(site["hotspots"])
        present = [h for h in hotspots if h in set(nums)]
        lost = [h for h in hotspots if h not in set(nums)]
        print(f"hotspots : site '{site['key']}' has {hotspots}")
        print(f"           present in this construct: {present}")
        if lost:
            print(f"\n  WARNING: this crop DROPS hotspots {lost}.")
            cations = [c for c in site.get("switch_cations", []) if c in lost]
            if cations:
                print(f"  Those include switch cation(s) {cations} -- the pH switch is\n"
                      f"  built on His packing against exactly those residues, so a\n"
                      f"  design made here is answering a different question.")
            print(f"  Pass --hotspots '{','.join('A%d' % h for h in present)}' to\n"
                  f"  af_design.py, or it will abort with\n"
                  f"    AssertionError: ERROR: positions {lost[0]} and chain A not found")
        hotspots = present

    # --- write --------------------------------------------------------------
    a.out.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(a.out), select=sel)

    # Residue numbering is PRESERVED on purpose. Renumbering is what let a
    # broken file look contiguous; keeping UniProt numbers means a hotspot is
    # written "A382" everywhere, in this file, in target_meta.json and in the
    # literature.
    written = sum(1 for ln in a.out.read_text().splitlines()
                  if ln.startswith("ATOM") and ln[12:16].strip() == "CA")
    print(f"\nwrote    : {a.out}  ({written} CA records, "
          f"{a.out.stat().st_size // 1024} KB)")
    if written != len(kept):
        print(f"ERROR: wrote {written} CA records but selected {len(kept)} residues",
              file=sys.stderr)
        return 1
    if hotspots:
        print(f"hotspot argument for af_design.py:\n"
              f"  --hotspots {','.join('A%d' % h for h in hotspots)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

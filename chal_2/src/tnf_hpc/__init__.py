"""
TNF-alpha pH-selective binder design -- the HPC half of challenge 02.

Vendored deliberately rather than imported from ``../chal_2_local``: this
directory is rsync'd to Kelvin-2 on its own, and a cross-project import would
break the moment the two are not copied together. The science modules are the
same code, validated on the laptop.

    linkage   the physics. Reverse-polarity proton linkage: the 0.504-per-
              histidine ceiling, the two levers, and why four histidines are
              needed where challenge 1 needed two.
    target    stage 0. 1TNF trimer, TNFR2 footprint measured from 3ALQ,
              human/mouse conservation, epitope grooves.
    pka       stage 4. Histidine rotamers, bound- and free-state pKa, S.
    design    stages 4-5. Composite ranking, sequence gates, quotas.
    tools     the HPC-only half: target cropping, RFdiffusion contigs, and the
              ProteinMPNN / ESMFold subprocess drivers.

    scaffold  }  retained only for their geometry helpers (NeRF `_place`,
    dock      }  `atom_line`, `pseudo_cb`), which `pka` and `tools` use.
                 Their generative roles -- parametric bundles and rigid-body
                 docking -- are replaced here by RFdiffusion.

Sign convention, and it is the opposite of ``chal_1``'s:

    S = log10( Kd(pH 6.0) / Kd(pH 7.4) )

S > 0 means weaker binding at pH 6.0, which is what challenge 2 asks for.

Runs on Python 3.9 with numpy 1.2x (the rfdiff env) and on 3.12 with numpy 2.x.
"""

__all__ = ["linkage", "target", "pka", "design", "tools", "scaffold", "dock"]
__version__ = "0.1.0"

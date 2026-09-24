# Exploratory scripts

Standalone one-off tests run on Young, kept for reference. These are not part
of the `mace_ch4` pipeline and are not covered by CI.

## `compare_4way_450K_02GPa.py`

Short cross-model sanity check of CH4 at 450 K / 0.2 GPa
(rho = 0.34586 g/cm3, NIST), 128 molecules: MACE-OFF23, MACE-OFF24,
flexible OPLS-AA-style LJ, and TraPPE-UA. 1 ps Langevin NVT equilibration,
then 0.5 ps NVE; writes COM RDF, single-origin COM MSD, energy and pressure
traces to `compare_4way_450K_02GPa_out/`.

Not self-contained -- it needs two files that are not in this repo:

- `flexible_lj.py` (`FlexibleCH4Calculator`), next to the script
- `shared_potentials.py` (box building, COM trajectory, RDF, MACE/UA
  calculators), found via `$SHARED_POTENTIALS_DIR`

The NVE window is far too short for a diffusion coefficient: the MSD is still
ballistic at 500 fs. Use it only as a qualitative check that the models agree
on structure, pressure, and energy conservation.

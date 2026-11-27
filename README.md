# ALP-ttbar Toy MC v3.0 — Post-Registration

**Δχ²_z Profiling of Removed Near-Band Vertices: Results**

DOI: [10.5281/zenodo.24581149](https://doi.org/10.5281/zenodo.24581149)
Preregistration: [10.5281/zenodo.24554495](https://doi.org/10.5281/zenodo.24554495)
Date: 2026-11-27
Author: W. S. Paik

## Summary

This post-registration reports the results of the v3.0 preregistered study,
which tested whether removed near-band vertices exhibit systematically higher
Δχ²_z than surviving vertices.  50 seeds (i = 50–99) under configuration E3
(f = 0.25) evaluated 11 predictions across four investigations (A–D).

**Overall: PARTIAL support (3/4 Core confirmed: A1, B1, D1).**

| ID | Tier | Measured | Result |
|----|------|----------|--------|
| A1 | Core | 3.99 | CONFIRMED |
| A2 | Core | 0.97× | REFUTED |
| A3 | Secondary | 1.92% | CONFIRMED |
| B1 | Core | 0.3525 | CONFIRMED |
| B2 | Diagnostic | 1.9432 | REFUTED |
| B3 | Secondary | 0.5881 | REFUTED |
| C1 | Diagnostic | 118.5% | CONFIRMED |
| C2 | Diagnostic | +0.779 | REFUTED |
| C3 | Secondary | 1.0000 | CONFIRMED |
| D1 | Core | 2.16 | CONFIRMED |
| D2 | Diagnostic | 4.08 | REPORTED |

Key insight: λ_fit (Δχ²_3D/2) dominates upper-tail removal (97% overlap),
while λ_pos (position-only) independently contributes (D1 = 2.16) but is
secondary.

## Files

| File | Description |
|------|-------------|
| `ALP-ttbar_Toy-MC_v3.0_Post-Registration_DeltaChi2z-Profiling-Results_2026-11-27.pdf` | Post-registration report (PDF) |
| `run_v30_postreg.py` | Experiment runner (all 4 investigations) |
| `generate_v30_postreg.py` | PDF generator |
| `v30_postreg_results.json` | Full results (JSON) |
| `v30_postreg_per_seed_statistics.csv` | Per-seed raw statistics (50 × 32) |
| `v30_postreg_prediction_scorecard.csv` | 11 predictions scorecard (CSV) |
| `v30_postreg_data.npz` | NumPy arrays for figure generation |
| `frozen/endcap_sim.py` | Frozen simulation module (MD5: 7daa81b2…) |
| `frozen/mtag_v24.py` | Frozen 17-band tag (MD5: 39451a21…) |
| `frozen/mtag_v22.py` | Frozen 9-band tag (MD5: a687701e…) |

## SHA-256

```
df4eda82bad3b2aa4ee1deb5631ea8ca56128fad64cef90ba0384f19c20083e5  ALP-ttbar_Toy-MC_v3.0_Post-Registration_DeltaChi2z-Profiling-Results_2026-11-27.pdf
```

## Execution

```bash
# Dependencies
pip install numpy scipy scikit-learn matplotlib reportlab

# Run experiment (50 seeds, ~60s)
python run_v30_postreg.py

# Generate PDF report
python generate_v30_postreg.py
```

Requires `frozen/` directory containing the three frozen modules.

## Frozen Module MD5 Verification

All three frozen modules were MD5-verified before execution:

```
7daa81b2723a1c13df68ee69b428c2d0  endcap_sim.py
39451a2189d71ec4a9d5990eacb54c6e  mtag_v24.py
a687701efe4476a999e27b32341c8c39  mtag_v22.py
```

## DOI Chain

| Version | DOI |
|---------|-----|
| **v3.0 post-reg** | **10.5281/zenodo.24581149** |
| v3.0 prereg | 10.5281/zenodo.24554495 |
| v2.9 post-reg | 10.5281/zenodo.24527398 |
| v2.9 prereg | 10.5281/zenodo.24466187 |
| v2.8 post-reg | 10.5281/zenodo.24437855 |
| v2.8 prereg | 10.5281/zenodo.24414960 |
| v2.7 post-reg | 10.5281/zenodo.24408990 |
| v2.7 prereg | 10.5281/zenodo.24388228 |
| v2.6 post-reg | 10.5281/zenodo.24373115 |
| v2.6 prereg | 10.5281/zenodo.24361872 |
| v2.5 results | 10.5281/zenodo.24341050 |
| v2.5 prereg | 10.5281/zenodo.24329044 |
| v2.4 results | 10.5281/zenodo.24302619 |
| v2.4 prereg | 10.5281/zenodo.24296796 |
| v2.3 results | 10.5281/zenodo.24273586 |
| v2.3 prereg | 10.5281/zenodo.24244893 |
| v2.2 results | 10.5281/zenodo.24238224 |
| v2.2 prereg | 10.5281/zenodo.24189990 |
| v2.1 results | 10.5281/zenodo.24136462 |
| v2.1 prereg | 10.5281/zenodo.24063091 |
| v2.0 | 10.5281/zenodo.24056097 |

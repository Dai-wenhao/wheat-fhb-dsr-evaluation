# Wheat FHB diseased spikelet rate evaluation

Statistical reproduction code for comparing visible and whole-spike evaluation references and measuring within-spike disease progression from RGB images.

## Setup

Use Python 3.12 and install the tested dependencies:

```sh
python -m pip install -r requirements.txt
```

The scripts run on CPU from saved predictions. They do not fit or run image models. Both use 10,000 bootstrap replicates and seed 42 by default. MAE and MAE differences in the output are on the 0–1 scale. Multiply by 100 to report percentage points.

## Dryad progression analysis

```sh
python reproduce_dryad.py --out-dir results/dryad
```

This command runs from a fresh checkout using the two CSV files in `data/dryad/`. No FHBDSR files or image downloads are required.

| Input | Contents |
| --- | --- |
| `dryad_perimage_pred.csv` | Spike identifier, day after inoculation, visual severity from Dryad, and zero-shot prediction. |
| `dryad_perimage_pred_rotation_ft.csv` | The same observation identifiers and visual severity, plus isolate, zero-based holdout fold and fine-tuned prediction. |

The two files are matched by `spike_id` and `dai`. The spike identifier combines isolate, cultivar, replicate and head. `sev`, `pred` and `pred_ft` are proportions. Zero-shot predictions were generated with a model trained on FHBDSR. For fine-tuning, the same source model was adapted to Dryad using the training isolates in each rotation. The original observations are credited to [McConachie et al., Dryad](https://doi.org/10.5061/dryad.tht76hf6g). The raw image archive and workbook are available from that DOI.

| Output | Manuscript result |
| --- | --- |
| `dryad_intervals.json` | Table 5 MAEs and paired intervals for zero-shot, fine-tuned and training-period median predictions. |
| `dryad_revision_checks.json` | Cultivar × period baseline, isolate-weighted comparison, observation availability and fold sizes. |
| `dryad_pairs.csv`, `dryad_fold_observations.csv` | Paired changes and observation-level fold membership used in the analysis. |

The expected pair counts are 858 for days 7 to 10 and 856 for days 10 to 14. The fine-tuned MAEs are approximately 7.83 and 7.18 percentage points, respectively.

## FHBDSR static analysis

Obtain access to the data through the [FHBDSR dataset request procedure](https://github.com/WeizhenLiuBioinform/Wheat-FHB-DSR-Measurement#dataset) and cite [Wu et al.](https://doi.org/10.1007/s42994-025-00245-0). The FHBDSR images, reference records and FHBDSR-derived analysis inputs are not distributed in this repository. The command below uses separately supplied inputs. Access to the original dataset alone does not supply the study-specific predictions.

```sh
python reproduce.py --project-dir local_fhbdsr --validation-cache calibration_validation_counts.csv --out-dir results/fhbdsr
```

Prepare these files under `local_fhbdsr/`, with the validation-count file passed separately:

| Path | Contents |
| --- | --- |
| `data/FHBDSR/groundtruth.csv` | Image and spike identifiers, side, whole-spike counts and diseased proportion. |
| `data/FHBDSR/oracle_single.csv` | Annotated visible diseased and healthy counts and visible diseased proportion. |
| `runs/detector_perimage_counts_oof.csv` | Saved image-level detector counts and outer folds. |
| `runs/probe_regression_oof_s42.npz` | Saved corrected regression predictions aligned to the whole-spike reference rows. Override with `--regression-oof` if needed. |
| `runs/detector_oof_foldmap.json` | Outer test spike IDs for each fold. |
| `runs/_det_oof_tmp/train_f0.txt` through `train_f4.txt` and `val_f0.txt` through `val_f4.txt` | Training and internal validation image membership. The script uses the image basenames. |
| `calibration_validation_counts.csv` | Per-image detector counts on each fold's internal validation images, used to refit affine calibration. |

`static.json` contains Tables 2–4 comparisons for reference switching, view aggregation and manual counting. `revision_checks.json` contains the static severity-stratified checks, zero-box count and calibration clipping sensitivity.

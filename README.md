# Wheat FHB diseased spikelet rate evaluation

Statistical reproduction code for comparing visible and whole-spike evaluation references and measuring within-spike disease progression from RGB images.

## Setup

Use Python 3.12 and install the tested dependencies:

```sh
python -m pip install -r requirements.txt
```

The two statistical scripts run on CPU from saved predictions. Both use 10,000 bootstrap replicates and seed 42 by default. MAE and MAE differences in the output are on the 0–1 scale. Multiply by 100 to report percentage points. The separate `modeling/` scripts explain how to regenerate the predictions from the original images.

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

Obtain access to the data through the [FHBDSR dataset request procedure](https://github.com/WeizhenLiuBioinform/Wheat-FHB-DSR-Measurement#dataset) and cite [Wu et al.](https://doi.org/10.1007/s42994-025-00245-0). The FHBDSR images, reference records and FHBDSR-derived analysis inputs are kept outside this repository. Generate them locally by following the steps below.

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

## Regenerate predictions from the original images

Run the commands below from the repository root. Use a separate Python 3.10 environment with a CUDA-capable GPU. The original training environment used PyTorch 2.12.1 and torchvision 0.27.1 with CUDA 12.6 builds. Install those builds through the appropriate PyTorch package index, then install `modeling/requirements.txt`. The YOLO workflow initializes from `yolo26n.pt`; the regression workflow initializes from torchvision's ImageNet ConvNeXt-Tiny weights. These upstream weights are obtained through their respective libraries. Trained study checkpoints are generated under `build/runs/`.

1. Place your authorized FHBDSR copy under `build/source/fhbdsr/` with the original `YOLO_Dataset/images` and `YOLO_Dataset/labels` subdirectories. Run:

   ```sh
   python modeling/prepare_fhbdsr.py
   python modeling/prepare_visible_reference.py
   ```

   The first script extracts whole-spike counts from the image filenames. The second counts the visible healthy and diseased boxes from the YOLO annotations.

2. Generate the grouped out-of-fold predictions and run the static analysis:

   ```sh
   python modeling/probe_detector_oof.py
   python modeling/probe_regression_oof.py
   python reproduce.py --project-dir build --validation-cache build/runs/calibration_validation_counts.csv --out-dir results/fhbdsr
   ```

   The detector script writes the outer test predictions, fold assignments, training/validation manifests and the separate inner validation counts. The regression script writes the corrected grouped-fold prediction array. The final command refits the affine calibrators exclusively on the inner validation counts and produces the statistical outputs.

3. For image-level regeneration of the Dryad predictions, place the [Dryad workbook and extracted image directory](https://doi.org/10.5061/dryad.tht76hf6g) directly under `build/source/dryad/`. Complete step 1, then run:

   ```sh
   python modeling/probe_external_dryad.py
   python modeling/probe_dryad_progression.py
   python modeling/probe_dryad_rotation_ft.py
   python reproduce_dryad.py --runs build/runs --out-dir results/dryad
   ```

   The first command trains the fixed FHBDSR source model used for the Dryad experiment. The second creates the zero-shot predictions and preprocessing cache. The third starts each isolate rotation from the same source checkpoint and writes the fine-tuned predictions. The public statistical replay at the top of this README uses the deposited predictions and is the direct route to the reported Table 5 values. Retraining can produce numerical differences across devices or software builds.

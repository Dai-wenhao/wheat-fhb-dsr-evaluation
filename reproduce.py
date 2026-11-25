"""Replay the wheat Fusarium head blight evaluation from saved predictions.

Dependencies: Python, NumPy (tested with 1.26.4), pandas (tested with 2.2.2).
Example:
  python reproduce.py --project-dir project --validation-cache calibration_validation_counts.csv

Required local inputs under --project-dir:
  data/FHBDSR/groundtruth.csv and data/FHBDSR/oracle_single.csv
  runs/detector_perimage_counts_oof.csv and runs/detector_oof_foldmap.json
  runs/probe_regression_oof_s42.npz, or its path via --regression-oof
  runs/_det_oof_tmp/train_f{0..4}.txt and val_f{0..4}.txt
  runs/dryad_perimage_pred.csv and runs/dryad_perimage_pred_rotation_ft.csv
Pass the fold-specific validation counts CSV via --validation-cache.
These inputs are separate from the script. No model training or inference is run.
Public data sources:
  FHBDSR: https://doi.org/10.1007/s42994-025-00245-0
  Dryad: https://doi.org/10.5061/dryad.tht76hf6g
Reported MAE and MAE differences are proportions. Multiply by 100 for percentage points.
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd


IMAGE_KEY = ["image", "spike", "side"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_project(root, regression_oof):
    inputs = {
        "whole_reference": root / "data/FHBDSR/groundtruth.csv",
        "visible_reference": root / "data/FHBDSR/oracle_single.csv",
        "detector_oof": root / "runs/detector_perimage_counts_oof.csv",
        "regression_oof": regression_oof,
        "detector_foldmap": root / "runs/detector_oof_foldmap.json",
    }
    gt = pd.read_csv(inputs["whole_reference"])
    visible = pd.read_csv(inputs["visible_reference"])
    detector = pd.read_csv(inputs["detector_oof"])
    with np.load(inputs["regression_oof"]) as regression:
        require(np.array_equal(gt["spike"], regression["spike"]),
                "Regression array has a different spike order than groundtruth.csv.")
        if "side" in regression.files:
            require(np.array_equal(gt["side"], regression["side"]),
                    "Regression array has a different side order than groundtruth.csv.")
        require(np.allclose(gt["dsrc"], regression["dsrc"], atol=1e-7, rtol=0),
                "Regression reference values differ from groundtruth.csv.")
        gt["regression"] = regression["full"]

    images = gt.merge(
        visible[IMAGE_KEY + ["n_diseased", "n_healthy", "single_dsr"]],
        on=IMAGE_KEY, validate="one_to_one",
    ).merge(
        detector[IMAGE_KEY + ["n_diseased", "n_healthy", "fold", "gt_d", "gt_t"]],
        on=IMAGE_KEY, suffixes=("_visible", "_detector"), validate="one_to_one",
    )
    require(len(images) == len(gt) == len(visible) == len(detector) == 620,
            "Expected one matched prediction and reference for each of the 620 images.")
    require(images.groupby("spike")["side"].apply(lambda x: set(x) == {1, 2}).all(),
            "Every spike must contain one image per side.")
    require(images.groupby("spike")[["total", "diseased", "dsrc", "fold"]]
            .nunique().eq(1).all().all(), "Sides of a spike disagree on their labels or fold.")
    require((images["diseased"] == images["gt_d"]).all()
            and (images["total"] == images["gt_t"]).all()
            and np.allclose(images["dsrc"], images["diseased"] / images["total"]),
            "Stored whole-spike counts or ratios disagree.")

    # The regression script shuffled first appearances of spike IDs with seed 42.
    spikes = gt["spike"].unique()
    np.random.default_rng(42).shuffle(spikes)
    folds = {str(f): sorted(map(int, fold)) for f, fold in enumerate(np.array_split(spikes, 5))}
    expected_fold = {spike: int(f) for f, fold in folds.items() for spike in fold}
    require(json.loads(inputs["detector_foldmap"].read_text(encoding="utf-8")) == folds
            and images["spike"].map(expected_fold).eq(images["fold"]).all(),
            "Detector and regression outer folds differ.")

    validation = []
    for fold in range(5):
        split = {}
        for part in ("train", "val"):
            manifest = root / f"runs/_det_oof_tmp/{part}_f{fold}.txt"
            inputs[f"detector_{part}_{fold}"] = manifest
            names = [PureWindowsPath(line).name
                     for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            split[part] = images[images["image"].isin(names)]
            require(len(split[part]) == len(names), f"Unmatched {part} images in fold {fold}.")
        training = set(split["train"]["spike"])
        val = set(split["val"]["spike"])
        test = set(folds[str(fold)])
        require(len(val) == 37 and not (training & val or training & test or val & test)
                and training | val | test == set(spikes),
                f"Unexpected split sizes or overlap in fold {fold}.")
        if regression_oof is not None:
            inner_spikes = gt.loc[~gt["spike"].isin(test), "spike"].unique()
            regression_val = np.random.default_rng(4200 + fold).choice(inner_spikes, size=37, replace=False)
            require(set(regression_val) == val, f"Regression and detector validation sets differ in fold {fold}.")
        validation.append(split["val"][IMAGE_KEY + ["dsrc"]].assign(fold=fold))

    images["detector"] = images["n_diseased_detector"] / (
        images["n_diseased_detector"] + images["n_healthy_detector"]
    )
    required_values = ["dsrc", "single_dsr", "detector", "regression"]
    require(np.isfinite(images[required_values].to_numpy()).all(), "Non-finite main predictions.")
    images["count_inconsistent"] = (
        images["n_diseased_visible"].gt(images["diseased"])
        | images["n_healthy_visible"].gt(images["total"] - images["diseased"])
    )
    return images.sort_values(["spike", "side"]), pd.concat(validation, ignore_index=True), inputs


def validation_predictions(expected, cache):
    key = ["fold"] + IMAGE_KEY
    counts = pd.read_csv(cache)
    val = expected.merge(counts, on=key, validate="one_to_one")
    require(len(val) == len(expected) == len(counts), "Validation cache differs from the manifests.")
    val["detector"] = val["n_diseased"] / (val["n_diseased"] + val["n_healthy"])
    require(np.isfinite(val["detector"]).all(), "A validation image has no counted spikelets.")
    return val


def apply_calibration(images, validation):
    images = images.copy()
    fit_info = []
    for fold, val in validation.groupby("fold"):
        val_spikes = val.groupby("spike")[["detector", "dsrc"]].mean()
        test = images["fold"].eq(fold)
        info = {"fold": int(fold), "n_validation_spikes": len(val_spikes)}
        for name, scores in (("single", val), ("dual", val_spikes)):
            design = np.column_stack([np.ones(len(scores)), scores["detector"]])
            intercept, slope = np.linalg.lstsq(design, scores["dsrc"], rcond=None)[0]
            images.loc[test, f"calibrated_{name}"] = intercept + slope * images.loc[test, "detector"]
            info[name] = {
                "intercept": float(intercept), "slope": float(slope),
                "validation_raw_bias": float((scores["detector"] - scores["dsrc"]).mean()),
                "test_raw_bias": float((images.loc[test, "detector"] - images.loc[test, "dsrc"]).mean()),
            }
        fit_info.append(info)
    return images, fit_info


def point_table(predictions, references):
    table = {}
    for name, forecast in predictions.items():
        table[name] = {}
        for target, reference in references.items():
            delta = (forecast - reference).ravel()
            table[name][target] = {
                "mae": float(np.abs(delta).mean()),
                "pearson_r": float(np.corrcoef(forecast.ravel(), reference.ravel())[0, 1]),
                "bias": float(delta.mean()),
            }
    return table


def paired_data(images, columns):
    return images[columns].to_numpy().reshape(-1, 2, len(columns))


def with_intervals(data, statistic, labels, draws):
    point = statistic(data)
    replicates = np.stack([statistic(data[index]) for index in draws])
    bounds = np.quantile(replicates, [0.025, 0.975], axis=0).T
    return {name: {"estimate": float(value), "ci95": interval.tolist()}
            for name, value, interval in zip(labels, point, bounds)}


def reference_results(images, draws):
    data = paired_data(images, ["detector", "regression", "single_dsr", "dsrc"])

    def contrasts(x):
        flat = x.reshape(-1, x.shape[-1])
        err = np.abs(flat[:, :2, None] - flat[:, None, 2:])
        gap_visible, gap_whole = (err[:, 0] - err[:, 1]).mean(axis=0)
        corr = np.corrcoef(flat, rowvar=False)
        r_visible, r_whole = corr[0, 2:] - corr[1, 2:]
        return [gap_visible, gap_whole, gap_whole - gap_visible,
                r_visible, r_whole, r_visible - r_whole]

    return {
        "points": point_table(
            {"detector": data[:, :, 0], "regression": data[:, :, 1]},
            {"visible": data[:, :, 2], "whole": data[:, :, 3]},
        ),
        "contrasts": with_intervals(
            data, contrasts,
            ["mae_det_minus_reg_visible", "mae_det_minus_reg_whole", "mae_gap_change_whole_minus_visible",
             "r_det_minus_reg_visible", "r_det_minus_reg_whole", "r_gap_change_visible_minus_whole"],
            draws,
        ),
    }


def view_and_calibration_results(images, draws):
    data = paired_data(images, ["detector", "regression", "calibrated_single", "calibrated_dual", "dsrc"])
    single = {"detector": data[:, :, 0], "regression": data[:, :, 1], "calibrated": data[:, :, 2]}
    dual = {name: score.mean(axis=1) for name, score in single.items()}
    dual["calibrated"] = data[:, :, 3].mean(axis=1)
    single_target, dual_target = data[:, :, 4], data[:, 0, 4]
    single_error = {name: np.abs(score - single_target).mean(axis=1) for name, score in single.items()}
    dual_error = {name: np.abs(score - dual_target) for name, score in dual.items()}
    single_gap = single_error["detector"] - single_error["regression"]
    dual_gap = dual_error["detector"] - dual_error["regression"]
    changes = {
        "dual_minus_single_detector": dual_error["detector"] - single_error["detector"],
        "dual_minus_single_regression": dual_error["regression"] - single_error["regression"],
        "single_minus_dual_detector_gap": single_gap - dual_gap,
        "dual_det_minus_reg": dual_gap,
        "dual_cal_minus_raw": dual_error["calibrated"] - dual_error["detector"],
        "dual_cal_minus_reg": dual_error["calibrated"] - dual_error["regression"],
        "single_cal_minus_raw": single_error["calibrated"] - single_error["detector"],
    }
    return {
        "single": point_table(single, {"whole": single_target}),
        "dual": point_table(dual, {"whole": dual_target}),
        "mae_contrasts": with_intervals(np.column_stack(list(changes.values())),
                                       lambda x: x.mean(axis=0), changes, draws),
    }


def manual_count_results(images, draws):
    data = paired_data(images, ["single_dsr", "dsrc", "n_diseased_visible", "n_healthy_visible"])

    def estimates(x):
        counts = x[:, :, 2:].sum(axis=1)
        dual = counts[:, 0] / counts.sum(axis=1)
        single_r = np.corrcoef(x[:, :, 0].ravel(), x[:, :, 1].ravel())[0, 1]
        dual_r = np.corrcoef(dual, x[:, 0, 1])[0, 1]
        single_mae = np.abs(x[:, :, 0] - x[:, :, 1]).mean()
        dual_mae = np.abs(dual - x[:, 0, 1]).mean()
        return np.array([single_r, dual_r, single_mae, dual_mae, dual_r - single_r, dual_mae - single_mae])

    return with_intervals(data, estimates, [
        "single_r", "dual_r", "single_mae", "dual_mae", "dual_minus_single_r", "dual_minus_single_mae",
    ], draws)


def view_error_cancellation(images):
    paired = paired_data(images, ["detector", "regression", "dsrc"])
    result = {}
    for column, method in enumerate(("detector", "regression")):
        errors = paired[:, :, column] - paired[:, :, 2]
        opposite = errors[:, 0] * errors[:, 1] < 0
        gain = np.abs(errors).mean(axis=1) - np.abs(errors.mean(axis=1))
        result[method] = {
            "opposite_sign_spikes": int(opposite.sum()),
            "mean_mae_gain_among_opposite_sign_spikes": float(gain[opposite].mean()) if opposite.any() else None,
        }
    return result


PERIODS = {(7, 10): "7->10", (10, 14): "10->14"}
ERROR_COLUMNS = ["error_zs", "error_base", "error_ft"]


def load_pairs(runs):
    zero_shot = pd.read_csv(runs / "dryad_perimage_pred.csv")
    fine_tuned = pd.read_csv(runs / "dryad_perimage_pred_rotation_ft.csv")
    observations = zero_shot.merge(fine_tuned, on=["spike_id", "dai"], validate="one_to_one")
    assert len(observations) == len(zero_shot) == len(fine_tuned) == 2576
    np.testing.assert_allclose(observations["sev_x"], observations["sev_y"], atol=1e-7, rtol=0)
    assert observations.groupby("spike_id")[["isolate", "fold"]].nunique().eq(1).all().all()

    observations = observations.sort_values(["spike_id", "dai"])
    columns = ["dai", "sev_x", "pred", "pred_ft"]
    previous = observations.groupby("spike_id")[columns].shift()
    pairs = observations[["spike_id", "isolate", "fold"]].copy()
    pairs["period"] = [PERIODS.get((start, end)) for start, end in zip(previous["dai"], observations["dai"])]
    for name in columns[1:]:
        pairs[name] = observations[name] - previous[name]
    pairs = pairs.dropna(subset=["period"]).copy()
    assert pairs.groupby("period").size().to_dict() == {"7->10": 858, "10->14": 856}
    assert np.isfinite(pairs[columns[1:]].to_numpy()).all()
    assert pairs.groupby("isolate")["fold"].nunique().eq(1).all()
    assert pairs.groupby("fold")["isolate"].nunique().to_dict() == dict.fromkeys(range(5), 5)

    pairs["error_zs"] = (pairs["pred"] - pairs["sev_x"]).abs()
    pairs["error_ft"] = (pairs["pred_ft"] - pairs["sev_x"]).abs()
    for fold in range(5):
        test = pairs["fold"].eq(fold)
        train = ~pairs["fold"].isin([fold, (fold + 1) % 5])
        medians = pairs.loc[train].groupby("period")["sev_x"].median()
        pairs.loc[test, "error_base"] = (pairs.loc[test, "period"].map(medians) - pairs.loc[test, "sev_x"]).abs()
    assert np.isfinite(pairs[ERROR_COLUMNS].to_numpy()).all()
    return pairs


def paired_intervals(pairs, n_resamples, seed):
    # Every replicate draws five strains within each held-out fold. Multiplicity
    # weights retain both periods and all spikes from each sampled strain.
    strains = pairs[["fold", "isolate"]].drop_duplicates().sort_values(["fold", "isolate"])
    index = pd.MultiIndex.from_frame(strains)
    rng = np.random.default_rng(seed)
    draws = []
    for fold in range(5):
        group = np.flatnonzero(strains["fold"].to_numpy() == fold)
        draws.append(rng.choice(group, (n_resamples, len(group)), replace=True))
    draws = np.concatenate(draws, axis=1)

    results = {}
    for period, records in pairs.groupby("period"):
        grouped = records.groupby(["fold", "isolate"])
        counts = grouped.size().reindex(index, fill_value=0).to_numpy()
        sums = grouped[ERROR_COLUMNS].sum().reindex(index, fill_value=0).to_numpy()
        observed = sums.sum(axis=0) / counts.sum()
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
        contrasts = {
            name: {
                "estimate": float(observed[i] - observed[2]),
                "ci95": np.quantile(boot[:, i] - boot[:, 2], [0.025, 0.975]).tolist(),
            }
            for i, name in enumerate(("gain_vs_zero_shot", "gain_vs_baseline"))
        }
        results[period] = {
            "n": len(records),
            "mae_zs": float(observed[0]),
            "mae_base": float(observed[1]),
            "mae_ft": float(observed[2]),
            **contrasts,
        }
    return results


def summary(point, bootstrap):
    return {"estimate": float(point), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}


def static_checks(spikes, draws, seed):
    visible = (spikes.detector - spikes.single_dsr).abs() - (spikes.regression - spikes.single_dsr).abs()
    whole = (spikes.detector - spikes.dsrc).abs() - (spikes.regression - spikes.dsrc).abs()
    ears = spikes.assign(visible_gap=visible, whole_gap=whole, shift=whole - visible).groupby('spike').agg(
        severity=('dsrc', 'first'), diseased=('diseased', 'first'), total=('total', 'first'),
        conflict=('count_inconsistent', 'any'), visible_gap=('visible_gap', 'mean'),
        whole_gap=('whole_gap', 'mean'), shift=('shift', 'mean'))
    rng = np.random.default_rng(seed)
    bins = pd.cut(ears.diseased / ears.total, [0, .20, .40, .60, np.inf], right=False,
                  labels=['0–<20', '20–<40', '40–<60', '60–100'])
    fields = ['visible_gap', 'whole_gap', 'shift']
    strata = {}
    common_draws = {'all': [], 'screened': []}
    weights = []
    for label in bins.cat.categories:
        group = ears.loc[bins.eq(label)]
        strata[label] = {'n': len(group), 'conflicts': int(group.conflict.sum())}
        groups = {'all': group, 'screened': group.loc[~group.conflict]}
        for name, subset in groups.items():
            arr = subset[fields].to_numpy()
            if not len(arr):
                strata[label][name] = {'n': 0}
                continue
            samples = arr[rng.integers(len(arr), size=(draws, len(arr)))].mean(axis=1)
            point = arr.mean(axis=0)
            strata[label][name] = {'n': len(arr), **{
                col: summary(point[i], samples[:, i]) for i, col in enumerate(fields)}}
            if len(groups['screened']):
                common_draws[name].append((point, samples))
        if len(groups['screened']):
            weights.append(len(group))
    standardized = {}
    for name, samples in common_draws.items():
        point = np.average(np.stack([entry[0] for entry in samples]), axis=0, weights=weights)
        intervals = np.average(np.stack([entry[1] for entry in samples]), axis=0, weights=weights)
        standardized[name] = {col: summary(point[i], intervals[:, i]) for i, col in enumerate(fields)}
    return {'strata': strata, 'standardized': standardized, 'weight_counts': weights}


def calibration_checks(images, draws, seed):
    rng = np.random.default_rng(seed)
    results = {}
    for unit in ('single', 'dual'):
        col = f'calibrated_{unit}'
        data = images[['spike', col, 'dsrc', 'detector']]
        if unit == 'dual':
            data = data.groupby('spike', as_index=False).mean()
        raw = (data.detector - data.dsrc).abs()
        uncapped = (data[col] - data.dsrc).abs()
        clipped = (data[col].clip(0, 1) - data.dsrc).abs()
        changes = data.assign(clip_minus_original=clipped - uncapped, clip_minus_detector=clipped - raw)
        means = changes.groupby('spike')[['clip_minus_original', 'clip_minus_detector']].mean().to_numpy()
        boot = means[rng.integers(len(means), size=(draws, len(means)))].mean(axis=1)
        results[unit] = {
            'n': len(data), 'below': int(data[col].lt(0).sum()), 'above': int(data[col].gt(1).sum()),
            'minimum': float(data[col].min()), 'maximum': float(data[col].max()),
            'clipped_mae': float(clipped.mean()),
            'clip_minus_original': summary(means[:, 0].mean(), boot[:, 0]),
            'clip_minus_detector': summary(means[:, 1].mean(), boot[:, 1]),
        }
    return results


def dryad_checks(runs, draws, seed, out):
    pairs = load_pairs(runs)
    observations = pd.read_csv(runs / 'dryad_perimage_pred_rotation_ft.csv')
    parts = observations.spike_id.str.split('_', n=3, expand=True)
    assert parts.shape[1] == 4 and parts.notna().all().all()
    assert parts[0].eq(observations.isolate.astype(str)).all()
    observations['cultivar'] = parts[1]
    pairs = pairs.merge(observations[['spike_id', 'cultivar']].drop_duplicates(), on='spike_id', validate='many_to_one')
    assert pairs.cultivar.nunique() == 6

    fold_sizes = []
    baselines = []
    for fold in range(5):
        val_fold = (fold + 1) % 5
        fit_obs = observations.loc[~observations.fold.isin([fold, val_fold])]
        val_obs = observations.loc[observations.fold.eq(val_fold)]
        test_obs = observations.loc[observations.fold.eq(fold)]
        fold_sizes.append({'fold': fold + 1, **{
            k: [len(f), f.spike_id.nunique(), f.isolate.nunique()]
            for k, f in [('fit', fit_obs), ('validation', val_obs), ('test', test_obs)]}})
        fit = pairs.loc[~pairs.fold.isin([fold, val_fold])]
        test = pairs.loc[pairs.fold.eq(fold)].copy()
        model = fit.groupby(['cultivar', 'period']).agg(
            median=('sev_x', 'median'), strains=('isolate', 'nunique'), n=('sev_x', 'size'))
        key = pd.MultiIndex.from_frame(test[['cultivar', 'period']])
        subset = model.reindex(key)
        fallback = subset.strains.fillna(0).to_numpy() < 3
        time_median = test.period.map(fit.groupby('period').sev_x.median())
        test['prediction_period'] = time_median
        test['prediction_cultivar_period'] = np.where(fallback, time_median, subset['median'].to_numpy())
        test['cultivar_period_fallback'] = fallback
        baselines.append(test)

    pairs = pd.concat(baselines, ignore_index=True)
    pairs['error_cultivar_period'] = (pairs.prediction_cultivar_period - pairs.sev_x).abs()
    assert np.isfinite(pairs.error_cultivar_period).all()
    columns = [*ERROR_COLUMNS, 'error_cultivar_period']
    strains = pairs[['fold', 'isolate']].drop_duplicates().sort_values(['fold', 'isolate'])
    idx = pd.MultiIndex.from_frame(strains)
    rng = np.random.default_rng(seed)
    by_fold = [np.flatnonzero(strains.fold.to_numpy() == f) for f in range(5)]
    sampling = np.concatenate([rng.choice(group, (draws, len(group)), replace=True) for group in by_fold], axis=1)
    intervals = {}
    for period, group in pairs.groupby('period'):
        grouped = group.groupby(['fold', 'isolate'])
        counts = grouped.size().reindex(idx, fill_value=0).to_numpy()
        sums = grouped[columns].sum().reindex(idx, fill_value=0).to_numpy()
        assert (counts > 0).all()
        point = sums.sum(axis=0) / counts.sum()
        boot = sums[sampling].sum(axis=1) / counts[sampling].sum(axis=1)[:, None]
        isolate_mae = sums / counts[:, None]
        macro = isolate_mae.mean(axis=0)
        macro_boot = isolate_mae[sampling].mean(axis=1)
        intervals[period] = {'n': len(group), 'fallbacks': int(group.cultivar_period_fallback.sum()),
            'mae': dict(zip(columns, point.tolist())),
            'gain': summary(point[3] - point[2], boot[:, 3] - boot[:, 2]),
            'macro_mae': dict(zip(columns, macro.tolist())),
            'macro_gain': {columns[i]: summary(macro[i] - macro[2], macro_boot[:, i] - macro_boot[:, 2]) for i in (0, 1, 3)},
            'signed_change_counts': [int(group.sev_x.lt(0).sum()), int(group.sev_x.eq(0).sum()), int(group.sev_x.gt(0).sum())]}
    masks = observations.groupby('spike_id').dai.apply(lambda values: tuple(sorted(values)))
    patterns = {','.join(map(str, items)): int(masks.map(lambda actual: actual == items).sum())
                for k in (1, 2, 3) for items in combinations((7, 10, 14), k)}
    assert sum(patterns.values()) == 861 and sum(len(k.split(',')) * n for k, n in patterns.items()) == 2576
    pairs.to_csv(out / 'dryad_pairs.csv', index=False)
    observations.sort_values(['fold', 'spike_id', 'dai']).to_csv(out / 'dryad_fold_observations.csv', index=False)
    return {'periods': intervals, 'patterns': patterns, 'fold_sizes': fold_sizes}



def evaluate_static(images, inputs, calibration, args):
    cache = args.validation_cache
    flagged = images.loc[images["count_inconsistent"], "spike"].unique()
    subsets = {"all": images, "excluding_count_inconsistent_spikes": images[~images["spike"].isin(flagged)]}
    result = {
        "bootstrap": {"unit": "spike", "method": "paired percentile with replacement",
                      "resamples": args.resamples, "seed": args.seed},
        "input_sha256": {name: digest(path) for name, path in inputs.items()},
        "validation_cache_sha256": digest(cache),
        "calibration_fits": calibration,
        "count_inconsistent_images": int(images["count_inconsistent"].sum()),
        "count_inconsistent_spikes": len(flagged),
        "fold_mae_gaps": {},
        "subsets": {},
    }
    for fold, group in images.groupby("fold"):
        errors = [np.abs(group["detector"] - group[reference]) - np.abs(group["regression"] - group[reference])
                  for reference in ("single_dsr", "dsrc")]
        visible_gap, whole_gap = (float(error.mean()) for error in errors)
        result["fold_mae_gaps"][str(fold + 1)] = [visible_gap, whole_gap, whole_gap - visible_gap]
    for name, group in subsets.items():
        n = group["spike"].nunique()
        draws = np.random.default_rng(args.seed).integers(n, size=(args.resamples, n))
        result["subsets"][name] = {
            "n_images": len(group), "n_spikes": n,
            "reference_comparison": reference_results(group, draws),
            "view_and_calibration": view_and_calibration_results(group, draws),
            "manual_count_comparison": manual_count_results(group, draws),
            "view_error_cancellation": view_error_cancellation(group),
        }
    return result


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--regression-oof", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")
    runs = args.project_dir / "runs"
    regression = args.regression_oof or runs / "probe_regression_oof_s42.npz"
    images, validation_rows, inputs = load_project(args.project_dir, regression)
    validation = validation_predictions(validation_rows, args.validation_cache)
    images, calibration = apply_calibration(images, validation)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.out_dir / "static.json", evaluate_static(images, inputs, calibration, args))
    write_json(args.out_dir / "dryad_intervals.json",
               paired_intervals(load_pairs(runs), args.resamples, args.seed))

    counts = pd.read_csv(inputs["detector_oof"])
    result = {
        "static": static_checks(images, args.resamples, args.seed),
        "zero_boxes": int((counts.n_diseased + counts.n_healthy).eq(0).sum()),
        "calibration": calibration_checks(images, args.resamples, args.seed),
        "dryad": dryad_checks(runs, args.resamples, args.seed, args.out_dir),
        "input_sha256": {name: digest(path) for name, path in inputs.items()},
        "bootstrap": {"resamples": args.resamples, "seed": args.seed},
    }
    write_json(args.out_dir / "revision_checks.json", result)
    print(f"Wrote evaluation results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

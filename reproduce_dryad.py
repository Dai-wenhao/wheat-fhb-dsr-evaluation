"""Recompute Dryad disease progression statistics from published predictions."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


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




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parent / "data/dryad")
    parser.add_argument("--out-dir", type=Path, default=Path("dryad_results"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("--resamples must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "dryad_intervals.json": paired_intervals(load_pairs(args.runs), args.resamples, args.seed),
        "dryad_revision_checks.json": dryad_checks(args.runs, args.resamples, args.seed, args.out_dir),
    }
    for filename, result in results.items():
        path = args.out_dir / filename
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()

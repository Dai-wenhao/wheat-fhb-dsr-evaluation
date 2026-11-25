#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train grouped detector folds and save outer test and inner validation counts."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def count_classes(model, image, args):
    result = model.predict(str(image), imgsz=args.imgsz, conf=args.conf,
                           device=args.device, save=False, verbose=False)[0]
    classes = result.boxes.cls.cpu().numpy()
    return int((classes == 1).sum()), int((classes == 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=r'build/data/FHBDSR/groundtruth.csv')
    ap.add_argument('--imgdir', default=r'build/source/fhbdsr/YOLO_Dataset/images')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15, help='早停（原检测器 ep20 达峰，60/15 足够）')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--out', default=r'build/runs/detector_perimage_counts_oof.csv')
    ap.add_argument('--foldmap', default=r'build/runs/detector_oof_foldmap.json')
    ap.add_argument('--max-folds', type=int, default=None, help='只跑前 N 折（冒烟测试用）')
    ap.add_argument('--device', default='0')
    ap.add_argument('--initial-weights', default='yolo26n.pt')
    ap.add_argument('--validation-cache', type=Path)
    ap.add_argument('--reuse-fold-weights', action='store_true')
    args = ap.parse_args()
    runs = Path(args.out).resolve().parent
    runs.mkdir(parents=True, exist_ok=True)
    Path(args.foldmap).parent.mkdir(parents=True, exist_ok=True)
    validation_cache = args.validation_cache or runs / 'calibration_validation_counts.csv'
    validation_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.max_folds is not None and args.max_folds < 1:
        ap.error('--max-folds must be positive')

    from ultralytics import YOLO

    gt = pd.read_csv(args.gt)
    img_dir = Path(args.imgdir).resolve()

    # ---------- 复现回归的穗级 5 折 ----------
    spikes = gt['spike'].unique()  # 310 穗，按 groundtruth 出现顺序
    rng = np.random.default_rng(args.seed)
    rng.shuffle(spikes)
    folds = np.array_split(spikes, args.k)
    fold_of_spike = {int(sp): f for f, arr in enumerate(folds) for sp in arr}
    gt['fold'] = gt['spike'].map(fold_of_spike)
    Path(args.foldmap).write_text(json.dumps(
        {str(f): sorted(int(s) for s in arr) for f, arr in enumerate(folds)},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'穗级 {args.k} 折切分完成，每折 {len(folds[0])} 穗，foldmap 落盘', flush=True)

    tmpdir = runs / '_det_oof_tmp'
    tmpdir.mkdir(parents=True, exist_ok=True)

    rows = []
    validation_rows = []
    n_folds = args.k if args.max_folds is None else min(args.max_folds, args.k)
    for f in range(n_folds):
        t0 = time.time()
        te = gt[gt['fold'] == f]
        tr_all = gt[gt['fold'] != f]
        tr_spikes = tr_all['spike'].unique()
        rng2 = np.random.default_rng(args.seed * 100 + f)
        val_spikes = rng2.choice(tr_spikes, size=max(8, int(0.15 * len(tr_spikes))), replace=False)
        val_set = set(val_spikes.tolist())
        tr_df = tr_all[~tr_all['spike'].isin(val_set)]
        va_df = tr_all[tr_all['spike'].isin(val_set)]

        tr_txt = tmpdir / f'train_f{f}.txt'
        va_txt = tmpdir / f'val_f{f}.txt'
        tr_txt.write_text('\n'.join(str(img_dir / im) for im in tr_df['image']), encoding='utf-8')
        va_txt.write_text('\n'.join(str(img_dir / im) for im in va_df['image']), encoding='utf-8')
        yaml_path = tmpdir / f'data_f{f}.yaml'
        yaml_path.write_text(
            f'path: {json.dumps(img_dir.parent.as_posix())}\n'
            f'train: {json.dumps(tr_txt.as_posix())}\n'
            f'val: {json.dumps(va_txt.as_posix())}\n'
            f'nc: 2\nnames: [\'healthy\', \'diseased\']\n', encoding='utf-8')

        print(f'\n===== fold {f} ===== train={len(tr_df)}图({tr_df.spike.nunique()}穗) '
              f'val={len(va_df)}图({va_df.spike.nunique()}穗) test={len(te)}图({te.spike.nunique()}穗)',
              flush=True)

        best = runs / f'det_oof_fold{f}' / 'weights' / 'best.pt'
        if not args.reuse_fold_weights:
            m = YOLO(args.initial_weights)
            m.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, project=str(runs), name=f'det_oof_fold{f}',
                    pretrained=True, seed=args.seed, amp=True, workers=0, patience=args.patience,
                    exist_ok=True, verbose=False)
            best = Path(m.trainer.best)
        model = YOLO(str(best))
        for row in va_df.itertuples(index=False):
            nd, nh = count_classes(model, img_dir / row.image, args)
            validation_rows.append((f, row.image, row.spike, row.side, nd, nh))
        for _, row in te.iterrows():
            nd, nh = count_classes(model, img_dir / row['image'], args)
            rows.append(dict(image=row['image'], spike=row['spike'], side=row['side'],
                             n_diseased=nd, n_healthy=nh, fold=f,
                             gt_d=row['diseased'], gt_t=row['total'], split=row['split']))
        print(f'  fold {f} 完成: {(time.time()-t0)/60:.1f} min', flush=True)

    validation_columns = ['fold', 'image', 'spike', 'side', 'n_diseased', 'n_healthy']
    pd.DataFrame(validation_rows, columns=validation_columns).to_csv(validation_cache, index=False)
    print(f'Wrote validation counts to {validation_cache}')
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'\nWrote {len(out_df)} outer test predictions to {args.out}')
    out_df['pdsr'] = out_df['n_diseased'] / (out_df['n_diseased'] + out_df['n_healthy'])
    out_df['gdsr'] = out_df['gt_d'] / out_df['gt_t']
    ok = (out_df['n_diseased'] + out_df['n_healthy'] > 0)
    r_img = float(np.corrcoef(out_df.loc[ok, 'pdsr'], out_df.loc[ok, 'gdsr'])[0, 1])
    print(f'Image-level Pearson r against the whole-spike reference: {r_img:.4f}')


if __name__ == '__main__':
    main()

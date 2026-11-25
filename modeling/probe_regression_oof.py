#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改正版单视图回归 OOF：统一内层划分 + 全检查点（与检测器 OOF 完全对齐）。

相对 probe_ablation/probe_single_view_oof 的两处改正：
  1) 内层验证从「图级 74 张」改为「穗级 37 穗」，与 probe_detector_oof.py 完全一致
     （seed 42*100+f 抽 15% 非测试穗，pd.unique 保持出现顺序）。
  2) 检查点从「best-val 头 + 末轮主干」改为「best-val 同时保存主干和回归头」。

外层沿用 seed 42 穗级五折（shuffle + array_split）。recipe 与 probe_ablation 一致：
  ConvNeXt-T(features[3,5,7]) + 三尺度密度头，MSE 整穗 dsrc，softmax[:,0]，
  每折重初始化 backbone，Adam lr_head=1e-3 lr_bb=1e-4，batch 8，epochs 60，patience 15。

输出逐图 OOF(620)，供 probe_reference_shift.py --reg-npz 重算换参照表。

用法: python -I scripts/probe_regression_oof.py --seed 42 --k 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


class CountingHead(nn.Module):
    def __init__(self, chans=(192, 384, 768), mid=32, out=2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, mid, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(mid, out, 1)) for c in chans
        ])

    def forward(self, f3, f4, f5):
        s = 0
        for head, f in zip(self.heads, (f3, f4, f5)):
            s = s + head(f).mean(dim=(2, 3))
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default=r'build/data/FHBDSR/groundtruth.csv')
    ap.add_argument('--oracle', default=r'build/data/FHBDSR/oracle_single.csv')
    ap.add_argument('--imgdir', default=r'build/source/fhbdsr/YOLO_Dataset/images')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-bb', type=float, default=1e-4)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-npz', default=r'build/runs/probe_regression_oof_s42.npz')
    ap.add_argument('--out-json', default=r'build/runs/probe_regression_oof_s42.json')
    args = ap.parse_args()

    from ultralytics.data.augment import LetterBox

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gt = pd.read_csv(args.gt)
    oracle = pd.read_csv(args.oracle)
    df = gt.merge(oracle[['image', 'single_dsr']], on='image', how='inner').reset_index(drop=True)
    assert len(df) == 620
    img_dir = Path(args.imgdir)
    df['path'] = [str(img_dir / f) for f in df['image']]
    y = df['dsrc'].to_numpy(np.float32)

    lb = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, scaleup=False, stride=32)
    print('预加载图像缓存...', flush=True)
    cache = np.zeros((len(df), 3, args.imgsz, args.imgsz), dtype=np.float32)
    for i, p in enumerate(df['path']):
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = lb(image=im)
        im = np.ascontiguousarray(im.transpose(2, 0, 1)).astype(np.float32) / 255.0
        im = (im - IMAGENET_MEAN) / IMAGENET_STD
        cache[i] = im
    print('  缓存完成', flush=True)

    def load_tensor(idxs):
        return torch.from_numpy(np.ascontiguousarray(cache[idxs])).to(device)

    # ---------- 外层穗级五折（seed 42）----------
    spike_arr = df['spike'].to_numpy()
    spikes = df['spike'].unique()  # 310, 出现顺序
    rng = np.random.default_rng(args.seed)
    rng.shuffle(spikes)
    folds = np.array_split(spikes, args.k)
    fold_of_spike = {int(sp): f for f, arr in enumerate(folds) for sp in arr}
    fold_img = np.array([fold_of_spike[int(s)] for s in spike_arr])  # (620,)

    oof = np.zeros(len(df))
    for f in range(args.k):
        te = np.where(fold_img == f)[0]
        tr_all = np.where(fold_img != f)[0]
        # ---------- 穗级内层验证（与检测器完全一致）----------
        tr_spikes = pd.unique(spike_arr[tr_all])  # 非测试穗，出现顺序
        rng2 = np.random.default_rng(args.seed * 100 + f)
        val_spikes = rng2.choice(tr_spikes, size=max(8, int(0.15 * len(tr_spikes))), replace=False)
        val_set = set(val_spikes.tolist())
        in_val = np.isin(spike_arr[tr_all], list(val_set))
        val = tr_all[in_val]
        tr = tr_all[~in_val]
        print(f'fold {f}: tr={len(tr)}图({len(pd.unique(spike_arr[tr]))}穗) '
              f'val={len(val)}图({len(pd.unique(spike_arr[val]))}穗) test={len(te)}图', flush=True)

        m = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
        layers = [m.features[3], m.features[5], m.features[7]]
        for p in m.parameters():
            p.requires_grad_(True)
        bb_params = [p for p in m.parameters() if p.requires_grad]
        captured = {}

        def make(i):
            def hook(mod, inp, out):
                captured[i] = out
            return hook
        for i, lay in enumerate(layers):
            lay.register_forward_hook(make(i))

        def forward_feats(x):
            m(x)
            return captured[0], captured[1], captured[2]

        head = CountingHead((192, 384, 768)).to(device)
        opt = torch.optim.Adam([
            {'params': head.parameters(), 'lr': args.lr_head},
            {'params': bb_params, 'lr': args.lr_bb},
        ])
        mse = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda', enabled=True)

        def predict(idxs):
            m.eval(); head.eval()
            preds = []
            with torch.no_grad():
                for s in range(0, len(idxs), args.batch):
                    bidx = idxs[s:s + args.batch]
                    x = load_tensor(bidx)
                    with torch.autocast('cuda', dtype=torch.float16):
                        preds.append(torch.softmax(head(*forward_feats(x)), dim=1)[:, 0].float().cpu().numpy())
            return np.concatenate(preds)

        best_val, best_head, best_bb, bad = float('inf'), None, None, 0
        for ep in range(1, args.epochs + 1):
            m.train(); head.train()
            order = tr.copy(); np.random.shuffle(order)
            for s in range(0, len(order), args.batch):
                bidx = order[s:s + args.batch]
                x = load_tensor(bidx)
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = torch.softmax(head(*forward_feats(x)), dim=1)[:, 0]
                    yb = torch.from_numpy(y[bidx]).to(device).to(pred.dtype)
                    loss = mse(pred, yb)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(head.parameters()) + bb_params, 10.0)
                scaler.step(opt)
                scaler.update()
            vp = predict(val)
            vl = float(np.mean((vp - y[val]) ** 2))
            if vl < best_val:
                best_val, best_head, best_bb, bad = vl, \
                    {k: v.clone() for k, v in head.state_dict().items()}, \
                    {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= args.patience:
                break
        # 全检查点：best-val 同时载回主干和回归头
        head.load_state_dict(best_head)
        m.load_state_dict(best_bb)
        te_pred = predict(te)
        oof[te] = te_pred
        print(f'  fold {f}: best_val={best_val:.6f}  r_te={pearson(te_pred, y[te]):.4f}', flush=True)

    r_img = pearson(oof, y)
    d = df[['spike']].copy(); d['p'] = oof; d['y'] = y
    g = d.groupby('spike').agg(p=('p', 'mean'), y=('y', 'first'))
    r_spk = pearson(g['p'], g['y'])
    print(f'\n回归 OOF(seed{args.seed}, 穗级内层验证+全检查点): 图级 r={r_img:.4f}  穗级 r={r_spk:.4f}')

    np.savez(args.out_npz, full=oof, dsrc=y, spike=df['spike'].to_numpy(), side=df['side'].to_numpy())
    Path(args.out_json).write_text(json.dumps(
        {'seed': args.seed, 'k': args.k, 'r_img': r_img, 'r_spike': r_spk},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n落盘: {args.out_npz} + {args.out_json}')


if __name__ == '__main__':
    main()

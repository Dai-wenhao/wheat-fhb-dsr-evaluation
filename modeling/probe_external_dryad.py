#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贡献② 外部跨域泛化检验（T1-A）：FHBDSR 整穗 DSR 回归 → McConachie Dryad 2576 单穗图 zero-shot。

模型：ConvNeXt-T 三尺度密度头（与 probe_oof_crosscheck.py 同 recipe，FHBDSR OOF r≈0.964）。
流程：
  1) 在全部 620 张 FHBDSR 图上训练②（目标=文件名整穗 dsrc，按穗留 15% 早停验证，存 ckpt）；
  2) 零 shot 推理 Dryad 2576 图（不接触 Dryad 训练）；
  3) 报 r(pred, 人标 Severity) + r(pred, McConachie 两自动方法 Threshold/Instance)，附人标与两方法的参照相关。

判据：若 r(pred, Severity) 接近 FHBDSR 域内成绩，说明②不是对 620 图过拟合，能跨相机/背景/品种泛化。
用法: python -I scripts/probe_external_dryad.py --seed 42
"""
import argparse
import json
import sys
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


def tok(v):
    """把 xlsx 单元格规范成文件名 token：去掉浮点尾 .0。"""
    s = str(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


class CountingHead(nn.Module):
    """三尺度密度头：softmax 归一后 [:,0] 当整穗 DSR 回归标量。"""

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
    ap.add_argument('--dryad-xlsx', default=r'build/source/dryad/Final_Single_Head_Data.xlsx')
    ap.add_argument('--dryad-imgdir', default=r'build/source/dryad/Database_Single_Head_Images')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-bb', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt', default=r'build/runs/probe_external_dryad_ckpt.pt')
    ap.add_argument('--out', default=r'build/runs/probe_external_dryad.json')
    args = ap.parse_args()

    from ultralytics.data.augment import LetterBox

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- 1. 训练 ② on FHBDSR ----------
    gt = pd.read_csv(args.gt)
    oracle = pd.read_csv(args.oracle)
    df = gt.merge(oracle[['image', 'single_dsr']], on='image', how='inner').reset_index(drop=True)
    assert len(df) == 620, f'FHBDSR 行数异常 {len(df)}'
    img_dir = Path(args.imgdir)
    df['path'] = [str(img_dir / f) for f in df['image']]
    y = df['dsrc'].to_numpy(np.float32)

    lb = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, scaleup=False, stride=32)
    print('预加载 FHBDSR 图像缓存...', flush=True)
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

    m = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    layers = [m.features[3], m.features[5], m.features[7]]
    chans = (192, 384, 768)
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

    head = CountingHead(chans).to(device)
    opt = torch.optim.Adam([
        {'params': head.parameters(), 'lr': args.lr_head},
        {'params': bb_params, 'lr': args.lr_bb},
    ])
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # 按穗留 15% 早停验证，其余全训练（zero-shot 用全量）
    spikes = df['spike'].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(spikes)
    n_val = max(8, int(0.15 * len(spikes)))
    val_spikes = set(spikes[:n_val].tolist())
    val_idx = df['spike'].isin(val_spikes).to_numpy()
    tr_idx = np.where(~val_idx)[0]
    va_idx = np.where(val_idx)[0]

    def predict(idxs):
        m.eval()
        head.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(idxs), args.batch):
                bidx = idxs[s:s + args.batch]
                x = load_tensor(bidx)
                with torch.autocast('cuda', dtype=torch.float16):
                    preds.append(torch.softmax(head(*forward_feats(x)), dim=1)[:, 0].float().cpu().numpy())
        pred = np.concatenate(preds)
        return float(np.mean((pred - y[idxs]) ** 2)), pred

    best_val, bad = float('inf'), 0
    for ep in range(1, args.epochs + 1):
        m.train()
        head.train()
        order = tr_idx.copy()
        np.random.shuffle(order)
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
        vl, _ = predict(va_idx)
        if vl < best_val:
            best_val, bad = vl, 0
            torch.save({'backbone': m.state_dict(), 'head': head.state_dict()}, args.ckpt)
        else:
            bad += 1
        if bad >= args.patience:
            print(f'  早停 @ ep{ep}', flush=True)
            break
    ck = torch.load(args.ckpt, map_location=device)
    m.load_state_dict(ck['backbone'])
    head.load_state_dict(ck['head'])
    print(f'FHBDSR 训练完成，ckpt={args.ckpt}', flush=True)

    # ---------- 2. zero-shot 推理 Dryad ----------
    xl = pd.read_excel(args.dryad_xlsx)
    print(f'\nDryad xlsx 列: {list(xl.columns)}', flush=True)
    dai = xl['DAI'].map(lambda v: tok(v).upper().replace('DAI', '').strip())
    xl['fname'] = (xl['Isolate'].map(tok) + '_' + xl['Cultivar'].map(tok) + '_' +
                   xl['Replication'].map(tok) + '_' + xl['Head'].map(tok) + '_' +
                   dai + 'DAI.png')
    dry_dir = Path(args.dryad_imgdir)
    xl['path'] = [str(dry_dir / f) for f in xl['fname']]
    exists = xl['path'].map(lambda p: Path(p).exists()).to_numpy()
    print(f'Dryad 行数={len(xl)}  图片命中={int(exists.sum())}', flush=True)
    print('  样例文件名:', xl['fname'].head(3).tolist(), flush=True)
    xl = xl[exists].reset_index(drop=True)

    m.eval()
    head.eval()
    preds = np.zeros(len(xl), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(xl), args.batch):
            e = min(s + args.batch, len(xl))
            ims = []
            for p in xl['path'].iloc[s:e]:
                im = cv2.imread(p)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = lb(image=im)
                im = np.ascontiguousarray(im.transpose(2, 0, 1)).astype(np.float32) / 255.0
                im = (im - IMAGENET_MEAN) / IMAGENET_STD
                ims.append(im)
            x = torch.from_numpy(np.ascontiguousarray(np.stack(ims))).to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                preds[s:e] = torch.softmax(head(*forward_feats(x)), dim=1)[:, 0].float().cpu().numpy()

    sev = pd.to_numeric(xl['Severity'], errors='coerce').to_numpy(np.float32)
    thr = pd.to_numeric(xl['Threshold Sev (%)'], errors='coerce').to_numpy(np.float32)
    ins = pd.to_numeric(xl['Instance Sev (%)'], errors='coerce').to_numpy(np.float32)

    def r_valid(p, t):
        ok = np.isfinite(t)
        return pearson(p[ok], t[ok]), int(ok.sum())

    print('\n' + '=' * 72)
    print('② 跨域 zero-shot：FHBDSR 训练 → Dryad 推理')
    print('=' * 72)
    r_ps, n_s = r_valid(preds, sev)
    r_pt, n_t = r_valid(preds, thr)
    r_pi, n_i = r_valid(preds, ins)
    r_st, _ = r_valid(sev, thr)
    r_si, _ = r_valid(sev, ins)
    r_ti, _ = r_valid(thr, ins)
    print(f'  r(②pred, 人标 Severity)       = {r_ps:.4f}  (n={n_s})')
    print(f'  r(②pred, McConachie Threshold) = {r_pt:.4f}  (n={n_t})')
    print(f'  r(②pred, McConachie Instance)  = {r_pi:.4f}  (n={n_i})')
    print(f'  参照: r(Severity,Threshold)={r_st:.4f}  r(Severity,Instance)={r_si:.4f}  r(Threshold,Instance)={r_ti:.4f}')

    out = {
        'seed': args.seed, 'n_fhbdsr': int(len(df)), 'n_dryad': int(len(xl)),
        'r_pred_severity': r_ps, 'r_pred_threshold': r_pt, 'r_pred_instance': r_pi,
        'ref_sev_thr': r_st, 'ref_sev_ins': r_si, 'ref_thr_ins': r_ti,
        'n_sev': n_s, 'n_thr': n_t, 'n_ins': n_i,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n落盘: {args.out}')


if __name__ == '__main__':
    main()

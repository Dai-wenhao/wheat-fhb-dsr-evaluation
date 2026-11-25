#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 1：Dryad 同穗病情变化测量。

问题：单图模型能否测准同一麦穗病情的"变化"（相邻 DAI 之间的病小穗率变化量），
而不仅是测准病情的"水平"。

做法：
  1) 用 FHBDSR 零 shot 模型对全部 Dryad 图逐图预测，落盘逐图 CSV；
  2) 按穗（Isolate_Cultivar_Replication_Head）配对，算相邻 DAI 的
     模型变化量 Δm 与人工变化量 Δs（Severity/100）；
  3) 时间基线：不看图像，用各时间段的人工平均变化量预测测试穗；
  4) 主报 corr(Δm, Δs)、MAE(Δm, Δs)、MAE(时间基线, Δs)、偏差，
     按时间段分开报，附穗级 bootstrap 95% CI。

判读：模型变化量的 MAE 显著小于时间基线，即图像提供了超出群体平均进展的个体信息。

用法: python -I scripts/probe_dryad_progression.py
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
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def tok(v):
    s = str(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


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


def build_model(device):
    m = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    layers = [m.features[3], m.features[5], m.features[7]]
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
    return m, head, forward_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'build/runs/probe_external_dryad_ckpt.pt')
    ap.add_argument('--xlsx', default=r'build/source/dryad/Final_Single_Head_Data.xlsx')
    ap.add_argument('--imgdir', default=r'build/source/dryad/Database_Single_Head_Images')
    ap.add_argument('--cache', default=r'build/runs/dryad_preproc_fp16.npy')
    ap.add_argument('--perimage', default=r'build/runs/dryad_perimage_pred.csv')
    ap.add_argument('--out', default=r'build/runs/dryad_progression.json')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # ---------- 读 Dryad，构建穗 ID ----------
    xl = pd.read_excel(args.xlsx)
    xl['spike_id'] = (xl['Isolate'].map(tok) + '_' + xl['Cultivar'].map(tok) + '_' +
                      xl['Replication'].map(tok) + '_' + xl['Head'].map(tok))
    xl['sev'] = pd.to_numeric(xl['Severity'], errors='coerce') / 100.0
    dai = xl['DAI'].map(lambda v: tok(v).upper().replace('DAI', '').strip())
    xl['dai'] = dai
    xl['fname'] = (xl['Isolate'].map(tok) + '_' + xl['Cultivar'].map(tok) + '_' +
                   xl['Replication'].map(tok) + '_' + xl['Head'].map(tok) + '_' +
                   dai + 'DAI.png')
    dry_dir = Path(args.imgdir)
    xl['path'] = [str(dry_dir / f) for f in xl['fname']]
    exists = xl['path'].map(lambda p: Path(p).exists()).to_numpy()
    xl = xl[exists].reset_index(drop=True)
    N = len(xl)
    print(f'Dryad 图片命中 N={N}')

    # ---------- 逐图预测（缓存到 CSV）----------
    perimage_path = Path(args.perimage)
    if perimage_path.exists():
        print(f'复用逐图预测 {perimage_path}')
        pred_df = pd.read_csv(perimage_path)
    else:
        from ultralytics.data.augment import LetterBox
        lb = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, scaleup=False, stride=32)

        ck = torch.load(args.ckpt, map_location=device)
        m, head, forward_feats = build_model(device)
        m.load_state_dict(ck['backbone'])
        head.load_state_dict(ck['head'])
        m.eval()
        head.eval()

        cache_path = Path(args.cache)
        if cache_path.exists():
            print(f'复用预处理缓存 {cache_path}')
            X = np.load(cache_path, mmap_mode='r')
            assert X.shape == (N, 3, args.imgsz, args.imgsz), f'缓存 shape 不匹配 {X.shape}'
        else:
            print('预处理 Dryad 图（无缓存）...')
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            X = np.lib.format.open_memmap(
                cache_path, mode="w+", dtype=np.float16, shape=(N, 3, args.imgsz, args.imgsz)
            )
            for i, p in enumerate(xl['path']):
                im = cv2.imread(p)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = lb(image=im)
                im = np.ascontiguousarray(im.transpose(2, 0, 1)).astype(np.float32) / 255.0
                im = ((im - IMAGENET_MEAN) / IMAGENET_STD).reshape(3, args.imgsz, args.imgsz)
                X[i] = im.astype(np.float16)
            X.flush()

        preds = np.zeros(N, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, N, args.batch):
                e = min(s + args.batch, N)
                x = torch.from_numpy(np.ascontiguousarray(X[s:e]).astype(np.float32)).to(device)
                with torch.autocast('cuda', dtype=torch.float16):
                    p = torch.softmax(head(*forward_feats(x)), dim=1)[:, 0].float().cpu().numpy()
                preds[s:e] = p
        pred_df = xl[['spike_id', 'dai', 'sev']].copy()
        pred_df['pred'] = preds
        pred_df.to_csv(perimage_path, index=False, encoding='utf-8-sig')
        print(f'逐图预测落盘 {perimage_path}')

    # ---------- 按穗配对，算相邻变化量 ----------
    pred_df['dai_num'] = pd.to_numeric(pred_df['dai'], errors='coerce')
    rows = []
    for sid, g in pred_df.groupby('spike_id'):
        g = g.sort_values('dai_num')
        for i in range(len(g) - 1):
            t1, t2 = g.iloc[i], g.iloc[i + 1]
            rows.append(dict(
                spike_id=sid, period=f'{t1["dai"]}->{t2["dai"]}',
                dm=t2['pred'] - t1['pred'],
                ds=t2['sev'] - t1['sev'],
            ))
    chg = pd.DataFrame(rows)
    chg = chg[np.isfinite(chg['ds']) & np.isfinite(chg['dm'])].reset_index(drop=True)
    print(f'配对变化量条数 = {len(chg)}，时间段 = {sorted(chg["period"].unique())}')

    # ---------- 时间基线：各时间段人工平均变化量 ----------
    base = chg.groupby('period')['ds'].mean().to_dict()

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def bias(a, b):
        return float(np.mean(a - b))

    def report(sub, label):
        r = pearson(sub['dm'], sub['ds'])
        m_model = mae(sub['dm'], sub['ds'])
        # 时间基线预测 = 该时间段平均变化量
        base_pred = sub['period'].map(base).to_numpy()
        m_base = mae(base_pred, sub['ds'])
        b = bias(sub['dm'], sub['ds'])
        # 穗级 bootstrap CI（按穗重采样）
        rng = np.random.default_rng(42)
        spikes = sub['spike_id'].unique()
        r_boot, m_boot = [], []
        for _ in range(500):
            ids = rng.choice(spikes, size=len(spikes), replace=True)
            ss = sub[sub['spike_id'].isin(ids)]
            r_boot.append(pearson(ss['dm'], ss['ds']))
            m_boot.append(mae(ss['dm'], ss['ds']))
        r_lo, r_hi = np.percentile(r_boot, [2.5, 97.5])
        print(f'  {label}: n={len(sub)}  corr={r:.3f} [CI {r_lo:.3f},{r_hi:.3f}]  '
              f'MAE_model={m_model:.4f}  MAE_time={m_base:.4f}  bias={b:+.4f}')
        return dict(n=int(len(sub)), corr=r, corr_ci=[float(r_lo), float(r_hi)],
                    mae_model=m_model, mae_time=m_base, bias=b)

    print('\n=== Dryad 同穗病情变化测量 ===')
    res = {'overall': report(chg, '整体')}
    res['per_period'] = {}
    for p in sorted(chg['period'].unique()):
        res['per_period'][p] = report(chg[chg['period'] == p], p)

    # 判读
    print('\n[判读]')
    m_model = res['overall']['mae_model']
    m_base = res['overall']['mae_time']
    r = res['overall']['corr']
    if m_model < m_base * 0.95 and r > 0.2:
        print(f'  模型变化量 MAE={m_model:.4f} < 时间基线 {m_base:.4f}，且 corr={r:.3f}>0.2')
        print('  → 图像提供了超出群体平均进展的个体信息，单图能追踪个体病情进展')
    elif r <= 0.2:
        print(f'  模型变化量 corr={r:.3f} 很低，MAE={m_model:.4f} ≈ 时间基线 {m_base:.4f}')
        print('  → 模型在该数据上能分病情高低，但对单穗进展的测量能力有限')
    else:
        print(f'  MAE_model={m_model:.4f} vs MAE_time={m_base:.4f}，corr={r:.3f}')
        print('  → 中等：需看分时间段结果')

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n落盘: {args.out}')


if __name__ == '__main__':
    main()

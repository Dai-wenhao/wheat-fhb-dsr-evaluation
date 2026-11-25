#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验2 主验证：Dryad 轮换留出菌株 fine-tune（5 组 × 5 菌株，每个菌株恰好被测一次，
由「从未在训练或调参中见过它」的模型测试）。同一穗的所有时点始终在同一划分。

设计：
  25 菌株按 split-seed 洗牌 → 5 组 × 5 菌株。第 f 折：test=组f，val=组(f+1)%5，train=其余3组。
  每折从 FHBDSR ckpt 重新初始化，fine-tune 目标=人标 Severity（/100），早停于 val MSE。
  落盘每折 test 菌株逐图预测，供 probe_dryad_rotation_analyze.py 回答三问。

zero-shot 复用 runs/dryad_perimage_pred.csv（FHBDSR 权重，全 2576 图，常量，跨折不变）。

用法: python -I scripts/probe_dryad_rotation_ft.py --n-groups 5 --split-seed 42 --train-seed 42
"""
import argparse
import json
import sys
import time
import traceback
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

LOG = None


def log(msg):
    msg = str(msg)
    print(msg, flush=True)
    if LOG is not None:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


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
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--dryad-xlsx', default=r'build/source/dryad/Final_Single_Head_Data.xlsx')
    ap.add_argument('--dryad-imgdir', default=r'build/source/dryad/Database_Single_Head_Images')
    ap.add_argument('--ckpt', default=r'build/runs/probe_external_dryad_ckpt.pt')
    ap.add_argument('--cache', default=r'build/runs/dryad_preproc_fp16.npy')
    ap.add_argument('--out', default=r'build/runs/probe_dryad_rotation_ft.json')
    ap.add_argument('--log', default=r'build/runs/probe_dryad_rotation_ft.log')
    ap.add_argument('--perimage-out', default=r'build/runs/dryad_perimage_pred_rotation_ft.csv')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-bb', type=float, default=1e-4)
    ap.add_argument('--n-groups', type=int, default=5, help='轮换组数（25 菌株 / 5 = 每折 test 5 菌株）')
    ap.add_argument('--split-seed', type=int, default=42, help='控菌株分组洗牌')
    ap.add_argument('--train-seed', type=int, default=42, help='控训练随机性')
    args = ap.parse_args()

    LOG = args.log
    Path(LOG).write_text('', encoding='utf-8')

    from ultralytics.data.augment import LetterBox

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # ---------- 读 Dryad，构建穗 ID + 路径 ----------
    xl = pd.read_excel(args.dryad_xlsx)
    dai = xl['DAI'].map(lambda v: tok(v).upper().replace('DAI', '').strip())
    xl['spike_id'] = (xl['Isolate'].map(tok) + '_' + xl['Cultivar'].map(tok) + '_' +
                      xl['Replication'].map(tok) + '_' + xl['Head'].map(tok))
    xl['dai'] = dai
    xl['fname'] = (xl['Isolate'].map(tok) + '_' + xl['Cultivar'].map(tok) + '_' +
                   xl['Replication'].map(tok) + '_' + xl['Head'].map(tok) + '_' +
                   dai + 'DAI.png')
    dry_dir = Path(args.dryad_imgdir)
    xl['path'] = [str(dry_dir / f) for f in xl['fname']]
    exists = xl['path'].map(lambda p: Path(p).exists()).to_numpy()
    xl = xl[exists].reset_index(drop=True)
    N = len(xl)
    log(f'Dryad 图片命中 N={N}')

    sev = pd.to_numeric(xl['Severity'], errors='coerce').to_numpy(np.float32) / 100.0
    iso_arr = xl['Isolate'].to_numpy()

    # ---------- 预处理缓存 ----------
    cache_path = Path(args.cache)
    assert cache_path.exists(), f'预处理缓存缺失 {cache_path}'
    X = np.load(cache_path, mmap_mode='r')
    assert X.shape == (N, 3, args.imgsz, args.imgsz), f'缓存 shape 不匹配 {X.shape}'
    log(f'复用预处理缓存 {cache_path}')

    def load_batch(idxs):
        return torch.from_numpy(np.ascontiguousarray(X[idxs]).astype(np.float32)).to(device)

    # ---------- 轮换分组 ----------
    isolates = np.array(sorted(set(iso_arr.tolist())))
    assert len(isolates) == 25, f'菌株数 {len(isolates)} != 25'
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(isolates)
    groups = np.array_split(isolates, args.n_groups)
    group_of = {iso: i for i, grp in enumerate(groups) for iso in grp}
    for i, grp in enumerate(groups):
        log(f'  组{i}: {sorted(grp.tolist())}')
    iso_idx = iso_arr

    # ---------- 逐折 fine-tune ----------
    per_rows = []
    summary = {'n_groups': args.n_groups, 'split_seed': args.split_seed, 'train_seed': args.train_seed,
               'folds': []}

    for f in range(args.n_groups):
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        test_iso = set(groups[f].tolist())
        val_iso = set(groups[(f + 1) % args.n_groups].tolist())
        train_iso = set(isolates.tolist()) - test_iso - val_iso
        te = np.where(np.isin(iso_idx, list(test_iso)))[0]
        va = np.where(np.isin(iso_idx, list(val_iso)))[0]
        tr = np.where(np.isin(iso_idx, list(train_iso)))[0]
        log(f'\n===== fold {f} ===== test={sorted(test_iso)} val={sorted(val_iso)} '
            f'train={len(train_iso)}株 ({len(tr)}图) val={len(va)}图 test={len(te)}图')

        # 每折从 FHBDSR ckpt 重新初始化（干净）
        ck = torch.load(args.ckpt, map_location=device)
        m, head, forward_feats = build_model(device)
        m.load_state_dict(ck['backbone'])
        head.load_state_dict(ck['head'])

        def predict(idxs):
            m.eval()
            head.eval()
            preds = np.zeros(len(idxs), dtype=np.float32)
            with torch.no_grad():
                for s in range(0, len(idxs), args.batch):
                    bidx = idxs[s:s + args.batch]
                    x = load_batch(bidx)
                    with torch.autocast('cuda', dtype=torch.float16):
                        out = torch.softmax(head(*forward_feats(x)), dim=1)[:, 0].float().cpu().numpy()
                    preds[s:s + len(bidx)] = out
            return preds

        def mse_on(idxs):
            m.eval()
            head.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(idxs), args.batch):
                    bidx = idxs[s:s + args.batch]
                    x = load_batch(bidx)
                    yb = torch.from_numpy(sev[bidx]).to(device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        p = torch.softmax(head(*forward_feats(x)), dim=1)[:, 0]
                        tot += float(nn.functional.mse_loss(p.float(), yb)) * len(bidx)
                    cnt += len(bidx)
            return tot / cnt

        for p in m.parameters():
            p.requires_grad_(True)
        bb_params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam([
            {'params': head.parameters(), 'lr': args.lr_head},
            {'params': bb_params, 'lr': args.lr_bb},
        ])
        mse = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda', enabled=True)

        best_val, bad, best_head, best_bb = float('inf'), 0, None, None
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            m.train()
            head.train()
            order = tr.copy()
            np.random.shuffle(order)
            for s in range(0, len(order), args.batch):
                bidx = order[s:s + args.batch]
                x = load_batch(bidx)
                yb = torch.from_numpy(sev[bidx]).to(device)
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = torch.softmax(head(*forward_feats(x)), dim=1)[:, 0]
                    loss = mse(pred, yb.to(pred.dtype))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(head.parameters()) + bb_params, 10.0)
                scaler.step(opt)
                scaler.update()
            vl = mse_on(va)
            if vl < best_val:
                best_val, bad, best_head, best_bb = vl, 0, \
                    {k: v.clone() for k, v in head.state_dict().items()}, \
                    {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
            if ep % 5 == 0 or ep == 1:
                log(f'  ep {ep:3d}  val_mse={vl:.4f}  best={best_val:.4f}  ({time.time()-t0:.0f}s)')
            if bad >= args.patience:
                log(f'  早停 @ ep{ep}')
                break

        head.load_state_dict(best_head)
        m.load_state_dict(best_bb)
        pred_ft = predict(te)

        te_xl = xl.iloc[te]
        for j in range(len(te)):
            per_rows.append({
                'spike_id': te_xl.iloc[j]['spike_id'],
                'dai': te_xl.iloc[j]['dai'],
                'sev': float(sev[te[j]]),
                'isolate': te_xl.iloc[j]['Isolate'],
                'fold': f,
                'pred_ft': float(pred_ft[j]),
            })
        summary['folds'].append({'fold': f, 'n_test_img': int(len(te)), 'best_val_mse': best_val})
        log(f'  fold {f} 完成: best_val_mse={best_val:.4f}  test图={len(te)}')

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(args.perimage_out, index=False, encoding='utf-8-sig')
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'\n落盘: {args.perimage_out} (n={len(per_df)}) + {args.out}')
    log('完成轮换 fine-tune')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('=' * 72, flush=True)
        traceback.print_exc()
        if LOG is not None:
            with open(LOG, 'a', encoding='utf-8') as f:
                f.write('\n' + '=' * 72 + '\n')
                traceback.print_exc(file=f)
        sys.exit(1)

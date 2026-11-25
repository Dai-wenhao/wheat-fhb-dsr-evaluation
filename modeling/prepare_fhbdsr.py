#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FHBDSR-Net 数据准备（第 0 步）：
  1) 从文件名解析真值 —— 文件名 `[穗号]+[面]+[总小穗数].[病小穗数]` 是人工数出的 DSR 真值，
     框标注只覆盖 ~53%，不能当计数真值；
  2) 按「穗号」分层切分 train/val/test —— 同一穗的两面(+1/+2)必须同 split，否则真值泄漏。
输出 data/FHBDSR/：groundtruth.csv / splits.json / train.txt / val.txt / test.txt / data.yaml
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def parse_stem(stem: str):
    """解析文件名 stem -> (spike, side, total, diseased)。容错两个坏文件名。"""
    m = re.search(r'\.(\d+)$', stem)              # 病小穗数 = 末尾 .数字
    diseased = int(m.group(1))
    rest = stem[:m.start()]
    m2 = re.search(r'(\d+)$', rest)               # 总小穗数 = 病数前最后一个数字
    total = int(m2.group(1))
    nums = re.findall(r'\d+', rest[:m2.start()])  # 穗号、面
    return int(nums[0]), int(nums[1]), total, diseased


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'build/source/fhbdsr')
    ap.add_argument('--out', default=r'build/data/FHBDSR')
    ap.add_argument('--ratios', default='0.8,0.1,0.1', help='train,val,test 比例')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(',')]
    assert abs(sum(ratios) - 1.0) < 1e-6, 'ratios 须和为 1'

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = data / 'YOLO_Dataset' / 'images'

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == '.jpg')
    if len(images) != 620:
        ap.error(f"Expected 620 FHBDSR images in {img_dir}; found {len(images)}.")
    rows = []
    for img in images:
        stem = img.stem
        spike, side, total, diseased = parse_stem(stem)
        rows.append(dict(image=img.name, stem=stem, spike=spike, side=side,
                         total=total, diseased=diseased,
                         dsrc=diseased / total,
                         malformed=re.fullmatch(r'\d+\+\d+\+\d+\.\d+', stem) is None))
    df = pd.DataFrame(rows)

    # 按穗分层切分（同一穗两面必须同 split，否则整穗真值泄漏），8:1:1 在穗级
    rng = random.Random(args.seed)
    spikes = sorted(int(s) for s in df['spike'].unique())
    rng.shuffle(spikes)
    n = len(spikes)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    split_of_spike = {}
    for s in spikes[:n_train]:
        split_of_spike[s] = 'train'
    for s in spikes[n_train:n_train + n_val]:
        split_of_spike[s] = 'val'
    for s in spikes[n_train + n_val:]:
        split_of_spike[s] = 'test'

    df['split'] = df['spike'].map(split_of_spike)
    df = df.sort_values(['spike', 'side']).reset_index(drop=True)

    df.to_csv(out / 'groundtruth.csv', index=False)
    splits = {k: sorted([s for s, v in split_of_spike.items() if v == k])
              for k in ('train', 'val', 'test')}
    (out / 'splits.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')

    for k in ('train', 'val', 'test'):
        imgs = df[df['split'] == k]['image'].tolist()
        (out / f'{k}.txt').write_text(
            '\n'.join(str(img_dir / im) for im in imgs) + '\n', encoding='utf-8')

    (out / 'data.yaml').write_text(
        f"path: {data / 'YOLO_Dataset'}\n"
        f"train: {out / 'train.txt'}\n"
        f"val: {out / 'val.txt'}\n"
        f"test: {out / 'test.txt'}\n"
        f"nc: 2\nnames: ['healthy', 'diseased']\n",
        encoding='utf-8')

    per_spike = df.groupby('spike').first()  # 仅用于穗级真值统计

    print('=' * 60)
    print('FHBDSR-Net 数据准备报告')
    print('=' * 60)
    print(f'图片总数: {len(df)}  (穗数 {len(per_spike)}, 每穗 2 面)')
    print('坏文件名 (2 个, 已容错解析):')
    for _, r in df[df['malformed']].iterrows():
        print(f'   {r["image"]:20s} -> 穗{r["spike"]} 面{r["side"]} 总{r["total"]} 病{r["diseased"]}  DSR={r["dsrc"]:.3f}')
    print(f'病小穗真值: 总 {per_spike["diseased"].sum()}, 均值 {per_spike["diseased"].mean():.2f}, '
          f'范围 [{per_spike["diseased"].min()}, {per_spike["diseased"].max()}]')
    print(f'总小穗真值: 均值 {per_spike["total"].mean():.2f}, '
          f'范围 [{per_spike["total"].min()}, {per_spike["total"].max()}]')
    print(f'DSR 真值: 均值 {per_spike["dsrc"].mean():.3f}, 中位 {per_spike["dsrc"].median():.3f}')
    print('-' * 60)
    for k in ('train', 'val', 'test'):
        sub = df[df['split'] == k]
        print(f'{k:5s}: 图 {len(sub):3d}  病小穗均值 {sub["diseased"].mean():.2f}  DSR均值 {sub["dsrc"].mean():.3f}')
    print('=' * 60)
    print(f'输出目录: {out}')


if __name__ == '__main__':
    main()

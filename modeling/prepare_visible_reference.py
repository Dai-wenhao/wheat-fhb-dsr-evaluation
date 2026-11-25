"""Convert YOLO box annotations to the visible spikelet reference."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("build/data/FHBDSR/groundtruth.csv"))
    parser.add_argument("--labels", type=Path, default=Path("build/source/fhbdsr/YOLO_Dataset/labels"))
    parser.add_argument("--out", type=Path, default=Path("build/data/FHBDSR/oracle_single.csv"))
    args = parser.parse_args()

    reference = pd.read_csv(args.gt, usecols=["image", "spike", "side", "total", "diseased", "dsrc"])
    rows = []
    for image in reference["image"]:
        label = args.labels / (Path(image).stem + ".txt")
        counts = [0, 0]
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            category = int(line.split()[0])
            if category not in (0, 1):
                raise ValueError(f"Unexpected class in {label}")
            counts[category] += 1
        if sum(counts) == 0:
            raise ValueError(f"No annotated spikelets in {label}")
        rows.append((counts[1], counts[0], sum(counts), counts[1] / sum(counts)))
    visible = pd.DataFrame(rows, columns=["n_diseased", "n_healthy", "n_total", "single_dsr"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([reference, visible], axis=1).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

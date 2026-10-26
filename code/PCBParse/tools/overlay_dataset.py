#!/usr/bin/env python3
"""Draw the labels of N random dataset cases on their images, for a visual spot check.

    python tools/overlay_dataset.py /mnt/c/EDA/dataset_v9 --n 20 --out /mnt/c/EDA/check_v9
    python tools/overlay_dataset.py /mnt/c/EDA/dataset_v9 --cases case_0054 case_0082 --out ...
"""
import argparse
import glob
import json
import os
import random
import sys

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'tools'))
from preannotate import overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ids = a.cases or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.dataset, 'case_*.json')))
    if not a.cases:
        random.Random(a.seed).shuffle(ids)
        ids = sorted(ids[:a.n])
    os.makedirs(a.out, exist_ok=True)
    for cid in ids:
        img = cv2.imread(os.path.join(a.dataset, cid + '.png'))
        if img is None:
            print('missing image', cid)
            continue
        J = json.load(open(os.path.join(a.dataset, cid + '.json'), encoding='utf-8'))
        ov = overlay(img[:, :, ::-1], J)
        cv2.imwrite(os.path.join(a.out, cid + '_overlay.png'), ov)
        print(cid, len(J['components']), 'components', len(J['nets']), 'nets')


if __name__ == '__main__':
    main()

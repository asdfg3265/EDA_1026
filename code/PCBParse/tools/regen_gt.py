#!/usr/bin/env python3
"""Regenerate the labels (JSON) of an existing dataset from the raw bundles, without
re-rendering the images (same crop / scale as build_dataset, recomputed from the seeds).

    python tools/regen_gt.py --raw /mnt/c/EDA/raw --dataset /mnt/c/EDA/dataset_v9

Old labels are moved to <dataset>/_gt_old/.  Cases whose image size no longer matches are
reported and left untouched.  Run tools/validate_dataset.py (or score_dataset) afterwards.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from easypro.model import load_bundle  # noqa: E402
from easypro.make import make_gt_only  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--bundles', default=None, help='bundle folder (default <raw>/bundles)')
    a = ap.parse_args()
    bdir = a.bundles or os.path.join(a.raw, 'bundles')
    rows = list(csv.DictReader(open(os.path.join(a.dataset, 'cases.csv'), encoding='utf-8-sig')))
    want = {r['project'] for r in rows}
    by_url = {}
    for bf in sorted(glob.glob(os.path.join(bdir, '*.json'))):
        try:
            b = json.load(open(bf, encoding='utf-8'))
        except Exception:
            continue
        url = (b.get('project') or {}).get('url')
        if url in want and url not in by_url:
            by_url[url] = bf
    old = os.path.join(a.dataset, '_gt_old')
    os.makedirs(old, exist_ok=True)
    done = changed = miss = 0
    cache = {}
    for r in rows:
        cid = r['case_id']
        bf = by_url.get(r['project'])
        if bf is None:
            print(cid, 'bundle not found'); miss += 1
            continue
        if bf not in cache:
            cache.clear()
            b = json.load(open(bf, encoding='utf-8'))
            cache[bf] = (b, load_bundle(b))
        bundle, sheets = cache[bf]
        proj = bundle.get('project') or {}
        res = None
        for si, (title, sh) in enumerate(sheets):
            if (title or '') != (r['sheet'] or ''):
                continue
            rng = random.Random(hashlib.md5(f"{proj.get('uuid')}{si}".encode()).hexdigest())
            scale = rng.uniform(1.90, 2.05)
            try:
                out = make_gt_only(sh, scale=scale, seed=rng.random())
            except Exception as e:
                print(cid, 'failed', e)
                continue
            if out and out[1] == (int(r['width']), int(r['height'])):
                res = out[0]
                break
        if res is None:
            print(cid, 'no sheet with the same size'); miss += 1
            continue
        js = os.path.join(a.dataset, cid + '.json')
        new = json.dumps(res, ensure_ascii=False, indent=1)
        if os.path.exists(js):
            cur = open(js, encoding='utf-8').read()
            if cur != new:
                shutil.copy2(js, os.path.join(old, cid + '.json'))
                changed += 1
        with open(js, 'w', encoding='utf-8') as f:
            f.write(new)
        done += 1
    print(f'regenerated {done}, changed {changed}, not found {miss}')


if __name__ == '__main__':
    main()

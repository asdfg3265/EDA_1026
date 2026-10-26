#!/usr/bin/env python3
"""Score a generated dataset with the parser in parallel (one process per case, crash/OOM safe).

    python tools/score_dataset.py /mnt/c/EDA/dataset_v7 --workers 8 [--limit 600] [--seed 1]

Writes <dataset>/parser_scores.csv (case_id,score,comp,pin,conn,wire,n_components) and prints
the mean.  Already scored cases are skipped, so the command can be interrupted and resumed.
--limit N scores a random sample of N cases (useful for a quick estimate).
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

CODE = ('import sys,json; sys.path.insert(0, sys.argv[3]); from pcbparse.pipeline import parse_image; '
        'json.dump(parse_image(sys.argv[1]), open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)')


def run_one(ds, cid, pred_dir, timeout):
    from pcbparse.scorer_official import score_case
    pj = os.path.join(pred_dir, cid + '_Pred.json')
    env = dict(os.environ, OMP_NUM_THREADS='1', PCB_OCR_THREADS='1')
    try:
        subprocess.run([sys.executable, '-c', CODE, os.path.join(ds, cid + '.png'), pj, HERE],
                       timeout=timeout, check=True, capture_output=True, env=env)
        P = json.load(open(pj, encoding='utf-8'))
    except Exception as e:
        P = {'components': {}, 'pins': {}, 'nets': {}}
    G = json.load(open(os.path.join(ds, cid + '.json'), encoding='utf-8'))
    r = score_case(G, P)
    return [cid, round(r['score'], 2), round(r['comp'], 4), round(r['pin'], 4), round(r['conn'], 4),
            round(r['wire'], 4), len(G['components'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=600)
    a = ap.parse_args()
    ds = a.dataset
    ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ds, 'case_*.json')))
    if a.limit:
        random.Random(a.seed).shuffle(ids)
        ids = sorted(ids[:a.limit])
    out_csv = os.path.join(ds, 'parser_scores.csv')
    done = {}
    if os.path.exists(out_csv):
        for row in csv.reader(open(out_csv, encoding='utf-8')):
            if row and row[0].startswith('case_'):
                done[row[0]] = row
    todo = [c for c in ids if c not in done]
    pred_dir = os.path.join(ds, '_pred')
    os.makedirs(pred_dir, exist_ok=True)
    new = not os.path.exists(out_csv)
    f = open(out_csv, 'a', newline='', encoding='utf-8')
    w = csv.writer(f)
    if new:
        w.writerow(['case_id', 'score', 'comp', 'pin', 'conn', 'wire', 'n_components'])
    print(f'{len(ids)} cases, {len(done)} already scored, {len(todo)} to go, {a.workers} workers', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, ds, c, pred_dir, a.timeout) for c in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            row = fu.result()
            w.writerow(row)
            f.flush()
            done[row[0]] = row
            if k % 20 == 0 or k == len(todo):
                sc = [float(r[1]) for c, r in done.items() if c in set(ids)]
                print(f'[{k}/{len(todo)}] mean so far {sum(sc) / len(sc):.2f}', flush=True)
    f.close()
    sc = [float(done[c][1]) for c in ids if c in done]
    print('mean', round(sum(sc) / max(1, len(sc)), 2), 'n', len(sc))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Turn downloaded EasyEDA Pro bundles into competition cases (PNG + GT JSON).

    python tools/build_dataset.py --raw F:/EDA比赛六/raw --out F:/EDA比赛六/dataset \
        [--train /path/to/200_train_cases] [--score] [--start 1]

Each output case: <out>/case_XXXX.png, <out>/case_XXXX.json; <out>/cases.csv lists the
source project, licence and author (needed for the data description document).
Cases are kept only if they satisfy the official validity rules (>=1 IC with enough
connected pins, >=5 components, >=5 nets), are not near-duplicates of each other or of
the 200 official training cases.  With --score the parser is run on every case and its
official-protocol score is recorded (useful to pick cases for the test-set window).
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from easypro.model import load_bundle  # noqa: E402
from easypro.make import make_case  # noqa: E402
from easypro.raster import Rasterizer  # noqa: E402


def min_pins(N):
    if N <= 16:
        return max(3, math.ceil(N * 0.20))
    if N <= 32:
        return max(5, math.ceil(N * 0.20))
    if N <= 64:
        return max(8, math.ceil(N * 0.15))
    if N <= 144:
        return max(12, math.ceil(N * 0.10))
    return max(16, math.ceil(N * 0.08))


def check(gt):
    comps = gt['components']
    nets = gt['nets']
    if len([k for k, c in comps.items() if c['type'] not in ('gnd', 'v', 'net_bidirection')]) < 5:
        return False, 'components<5'
    sets = {tuple(sorted(n['hyperGraph'].strip('()').split(','))) for n in nets.values()}
    if len([s for s in sets if len(s) >= 2]) < 5:
        return False, 'nets<5'
    used = {}
    for n in nets.values():
        for t in n['hyperGraph'].strip('()').split(','):
            if '.' in t:
                c, p = t.rsplit('.', 1)
                used.setdefault(c, set()).add(p)
    ok_ic = False
    for k, c in comps.items():
        if c['type'] == 'box':
            N = len(gt['pins'].get(k, {}))
            need = min(min_pins(N), N)
            if N >= 3 and len(used.get(k, ())) >= need:
                ok_ic = True
    if not ok_ic:
        return False, 'no valid IC'
    return True, 'ok'


def signature(gt):
    toks = sorted(f"{k}:{c.get('Name') or ''}:{c.get('value') or ''}" for k, c in gt['components'].items()
                  if c['type'] not in ('gnd', 'v'))
    return set(toks)


def jacc(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train', default=None, help='folder with the 200 official training cases (json) for de-duplication')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--max-side', type=int, default=4200)
    ap.add_argument('--per-project', type=int, default=8, help='max cases taken from one project')
    ap.add_argument('--overlay', action='store_true', help='also write review overlays to <out>_review/')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train_sigs = []
    if a.train:
        for f in glob.glob(os.path.join(a.train, '**', '*.json'), recursive=True):
            try:
                train_sigs.append(signature(json.load(open(f, encoding='utf-8'))))
            except Exception:
                pass
        print('training signatures', len(train_sigs))
    R = Rasterizer()
    seen = []
    rows = []
    k = a.start
    for bf in sorted(glob.glob(os.path.join(a.raw, 'bundles', '*.json'))):
        try:
            bundle = json.load(open(bf, encoding='utf-8'))
            sheets = load_bundle(bundle)
        except Exception as e:
            print('bad bundle', bf, e)
            continue
        proj = bundle.get('project') or {}
        proj_sigs, proj_titles = [], set()
        for si, (title, sh) in enumerate(sheets):
            if len(proj_sigs) >= a.per_project:
                break
            if sum(1 for c in sh.components if c.symbol is None) > 0:
                continue
            # embedded pictures (module photos as symbol bodies, logos) are not rendered
            if any(c.symbol.has_image and c.symbol.stype != 20 and not str(c.symbol.title).lower().startswith('sheet-symbol')
                   for c in sh.components):
                print('skip (embedded image)', proj.get('path'), title)
                continue
            cid = f'case_{k:04d}'
            rng = random.Random(hashlib.md5(f"{proj.get('uuid')}{si}".encode()).hexdigest())
            scale = rng.uniform(1.90, 2.05)
            png = os.path.join(a.out, cid + '.png')
            js = os.path.join(a.out, cid + '.json')
            try:
                res = make_case(sh, R, png, js, scale=scale, seed=rng.random())
            except Exception as e:
                print('render failed', proj.get('path'), e)
                continue
            if res is None:
                continue
            gt, (W, H) = res
            ok, why = check(gt)
            if ok and max(W, H) > a.max_side:
                ok, why = False, 'too large'
            sig = signature(gt)
            if ok and any(jacc(sig, s) > 0.7 for s in train_sigs):
                ok, why = False, 'similar to training case'
            if ok and any(jacc(sig, s) > 0.85 for s in seen):
                ok, why = False, 'duplicate'
            # same project: board versions repeat the same sheet -> stricter rule
            tkey = (title or '').strip().lower()
            if ok and (tkey and tkey in proj_titles and any(jacc(sig, s) > 0.4 for s in proj_sigs)
                       or any(jacc(sig, s) > 0.6 for s in proj_sigs)):
                ok, why = False, 'same-project duplicate'
            if not ok:
                for p in (png, js):
                    if os.path.exists(p):
                        os.remove(p)
                continue
            seen.append(sig)
            proj_sigs.append(sig)
            proj_titles.add(tkey)
            if a.overlay:
                import cv2
                sys.path.insert(0, os.path.join(HERE, 'tools'))
                from preannotate import overlay
                rv = a.out.rstrip('/\\') + '_review'
                os.makedirs(rv, exist_ok=True)
                img = cv2.imread(png)[:, :, ::-1]
                cv2.imwrite(os.path.join(rv, cid + '_overlay.png'), overlay(img, gt))
            rows.append({'case_id': cid, 'project': proj.get('url'), 'license': proj.get('license'),
                         'author': proj.get('author'), 'sheet': title, 'width': W, 'height': H,
                         'components': len(gt['components']), 'nets': len(gt['nets'])})
            print(cid, proj.get('path'), title, W, H, len(gt['components']), 'comps', flush=True)
            k += 1
    R.close()
    with open(os.path.join(a.out, 'cases.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['case_id'])
        w.writeheader()
        w.writerows(rows)
    print('cases', len(rows))
    if a.score and rows:
        score_cases(a.out, [r['case_id'] for r in rows])


def score_cases(out_dir, ids, timeout=600):
    """Run the parser on each case in a separate process (a crash/OOM only loses that case)."""
    import subprocess
    from pcbparse.scorer_official import score_case
    pred_dir = os.path.join(out_dir, '_pred')
    os.makedirs(pred_dir, exist_ok=True)
    code = ('import sys,json; sys.path.insert(0, sys.argv[3]); from pcbparse.pipeline import parse_image; '
            'json.dump(parse_image(sys.argv[1]), open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)')
    out = []
    for cid in ids:
        pj = os.path.join(pred_dir, cid + '_Pred.json')
        try:
            subprocess.run([sys.executable, '-c', code, os.path.join(out_dir, cid + '.png'), pj, HERE],
                           timeout=timeout, check=True, capture_output=True)
            P = json.load(open(pj, encoding='utf-8'))
            G = json.load(open(os.path.join(out_dir, cid + '.json'), encoding='utf-8'))
            sc = round(score_case(G, P)['score'], 2)
        except Exception as e:
            print(cid, 'parser failed', type(e).__name__, flush=True)
            sc = 0.0
        out.append((cid, sc))
        print(cid, sc, flush=True)
    with open(os.path.join(out_dir, 'parser_scores.csv'), 'w', encoding='utf-8') as f:
        f.write('case_id,score\n' + '\n'.join(f'{c},{s}' for c, s in out))
    if out:
        print('mean', round(sum(s for _, s in out) / len(out), 2))


if __name__ == '__main__':
    main()

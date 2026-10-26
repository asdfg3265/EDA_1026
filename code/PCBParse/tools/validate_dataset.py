#!/usr/bin/env python3
"""Validate a contributed dataset folder against the rules of 《数据提交说明》.

Usage:
    python tools/validate_dataset.py dataset/            # folder with case_xxxx.png + case_xxxx.json
    python tools/validate_dataset.py dataset/ --strict   # also fail on warnings

Checks (automatic part of the official pipeline):
  * file naming (ASCII letters/digits/underscore, png+json pairs with the same case_id)
  * JSON schema: components / pins / nets, required fields and types
  * absolute coordinates are integer pixels (origin bottom-left, x right, y up)
  * references: pins keys exist in components, hyperGraph tokens exist in pins
  * no duplicated connections, every edge has exactly two points
  * validity: >=1 IC (type box) with name, >=5 components, >=5 unique nets,
    IC effective pin count >= threshold(N) (N = IC total pins)
"""
import json
import math
import os
import re
import sys
from collections import defaultdict

CASE_RE = re.compile(r'^[A-Za-z0-9_]+$')


def ic_min_pins(n):
    if n <= 16:
        v = max(3, math.ceil(n * 0.20))
    elif n <= 32:
        v = max(5, math.ceil(n * 0.20))
    elif n <= 64:
        v = max(8, math.ceil(n * 0.15))
    elif n <= 144:
        v = max(12, math.ceil(n * 0.10))
    else:
        v = max(16, math.ceil(n * 0.08))
    return v


def parse_hg(s):
    s = (s or '').strip()
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            if '.' not in tok:
                out.append((tok, None))
            else:
                c, p = tok.rsplit('.', 1)
                out.append((c, p))
    return out


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool) or (isinstance(v, float) and float(v).is_integer())


def check_case(case_id, J, W=None, H=None):
    err, warn = [], []
    for k in ('components', 'pins', 'nets'):
        if k not in J or not isinstance(J[k], dict):
            err.append(f'missing top-level "{k}"')
    if err:
        return err, warn, {}
    comps, pins, nets = J['components'], J['pins'], J['nets']
    # components
    for k, c in comps.items():
        for f in ('Name', 'type', 'bbox'):
            if f not in c:
                err.append(f'component {k}: missing {f}')
        b = c.get('bbox')
        if not (isinstance(b, list) and len(b) == 4):
            err.append(f'component {k}: bbox must be [x1,y1,x2,y2]')
            continue
        if not all(is_int(v) for v in b):
            err.append(f'component {k}: bbox must be integer pixels')
        if b[0] > b[2] or b[1] > b[3]:
            err.append(f'component {k}: bbox needs x1<=x2 and y1<=y2 (bottom-left origin)')
        if W and H and (min(b) < 0 or b[2] > W or b[3] > H):
            warn.append(f'component {k}: bbox outside image')
    # pins
    for k, ps in pins.items():
        if k not in comps:
            err.append(f'pins: component "{k}" not in components')
        for pk, p in ps.items():
            if not pk.startswith('pin_'):
                err.append(f'pins {k}.{pk}: key must be pin_<number>')
            pt = p.get('point', {})
            if not (is_int(pt.get('x')) and is_int(pt.get('y'))):
                err.append(f'pins {k}.{pk}: point must be integer pixels')
            if 'pinname' not in p:
                err.append(f'pins {k}.{pk}: missing pinname')
    # nets
    seen_sets = {}
    pin_in_net = defaultdict(int)
    for nk, n in nets.items():
        toks = parse_hg(n.get('hyperGraph', ''))
        if not toks:
            warn.append(f'net {nk}: empty hyperGraph')
        for c, p in toks:
            if p is None:
                err.append(f'net {nk}: token "{c}" has no pin number')
                continue
            if c not in pins or f'pin_{p}' not in pins.get(c, {}):
                err.append(f'net {nk}: {c}.{p} not found in pins')
            pin_in_net[(c, p)] += 1
        key = frozenset(toks)
        if key in seen_sets:
            err.append(f'net {nk}: duplicate of {seen_sets[key]}')
        seen_sets[key] = nk
        if len(toks) != len(set(toks)):
            err.append(f'net {nk}: repeated pin in hyperGraph')
        edges = n.get('edges', {})
        if not isinstance(edges, dict):
            err.append(f'net {nk}: edges must be an object')
            continue
        for ek, e in edges.items():
            if not (isinstance(e, list) and len(e) == 2):
                err.append(f'net {nk}.{ek}: each edge needs exactly 2 points')
                continue
            for p in e:
                if not (is_int(p.get('x')) and is_int(p.get('y'))):
                    err.append(f'net {nk}.{ek}: edge points must be integer pixels')
    for (c, p), cnt in pin_in_net.items():
        if cnt > 1:
            err.append(f'pin {c}.{p} appears in {cnt} nets (nets must be merged)')
    # validity statistics
    stats = {}
    ics = [k for k, c in comps.items() if c.get('type') in ('box', 'ic')]
    stats['components'] = len(comps)
    stats['nets'] = len([1 for n in nets.values() if len(parse_hg(n.get('hyperGraph', ''))) >= 2])
    stats['ics'] = len(ics)
    if not ics:
        err.append('no IC (type "box") in the case')
    ok_ic = 0
    for k in ics:
        name = comps[k].get('Name')
        if not name or name == k:
            warn.append(f'IC {k}: Name should be the part name/model shown in the image')
        N = len(pins.get(k, {}))
        conn = sum(1 for pk in pins.get(k, {}) if pin_in_net.get((k, pk[4:]), 0) and
                   any(len(parse_hg(n['hyperGraph'])) >= 2 and (k, pk[4:]) in parse_hg(n['hyperGraph']) for n in nets.values()))
        need = ic_min_pins(N) if N else 3
        stats[f'ic_{k}'] = f'{conn}/{N} connected (need {need})'
        if N and conn >= need:
            ok_ic += 1
        else:
            warn.append(f'IC {k}: only {conn} connected pins of {N} (need {need})')
    if ics and ok_ic == 0:
        err.append('no IC satisfies the minimum connected-pin rule')
    if len(comps) < 5:
        err.append(f'only {len(comps)} components (need >= 5)')
    if stats['nets'] < 5:
        err.append(f'only {stats["nets"]} nets with >= 2 pins (need >= 5)')
    return err, warn, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--strict', action='store_true')
    a = ap.parse_args()
    files = os.listdir(a.folder)
    pngs = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.png')}
    jsons = {os.path.splitext(f)[0] for f in files if f.lower().endswith('.json')}
    # also accept one sub-folder per case
    for d in files:
        p = os.path.join(a.folder, d)
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.lower().endswith('.png'):
                    pngs.add(os.path.join(d, os.path.splitext(f)[0]))
                if f.lower().endswith('.json'):
                    jsons.add(os.path.join(d, os.path.splitext(f)[0]))
    bad = 0
    good = 0
    for cid in sorted(pngs | jsons):
        name = os.path.basename(cid)
        msgs = []
        if not CASE_RE.match(name):
            msgs.append(f'case_id "{name}" must use only ASCII letters, digits and "_"')
        if cid not in pngs:
            msgs.append('missing PNG')
        if cid not in jsons:
            msgs.append('missing JSON')
            print(f'[FAIL] {cid}: ' + '; '.join(msgs))
            bad += 1
            continue
        W = H = None
        if cid in pngs:
            try:
                from PIL import Image
                W, H = Image.open(os.path.join(a.folder, cid + '.png')).size
            except Exception:
                pass
        try:
            J = json.load(open(os.path.join(a.folder, cid + '.json'), encoding='utf-8'))
        except Exception as e:
            print(f'[FAIL] {cid}: invalid JSON ({e})')
            bad += 1
            continue
        err, warn, stats = check_case(name, J, W, H)
        err = msgs + err
        if err or (a.strict and warn):
            bad += 1
            print(f'[FAIL] {cid}: {stats}')
            for e in err:
                print('      E', e)
        else:
            good += 1
            print(f'[ OK ] {cid}: {stats}')
        for w in warn:
            print('      W', w)
    print(f'\n{good} valid, {bad} invalid')
    if good < 20:
        print('NOTE: at least 20 valid cases are required for the data bonus')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

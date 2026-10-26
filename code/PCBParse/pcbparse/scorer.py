"""Local re-implementation of the competition scorer (赛题六 AI驱动的PCB原理图自动解析).

Implements the rules from the task guide + Q&A:
  * ComponentF1  : key match; type equal; bbox max-coordinate error <= 20 px;
                   text: box/ic -> Name, r/c -> value (strict).  Anonymous power/ground
                   components (key starts with '∅') are matched by coordinates.
  * PinF1        : component key matched, then pin number; pinname strict + dist <= 5 px.
                   Non-box pins: number/name not scored -> matched by position only.
  * NetHypergraphF1 : nets as pin sets; same-name GND/VCC nets merged; Hungarian match.
                   Non-box component pins are not distinguished (token = component key).
  * NetLineF1    : inside matched net pairs, edges Hungarian-matched by endpoints (5 px).
Score = 100*(0.3*C + 0.25*P + 0.35*N + 0.1*L)
The official code may differ in edge cases; this is used for relative comparison.
"""
import json, re, sys, os, math
import numpy as np
from scipy.optimize import linear_sum_assignment

BBOX_TH = 20.0
PIN_TH = 5.0
LINE_TH = 5.0
ANON_PREFIX = '∅'
POWER_TYPES = {'gnd', 'v'}
BOX_TYPES = {'box', 'ic'}

TYPE_FAMILY = {
    'bjt_npn': 'bjt', 'bjt_pnp': 'bjt', 'mosfet_npn': 'mosfet', 'mosfet_pnp': 'mosfet',
    'mosfet_n': 'mosfet', 'mosfet_p': 'mosfet',
    'amp_3pin': 'amp', 'amp_5pin': 'amp', 'amp_7pin': 'amp',
    'crystal_2pin': 'crystal', 'crystal_3pin': 'crystal', 'crystal_4pin': 'crystal',
}


def norm_type(t, family=False):
    t = (t or '').strip().lower()
    if t == 'ic':
        t = 'box'
    if family:
        t = TYPE_FAMILY.get(t, t)
    return t


def f1(tp, npred, ngt):
    if npred == 0 and ngt == 0:
        return 1.0
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def is_anon(k):
    return k.startswith(ANON_PREFIX)


def bbox_ok(a, b, th=BBOX_TH):
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= th


def bbox_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def text_ok(gt, pr):
    t = norm_type(gt.get('type'))
    if t in BOX_TYPES:
        return (gt.get('Name') or '') == (pr.get('Name') or '')
    if t in ('r', 'c', 'resistor', 'capacitor'):
        return (gt.get('value') or '') == (pr.get('value') or '')
    return True


def parse_hg(s):
    s = (s or '').strip()
    if s.startswith('('):
        s = s[1:]
    if s.endswith(')'):
        s = s[:-1]
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '.' in tok:
            c, p = tok.rsplit('.', 1)
        else:
            c, p = tok, ''
        out.append((c, p))
    return out


def match_components(G, P, family=False):
    """Return mapping pred_key->gt_key (identity for named, coordinate-based for anon)."""
    gc, pc = G['components'], P['components']
    mapping = {}
    for k in gc:
        if not is_anon(k) and k in pc:
            mapping[k] = k
    # anonymous: hungarian on center distance within the same (family) type
    ga = [k for k in gc if is_anon(k)]
    pa = [k for k in pc if is_anon(k) or (k not in gc and norm_type(pc[k].get('type')) in POWER_TYPES)]
    pa = [k for k in pa if k not in mapping]
    if ga and pa:
        C = np.full((len(ga), len(pa)), 1e6)
        for i, g in enumerate(ga):
            cg = bbox_center(gc[g]['bbox'])
            for j, p in enumerate(pa):
                if norm_type(gc[g]['type'], True) != norm_type(pc[p].get('type'), True):
                    continue
                cp = bbox_center(pc[p]['bbox'])
                d = math.hypot(cg[0] - cp[0], cg[1] - cp[1])
                if d <= 60:
                    C[i, j] = d
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            if C[i, j] < 1e5:
                mapping[pa[j]] = ga[i]
    return mapping


def component_score(G, P, mapping, family=False):
    gc, pc = G['components'], P['components']
    tp = 0
    det = []
    for pk, gk in mapping.items():
        g, p = gc[gk], pc[pk]
        ok_t = norm_type(g.get('type'), family) == norm_type(p.get('type'), family)
        ok_b = bbox_ok(g['bbox'], p['bbox'])
        ok_x = text_ok(g, p)
        if ok_t and ok_b and ok_x:
            tp += 1
        else:
            det.append((gk, pk, ok_t, ok_b, ok_x))
    return f1(tp, len(pc), len(gc)), tp, len(pc), len(gc), det


def pin_score(G, P, mapping):
    gc = G['components']
    gp, pp = G.get('pins', {}), P.get('pins', {})
    ngt = sum(len(v) for v in gp.values())
    npred = sum(len(v) for v in pp.values())
    tp = 0
    errs = []
    for pk, gk in mapping.items():
        gpins = gp.get(gk, {})
        ppins = pp.get(pk, {})
        if not gpins or not ppins:
            continue
        t = norm_type(gc[gk].get('type'))
        if t in BOX_TYPES:
            for num, g in gpins.items():
                p = ppins.get(num)
                if p is None:
                    errs.append((gk, num, 'missing'))
                    continue
                d = math.hypot(g['point']['x'] - p['point']['x'], g['point']['y'] - p['point']['y'])
                if (g.get('pinname') or '') == (p.get('pinname') or '') and d <= PIN_TH:
                    tp += 1
                else:
                    errs.append((gk, num, g.get('pinname'), p.get('pinname'), round(d, 1)))
        else:
            gl = list(gpins.values())
            pl = list(ppins.values())
            C = np.zeros((len(gl), len(pl)))
            for i, g in enumerate(gl):
                for j, p in enumerate(pl):
                    C[i, j] = math.hypot(g['point']['x'] - p['point']['x'], g['point']['y'] - p['point']['y'])
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] <= PIN_TH:
                    tp += 1
                else:
                    errs.append((gk, 'nonbox', round(C[i, j], 1)))
    return f1(tp, npred, ngt), tp, npred, ngt, errs


def build_nets(J, comp_types, keymap=None, power_merge=True):
    """Return list of (token_multiset_as_list, edges list). keymap maps component key -> canonical key."""
    comps = J['components']
    nets = []
    for nk, n in J.get('nets', {}).items():
        toks = []
        power_names = set()
        for c, p in parse_hg(n.get('hyperGraph', '')):
            ck = keymap.get(c, c) if keymap is not None else c
            t = comp_types.get(ck)
            if t is None:
                t = norm_type(comps.get(c, {}).get('type'))
            if t in BOX_TYPES:
                toks.append(f'{ck}.{p}')
            else:
                toks.append(f'{ck}')
            if c in comps and norm_type(comps[c].get('type')) in POWER_TYPES:
                v = comps[c].get('value') or comps[c].get('Name') or ''
                power_names.add((norm_type(comps[c].get('type')), v))
        edges = []
        for e in n.get('edges', {}).values():
            if isinstance(e, list) and len(e) == 2:
                edges.append(((float(e[0]['x']), float(e[0]['y'])), (float(e[1]['x']), float(e[1]['y']))))
        nets.append({'toks': toks, 'edges': edges, 'power': power_names})
    if power_merge:
        # union nets sharing same power name
        parent = list(range(len(nets)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        seen = {}
        for i, n in enumerate(nets):
            for pn in n['power']:
                if pn in seen:
                    parent[find(i)] = find(seen[pn])
                else:
                    seen[pn] = i
        groups = {}
        for i in range(len(nets)):
            groups.setdefault(find(i), []).append(i)
        merged = []
        for g in groups.values():
            toks, edges = [], []
            for i in g:
                toks += nets[i]['toks']
                edges += nets[i]['edges']
            merged.append({'toks': toks, 'edges': edges})
        nets = merged
    return nets


def multiset_inter(a, b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


def edge_match(ge, pe):
    if not ge or not pe:
        return 0
    C = np.zeros((len(ge), len(pe)))
    for i, g in enumerate(ge):
        for j, p in enumerate(pe):
            d1 = max(math.dist(g[0], p[0]), math.dist(g[1], p[1]))
            d2 = max(math.dist(g[0], p[1]), math.dist(g[1], p[0]))
            C[i, j] = min(d1, d2)
    r, c = linear_sum_assignment(C)
    return int(sum(1 for i, j in zip(r, c) if C[i, j] <= LINE_TH))


def net_scores(G, P, mapping):
    gtypes = {k: norm_type(v.get('type')) for k, v in G['components'].items()}
    gn = build_nets(G, gtypes)
    ptypes = {}
    for k, v in P['components'].items():
        ptypes[mapping.get(k, k)] = norm_type(v.get('type'))
    # tokens for pred: map keys to gt keys; types: use gt types when mapped (so token forms agree)
    pt2 = dict(ptypes)
    for pk, gk in mapping.items():
        pt2[gk] = gtypes.get(gk, ptypes.get(gk))
    pn = build_nets(P, pt2, keymap=mapping)
    ngt = sum(len(n['toks']) for n in gn)
    npred = sum(len(n['toks']) for n in pn)
    if gn and pn:
        C = np.zeros((len(gn), len(pn)))
        for i, g in enumerate(gn):
            for j, p in enumerate(pn):
                C[i, j] = -multiset_inter(g['toks'], p['toks'])
        r, c = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(r, c) if C[i, j] < 0]
    else:
        pairs = []
    tp = sum(-C[i, j] for i, j in pairs) if pairs else 0
    nf1 = f1(tp, npred, ngt)
    # lines
    gte = sum(len(n['edges']) for n in gn)
    pre = sum(len(n['edges']) for n in pn)
    ltp = sum(edge_match(gn[i]['edges'], pn[j]['edges']) for i, j in pairs)
    lf1 = f1(ltp, pre, gte)
    return nf1, (tp, npred, ngt), lf1, (ltp, pre, gte)


def score_case(G, P, family=False, verbose=False):
    mapping = match_components(G, P, family)
    cf1, ctp, cnp, cng, cdet = component_score(G, P, mapping, family)
    pf1, ptp, pnp, png, perr = pin_score(G, P, mapping)
    nf1, ninfo, lf1, linfo = net_scores(G, P, mapping)
    s = 100 * (0.3 * cf1 + 0.25 * pf1 + 0.35 * nf1 + 0.1 * lf1)
    res = dict(score=s, comp=cf1, pin=pf1, net=nf1, line=lf1,
               comp_info=(ctp, cnp, cng), pin_info=(ptp, pnp, png), net_info=ninfo, line_info=linfo)
    if verbose:
        res['comp_err'] = cdet
        res['pin_err'] = perr
        res['unmatched_pred'] = [k for k in P['components'] if k not in mapping]
        res['unmatched_gt'] = [k for k in G['components'] if k not in mapping.values()]
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('gt_dir')
    ap.add_argument('pred_dir')
    ap.add_argument('-v', action='store_true')
    ap.add_argument('--family', action='store_true')
    a = ap.parse_args()
    tot = []
    for case in sorted(os.listdir(a.gt_dir), key=lambda s: (len(s), s)):
        d = os.path.join(a.gt_dir, case)
        if not os.path.isdir(d):
            continue
        gj = [f for f in os.listdir(d) if f.endswith('.json')]
        if not gj:
            continue
        G = json.load(open(os.path.join(d, gj[0]), encoding='utf-8'))
        pf = os.path.join(a.pred_dir, case, case + '.json')
        if not os.path.exists(pf):
            print(case, 'MISSING')
            tot.append(0)
            continue
        P = json.load(open(pf, encoding='utf-8'))
        r = score_case(G, P, a.family, a.v)
        tot.append(r['score'])
        print(f"{case:>24s} score={r['score']:6.2f} comp={r['comp']:.3f}{r['comp_info']} pin={r['pin']:.3f}{r['pin_info']} "
              f"net={r['net']:.3f}{r['net_info']} line={r['line']:.3f}{r['line_info']}")
        if a.v:
            for k in ('comp_err', 'unmatched_pred', 'unmatched_gt', 'pin_err'):
                if r[k]:
                    print('    ', k, r[k][:40])
    print(f'MEAN {np.mean(tot):.2f} over {len(tot)} cases')


if __name__ == '__main__':
    main()

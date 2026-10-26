"""Replica of the evaluation-platform scorer (protocol 2026-10-08-global-power-rails), reverse-engineered from the first official
submission (2026-10-17, 13 open cases; reproduces every per-case CompF1/PinF1/ConnF1/WireF1).

  CompF1 : key match (∅ power/ground matched by coordinates); type equal; bbox max-corner
           error <= 20 px; Name AND value strictly equal for every component (∅ Name ignored).
  PinF1  : only box/ic pins; pred pin with same pin number, same pinname, distance <= 10 px.
  ConnF1 : nets = token sets (box pin -> 'U1.5', other components -> key, power/ground ->
           'PWR:<value>'); nets sharing a power name merged; Hungarian match between nets,
           TP = sum of Dice(set_g, set_p) over matched pairs; F1 over net counts.
  WireF1 : inside matched net pairs, edges Hungarian-matched by endpoints (<= 5 px).
  Final  = 100*(0.3 C + 0.25 P + 0.35 N + 0.1 W)
"""
import json, os, sys, math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .scorer import (match_components, norm_type, f1, bbox_ok, parse_hg, edge_match,
                     BOX_TYPES, POWER_TYPES, is_anon)

PIN_TH = 20.0   # protocol: spatial_match_threshold_px = 20 (U1.pin_9 at 21.0 px failed)
# Scoring-rule update (organisers, 2026-10-18; the platform still runs the old one):
# one spatial threshold of 20 px for bbox, IC pins AND wire endpoints (was 5 px for wires).
# PCB_SCORE_RULE=platform reproduces the current platform numbers.
from . import scorer as _sc
# The platform as of 2026-10-21 (protocol '2026-10-18-independent-thresholds-only', read from
# its per-case report): component bbox 20 px (every coordinate), IC pin 5 px (Euclidean),
# wire endpoints 5 px; net-label types are compared exactly (net_input / net_output /
# net_bidirection).  This is the default now.  PCB_SCORE_RULE=all20 gives the 20 px-for-all
# variant described in the organisers' note.
RULE = os.environ.get('PCB_SCORE_RULE', 'spec1022')
# 'spec1022': the organisers' written evaluation spec of 2026-10-22 ('测评要求与匹配过程说明'):
# thresholds as 'platform'; VCC/GND AND net_input/net_output/net_short/net_bidirection are
# matched one-to-one by category and position (auto-generated keys / names are not compared,
# type and value are); in the connectivity graph a net label is a global-net reference (its
# value), so branches sharing a label name are merged like power rails.
NET_BY_POS = RULE == 'spec1022'
_sc.LINE_TH = 20.0 if RULE == 'all20' else 5.0
PIN_TH = 20.0 if RULE == 'all20' else 5.0


def comp_f1(G, P, m):
    gc, pc = G['components'], P['components']
    tp, errs = 0, []
    for pk, gk in m.items():
        g, p = gc[gk], pc[pk]
        reasons = []
        if norm_type(g.get('type'), NET_BY_POS) != norm_type(p.get('type'), NET_BY_POS):
            reasons.append(f"type {g.get('type')}!={p.get('type')}")
        if not bbox_ok(g['bbox'], p['bbox']):
            reasons.append('bbox ' + str(round(max(abs(a - b) for a, b in zip(g['bbox'], p['bbox'])), 1)))
        if not is_anon(gk) and not (NET_BY_POS and norm_type(g.get('type')).startswith('net')) \
                and (g.get('Name') or '') != (p.get('Name') or ''):
            reasons.append(f"Name {g.get('Name')!r}!={p.get('Name')!r}")
        if (g.get('value') or '') != (p.get('value') or ''):
            reasons.append(f"value {g.get('value')!r}!={p.get('value')!r}")
        if reasons:
            errs.append((gk, pk, reasons))
        else:
            tp += 1
    return f1(tp, len(pc), len(gc)), (tp, len(pc), len(gc)), errs


def pin_f1(G, P, m):
    inv = {v: k for k, v in m.items()}
    tp = npred = ngt = 0
    errs = []
    for gk, g in G['components'].items():
        if norm_type(g.get('type')) in BOX_TYPES:
            gp = G['pins'].get(gk, {})
            ngt += len(gp)
            pk = inv.get(gk)
            pp = P['pins'].get(pk, {}) if pk else {}
            for n, a in gp.items():
                b = pp.get(n)
                if b is None:
                    errs.append((gk, n, 'missing'))
                    continue
                d = math.hypot(a['point']['x'] - b['point']['x'], a['point']['y'] - b['point']['y'])
                if (a.get('pinname') or '') == (b.get('pinname') or '') and d <= PIN_TH:
                    tp += 1
                else:
                    errs.append((gk, n, a.get('pinname'), b.get('pinname'), round(d, 1)))
    for pk, p in P['components'].items():
        if norm_type(p.get('type')) in BOX_TYPES:
            npred += len(P['pins'].get(pk, {}))
    return f1(tp, npred, ngt), (tp, npred, ngt), errs


def _nets(J, keymap):
    comps = J['components']
    out = []
    for n in J.get('nets', {}).values():
        toks, pw = set(), set()
        for c, p in parse_hg(n.get('hyperGraph', '')):
            t = norm_type(comps.get(c, {}).get('type'))
            if t in POWER_TYPES:
                # protocol 'global-power-rails': all grounds share GND; supplies share exact value (null -> VCC)
                tok = 'PWR:GND' if t == 'gnd' else 'PWR:' + (comps[c].get('value') or 'VCC')
                pw.add(tok)
            elif NET_BY_POS and t.startswith('net') and (comps[c].get('value') or not c.startswith('∅')):
                tok = 'NET:' + (comps[c].get('value') or comps[c].get('Name') or c)
                pw.add(tok)
            elif t.startswith('net'):
                tok = keymap.get(c, c)          # an anonymous port without a name (net_short): by position
            else:
                ck = keymap.get(c, c)
                tok = f'{ck}.{p}' if t in BOX_TYPES else ck
            toks.add(tok)
        edges = []
        for e in n.get('edges', {}).values():
            if isinstance(e, list) and len(e) == 2:
                edges.append(((float(e[0]['x']), float(e[0]['y'])), (float(e[1]['x']), float(e[1]['y']))))
        out.append({'toks': toks, 'pw': pw, 'edges': edges})
    # merge nets sharing a power name
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if out[i]['pw'] & out[j]['pw']:
                    out[i] = {'toks': out[i]['toks'] | out[j]['toks'], 'pw': out[i]['pw'] | out[j]['pw'],
                              'edges': out[i]['edges'] + out[j]['edges']}
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    return out


def conn_wire_f1(G, P, m):
    gn, pn = _nets(G, {}), _nets(P, m)
    pairs, tp = [], 0.0
    if gn and pn:
        M = np.array([[2 * len(g['toks'] & p['toks']) / max(1, len(g['toks']) + len(p['toks'])) for p in pn] for g in gn])
        r, c = linear_sum_assignment(-M)
        pairs = [(i, j) for i, j in zip(r, c) if M[i, j] > 0]
        tp = float(sum(M[i, j] for i, j in pairs))
    cf = f1(tp, len(pn), len(gn))
    gte = sum(len(n['edges']) for n in gn)
    pre = sum(len(n['edges']) for n in pn)
    ltp = sum(edge_match(gn[i]['edges'], pn[j]['edges']) for i, j in pairs)
    detail = {'unmatched_pred_nets': [sorted(pn[j]['toks']) for j in range(len(pn)) if j not in {b for _, b in pairs}],
              'unmatched_gt_nets': [sorted(gn[i]['toks']) for i in range(len(gn)) if i not in {a for a, _ in pairs}],
              'imperfect': [(sorted(gn[i]['toks'] - pn[j]['toks']), sorted(pn[j]['toks'] - gn[i]['toks']))
                            for i, j in pairs if gn[i]['toks'] != pn[j]['toks']]}
    return cf, (round(tp, 3), len(pn), len(gn)), f1(ltp, pre, gte), (ltp, pre, gte), detail


def _match_spec(G, P):
    """Identifier match for ordinary parts; category + position for power/ground symbols
    and net ports (their keys are auto-numbered)."""
    gc, pc = G['components'], P['components']
    def pos_kind(comp, key):
        t = norm_type(comp.get('type'))
        if t in POWER_TYPES:
            return t
        if t.startswith('net'):
            return 'net'
        return 'anon' if is_anon(key) else None
    m = {}
    for k in gc:
        if pos_kind(gc[k], k) is None and k in pc and pos_kind(pc[k], k) is None:
            m[k] = k
    ga = [k for k in gc if pos_kind(gc[k], k) is not None]
    pa = [k for k in pc if k not in m and (pos_kind(pc[k], k) is not None or k not in gc)]
    if ga and pa:
        C = np.full((len(ga), len(pa)), 1e6)
        for i, g in enumerate(ga):
            kg = pos_kind(gc[g], g)
            bg = gc[g]['bbox']
            cg = ((bg[0] + bg[2]) / 2.0, (bg[1] + bg[3]) / 2.0)
            for j, p in enumerate(pa):
                kp = pos_kind(pc[p], p)
                if kp is None:
                    continue
                if (kg in ('net',) or kp in ('net',)) and kg != kp:
                    continue
                if kg in POWER_TYPES and norm_type(gc[g].get('type'), True) != norm_type(pc[p].get('type'), True):
                    continue
                bp = pc[p]['bbox']
                d = math.hypot(cg[0] - (bp[0] + bp[2]) / 2.0, cg[1] - (bp[1] + bp[3]) / 2.0)
                if d <= 60:
                    C[i, j] = d
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            if C[i, j] < 1e5:
                m[pa[j]] = ga[i]
    return m


def score_case(G, P):
    m = _match_spec(G, P) if NET_BY_POS else match_components(G, P)
    c, ci, ce = comp_f1(G, P, m)
    p, pi, pe = pin_f1(G, P, m)
    n, ni, w, wi, nd = conn_wire_f1(G, P, m)
    s = 100 * (0.3 * c + 0.25 * p + 0.35 * n + 0.1 * w)
    return dict(score=s, comp=c, pin=p, conn=n, wire=w, ci=ci, pi=pi, ni=ni, wi=wi, comp_err=ce, pin_err=pe, net=nd,
                unmatched_pred=[k for k in P['components'] if k not in m],
                unmatched_gt=[k for k in G['components'] if k not in m.values()])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('gt_dir'); ap.add_argument('pred_dir'); ap.add_argument('-v', action='store_true')
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
        cands = [os.path.join(a.pred_dir, case, case + '.json'), os.path.join(a.pred_dir, case + '_Pred.json')]
        pf = next((p for p in cands if os.path.exists(p)), None)
        if pf is None:
            print(case, 'MISSING'); tot.append(0); continue
        r = score_case(G, json.load(open(pf, encoding='utf-8')))
        tot.append(r['score'])
        print(f"{case:>6s} Comp {r['comp']:.4f}{r['ci']} Pin {r['pin']:.4f}{r['pi']} Conn {r['conn']:.4f}{r['ni']} "
              f"Wire {r['wire']:.4f}{r['wi']} Final {r['score']:.4f}")
        if a.v:
            for k in ('comp_err', 'unmatched_pred', 'unmatched_gt', 'pin_err'):
                if r[k]:
                    print('     ', k, r[k][:30])
            for k, v in r['net'].items():
                if v:
                    print('     ', k, v[:10])
    print(f'MEAN {np.mean(tot):.4f} over {len(tot)} cases')


if __name__ == '__main__':
    main()

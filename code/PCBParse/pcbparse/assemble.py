"""Assemble primitives, symbols and texts into the competition JSON."""
import os
import re
import math
import numpy as np
from collections import Counter
import cv2
from scipy.optimize import linear_sum_assignment

from .textclass import is_desig, prefix_type, desig_prefix, is_gnd_name, is_power_name, is_value
from .text import redecode
from .wiregraph import UF, decompose_edges

UPPER_DIGITS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def box_gap(a, b):
    """Gap between two boxes [x0,y0,x1,y1] (0 when overlapping)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


_KNOWN_LONG_PREFIX = {'LED', 'USB', 'BAT', 'ANT', 'ESD', 'MIC', 'KEY', 'BTN', 'LCD', 'SIM', 'PWR', 'RLY', 'TVS', 'NTC', 'PTC',
                      'OLED', 'XTAL', 'BUZ', 'CON', 'SPK', 'MOT', 'FPC', 'HDR', 'DCDC', 'LDO', 'OSC', 'IND', 'SWD', 'TFT',
                      'ADC', 'DAC', 'MCU', 'CAN', 'ETH', 'RJ', 'SD', 'TF', 'IR', 'HDMI', 'POW', 'RST', 'BOOT', 'JTAG'}


DESIG_VOLT_TAIL = os.environ.get('PCB_DESIG_VOLT_TAIL', '1') == '1'
DESIG_STACK_PRIOR = os.environ.get('PCB_DESIG_STACK_PRIOR', '1') == '1'


def desig_implausibility(t):
    """Part numbers that happen to look like designators (GTX312L, AMS1117) are penalised."""
    m = re.match(r'^([A-Z]+)(\d+)([A-Z]?)', t or '')
    if not m:
        return 0.0
    pre, num, tail = m.groups()
    pen = 0.0
    if tail and len(num) >= 2:
        pen += 0.8
    if DESIG_VOLT_TAIL and tail == 'V' and len(pre) >= 2:
        pen += 0.8                       # DCDC12V, LDO5V: a voltage, not a designator
    if len(pre) >= 3 and not any(pre.startswith(k) for k in _KNOWN_LONG_PREFIX):
        pen += 0.6
    if len(num) >= 4:
        pen += 0.6
    if DESIG_PREFIX_PEN and pre not in _EXACT_PREFIXES and pre not in _KNOWN_LONG_PREFIX and len(pre) >= 2:
        # a letter prefix no designator table uses (RS1M, SS34, US1M are diodes' part
        # numbers printed next to D12): weaker than a real designator at the same distance
        pen += 0.3 + (0.3 if tail else 0.0)
    return pen


def _ext_bbox(c):
    xs = [c.bbox[0], c.bbox[2]] + [pn.tip[0] for pn in c.pins]
    ys = [c.bbox[1], c.bbox[3]] + [pn.tip[1] for pn in c.pins]
    return [min(xs), min(ys), max(xs), max(ys)]


def tbox(it):
    return [it.x0, it.y0, it.x1, it.y1]


class Comp:
    def __init__(self, gid, bbox, pins):
        self.gids = [gid]
        self.bbox = list(bbox)
        self.pins = list(pins)
        self.type = None
        self.key = None
        self.name = None
        self.value = None
        self.desig_item = None
        self.label_item = None
        self.anon = False

    def __repr__(self):
        return f"Comp({self.key},{self.type},{self.bbox},{len(self.pins)}p)"


def group_body_info(S, P):
    """Per-group body geometry features."""
    glab = S['glab']
    info = {}
    for gid, g in S['groups'].items():
        x0, y0, x1, y1 = g['bbox']
        sub = (glab[y0:y1 + 1, x0:x1 + 1] == gid)
        info[gid] = {'mask': sub, 'off': (x0, y0)}
    return info


def _rect_main(c, S, P):
    """The part is a rectangle with pins: its rectangular hole is most of its body (a
    transformer with a small box drawn inside is still a transformer)."""
    r = rect_hole(c.gids[0], S, P)
    if not r:
        return False
    if not RECT_MAIN:
        return True
    x0, y0, x1, y1 = c.bbox
    return (r[2] - r[0]) * (r[3] - r[1]) >= 0.5 * max(1.0, (x1 - x0) * (y1 - y0))


RECT_MAIN = os.environ.get('PCB_RECT_MAIN', '1') == '1'


def rect_hole(gid, S, P):
    """Largest rectangular hole of a body group (IC body outline), image coords."""
    g = S['groups'][gid]
    if 'rect' in g:
        return g['rect']
    x0, y0, x1, y1 = g['bbox']
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == gid).astype(np.uint8)
    best = None
    holes = []
    npins = len([pn for pn in g.get('pins', []) if pn.o in ('h', 'v')])
    saved = None
    for attempt in (0, 1):
        if attempt == 1:
            if not RECT_CLOSE:
                break
            if best is not None and not (npins >= 8 and not _pins_on_rect(g, best[1], P)):
                break
            # one-pixel cuts in the outline where pin stubs / touching glyphs were removed;
            # a hole found only this way must be the body outline (its edges carry the pins
            # or it fills the group); a pin-less small hole found before is kept otherwise
            saved = (best, holes)
            best = None
            m = cv2.morphologyEx(np.pad(m, 1), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))[1:-1, 1:-1]
            holes = []
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for c, hh in zip(cnts, hier[0]):
            if hh[3] < 0:
                continue
            a = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            if a < 0.8 * bw * bh:
                continue
            rb = [x0 + bx, y0 + by, x0 + bx + bw - 1, y0 + by + bh - 1]
            if min(bw, bh) >= 0.3 * P.u:
                holes.append(rb)
            if bw < 0.9 * P.u or bh < 0.9 * P.u:
                continue
            if attempt == 1 and not (_pins_on_rect(g, rb, P) or (bw >= 0.7 * (x1 - x0) and bh >= 0.7 * (y1 - y0))):
                continue
            if best is None or a > best[0]:
                best = (a, rb)
    if best is None and saved is not None and saved[0] is not None:
        best, holes = saved
    if best is None and RECT_RED_FALLBACK and (x1 - x0) >= 1.5 * P.u and (y1 - y0) >= 1.5 * P.u:
        # outline pieces lost to the stub removal (a top edge taken for a pin line): look
        # for the body's rectangle in the raw symbol-colour mask inside the group's box
        mr = (P.red[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)
        mr = cv2.morphologyEx(np.pad(mr, 1), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))[1:-1, 1:-1]
        cnts2, hier2 = cv2.findContours(mr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier2 is not None:
            for c, hh in zip(cnts2, hier2[0]):
                if hh[3] < 0:
                    continue
                a = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                rb2 = [x0 + bx, y0 + by, x0 + bx + bw - 1, y0 + by + bh - 1]
                if a < 0.8 * bw * bh or not (_pins_on_rect(g, rb2, P) or (bw >= 0.7 * (x1 - x0) and bh >= 0.7 * (y1 - y0))):
                    continue
                if best is None or a > best[0]:
                    best = (a, [x0 + bx, y0 + by, x0 + bx + bw - 1, y0 + by + bh - 1])
    if best is not None and RECT_COMPARTMENTS and len(holes) > 1:
        # an IC body split into compartments by divider lines (connector banks, 'BANK 1'):
        # the outline is the union of the stacked / side-by-side compartments
        U = list(best[1])
        tol = 2 * P.t + 3
        grown = True
        while grown:
            grown = False
            for h in holes:
                if h[0] >= U[0] and h[1] >= U[1] and h[2] <= U[2] and h[3] <= U[3]:
                    continue
                same_x = abs(h[0] - U[0]) <= tol and abs(h[2] - U[2]) <= tol
                same_y = abs(h[1] - U[1]) <= tol and abs(h[3] - U[3]) <= tol
                if same_x and (0 <= h[1] - U[3] <= tol or 0 <= U[1] - h[3] <= tol):
                    U = [min(U[0], h[0]), min(U[1], h[1]), max(U[2], h[2]), max(U[3], h[3])]
                    grown = True
                elif same_y and (0 <= h[0] - U[2] <= tol or 0 <= U[0] - h[2] <= tol):
                    U = [min(U[0], h[0]), min(U[1], h[1]), max(U[2], h[2]), max(U[3], h[3])]
                    grown = True
        best = (best[0], U)
    g['rect'] = best[1] if best else None
    return g['rect']


RECT_COMPARTMENTS = os.environ.get('PCB_RECT_COMP', '1') == '1'


def _pins_on_rect(g, rb, P):
    """Half or more of the group's pins start on the edges of rectangle rb: it is the body
    outline (an IC), not a small box drawn inside a larger symbol."""
    pins = [pn for pn in g.get('pins', []) if pn.o in ('h', 'v')]
    if len(pins) < 4:
        return False
    tol = 1.5 * P.t + 3
    n = 0
    for pn in pins:
        bx, by = pn.base
        if pn.o == 'h' and rb[1] - tol <= by <= rb[3] + tol and (abs(bx - rb[0]) <= tol + P.t or abs(bx - rb[2]) <= tol + P.t):
            n += 1
        elif pn.o == 'v' and rb[0] - tol <= bx <= rb[2] + tol and (abs(by - rb[1]) <= tol + P.t or abs(by - rb[3]) <= tol + P.t):
            n += 1
    return n >= 0.5 * len(pins)
RECT_CLOSE = os.environ.get('PCB_RECT_CLOSE', '1') == '1'
RECT_RED_FALLBACK = os.environ.get('PCB_RECT_RED', '1') == '1'
SMALL_EDGE_PINS = os.environ.get('PCB_SMALL_EDGE_PINS', '1') == '1'
PINLESS_PEN = float(os.environ.get('PCB_PINLESS_PEN', '1.5'))


def clean_pins(S, P):
    u = P.u
    t = P.t
    markers = list(P.gray_dots) + list(getattr(P, 'noconn', []))
    big_rects = []
    if SMALL_EDGE_PINS:
        for gid_, g_ in S['groups'].items():
            bb_ = g_['bbox']
            if max(bb_[2] - bb_[0], bb_[3] - bb_[1]) > 4 * u:
                r_ = rect_hole(gid_, S, P)
                if r_:
                    big_rects.append((gid_, r_))

    def in_big_rect(gid_, bb_):
        # segments of a 7-segment display (or anything drawn inside a larger outline)
        return any(o != gid_ and r_[0] - 2 <= bb_[0] and bb_[2] <= r_[2] + 2 and r_[1] - 2 <= bb_[1] and bb_[3] <= r_[3] + 2
                   for o, r_ in big_rects)
    for gid, g in S['groups'].items():
        x0, y0, x1, y1 = g['bbox']
        keep = []
        rect = None
        rect0 = rect_hole(gid, S, P)
        bx0, by0, bx1, by1 = rect0 if rect0 else (x0, y0, x1, y1)
        for pn in g['pins']:
            if pn.kind == 'free':
                tx, ty = pn.tip
                inside = (bx0 - 0.15 * u <= tx <= bx1 + 0.15 * u) and (by0 - 0.15 * u <= ty <= by1 + 0.15 * u)
                if inside or pn.length < 0.45 * u:
                    pn.group = -1
                    continue
                marked = any(math.hypot(mx - tx, my - ty) <= 1.8 * t + 2 for mx, my in markers)
                marked = marked or getattr(pn, 'bar', False)       # a switch's pin bar end
                if not marked and SMALL_EDGE_PINS and max(x1 - x0, y1 - y0) <= 4 * u and not in_big_rect(gid, (x0, y0, x1, y1)):
                    # a small symbol (net flag pair) whose stub leaves its outline at the edge
                    bx, by = pn.base
                    tol = t + 2
                    if pn.o == 'h' and (abs(bx - x0) <= tol or abs(bx - x1) <= tol) and y0 - tol <= by <= y1 + tol:
                        marked = True
                    if pn.o == 'v' and (abs(by - y0) <= tol or abs(by - y1) <= tol) and x0 - tol <= bx <= x1 + tol:
                        marked = True
                if not marked:
                    if rect is None:
                        rect = rect_hole(gid, S, P) or False
                    ok = False
                    if rect:
                        rx0, ry0, rx1, ry1 = rect
                        bx, by = pn.base
                        tol = 1.5 * t + 2
                        if pn.o == 'h' and ry0 - tol <= by <= ry1 + tol and (abs(bx - rx0) <= tol + t or abs(bx - rx1) <= tol + t):
                            ok = True
                        if pn.o == 'v' and rx0 - tol <= bx <= rx1 + tol and (abs(by - ry0) <= tol + t or abs(by - ry1) <= tol + t):
                            ok = True
                    if not ok:
                        pn.group = -1
                        continue
            keep.append(pn)
        g['pins'] = keep
    S['pins'] = [pn for pn in S['pins'] if pn.group != -1 and pn.group in S['groups']]


def shape_features(gid, S, P):
    """Simple shape descriptors of a body group."""
    g = S['groups'][gid]
    glab = S['glab']
    x0, y0, x1, y1 = g['bbox']
    m = (glab[y0:y1 + 1, x0:x1 + 1] == gid).astype(np.uint8)
    h, w = m.shape
    f = {'w': w, 'h': h, 'area': int(m.sum())}
    # closed contours (holes)
    cnts, hier = cv2.findContours(m.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is not None:
        for c, hh in zip(cnts, hier[0]):
            if hh[3] >= 0:
                a = cv2.contourArea(c)
                if a > (1.2 * P.t) ** 2:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    rect = a / max(bw * bh, 1)
                    holes.append((a, bw, bh, rect))
    f['holes'] = holes
    f['fill'] = m.sum() / max(w * h, 1)
    return f


_P_CTX = [None]
_S_CTX = [None]
SPLIT_FLAG_STACKS = True
ORPHAN_DESIG = True
MERGE_CONTACT_ORPHANS = True
MERGE_HOST_TYPES = ('mosfet', 'bjt', 'box', 'switch', 'amp', 'transfomer', 'crystal')


DESIG_CONSIST = os.environ.get('PCB_DESIG_CONSIST', '0') == '1'   # off: no measurable gain, -7 on one holdout case
HIDDEN_GND_NONE = os.environ.get('PCB_GND_NONE', '0') == '1'   # public GT: hidden ground name -> 'GND'
TRI_AGND = os.environ.get('PCB_TRI_AGND', '1') == '1'
ORPHAN_SERIES = os.environ.get('PCB_ORPHAN_SERIES', '1') == '1'
FAR_FROM_TIPS = os.environ.get('PCB_FAR_FROM_TIPS', '1') == '1'
FLAG_VOLT = os.environ.get('PCB_FLAG_VOLT', '1') == '1'   # a net port named 3.3V / 5V
IO_ALT = os.environ.get('PCB_IO_ALT', '1') == '1'
LIB_MIN4 = os.environ.get('PCB_LIB_MIN4', '1') == '1'   # 'USB-' is a net label, not the library part 'USB'
BALL_NUMS = os.environ.get('PCB_BALL_NUMS', '1') == '1'   # die-prefixed ball numbers: 1C12, 1AB3
# colour style of the source drawing (style.analyse): set per image by the pipeline
SRC_STYLE = ['easyeda']


def _tri_ground(c, P):
    """EasyEDA's analog-ground symbol: a hollow triangle (one closed, roughly half-filled
    cell) instead of stacked bars; its net name (AGND) is not printed.  Measured on the
    dev + holdout labels: 94 of 95 nameless triangles are AGND, no GND is drawn so."""
    if not TRI_AGND or SRC_STYLE[0] not in ('easyeda', 'off'):
        return False
    f = getattr(c, 'feat', None)
    if not f or len(f.get('holes') or []) != 1:
        return False
    a, bw, bh, rect = f['holes'][0]
    return 0.35 <= rect <= 0.65 and a >= 0.12 * f['w'] * f['h'] and max(bw, bh) >= 0.5 * P.u


def _desig_consistency(C, r, cidx, items, comps, u, w=0.5, cap=1.0):
    """Second pass of the designator assignment.  On one sheet, small parts of one kind and
    orientation carry their designator at the same place (e.g. above-left of every
    horizontal capacitor).  Learn that offset from the unambiguous first-pass pairs and
    penalise assignments that deviate from it: a text between two capacitors goes to the
    one it is 'typically' placed for, not merely the nearer one."""
    nC = len(comps)
    pairs = [(i, j) for i, j in zip(r, cidx) if j < nC and C[i, j] < 1e5]
    if len(pairs) < 3:
        return None

    def grp(i, j):
        c = comps[j]
        pt = prefix_type(items[i].text)
        if len(c.pins) != 2 or pt not in ('r', 'c', 'l', 'd', 'led', 'fuse'):
            return None                    # two-pin passives only: IC labels vary with body size
        o = 'h' if c.feat['w'] >= c.feat['h'] else 'v'
        return (pt, o)

    def off(i, j):
        c = comps[j]
        cx, cy = (c.bbox[0] + c.bbox[2]) / 2.0, (c.bbox[1] + c.bbox[3]) / 2.0
        return np.array([(items[i].cx - cx) / u, (items[i].cy - cy) / u])
    samples = {}
    for i, j in pairs:
        g = grp(i, j)
        if g is None:
            continue
        row = np.sort(C[i, :nC]); col = np.sort(C[:, j])
        m_row = (row[1] - row[0]) if len(row) > 1 else 9.0
        m_col = (col[1] - col[0]) if len(col) > 1 else 9.0
        if C[i, j] <= row[0] + 1e-9 and C[i, j] <= col[0] + 1e-9 and min(m_row, m_col) >= 0.4:
            samples.setdefault(g, []).append(off(i, j))
    typ = {}
    for g, lst in samples.items():
        if len(lst) < 3:
            continue
        A = np.array(lst)
        med = np.median(A, axis=0)
        spread = float(np.median(np.linalg.norm(A - med, axis=1)))
        if spread <= 0.5:
            typ[g] = med
    if not typ:
        return None
    C2 = C.copy()
    changed = False
    for i in range(C.shape[0]):
        for j in range(nC):
            if C[i, j] >= 1e5:
                continue
            g = grp(i, j)
            if g in typ:
                C2[i, j] += w * min(float(np.linalg.norm(off(i, j) - typ[g])), cap)
                changed = True
    return C2 if changed else None


DESIG_CHAIN = os.environ.get('PCB_DESIG_CHAIN', '1') == '1'


def _desig_chains(C, r, cidx, items, comps, u):
    """A row of equal parts with the designators all set one step to the side (a bank of
    decoupling capacitors, each label a little right of its part and over the next one): the
    nearest-text assignment shifts every label onto the neighbouring part, which leaves the
    first part without a label and the last label without a part.  Every part carries a
    designator, so such a chain is re-assigned one step over when the shifted pairs all show
    one and the same label offset."""
    nC = len(comps)
    at = {i: j for i, j in zip(r, cidx) if j < nC and C[i, j] < 1e5}
    ac = {j: i for i, j in at.items()}

    def kind(j):
        c = comps[j]
        if len(c.pins) != 2:
            return None
        return 'h' if c.feat['w'] >= c.feat['h'] else 'v'

    def off(i, j):
        c = comps[j]
        return ((items[i].cx - (c.bbox[0] + c.bbox[2]) / 2.0) / u, (items[i].cy - (c.bbox[1] + c.bbox[3]) / 2.0) / u)

    def gap(i, j):
        return box_gap(comps[j].bbox, tbox(items[i])) / u

    changed = False
    for i0 in range(len(items)):
        if i0 in at:
            continue
        pt = prefix_type(items[i0].text)
        if pt not in ('r', 'c', 'l', 'd', 'led', 'fuse', 'box'):
            continue                        # ('U1' over a row of LED-shaped two-pin parts)
        # follow the chain: label -> nearest same-kind part -> that part's label -> ...
        path, seen, i, k0, size = [], set(), i0, None, None
        ok, displaced = False, None
        while len(path) < 60:
            best = None
            for j in range(nC):
                if j in seen or C[i, j] >= 1e5 or kind(j) is None:
                    continue
                if k0 is not None and (kind(j) != k0 or
                                       abs(comps[j].feat['w'] - size[0]) > 0.2 * size[0] + 2 or
                                       abs(comps[j].feat['h'] - size[1]) > 0.2 * size[1] + 2):
                    continue
                g = gap(i, j)
                if g <= 1.6 and (best is None or g < best[0]):
                    best = (g, j)
            if best is None:
                break
            j = best[1]
            if k0 is None:
                k0, size = kind(j), (comps[j].feat['w'], comps[j].feat['h'])
            seen.add(j)
            path.append((i, j))
            if j not in ac:
                ok = True
                break
            i = ac[j]
            if prefix_type(items[i].text) != pt:
                # the first part of the row holds a label of another kind ('Q3' of the
                # transistor beside the row): the row's own label takes it back
                if pt in ('r', 'c', 'l') and prefix_type(items[i].text) not in ('r', 'c', 'l', None):
                    ok, displaced = True, i
                break
        if not ok or len(path) < 3:
            continue
        # the shifted pairs must all show one label offset (the nearest-text pairs need not:
        # the parts of a row are not always evenly spaced)
        new = np.array([off(i, j) for i, j in path])
        if np.abs(new - np.median(new, axis=0)).max() > 0.25:
            continue
        if displaced is not None:
            del at[displaced]
        for i, j in path:
            at[i] = j
            ac[j] = i
        changed = True
    if not changed:
        return None
    rr = sorted(at)
    return rr, [at[i] for i in rr]


MERGE_SWITCH_PIECES = os.environ.get('PCB_SW_MERGE', '1') == '1'


def _merge_switch_pieces(comps, u):
    """Tactile switches are drawn as two separate pin-to-pin bars plus the lever: only the
    piece next to the designator gets it, the other bar (with its two pins) would be dropped.
    Fold an undesignated piece lying parallel to the switch (overlapping along the bars,
    within ~2.6u across) into it."""
    hosts = [c for c in comps.values() if c.type == 'switch' and c.key and len(c.pins) < 4]
    free = [c for c in comps.values() if c.type is None and c.key is None and not c.anon and c.pins
            and not getattr(c, 'flag', False) and len(c.pins) <= 3]
    for c in hosts:
        for p in list(free):
            if len(c.pins) + len(p.pins) > 4:
                continue
            ox = min(c.bbox[2], p.bbox[2]) - max(c.bbox[0], p.bbox[0])
            oy = min(c.bbox[3], p.bbox[3]) - max(c.bbox[1], p.bbox[1])
            wmin = max(1.0, min(c.bbox[2] - c.bbox[0], p.bbox[2] - p.bbox[0]))
            hmin = max(1.0, min(c.bbox[3] - c.bbox[1], p.bbox[3] - p.bbox[1]))
            gx = max(0.0, max(c.bbox[0], p.bbox[0]) - min(c.bbox[2], p.bbox[2]))
            gy = max(0.0, max(c.bbox[1], p.bbox[1]) - min(c.bbox[3], p.bbox[3]))
            if (ox >= 0.6 * wmin and gy <= 2.6 * u) or (oy >= 0.6 * hmin and gx <= 2.6 * u):
                c.bbox = [min(c.bbox[0], p.bbox[0]), min(c.bbox[1], p.bbox[1]),
                          max(c.bbox[2], p.bbox[2]), max(c.bbox[3], p.bbox[3])]
                c.gids = c.gids + p.gids
                c.pins = c.pins + p.pins
                p.pins = []
                p.type = None
                p.key = None
                free.remove(p)
                c.sw_merged = True
    if SW_PAIR_MERGE:
        # both bars took a designator: one of them is a net name written on the wire at a
        # pin ('KEY1' beside 'SW1'); keep the one off the wires
        P_ = _P_CTX[0]
        hosts = [c for c in comps.values() if c.type == 'switch' and c.key and c.pins and len(c.pins) <= 2
                 and c.desig_item is not None]
        done = set()
        for i, a in enumerate(hosts):
            for b in hosts[i + 1:]:
                if id(a) in done or id(b) in done or len(a.pins) + len(b.pins) > 4:
                    continue
                ox = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
                oy = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
                wmin = max(1.0, min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0]))
                hmin = max(1.0, min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1]))
                gx = max(0.0, max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2]))
                gy = max(0.0, max(a.bbox[1], b.bbox[1]) - min(a.bbox[3], b.bbox[3]))
                if not ((ox >= 0.8 * wmin and gy <= 2.0 * u) or (oy >= 0.8 * hmin and gx <= 2.0 * u)):
                    continue
                wa, wb = _on_wire(a.desig_item, P_), _on_wire(b.desig_item, P_)
                if wa == wb:
                    continue
                keep, drop = (b, a) if wa else (a, b)
                keep.bbox = [min(keep.bbox[0], drop.bbox[0]), min(keep.bbox[1], drop.bbox[1]),
                             max(keep.bbox[2], drop.bbox[2]), max(keep.bbox[3], drop.bbox[3])]
                keep.gids = keep.gids + drop.gids
                keep.pins = keep.pins + drop.pins
                drop.pins = []
                drop.type = None
                drop.key = None
                keep.sw_merged = True
                done.update([id(a), id(b)])


SW_PAIR_MERGE = os.environ.get('PCB_SW_PAIR_MERGE', '1') == '1'
MERGE_TOUCHING = os.environ.get('PCB_MERGE_TOUCH', '1') == '1'


def _merge_touching_pieces(comps, S, P):
    """A transistor's body diode drawn as a separate stroke group often takes the
    designator (it is the piece nearest to the text) while the transistor itself - gate,
    channel, the three real pins - stays undesignated and would be dropped.  The diode's
    'pins' end ON the transistor's drain / source lines: fold the undesignated piece they
    touch into the designated one and drop those internal joints."""
    glab = S['glab']
    H, W = glab.shape
    r = int(P.t + 2)
    gid2c = {}
    for c in comps.values():
        for g in c.gids:
            gid2c[g] = c

    def touching(pn, gids):
        x, y = int(round(pn.tip[0])), int(round(pn.tip[1]))
        win = glab[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
        return set(int(v) for v in np.unique(win)) & set(gids)
    for c in list(comps.values()):
        if c.key is None or c.anon or (c.type or prefix_type(c.key)) not in ('mosfet', 'bjt'):
            continue
        cand = {}
        for pn in c.pins:
            x, y = int(round(pn.tip[0])), int(round(pn.tip[1]))
            win = glab[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)]
            for g in np.unique(win):
                g = int(g)
                o = gid2c.get(g)
                if g <= 0 or g in c.gids or o is None or o is c:
                    continue
                if o.type is not None or o.key is not None or o.anon or getattr(o, 'flag', False) or not o.pins:
                    continue
                cand.setdefault(id(o), [o, 0])[1] += 1
        for o, k in cand.values():
            if k < 1 or len(o.pins) + len(c.pins) > 8:
                continue
            keep = [pn for pn in c.pins if not touching(pn, o.gids)] + [pn for pn in o.pins if not touching(pn, c.gids)]
            if len(keep) < 2:
                continue
            c.bbox = [min(c.bbox[0], o.bbox[0]), min(c.bbox[1], o.bbox[1]), max(c.bbox[2], o.bbox[2]), max(c.bbox[3], o.bbox[3])]
            c.gids = c.gids + o.gids
            for g in o.gids:
                gid2c[g] = c
            c.pins = keep
            o.pins = []
            o.type = None
            o.key = None
            if c.type is not None:
                c.type = 'bjt' if _is_bjt(c, S, P) else 'mosfet'


DOT_END_PIN = os.environ.get('PCB_DOT_END_PIN', '1') == '1'
DROP_NET_TIES = os.environ.get('PCB_NET_TIE', '1') == '1'
SPLIT_PAIRS = os.environ.get('PCB_SPLIT_PAIRS', '1') == '1'
DROP_FRAMES = os.environ.get('PCB_DROP_FRAMES', '1') == '1'
LINE_DROP = os.environ.get('PCB_LINE_DROP', '1') == '1'
MOUNT_HOLES = os.environ.get('PCB_MOUNT_HOLES', '1') == '1'   # screw / mounting-hole symbols -> 'm3螺丝'
NET_SHORT_OUT = os.environ.get('PCB_NET_SHORT', '1') == '1'   # short (net-tie) diamonds -> 'net_short' (official type list)
FAR_DESIG = os.environ.get('PCB_FAR_DESIG', '1') == '1'
SPLIT_STACKS = os.environ.get('PCB_SPLIT_STACKS', '1') == '1'
STACK_PITCH = os.environ.get('PCB_STACK_PITCH', '1') == '1'


def _desig_text(t):
    """Designator text with a trailing dot read off a body corner removed ('R5.' -> 'R5')."""
    t = t.replace(' ', '')
    if DESIG_TRAIL_PUNCT and len(t) >= 3 and t[-1] in '.,\'' and is_desig(t[:-1]) and re.search(r'\d$', t[:-1]):
        return t[:-1]
    if DESIG_TRAIL_PUNCT and re.fullmatch(r'LD0\d{1,3}', t):
        return 'LDO' + t[3:]                   # a regulator's LDO1 read with a zero
    if DESIG_TRAIL_PUNCT and re.fullmatch(r'[A-Z]{1,3}[-·][1-9]\d{0,3}', t) and is_desig(re.sub(r'[-·]', '', t)):
        return re.sub(r'[-.·]', '', t)         # a junction dot on the label read as '-' ('C-35')
    return t


STACK_CELLS = os.environ.get('PCB_STACK_CELLS', '1') == '1'
# new group ids must be unused in the label image too (ids of dropped groups stay there)
NEW_GID_GLAB = os.environ.get('PCB_NEW_GID_GLAB', '1') == '1'
HIDDEN_NOT_PIECE = os.environ.get('PCB_HIDDEN_NOT_PIECE', '1') == '1'
CELL_SUFFIX = os.environ.get('PCB_CELL_SUFFIX', '1') == '1'


def _stacked_cells(m, axis, cents, o0, t, u):
    """Resistor boxes stacked edge to edge: between every two pin rows a line runs across
    the whole body (the shared edge), and every cell is an empty box of resistor size.  A
    resistor network (small boxes inside one frame) or a DIP switch has no such full-width
    separators or has ink inside its cells."""
    prof = m.sum(axis=axis).astype(float)       # ink across the body at each position along the rows
    full = prof.max()
    span = m.shape[1] if axis == 1 else m.shape[0]
    if full < 1.5 * u or full > 4.0 * u:
        return False                           # body length of a resistor
    # the box's own edges: positions where the ink runs across the body
    lines = np.nonzero(prof >= 0.85 * full)[0]
    if not len(lines):
        return False
    rel = [cv - o0 for cv in cents]
    seps = []
    for i in range(len(rel) - 1):
        band = [p_ for p_ in lines if rel[i] + 0.25 * u < p_ < rel[i + 1] - 0.25 * u]
        if not band:
            return False
        seps.append(np.mean(band))
    edges = [p_ for p_ in lines if p_ < rel[0]] , [p_ for p_ in lines if p_ > rel[-1]]
    if not edges[0] or not edges[1]:
        return False                           # closed at both outer ends
    bounds = [max(edges[0])] + seps + [min(edges[1])]
    for i in range(len(bounds) - 1):
        a, b = int(round(bounds[i])), int(round(bounds[i + 1]))
        if not (0.55 * u <= b - a <= 1.6 * u):
            return False                       # cell height of a resistor box
        inner = prof[a + int(t) + 2:b - int(t) - 1]
        if len(inner) and inner.max() > 0.5 * full:
            return False                       # something drawn inside the cell
    return True


PIN_NUM_PEN = os.environ.get('PCB_PIN_NUM_PEN', '1') == '1'
BODY_RECT_NAME = os.environ.get('PCB_BODY_RECT_NAME', '1') == '1'


def _in_body_rect(c, x, y):
    """(x, y) lies within the part's drawn body rectangle, not just its box with the pin
    leads (a neighbour's part number between two pin leads is outside the body)."""
    if not BODY_RECT_NAME or _S_CTX[0] is None or not c.gids:
        return True
    try:
        r = rect_hole(c.gids[0], _S_CTX[0], _P_CTX[0])
    except Exception:
        return True
    if not r:
        return True
    return r[0] - 2 <= x <= r[2] + 2 and r[1] - 2 <= y <= r[3] + 2


def _on_pin_lead(c, it, u):
    """The text sits along one of the part's pin leads, against the line, between the body
    and the pin end: the place of a pin number, never of the part's designator."""
    cx_, cy_ = (c.bbox[0] + c.bbox[2]) / 2.0, (c.bbox[1] + c.bbox[3]) / 2.0
    for pn in c.pins:
        tx, ty = pn.tip
        if pn.o == 'h':
            a_, b_ = (tx - 1.5 * u, tx + 0.2 * u) if tx > cx_ else (tx - 0.2 * u, tx + 1.5 * u)
            if a_ <= it.x0 and it.x1 <= b_ and (0 <= ty - it.y1 <= 0.35 * u or 0 <= it.y0 - ty <= 0.35 * u):
                return True
        elif pn.o == 'v':
            a_, b_ = (ty - 1.5 * u, ty + 0.2 * u) if ty > cy_ else (ty - 0.2 * u, ty + 1.5 * u)
            if a_ <= it.y0 and it.y1 <= b_ and (0 <= tx - it.x1 <= 0.35 * u or 0 <= it.x0 - tx <= 0.35 * u):
                return True
    return False


ARROWS_TO_LED = os.environ.get('PCB_ARROWS_TO_LED', '1') == '1'
FOLD_NO_RES = os.environ.get('PCB_FOLD_NO_RES', '1') == '1'


def _zigzag(c, S, P):
    """A zigzag resistor: many short slanted strokes alternating in slope."""
    m = _body_mask(c, S)
    if m.sum() < 10:
        return False
    L = cv2.HoughLinesP(m.astype(np.uint8) * 255, 1, np.pi / 90, threshold=max(5, int(0.2 * P.u)),
                        minLineLength=max(4, int(0.25 * P.u)), maxLineGap=1)
    if L is None:
        return False
    pos = neg = 0
    for x1, y1, x2, y2 in np.asarray(L).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 20 < a < 80:
            pos += 1
        elif 100 < a < 160:
            neg += 1
    w, h = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
    return pos >= 3 and neg >= 3 and max(w, h) >= 2.2 * max(min(w, h), 1)


def _fold_light_arrows(comps, S, P):
    """An LED's light arrows drawn apart from the diode are a separate small piece; left
    alone they compete with the diode for its designator.  Fold a lone arrow pair into the
    two-pin part right next to it."""
    u = P.u
    for k, c in list(comps.items()):
        if c.type is not None or c.anon or c.key is not None or len(c.pins) > 1 or getattr(c, 'flag', False):
            continue
        w, h = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
        if max(w, h) > 1.6 * u or max(w, h) < 0.5 * u:
            continue
        m = _body_mask(c, S)
        if m.sum() < 10 or not (_arrow_pair_raw(m, u) or _arrow_ccs(np.pad(m, 1), u)):
            continue
        best = None
        for o in comps.values():
            if o is c or o.anon or o.type is not None or len(o.pins) not in (2, 3) or getattr(o, 'flag', False):
                continue
            ow, oh = o.bbox[2] - o.bbox[0], o.bbox[3] - o.bbox[1]
            if max(ow, oh) > 4.5 * u or (FOLD_NO_RES and len(o.pins) == 2 and (_resistor_box(o, S, P) or _zigzag(o, S, P))):
                continue                       # (a diode body is compact; a resistor is no LED)
            d = box_gap(c.bbox, o.bbox)
            if d <= 0.6 * u and (best is None or d < best[0]):
                best = (d, o)
        if best is None:
            continue
        o = best[1]
        o.bbox = [min(o.bbox[0], c.bbox[0]), min(o.bbox[1], c.bbox[1]), max(o.bbox[2], c.bbox[2]), max(o.bbox[3], c.bbox[3])]
        o.gids = o.gids + c.gids
        # a 'pin' where the arrows touched the diode is no pin
        x0, y0, x1, y1 = c.bbox
        o.pins = [pn for pn in o.pins if not (pn.kind in ('contact', 'dot', 'zero') and x0 - 3 <= pn.tip[0] <= x1 + 3
                                                and y0 - 3 <= pn.tip[1] <= y1 + 3)] or o.pins
        del comps[k]


def _split_stacked_rows(comps, S, P, texts):
    """Resistors stacked so tightly that their bodies touch form one body with 2N pins in N
    parallel rows.  With N designators of one kind beside it and a closed body in every row
    slice, cut the body into the N parts between the rows."""
    glab = S['glab']
    t, u = P.t, P.u
    for k, c in list(comps.items()):
        n2 = len(c.pins)
        if c.type is not None or c.anon or n2 < 4 or n2 % 2 or len(c.gids) != 1 or getattr(c, 'flag', False):
            continue
        g = c.gids[0]
        if g not in S['groups']:
            continue
        os_ = {pn.o for pn in c.pins}
        if os_ == {'h'}:
            cr, al, axis = (lambda p: p.tip[1]), (lambda p: p.tip[0]), 1
        elif os_ == {'v'}:
            cr, al, axis = (lambda p: p.tip[0]), (lambda p: p.tip[1]), 0
        else:
            continue
        pins = sorted(c.pins, key=cr)
        rows = [pins[i:i + 2] for i in range(0, n2, 2)]
        N = len(rows)
        if any(abs(cr(a) - cr(b)) > 0.3 * u or abs(al(a) - al(b)) < 1.5 * u for a, b in rows):
            continue
        cents = [(cr(a) + cr(b)) / 2 for a, b in rows]
        gaps = np.diff(cents)
        if gaps.min() < 0.6 * u or gaps.max() > (2.6 if STACK_PITCH else 2.0) * u or gaps.max() - gaps.min() > 0.3 * u:
            continue
        near = [prefix_type(_desig_text(it.text)) for it in texts if is_desig(_desig_text(it.text))
                and prefix_type(_desig_text(it.text)) in ('r', 'c', 'l') and box_gap(c.bbox, tbox(it)) <= 2.5 * u]
        x0, y0, x1, y1 = [int(v) for v in S['groups'][g]['bbox']]
        m = (glab[y0:y1 + 1, x0:x1 + 1] == g)
        o0 = y0 if axis == 1 else x0
        kind = max(set(near), key=near.count) if near else None
        if not near or max(near.count(x) for x in set(near)) < N:
            if not (STACK_CELLS and _stacked_cells(m, axis, cents, o0, t, u)):
                continue
            kind = 'r'
        cuts = [int(round((cents[i] + cents[i + 1]) / 2)) - o0 for i in range(N - 1)]
        # parts a few pixels apart: cut in the empty band between the pin rows
        prof = m.sum(axis=axis)
        banded = []
        for i in range(N - 1):
            lo_, hi_ = int(round(cents[i])) - o0 + 1, int(round(cents[i + 1])) - o0
            if hi_ - lo_ >= 3 and prof[lo_:hi_].min() <= 2 * t:
                banded.append(lo_ + int(np.argmin(prof[lo_:hi_])))
        if len(banded) == N - 1:
            cuts = banded
        bounds = [0] + cuts + [m.shape[0] if axis == 1 else m.shape[1]]
        mg = int(np.ceil(t + 1))
        ok = True
        for i in range(N):
            if len(banded) == N - 1:
                break                          # separated by empty bands: no closed body needed
            lo, hi = max(bounds[i] - mg, 0), bounds[i + 1] + mg
            sl = m[lo:hi, :] if axis == 1 else m[:, lo:hi]
            cnts, hier = cv2.findContours(sl.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hier is None or not any(hh[3] >= 0 and cv2.contourArea(cc) >= (0.3 * u) ** 2 for cc, hh in zip(cnts, hier[0])):
                ok = False                     # every slice must hold a closed body (resistor box)
                break
        if not ok:
            continue
        sub = glab[y0:y1 + 1, x0:x1 + 1]
        new_ids = [g]
        nxt = max(list(S['groups'].keys()) + [0, int(glab.max()) if NEW_GID_GLAB else 0]) + 1
        for i in range(1, N):
            new_ids.append(nxt)
            nxt += 1
        for i in range(1, N):
            sel = np.zeros_like(m)
            if axis == 1:
                sel[bounds[i]:bounds[i + 1], :] = True
            else:
                sel[:, bounds[i]:bounds[i + 1]] = True
            sub[m & sel] = new_ids[i]
        for i, gid in enumerate(new_ids):
            mm = (sub == gid)
            ys, xs = np.nonzero(mm)
            if not len(ys):
                ok = False
                break
            bb = [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max())]
            for pn in rows[i]:
                pn.group = gid
            if gid == g:
                S['groups'][g]['bbox'] = bb
                S['groups'][g]['pins'] = list(rows[i])
                S['groups'][g].pop('rect', None)
                c.bbox, c.pins = list(bb), list(rows[i])
                c.feat = shape_features(g, S, P)
                c.cell_kind = kind
            else:
                S['groups'][gid] = {'id': gid, 'bbox': bb, 'area': int(mm.sum()), 'pins': list(rows[i])}
                nc = Comp(gid, list(bb), list(rows[i]))
                nc.feat = shape_features(gid, S, P)
                nc.cell_kind = kind
                comps[gid] = nc


def _far_designators(comps, texts, used_text, P):
    """Rows of capacitors hanging from a supply rail often carry their designator and value
    above the rail, 2-4 units from the body.  Two-pin parts left without a designator get a
    free designator of a two-pin kind that is aligned with them across the rail."""
    u = P.u
    parts = [c for c in comps.values() if c.key is None and c.type is None and not c.anon and len(c.pins) == 2
             and not getattr(c, 'flag', False)]
    texts = [it for it in texts if prefix_type(it.text) in ('r', 'c', 'l', 'd', 'led', 'fuse')]
    if not parts or not texts:
        return
    C = np.full((len(parts), len(texts)), 1e6)
    for i, c in enumerate(parts):
        a, b = c.pins
        vert = abs(a.tip[1] - b.tip[1]) > abs(a.tip[0] - b.tip[0])
        cx, cy = (c.bbox[0] + c.bbox[2]) / 2, (c.bbox[1] + c.bbox[3]) / 2
        if max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) > 3 * u:
            continue
        for j, it in enumerate(texts):
            if vert:
                off, gap = abs(it.cx - cx), min(abs(it.y1 - c.bbox[1]), abs(it.y0 - c.bbox[3]))
                if FAR_FROM_TIPS:
                    # long leads: count from the lead's end, where the rail wire starts
                    gap = min([gap] + [abs(it.y1 - pn.tip[1]) for pn in c.pins if pn.tip[1] < c.bbox[1]]
                              + [abs(it.y0 - pn.tip[1]) for pn in c.pins if pn.tip[1] > c.bbox[3]])
            else:
                off, gap = abs(it.cy - cy), min(abs(it.x1 - c.bbox[0]), abs(it.x0 - c.bbox[2]))
                if FAR_FROM_TIPS:
                    gap = min([gap] + [abs(it.x1 - pn.tip[0]) for pn in c.pins if pn.tip[0] < c.bbox[0]]
                              + [abs(it.x0 - pn.tip[0]) for pn in c.pins if pn.tip[0] > c.bbox[2]])
            if off > 0.6 * u or gap > 4.5 * u:
                continue
            if getattr(c, 'cell_kind', None) and prefix_type(it.text) != c.cell_kind:
                continue                       # a resistor cell takes a resistor's label
            # nothing else in between: no other part's box crosses the line to the text
            x0, x1 = min(cx, it.cx), max(cx, it.cx)
            y0, y1 = min(cy, it.cy), max(cy, it.cy)
            if any(o is not c and o.bbox[0] < x1 and o.bbox[2] > x0 and o.bbox[1] < y1 and o.bbox[3] > y0
                   and not o.anon and o.type is None and len(o.pins) >= 2 for o in comps.values()):
                continue
            C[i, j] = gap / u + off / u
    if (C < 1e5).sum() == 0:
        return
    r, cc = linear_sum_assignment(C)
    for i, j in zip(r, cc):
        if C[i, j] < 1e5:
            c, it = parts[i], texts[j]
            c.desig_item = it
            c.key = it.text
            used_text.add(it.id)


def _split_grouped_pairs(comps, S, P, texts):
    """Two two-pin parts drawn close together (stacked capacitors a few pixels apart) are
    grouped into one 4-pin body by the proximity grouping.  When the four pins form two
    parallel rows (one pin at each end of each row), the body is empty along a line between
    the rows, and two designators of two-pin kinds sit nearby, split it before the
    designators are handed out."""
    glab = S['glab']
    t, u = P.t, P.u
    for k, c in list(comps.items()):
        if c.type is not None or c.anon or len(c.pins) != 4 or len(c.gids) != 1 or getattr(c, 'flag', False):
            continue
        g = c.gids[0]
        if g not in S['groups']:
            continue
        x0, y0, x1, y1 = [int(v) for v in S['groups'][g]['bbox']]
        if max(x1 - x0, y1 - y0) > 5 * u:
            continue
        near = [prefix_type(_desig_text(it.text)) for it in texts if is_desig(_desig_text(it.text))
                and prefix_type(_desig_text(it.text)) in ('r', 'c', 'l', 'd', 'led') and box_gap(c.bbox, tbox(it)) <= 2.5 * u]
        if not near or max(near.count(x) for x in set(near)) < 2:
            continue                           # two designators of one two-pin kind
        hs = [pn for pn in c.pins if pn.o == 'h']
        vs = [pn for pn in c.pins if pn.o == 'v']
        if len(hs) == 4:
            cr, al, axis = (lambda p: p.tip[1]), (lambda p: p.tip[0]), 1
        elif len(vs) == 4:
            cr, al, axis = (lambda p: p.tip[0]), (lambda p: p.tip[1]), 0
        else:
            continue
        pins = sorted(c.pins, key=cr)
        r1, r2 = pins[:2], pins[2:]
        if abs(cr(r1[0]) - cr(r1[1])) > 0.3 * u or abs(cr(r2[0]) - cr(r2[1])) > 0.3 * u:
            continue
        c1 = (cr(r1[0]) + cr(r1[1])) / 2; c2 = (cr(r2[0]) + cr(r2[1])) / 2
        if not (0.6 * u <= c2 - c1 <= 3.0 * u):
            continue
        if abs(al(r1[0]) - al(r1[1])) < 1.5 * u or abs(al(r2[0]) - al(r2[1])) < 1.5 * u:
            continue
        m = (glab[y0:y1 + 1, x0:x1 + 1] == g)
        prof = m.sum(axis=axis)                 # ink per row (horizontal parts) / column
        o0 = y0 if axis == 1 else x0
        a_, b_ = int(round(c1)) - o0 + 1, int(round(c2)) - o0
        if b_ - a_ < 3:
            continue
        seg = prof[a_:b_]
        cut = a_ + int(np.argmin(seg))
        if seg.min() > 2 * t:
            continue                           # one connected drawing (an LED pair overlapping)
        new = max(list(S['groups'].keys()) + [0, int(glab.max()) if NEW_GID_GLAB else 0]) + 1
        sub = glab[y0:y1 + 1, x0:x1 + 1]
        sel = np.zeros_like(m)
        if axis == 1:
            sel[cut + 1:, :] = True
        else:
            sel[:, cut + 1:] = True
        sub[m & sel] = new
        m1 = (sub == g); m2 = (sub == new)
        if m1.sum() < 6 or m2.sum() < 6:
            sub[m2] = g
            continue

        def bbox_of(mm):
            ys, xs = np.nonzero(mm)
            return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max())]
        bb1, bb2 = bbox_of(m1), bbox_of(m2)
        s1 = (bb1[2] - bb1[0], bb1[3] - bb1[1]); s2 = (bb2[2] - bb2[0], bb2[3] - bb2[1])
        if any(abs(a - b) > 0.35 * max(a, b) + 2 for a, b in zip(s1, s2)):
            sub[m2] = g                        # two copies of one symbol look alike
            continue
        for pn in r2:
            pn.group = new
        S['groups'][g]['bbox'] = bb1
        S['groups'][g]['pins'] = list(r1)
        S['groups'][g].pop('rect', None)
        S['groups'][new] = {'id': new, 'bbox': bb2, 'area': int(m2.sum()), 'pins': list(r2)}
        c.bbox, c.pins = list(bb1), list(r1)
        c.feat = shape_features(g, S, P)
        nc = Comp(new, list(bb2), list(r2))
        nc.feat = shape_features(new, S, P)
        comps[new] = nc


def _is_net_tie(c, P):
    """A small diamond between two wires (EasyEDA net-tie / 'short' symbol): it has no
    designator of its own and is not a part in the official labels; the text next to it is
    the net name written on the wire."""
    if len(c.pins) not in (1, 2):
        return False
    if not c.feat.get('holes'):
        return CUT_DIAMOND and _cut_diamond(c, P)
    w, h = c.feat['w'], c.feat['h']
    u = P.u
    if len(c.pins) == 2:
        a, b = c.pins
        dx, dy = abs(a.tip[0] - b.tip[0]), abs(a.tip[1] - b.tip[1])
        horiz = dx > dy
        if min(dx, dy) > P.t + 1:
            return False                        # the two ends on one axis
    else:
        horiz = w >= h                          # one end on a wire, the other touching a symbol
        dx = dy = None
    L, S_ = (w, h) if horiz else (h, w)
    if not (1.2 * u <= L <= 3.2 * u and 0.4 * u <= S_ <= 1.3 * u):
        return False
    if dx is not None and max(dx, dy) < 0.9 * L:
        return False
    big = max(c.feat['holes'], key=lambda x: x[0])
    ha, hw, hh, rect = big
    hL, hS = (hw, hh) if horiz else (hh, hw)
    if not (0.38 <= rect <= 0.65 and hL >= 0.35 * L and hS >= 0.5 * S_ and len(c.feat['holes']) == 1):
        return False
    # a diamond is widest in the middle; a diode's triangle at one end
    S = _S_CTX[0]
    if S is None:
        return False
    x0, y0, x1, y1 = [int(v) for v in S['groups'][c.gids[0]]['bbox']]
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == c.gids[0]).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    hole = [cc for cc, hh in zip(cnts, hier[0]) if hh[3] >= 0]
    if not hole:
        return False
    fill = np.zeros_like(m)
    cv2.drawContours(fill, [max(hole, key=cv2.contourArea)], -1, 1, -1)
    prof = fill.sum(0) if horiz else fill.sum(1)
    idx = np.nonzero(prof)[0]
    if len(idx) < 4:
        return False
    pos = (float(np.argmax(prof)) - idx[0]) / max(idx[-1] - idx[0], 1)
    return 0.3 <= pos <= 0.7


CUT_DIAMOND = os.environ.get('PCB_CUT_DIAMOND', '1') == '1'


def _cut_diamond(c, P):
    """A short whose two stubs were taken as one straight pin line through the diamond: the
    tips were cut away with it, so the group (just the diamond) has no enclosed region.  Look
    for the rhombus in the symbol ink itself.  Two pins on one axis, pin to pin about 2u."""
    if len(c.pins) != 2 or len(c.gids) != 1:
        return False
    u, t = P.u, P.t
    a, b = c.pins
    dx, dy = abs(a.tip[0] - b.tip[0]), abs(a.tip[1] - b.tip[1])
    horiz = dx > dy
    if min(dx, dy) > t + 1 or not (1.4 * u <= max(dx, dy) <= 3.2 * u):
        return False
    x0, y0, x1, y1 = [int(round(v)) for v in c.bbox]
    w, h = x1 - x0 + 1, y1 - y0 + 1
    L, S_ = (w, h) if horiz else (h, w)
    if not (0.7 * u <= L <= 1.4 * u and 0.45 * u <= S_ <= 1.0 * u):
        return False
    # the body sits between the two pins, centred on their axis
    ax = (a.tip[1] + b.tip[1]) / 2.0 if horiz else (a.tip[0] + b.tip[0]) / 2.0
    if abs(((y0 + y1) / 2.0 if horiz else (x0 + x1) / 2.0) - ax) > 0.15 * u:
        return False
    lo, hi = sorted((a.tip[0], b.tip[0]) if horiz else (a.tip[1], b.tip[1]))
    if not (lo < (x0 if horiz else y0) and (x1 if horiz else y1) < hi):
        return False
    red = getattr(P, 'red', None)
    if red is None:
        return False
    H, W = red.shape[:2]
    m = (red[max(y0 - 1, 0):min(y1 + 2, H), max(x0 - 1, 0):min(x1 + 2, W)] > 0).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    holes = [cc for cc, hh in zip(cnts, hier[0]) if hh[3] >= 0 and cv2.contourArea(cc) >= 4]
    if len(holes) != 1:
        return False
    bx, by, bw, bh = cv2.boundingRect(holes[0])
    r = cv2.contourArea(holes[0]) / max(bw * bh, 1)
    hL, hS = (bw, bh) if horiz else (bh, bw)
    if not (0.3 <= r <= 0.65 and 0.35 * L <= hL <= 0.9 * L and hS >= 0.4 * S_):
        return False
    fill = np.zeros_like(m)
    cv2.drawContours(fill, holes, -1, 1, -1)
    for prof in (fill.sum(0), fill.sum(1)):
        idx = np.nonzero(prof)[0]
        if len(idx) < 3:
            return False
        top = np.nonzero(prof >= prof.max() - 1)[0]
        pos = ((top[0] + top[-1]) / 2.0 - idx[0]) / max(idx[-1] - idx[0], 1)
        if not 0.3 <= pos <= 0.7:
            return False
    return True


MERGE_BODY_PARTS =os.environ.get('PCB_MERGE_BODY', '1') == '1'
MERGE_SKIP_ARROWS = os.environ.get('PCB_MERGE_SKIP_ARROWS', '1') == '1'
BODY_STUBS = os.environ.get('PCB_BODY_STUBS', '1') == '1'


def _points_at(b, host, S, P):
    """Do the arrows of piece b point at host (their heads, the heavier ends, nearer the host
    than their tails)?  Light leaves the LED: an LED's own arrows point away from it.
    Each separate stroke (one arrow) votes."""
    m = _body_mask(b, S)
    if len(b.gids) == 1 or not BODY_ALL_GIDS:
        x0, y0 = [int(round(v)) for v in S['groups'][b.gids[0]]['bbox'][:2]]
    else:                                  # the union box _body_mask used
        bbs = [S['groups'][g]['bbox'] for g in b.gids if g in S['groups']]
        x0 = int(round(min(q[0] for q in bbs)))
        y0 = int(round(min(q[1] for q in bbs)))
    n, lab = cv2.connectedComponents(m, connectivity=8)
    hb = host.bbox
    votes = 0
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 6:
            continue
        pts = np.stack([xs + x0, ys + y0], 1).astype(float)
        ctr = pts.mean(0)
        w, v = np.linalg.eigh(np.cov((pts - ctr).T))
        ax = v[:, int(np.argmax(w))]
        proj = (pts - ctr) @ ax
        lo, hi = proj.min(), proj.max()
        span = hi - lo
        if span < 3:
            continue
        n_lo = int((proj < lo + span / 3).sum())
        n_hi = int((proj > hi - span / 3).sum())
        if n_lo == n_hi:
            continue
        head = ctr + ax * (hi if n_hi > n_lo else lo)
        tail = ctr + ax * (lo if n_hi > n_lo else hi)
        dh = box_gap(hb, [head[0], head[1], head[0], head[1]])
        dt = box_gap(hb, [tail[0], tail[1], tail[0], tail[1]])
        votes += 1 if dh < dt else -1
    return votes > 0


def _arrow_blobs(c, S, P):
    """A small piece made of two or more separate strokes (a pair of LED arrows grouped by
    proximity); body diodes and drain stubs are one connected stroke."""
    if len(c.pins) > 1 or max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) > 1.6 * P.u:
        return False
    m = _body_mask(c, S)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= max(6, 1.5 * P.t * P.t)]
    if len(big) < 2:
        return False
    # every stroke diagonal (an arrow), not the straight bars of a transistor channel
    for i in big:
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if min(w, h) < 0.3 * P.u or max(w, h) > 2.5 * min(w, h):
            return False
    return True


BODY_LEADS = os.environ.get('PCB_BODY_LEADS', '1') == '1'


STUB_DUP_ENDS = os.environ.get('PCB_STUB_DUP_ENDS', '1') == '1'


class _PinCount:
    """A stand-in with a part's box and a corrected pin count (for the stub test)."""
    def __init__(self, c, n):
        self.bbox = c.bbox
        self.pins = [None] * n


def _merge_body_parts(comps, S, P):
    """A transistor drawn as several stroke groups (body diode, drain stub) tied together by
    direct contacts is one symbol: fold the small pieces whose every pin ends on another
    piece of the cluster into the one multi-pin piece - before any designator is handed out
    (else the body diode grabs a nearby net name: 'BAT0' battery)."""
    u = P.u
    r = P.t + 2.5

    def free(c):
        return not (c.anon or c.type is not None or getattr(c, 'flag', False) or c.key is not None)

    def small(c):
        return len(c.pins) <= 2 and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) <= 2.5 * u
    cdots = list(getattr(P, 'contact_dots', []) or [])

    def dotted(pn):                         # a real part-to-part contact carries a green dot
        return any(abs(x - pn.tip[0]) <= P.t + 3 and abs(y - pn.tip[1]) <= P.t + 3 for x, y, _ in cdots)
    owner = {}
    tips = []
    for k, c in comps.items():
        if not free(c):
            continue
        for pn in c.pins:
            owner[id(pn)] = k
            tips.append((pn.tip[0], pn.tip[1], k, pn))

    def others_of(k, pn):
        out = set()
        if BODY_STUBS and dotted(pn):
            return out
        m = getattr(pn, 'mate', None)
        if m is not None and id(m) in owner and owner[id(m)] != k:
            out.add(owner[id(m)])
        if STUB_DUP_ENDS and pn.kind == 'zero' and any(
                np.hypot(pn.tip[0] - x, pn.tip[1] - y) <= r_ + P.t + 1 for x, y, r_ in (getattr(P, 'dots', None) or [])):
            return out                     # ends at a junction dot: a wire connection, not a joint of the symbol
        if pn.kind in ('contact', 'zero'):
            for x, y, k2, q in tips:
                if k2 != k and q.kind in ('contact', 'zero') and abs(x - pn.tip[0]) <= r and abs(y - pn.tip[1]) <= r:
                    out.add(k2)
        return out
    parent = {k: k for k in comps}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    inner = {}
    links = {}

    def stubby(c, nl, nx):
        # a short stroke group of the symbol (drain stub with its wire end): two or more
        # internal joints and at most one end on a wire
        return (BODY_STUBS and len(c.pins) <= 3 and nl >= 2 and nx <= 1
                and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) <= 1.6 * u)
    def lead(c, nl, nx):
        # a straight lead of the symbol (drain / collector lead) joined to the body at one end
        # and ending on a wire at the other
        w_, h_ = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
        return (BODY_LEADS and len(c.pins) == 2 and nl == 1 and nx == 1
                and max(w_, h_) <= 1.6 * u and min(w_, h_) <= 0.4 * u)
    for k, c in comps.items():
        if not free(c) or not c.pins:
            continue
        ok = True
        ls = [others_of(k, pn) for pn in c.pins]
        nl = sum(1 for x in ls if x)
        nx = len(ls) - nl
        if STUB_DUP_ENDS and nx >= 2:
            # one wire end found twice (a junction dot and the wire just past it)
            ext = [pn.tip for pn, x in zip(c.pins, ls) if not x]
            uniq = []
            for p_ in ext:
                if not any(abs(p_[0] - q[0]) + abs(p_[1] - q[1]) <= 0.6 * u for q in uniq):
                    uniq.append(p_)
            if len(uniq) == 1 and len(c.pins) - nx + 1 <= 3:
                st = stubby(_PinCount(c, nl + 1), nl, 1)
            else:
                st = stubby(c, nl, nx) or lead(c, nl, nx)
        else:
            st = stubby(c, nl, nx) or lead(c, nl, nx)
        for pn, os_ in zip(c.pins, ls):
            links[id(pn)] = os_
            if not os_:
                ok = False
            elif small(c) or st:
                for o in os_:
                    parent[find(k)] = find(o)
        inner[k] = (ok and small(c)) or st
    groups = {}
    for k in comps:
        groups.setdefault(find(k), []).append(k)
    for members in groups.values():
        if len(members) < 2:
            continue
        hosts = [k for k in members if not inner.get(k)]
        if len(hosts) != 1:
            continue
        a = comps[hosts[0]]
        if len(a.pins) < 3 or max(a.bbox[2] - a.bbox[0], a.bbox[3] - a.bbox[1]) > 3.5 * u or rect_hole(a.gids[0], S, P):
            continue
        parts = [comps[k] for k in members if k != hosts[0]]
        if any(not (small(b) or inner.get(k)) for k, b in zip([k for k in members if k != hosts[0]], parts)):
            continue
        if MERGE_SKIP_ARROWS and any(len(b.pins) <= 1 and (_arrow_pair(b, S, P) or _arrow_blobs(b, S, P))
                                     and _points_at(b, a, S, P) for b in parts):
            continue                     # LED arrows pointing at the next LED belong to the one behind
        ms = set(members)
        a.pins = [pn for pn in a.pins if not (links.get(id(pn), set()) & ms)]
        for b in parts:
            a.pins += [pn for pn in b.pins if not links.get(id(pn)) and (len(b.pins) == 3 or BODY_LEADS)]   # a stub's / lead's wire end
            a.bbox = [min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]), max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3])]
            a.gids = a.gids + b.gids
        for k in members:
            if k != hosts[0]:
                del comps[k]


def _pins_at_end_dots(final, P):
    """A two-pin part drawn straight into a junction dot (an inductor's last arc ending on
    the dot) has no pin stub on that side: add a zero-length pin at the dot."""
    from .sym import Pin
    t = P.t
    for c in final:
        if c.type not in ('r', 'c', 'l', 'd', 'led', 'fuse') or len(c.pins) != 1:
            continue
        x0, y0, x1, y1 = c.bbox
        w, h = x1 - x0, y1 - y0
        pn = c.pins[0]
        horiz = w >= h
        best = None
        for (dx, dy, r) in P.dots:
            if horiz:
                ay = (y0 + y1) / 2
                if abs(dy - ay) > max(h / 2, r):
                    continue
                if pn.tip[0] <= (x0 + x1) / 2:
                    gap = dx - x1
                else:
                    gap = x0 - dx
            else:
                ax = (x0 + x1) / 2
                if abs(dx - ax) > max(w / 2, r):
                    continue
                if pn.tip[1] <= (y0 + y1) / 2:
                    gap = dy - y1
                else:
                    gap = y0 - dy
            if -r <= gap <= r + 2 * t + 2 and (best is None or abs(gap) < abs(best[0])):
                best = (gap, dx, dy)
        if best is None:
            continue
        _, dx, dy = best
        if horiz:
            base = (x1, dy) if dx > (x0 + x1) / 2 else (x0, dy)
            np_ = Pin((dx, dy), base, 'h', 'zero')
        else:
            base = (dx, y1) if dy > (y0 + y1) / 2 else (dx, y0)
            np_ = Pin((dx, dy), base, 'v', 'zero')
        np_.group = c.gids[0]
        c.pins.append(np_)


SPLIT_DOUBLES = os.environ.get('PCB_SPLIT_DOUBLES', '1') == '1'


def _split_double_parts(comps, blue, used_text, P):
    """Two identical two-pin parts drawn on top of each other, offset by a grid step (a
    copy-paste slip: two LEDs one unit apart) merge into one body with four pins in two
    rows.  When a second designator of the same kind sits unclaimed next to it, split the
    body into the two parts."""
    u = P.u
    free = [it for it in blue if it.id not in used_text and is_desig(it.text)]
    for key in list(comps.keys()):
        c = comps[key]
        if c.type not in ('led', 'd', 'r', 'c', 'l') or len(c.pins) != 4 or c.desig_item is None:
            continue
        x0, y0, x1, y1 = c.bbox
        horiz = sum(1 for p in c.pins if p.o == 'h') >= sum(1 for p in c.pins if p.o == 'v')
        cr = (lambda p: p.tip[1]) if horiz else (lambda p: p.tip[0])
        al = (lambda p: p.tip[0]) if horiz else (lambda p: p.tip[1])
        pins = sorted(c.pins, key=cr)
        r1, r2 = pins[:2], pins[2:]
        if abs(cr(r1[0]) - cr(r1[1])) > 0.3 * u or abs(cr(r2[0]) - cr(r2[1])) > 0.3 * u:
            continue
        off = (cr(r2[0]) + cr(r2[1])) / 2 - (cr(r1[0]) + cr(r1[1])) / 2
        if not (0.6 * u <= off <= 2.2 * u):
            continue
        # each row: one pin on each end
        if abs(al(r1[0]) - al(r1[1])) < 1.5 * u or abs(al(r2[0]) - al(r2[1])) < 1.5 * u:
            continue
        pt = prefix_type(c.key)
        cand = [it for it in free if prefix_type(it.text) == pt and it.text != c.key
                and box_gap(c.bbox, tbox(it)) <= 2.5 * u]
        if not cand:
            continue
        it2 = min(cand, key=lambda it: box_gap(c.bbox, tbox(it)))
        if horiz:
            hs = (y1 - y0) - off
            if hs < 0.4 * u:
                continue
            bb1, bb2 = [x0, y0, x1, y0 + hs], [x0, y1 - hs, x1, y1]
        else:
            hs = (x1 - x0) - off
            if hs < 0.4 * u:
                continue
            bb1, bb2 = [x0, y0, x0 + hs, y1], [x1 - hs, y0, x1, y1]
        # which half keeps the current designator: the one its text is nearer to
        d_keep1 = box_gap(bb1, tbox(c.desig_item)) + box_gap(bb2, tbox(it2))
        d_keep2 = box_gap(bb2, tbox(c.desig_item)) + box_gap(bb1, tbox(it2))
        mine, other = ((bb1, r1), (bb2, r2)) if d_keep1 <= d_keep2 else ((bb2, r2), (bb1, r1))
        c.bbox, c.pins = list(mine[0]), list(mine[1])
        nc = Comp(c.gids[0], list(other[0]), list(other[1]))
        nc.feat = c.feat
        nc.type = c.type
        nc.key = it2.text
        nc.desig_item = it2
        nc.split_double = True
        c.split_double = True
        used_text.add(it2.id)
        free.remove(it2)
        comps[('dbl', key)] = nc


def _merge_orphan_pieces(comps):
    """A symbol cut into pieces at internal joints (MOSFET + body diode, multi-part outlines):
    the pieces are tied together by mated 'contact' pins.  Pieces without a designator (they
    would be dropped) are folded into the one designated part they are tied to."""
    owner = {}
    for key, c in comps.items():
        for pn in c.pins:
            owner[id(pn)] = key
    parent = {k: k for k in comps}
    P = _P_CTX[0]
    dots = getattr(P, 'contact_dots', []) if P is not None else []
    tol = (1.5 * P.t + 3) if P is not None else 6

    def dotted(pn):                         # a real pin-to-pin connection carries a green dot
        return any(abs(x - pn.tip[0]) <= tol and abs(y - pn.tip[1]) <= tol for x, y, _ in dots)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for key, c in comps.items():
        if c.anon or getattr(c, 'flag', False):
            continue
        for pn in c.pins:
            m = getattr(pn, 'mate', None)
            if m is None or id(m) not in owner:
                continue
            k2 = owner[id(m)]
            o = comps[k2]
            if o.anon or getattr(o, 'flag', False) or k2 == key:
                continue
            if ((c.type is None) or (o.type is None)) and not dotted(pn):   # internal joints only
                parent[find(key)] = find(k2)
    groups = {}
    for k in comps:
        groups.setdefault(find(k), []).append(k)
    for members in groups.values():
        if len(members) < 2:
            continue
        named = [k for k in members if comps[k].type is not None]
        if len(named) != 1 or not str(comps[named[0]].type).startswith(MERGE_HOST_TYPES):
            continue
        host = comps[named[0]]
        inside = set(members)
        pieces = [comps[k] for k in members if k != named[0]]
        allpins = host.pins + [pn for pc in pieces for pn in pc.pins]
        keep = []
        for pn in allpins:
            m = getattr(pn, 'mate', None)
            if m is not None and owner.get(id(m)) in inside:
                continue                    # internal joint of the symbol
            keep.append(pn)
        for pc in pieces:
            host.bbox = [min(host.bbox[0], pc.bbox[0]), min(host.bbox[1], pc.bbox[1]),
                         max(host.bbox[2], pc.bbox[2]), max(host.bbox[3], pc.bbox[3])]
            host.gids = host.gids + pc.gids
            pc.type = None
            pc.key = None
            pc.pins = []
        host.pins = keep
_ORPHAN_RE = re.compile(r'^[A-Za-z\u4e00-\u9fff][A-Za-z_\u4e00-\u9fff+\-]*\d{0,3}$')
_CJK = re.compile(r'[\u4e00-\u9fff]')


DROP_CONTACT_ISLANDS = os.environ.get('PCB_DROP_ISLANDS', '0') == '1'   # off: decoupling caps with GND/VCC drawn straight on are such islands too


def drop_contact_islands(comps, P):
    """Groups that touch only each other (every pin a contact, none reaching a wire or a
    junction) are lettering or artwork drawn in the symbol colour (a large red title whose
    glyphs touch), not parts."""
    t = P.t
    keys = [k for k, c in comps.items() if c.pins]
    tips = [(pn.tip[0], pn.tip[1], k, pn.kind) for k in keys for pn in comps[k].pins]
    parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    r = 2 * t + 2
    for i, (x, y, k, kd) in enumerate(tips):
        for x2, y2, k2, kd2 in tips[i + 1:]:
            if k2 != k and abs(x - x2) <= r and abs(y - y2) <= r:
                parent[find(k)] = find(k2)
    clusters = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)
    out = dict(comps)
    for members in clusters.values():
        if all(pn.kind == 'contact' for k in members for pn in comps[k].pins):
            for k in members:
                del out[k]
    return out


def drop_red_text_parts(comps, texts, P):
    """Free annotations written in the symbol colour (red Chinese notes, formulas) form red
    'bodies'; one touching a wire gets a pin and would become a net label.  Drop small
    one-pin parts mostly covered by confidently read red text."""
    u = P.u
    reds = [it for it in texts if it.color == 'red' and it.text and it.conf >= 0.8
            and (_CJK.search(it.text) or sum(ch.isalnum() for ch in it.text) >= 3)]
    if not reds:
        return comps
    out = {}
    for key, c in comps.items():
        x0, y0, x1, y1 = c.bbox
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if len(c.pins) <= 1 and min(w, h) <= 0.9 * u:
            cov = cjk = 0.0
            for it in reds:
                ix = min(x1, it.x1) - max(x0, it.x0) + 1
                iy = min(y1, it.y1) - max(y0, it.y0) + 1
                if ix > 0 and iy > 0:
                    cov += ix * iy
                    if _CJK.search(it.text):
                        cjk += ix * iy
            if cov >= 0.4 * w * h or cjk >= 0.15 * w * h:   # Chinese glyphs are never symbol graphics
                continue
        out[key] = c
    return out


def _holes_dots_restored(c, S, P, pins):
    """Enclosed regions of a part's group with the junction dots under its pins painted back
    (feat format [(area, w, h, rect)])."""
    dots = getattr(P, 'dots', None) or []
    if len(c.gids) != 1 or not dots:
        return []
    gid = c.gids[0]
    x0, y0, x1, y1 = [int(v) for v in S['groups'][gid]['bbox']]
    paint = []
    for pn in pins:
        tx, ty = pn.tip
        near = [d for d in dots if abs(d[0] - tx) <= 1.2 * d[2] + 2 and abs(d[1] - ty) <= 1.2 * d[2] + 2]
        if near:
            paint.append(min(near, key=lambda d: (d[0] - tx) ** 2 + (d[1] - ty) ** 2))
    if not paint:
        return []
    R = max(int(np.ceil(1.6 * d[2])) + 1 for d in paint)
    H, W = S['glab'].shape
    X0, Y0, X1, Y1 = max(x0 - R - 1, 0), max(y0 - R - 1, 0), min(x1 + R + 1, W - 1), min(y1 + R + 1, H - 1)
    m = (S['glab'][Y0:Y1 + 1, X0:X1 + 1] == gid).astype(np.uint8)
    for dx, dy, dr in paint:
        cv2.circle(m, (int(round(dx - X0)), int(round(dy - Y0))), int(np.ceil(1.6 * dr)) + 1, 1, -1)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is not None:
        for cnt, hh in zip(cnts, hier[0]):
            if hh[3] >= 0:
                a = cv2.contourArea(cnt)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if a >= 4:
                    out.append((a, bw, bh, a / max(bw * bh, 1)))
    return out


def split_flag_stacks(comps, S, P, blue):
    """Net-port flags stacked edge to edge (a column of hexagons, one per signal) merge into
    one body group.  Recognise the stack - k parallel pins on one side at a regular pitch,
    k closed cells, a label beyond the far end of every cell - and split it into k flags."""
    u, t = P.u, P.t
    out = {}
    for key, c in comps.items():
        cp = c.pins
        if STACK_DROP_Z:
            # a zero-length contact where a wire runs along the stack's outer edge is not
            # one of its pins
            nz = [pn for pn in c.pins if pn.o != 'z']
            if 2 <= len(nz) < len(c.pins) and len({pn.o for pn in nz}) == 1:
                ax = 0 if nz[0].o == 'h' else 1
                line = float(np.median([pn.tip[ax] for pn in nz]))
                if all(abs(pn.tip[ax] - line) > 0.5 * u for pn in c.pins if pn.o == 'z'):
                    cp = nz
        k = len(cp)
        if k < 2 or len(c.gids) != 1:
            out[key] = c
            continue
        os_ = {pn.o for pn in cp}
        if STACK_Z_ORIENT and 'z' in os_ and len(os_ - {'z'}) <= 1:
            # flags whose tips touch the wire ends directly have zero-length pins: they
            # still lie on one line on the stack's pin side
            xs_t = [pn.tip[0] for pn in cp]
            ys_t = [pn.tip[1] for pn in cp]
            eo = os_ - {'z'}
            if (eo == {'h'} or not eo) and max(xs_t) - min(xs_t) <= 2 * t + 2:
                os_ = {'h'}
            elif (eo == {'v'} or not eo) and max(ys_t) - min(ys_t) <= 2 * t + 2:
                os_ = {'v'}
        if len(os_) != 1 or os_ & {'z'}:
            out[key] = c
            continue
        o = os_.pop()
        x0, y0, x1, y1 = c.bbox
        cbx = getattr(c, 'carved_boxes', None)
        if cbx:
            # extent of what is left after the grounds drawn onto the stack were cut out
            gx0_, gy0_, gx1_, gy1_ = S['groups'][c.gids[0]]['bbox']
            mm = (S['glab'][gy0_:gy1_ + 1, gx0_:gx1_ + 1] == c.gids[0]).copy()
            for bx0, by0, bx1, by1 in cbx:
                mm[max(0, int(by0) - gy0_):max(0, int(by1) - gy0_ + 1), max(0, int(bx0) - gx0_):max(0, int(bx1) - gx0_ + 1)] = False
            ys_, xs_ = np.nonzero(mm)
            if len(xs_):
                x0, y0, x1, y1 = gx0_ + xs_.min(), gy0_ + ys_.min(), gx0_ + xs_.max(), gy0_ + ys_.max()
        if o == 'h':
            along0, along1 = x0, x1
            tips_a = [pn.tip[0] for pn in cp]
            cross = sorted(pn.tip[1] for pn in cp)
            c0, c1 = y0, y1
        else:
            along0, along1 = y0, y1
            tips_a = [pn.tip[1] for pn in cp]
            cross = sorted(pn.tip[0] for pn in cp)
            c0, c1 = x0, x1
        L = along1 - along0
        mid = (along0 + along1) / 2
        side = [ta < mid for ta in tips_a]
        if not (all(side) or not any(side)) or max(tips_a) - min(tips_a) > 2 * t + 2:
            out[key] = c
            continue
        if not (1.2 * u <= L <= 8 * u):
            out[key] = c
            continue
        gaps = np.diff(cross)
        if gaps.min() < 0.6 * u or gaps.max() > 1.25 * gaps.min():
            out[key] = c
            continue
        pitch = float(np.median(gaps))
        if (c1 - c0) > k * pitch + 0.6 * pitch:
            out[key] = c
            continue
        holes = [hh for hh in c.feat['holes'] if hh[0] > 0.15 * pitch * L]
        dotted = False
        if len(holes) < k and FLAG_DOT_CLOSE and not cbx:
            # flags whose points sit on junction dots lost the point with the dot
            h2 = [hh for hh in _holes_dots_restored(c, S, P, cp) if hh[0] > 0.15 * pitch * L]
            if len(h2) > len(holes):
                holes, dotted = h2, True
        cell_rows = None
        if len(holes) < k:
            if not STACK_PARTIAL or len(holes) < 2:
                out[key] = c
                continue
            # some rows are not flags (a power port drawn into the stack): find which rows
            # own a closed cell
            gid = c.gids[0]
            gx0, gy0, gx1, gy1 = S['groups'][gid]['bbox']
            mk = (S['glab'][gy0:gy1 + 1, gx0:gx1 + 1] == gid).astype(np.uint8)
            bg = (1 - mk).astype(np.uint8)
            nlab, lab_, st_, cen_ = cv2.connectedComponentsWithStats(bg, connectivity=4)
            border = set(np.unique(np.r_[lab_[0], lab_[-1], lab_[:, 0], lab_[:, -1]]).tolist())
            cents = [cen_[i] for i in range(1, nlab) if i not in border and st_[i, cv2.CC_STAT_AREA] > 0.15 * pitch * L]
            cell_rows = set()
            for cx_, cy_ in cents:
                v = (cy_ + gy0) if o == 'h' else (cx_ + gx0)
                j = int(np.argmin([abs(v - cc) for cc in cross]))
                if abs(v - cross[j]) <= 0.35 * pitch:
                    cell_rows.add(j)
            if len(cell_rows) < max(2, k - max(1, k // 4)) or len(cell_rows) == k:
                out[key] = c
                continue
        # a label beyond the far end, aligned with each pin
        far_lo = all(side)              # pins on the low side -> labels beyond along1
        hits = 0
        for cc in cross:
            for it in blue:
                ic = it.cy if o == 'h' else it.cx
                if abs(ic - cc) > 0.35 * pitch:
                    continue
                if o == 'h':
                    gap = it.x0 - x1 if far_lo else x0 - it.x1
                else:
                    gap = it.y0 - y1 if far_lo else y0 - it.y1
                if -0.2 * u <= gap <= 1.0 * u:
                    hits += 1
                    break
        if hits < max(2, k - 1):
            out[key] = c
            continue
        gid = c.gids[0]
        gx0, gy0, gx1, gy1 = S['groups'][gid]['bbox']
        m = (S['glab'][gy0:gy1 + 1, gx0:gx1 + 1] == gid)
        for bx0, by0, bx1, by1 in (cbx or []):
            m[max(0, int(by0) - gy0):max(0, int(by1) - gy0 + 1), max(0, int(bx0) - gx0):max(0, int(bx1) - gx0 + 1)] = False
        pins = sorted(cp, key=lambda pn: pn.tip[1] if o == 'h' else pn.tip[0])
        rest = []
        rest_m = m.copy()
        for i, pn in enumerate(pins):
            cc = pn.tip[1] if o == 'h' else pn.tip[0]
            lo, hi = cc - pitch / 2, cc + pitch / 2
            if cell_rows is not None:
                if i not in cell_rows:
                    rest.append(pn)
                    continue
                a_, b_ = int(np.ceil(lo)), int(np.floor(hi))
                if o == 'h':
                    rest_m[max(a_ - gy0, 0):max(b_ - gy0 + 1, 0)] = False
                else:
                    rest_m[:, max(a_ - gx0, 0):max(b_ - gx0 + 1, 0)] = False
            if o == 'h':
                rows = m[max(int(np.ceil(lo)) - gy0, 0):max(int(np.floor(hi)) - gy0 + 1, 0)]
                ys, xs = np.nonzero(rows)
                if len(xs) == 0:
                    continue
                bb = [gx0 + xs.min(), max(int(np.ceil(lo)), gy0) + ys.min(), gx0 + xs.max(), max(int(np.ceil(lo)), gy0) + ys.max()]
            else:
                cols = m[:, max(int(np.ceil(lo)) - gx0, 0):max(int(np.floor(hi)) - gx0 + 1, 0)]
                ys, xs = np.nonzero(cols)
                if len(xs) == 0:
                    continue
                bb = [max(int(np.ceil(lo)), gx0) + xs.min(), gy0 + ys.min(), max(int(np.ceil(lo)), gx0) + xs.max(), gy0 + ys.max()]
            nc = Comp(gid, bb, [pn])
            nc.feat = c.feat
            nc.forced_flag = True
            nc.split_from = key
            if dotted:
                nc.dot_closed = True
            out[(key, i)] = nc
        if rest:
            # what is left (the rows without a cell) stays one part with its own box
            band = np.zeros_like(rest_m)
            for pn in rest:
                cc = pn.tip[1] if o == 'h' else pn.tip[0]
                a_, b_ = int(np.ceil(cc - pitch / 2 + t)), int(np.floor(cc + pitch / 2 - t))
                if o == 'h':
                    band[max(a_ - gy0, 0):max(b_ - gy0 + 1, 0)] = True
                else:
                    band[:, max(a_ - gx0, 0):max(b_ - gx0 + 1, 0)] = True
            rest_m &= band
            ys, xs = np.nonzero(rest_m)
            if len(xs) == 0:
                continue
            bb = [gx0 + int(xs.min()), gy0 + int(ys.min()), gx0 + int(xs.max()), gy0 + int(ys.max())]
            nc = Comp(gid, bb, rest)
            nc.feat = {'w': bb[2] - bb[0] + 1, 'h': bb[3] - bb[1] + 1, 'area': int(rest_m.sum()), 'holes': [],
                       'fill': float(rest_m.sum()) / max(1, (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1))}
            nc.split_from = key
            out[(key, 'rest')] = nc
    return out


STACK_Z_ORIENT = os.environ.get('PCB_STACK_Z_ORIENT', '1') == '1'
STACK_PARTIAL = os.environ.get('PCB_STACK_PARTIAL', '1') == '1'
STACK_DROP_Z = os.environ.get('PCB_STACK_DROP_Z', '1') == '1'
CARVE_FLAGS = os.environ.get('PCB_CARVE_FLAGS', '1') == '1'


def carve_attached_flags(comps, S, P, blue):
    """Net-port flags drawn touching a connector's pin ends (no wire in between) become part
    of the connector's stroke group.  Find them as arrow-shaped closed cells outside the IC
    rectangle with a label beyond their far end, cut each out as a one-pin flag and give the
    host a pin at the contact point."""
    from .sym import Pin
    u, t = P.u, P.t
    out = dict(comps)
    for key, c in comps.items():
        if len(c.gids) != 1 or (c.pins and len(c.pins) == 1 and not CARVE_FLAGS_SMALL):
            continue
        gid = c.gids[0]
        rect = rect_hole(gid, S, P)
        if rect and CARVE_FLAGS_SMALL and 1 <= len(c.pins) <= 2 and \
                min(rect[2] - rect[0], rect[3] - rect[1]) <= 1.3 * u and max(rect[2] - rect[0], rect[3] - rect[1]) <= 4 * u:
            rect = None                         # the 'rectangle' is the flag itself, not an IC body
        if rect and len(c.pins) == 1:
            continue
        small_host = False
        if not rect:
            # a two-pin part (LED, resistor) whose far end runs straight into a net flag
            gb_ = S['groups'][gid]['bbox']
            if not (CARVE_FLAGS_SMALL and 1 <= len(c.pins) <= 2 and max(gb_[2] - gb_[0], gb_[3] - gb_[1]) <= 8 * u):
                continue
            small_host = True
            rect = [gb_[2] + 10 * u, gb_[3] + 10 * u, gb_[2] + 10 * u, gb_[3] + 10 * u]   # no IC body to avoid
        gx0, gy0, gx1, gy1 = S['groups'][gid]['bbox']
        m = (S['glab'][gy0:gy1 + 1, gx0:gx1 + 1] == gid).astype(np.uint8)
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        k = 0
        for cnt, hh in zip(cnts, hier[0]):
            if hh[3] < 0:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            hx0, hy0, hx1, hy1 = gx0 + bx, gy0 + by, gx0 + bx + bw - 1, gy0 + by + bh - 1
            if not (hx1 < rect[0] - t or hx0 > rect[2] + t or hy1 < rect[1] - t or hy0 > rect[3] + t):
                continue                        # inside / overlapping the IC body
            long_, short = max(bw, bh), min(bw, bh)
            if not (0.35 * u <= short <= 1.2 * u and 1.8 <= long_ / max(short, 1) <= 6 and long_ <= 5 * u):
                continue
            fill = cv2.contourArea(cnt) / max(bw * bh, 1)
            if not (0.55 <= fill <= 0.92):
                continue
            horiz = bw >= bh
            cc = (hy0 + hy1) / 2 if horiz else (hx0 + hx1) / 2
            # label beyond one end, aligned with the axis; the contact is the other end
            lab = None
            for it in blue:
                ic = it.cy if horiz else it.cx
                if abs(ic - cc) > max(0.45 * short, 0.3 * u):
                    continue
                if horiz:
                    g_lo, g_hi = hx0 - it.x1, it.x0 - hx1
                else:
                    g_lo, g_hi = hy0 - it.y1, it.y0 - hy1
                for side, gg in (('lo', g_lo), ('hi', g_hi)):
                    if -0.2 * u <= gg <= 1.6 * u and (lab is None or gg < lab[0]):
                        lab = (gg, side)
            if lab is None:
                continue
            if small_host:
                # the flag is the end of the piece: its far side is the group's edge, and the
                # piece is more than the flag (a lone flag is a net label already)
                if horiz and not (abs(hx1 - gx1) <= 2 * t + 2 or abs(hx0 - gx0) <= 2 * t + 2):
                    continue
                if not horiz and not (abs(hy1 - gy1) <= 2 * t + 2 or abs(hy0 - gy0) <= 2 * t + 2):
                    continue
                if (bw if horiz else bh) > 0.6 * ((gx1 - gx0) if horiz else (gy1 - gy0)):
                    continue
            e = t                               # hole = inner edge; graphics bbox = outer edge
            bb = [hx0 - e, hy0 - e, hx1 + e, hy1 + e]
            if horiz:
                tipx = bb[2] if lab[1] == 'lo' else bb[0]
                tip, base, o = (tipx, cc), (tipx - (1 if lab[1] == 'lo' else -1) * 0.3 * u, cc), 'h'
            else:
                tipy = bb[3] if lab[1] == 'lo' else bb[1]
                tip, base, o = (cc, tipy), (cc, tipy - (1 if lab[1] == 'lo' else -1) * 0.3 * u), 'v'
            pn = Pin(tip, base, o, 'contact')
            pn.group = gid
            nc = Comp(gid, bb, [pn])
            nc.feat = {'w': bb[2] - bb[0], 'h': bb[3] - bb[1], 'area': int(cv2.contourArea(cnt)),
                       'holes': [(cv2.contourArea(cnt), bw, bh, fill)], 'fill': 0.3}
            nc.forced_flag = True
            nc.carved_from = key
            out[(key, 'flag', k)] = nc
            k += 1
            # the host's pin ends at the flag
            if not any(abs(p.tip[0] - tip[0]) + abs(p.tip[1] - tip[1]) <= 2 * t + 2 for p in c.pins):
                if small_host:
                    d_ = 1 if (tip[0] > (gx0 + gx1) / 2 if o == 'h' else tip[1] > (gy0 + gy1) / 2) else -1
                    b_ = (tip[0] - d_ * u, tip[1]) if o == 'h' else (tip[0], tip[1] - d_ * u)
                    hp = Pin(tip, b_, o, 'contact')
                elif o == 'h':
                    bx_ = rect[0] if tip[0] < rect[0] else rect[2]
                    hp = Pin(tip, (bx_, tip[1]), 'h', 'contact')
                else:
                    by_ = rect[1] if tip[1] < rect[1] else rect[3]
                    hp = Pin(tip, (tip[0], by_), 'v', 'contact')
                hp.group = gid
                hp.mate, pn.mate = pn, hp
                c.pins.append(hp)
            if small_host:
                # the host's box no longer covers the flag
                mm = m.copy()
                mm[max(int(bb[1]) - gy0, 0):int(bb[3]) - gy0 + 1, max(int(bb[0]) - gx0, 0):int(bb[2]) - gx0 + 1] = 0
                ys_, xs_ = np.nonzero(mm)
                if len(ys_):
                    c.bbox = [gx0 + int(xs_.min()), gy0 + int(ys_.min()), gx0 + int(xs_.max()), gy0 + int(ys_.max())]
        if k and not small_host:
            c.bbox = [min(c.bbox[0], rect[0]), min(c.bbox[1], rect[1]), max(c.bbox[2], rect[2]), max(c.bbox[3], rect[3])]
    return out


CARVE_GROUNDS = os.environ.get('PCB_CARVE_GND', '1') == '1'
CARVE_FLAGS_SMALL = os.environ.get('PCB_CARVE_FLAGS_SMALL', '1') == '1'


GND_FREE_ENDS = os.environ.get('PCB_GND_FREE', '1') == '1'


def _bar_ends_free(P, b):
    """True when the red mask is empty just beyond both ends of segment b and beside its
    ends across the bar (no stroke attached to the bar's end points)."""
    red = P.red
    H, W = red.shape
    t = P.t
    g = int(round(t + 2))
    for end, sgn in ((b.a, -1), (b.b, 1)):
        e = int(round(end + sgn * g))
        if b.o == 'h':
            y = int(round(b.c))
            ys = slice(max(0, y - int(t)), min(H, y + int(t) + 1))
            xs = slice(max(0, min(e, int(round(end))) ), min(W, max(e, int(round(end))) + 1))
            if not (0 <= e < W):
                continue
            if red[ys, e if 0 <= e < W else 0].any():
                return False
        else:
            x = int(round(b.c))
            xs = slice(max(0, x - int(t)), min(W, x + int(t) + 1))
            if not (0 <= e < H):
                continue
            if red[e, xs].any():
                return False
    return True


def _beyond_empty(P, b1, b2, sgn, mid, segs=None):
    """Past the shortest bar of a ground symbol there is nothing (transformer cores, coil
    leads and CJK strokes always have more strokes there)."""
    red = P.red
    H, W = red.shape
    u, t = P.u, P.t
    c0 = b2.c + sgn * (t + 1.5)
    c1 = b2.c + sgn * 0.4 * u
    lo, hi = min(c0, c1), max(c0, c1)
    a, b = mid - b1.length / 2, mid + b1.length / 2
    if b2.o == 'h':
        sub = red[max(0, int(lo)):min(H, int(hi) + 1), max(0, int(a)):min(W, int(b) + 1)]
    else:
        sub = red[max(0, int(a)):min(H, int(b) + 1), max(0, int(lo)):min(W, int(hi) + 1)]
    # a tiny last bar (some ground symbols have a 4th, dot-sized one) is allowed
    n = int((sub > 0).sum())
    lim = max(3, 0.5 * b2.length * t)
    if n > lim and segs is not None and GND_PASSING:
        # a perpendicular line that runs on past the window on both sides (the edge of a
        # net flag stacked right next to the symbol) passes by - it is not part of it
        k = 0
        for s_ in segs:
            if s_.o == b2.o or not (a - t <= s_.c <= b + t):
                continue
            # it comes from outside and stops short of the bars (a transformer's leads
            # run on across the coil)
            if sgn < 0 and s_.b > b2.c - t or sgn > 0 and s_.a < b2.c + t:
                continue
            if s_.a <= lo - 0.5 * u and s_.b >= hi - 1 or s_.b >= hi + 0.5 * u and s_.a <= lo + 1:
                n -= int(min(s_.b, hi) - max(s_.a, lo) + 1) * int(round(t))
                k += 1
        return k == 1 and n <= lim
    return n <= lim


GND_PASSING = os.environ.get('PCB_GND_PASSING', '1') == '1'
GND_HOST_BBOX = os.environ.get('PCB_GND_HOST_BBOX', '1') == '1'
GND_FIRST = os.environ.get('PCB_GND_FIRST', '1') == '1'
GND_BASE_PIX = os.environ.get('PCB_GND_BASE_PIX', '1') == '1'
GND_HOST_FIX = os.environ.get('PCB_GND_HOST_FIX', '1') == '1'


def _ground_stacks(S, P):
    """Ground symbols as stroke patterns: a base bar, then >= 2 shorter parallel bars
    centred on the same line and getting shorter step by step, plus a perpendicular stem
    from the base bar's middle.  The base bar may be cut into pieces or run on into another
    stroke (a box edge it touches), so it is checked by coverage, not as one segment.
    Returns [(bars, stem, tip, orientation, side)]."""
    u, t = P.u, P.t
    segs = S['rsegs']
    out = []
    seen = set()
    for o in 'hv':
        bars = sorted([s_ for s_ in segs if s_.o == o and 0.2 * u <= s_.length <= 2.6 * u], key=lambda s_: s_.c)
        allo = [s_ for s_ in segs if s_.o == o]
        po = 'v' if o == 'h' else 'h'
        stems = [s_ for s_ in segs if s_.o == po and 0.3 * u <= s_.length <= 3.0 * u]
        for i, b1 in enumerate(bars):
            mid = (b1.a + b1.b) / 2
            if b1.length < 0.6 * u:             # digit strokes ('8', '9') are shorter
                continue
            for sgn in (1, -1):
                nxt = [b for b in (bars[i + 1:] if sgn > 0 else bars[:i][::-1])
                       if 0.12 * u <= (b.c - b1.c) * sgn <= 0.55 * u and abs((b.a + b.b) / 2 - mid) <= t + 2
                       and 0.25 * u <= b.length <= b1.length - 0.5 * t]
                if not nxt:
                    continue
                b2 = min(nxt, key=lambda b: (b.c - b1.c) * sgn)
                if b2.length > 0.85 * b1.length:
                    continue
                # bars of a ground symbol end free: nothing continues past either end
                # (transformer cores, coil leads and CJK strokes do)
                if GND_FREE_ENDS and not _beyond_empty(P, b1, b2, sgn, mid, segs):
                    continue
                step = (b2.c - b1.c) * sgn
                base_c = b1.c - sgn * step
                half = b1.length / 2 + 0.15 * u
                cov = 0.0
                for s_ in allo:
                    if abs(s_.c - base_c) <= t + 1:
                        cov += max(0.0, min(s_.b, mid + half) - max(s_.a, mid - half))
                if cov < 0.85 * 2 * half and GND_BASE_PIX:
                    # the base bar may be lost as a span when the first bar runs so close
                    # beside it that the two touch in the mask: look at the pixels
                    red = P.red
                    ci = int(round(base_c))
                    lo_, hi_ = int(round(mid - half)), int(round(mid + half))
                    if o == 'h':
                        col = red[max(0, ci - 1):ci + 2, max(0, lo_):hi_ + 1].any(0)
                    else:
                        col = red[max(0, lo_):hi_ + 1, max(0, ci - 1):ci + 2].any(1)
                    cov = float(col.sum())
                if cov < 0.85 * 2 * half:
                    continue
                for st in stems:
                    if abs(st.c - mid) > t + 1:
                        continue
                    near_end = st.b if sgn > 0 else st.a
                    far_end = st.a if sgn > 0 else st.b
                    if abs(near_end - base_c) <= t + 2:
                        key = (o, round(mid), round(base_c))
                        if key in seen:
                            break
                        seen.add(key)
                        tip = (mid, far_end) if o == 'h' else (far_end, mid)
                        out.append(((base_c, mid - half, mid + half), [b1, b2], st, tip, o, sgn))
                        break
    return out


def mid_of(sg):
    return (sg.a + sg.b) / 2


CARVE_LONE_HOST = os.environ.get('PCB_CARVE_LONE_HOST', '0') == '1'   # off: red notes and part rows carve false grounds
CARVE_NO_ARROWS = os.environ.get('PCB_CARVE_NO_ARROWS', '0') == '1'   # off: flags' slanted edges beside grounds


def _slanted_in(glab, gids, bb, u):
    """A slanted stroke at least half a unit long within the box (a ground symbol is drawn
    with straight bars only)."""
    H, W = glab.shape
    m_ = int(round(0.2 * u))
    x0, y0 = max(int(bb[0]) - m_, 0), max(int(bb[1]) - m_, 0)
    x1, y1 = min(int(bb[2]) + m_, W - 1), min(int(bb[3]) + m_, H - 1)
    if x1 <= x0 or y1 <= y0:
        return False
    m = np.isin(glab[y0:y1 + 1, x0:x1 + 1], list(gids)).astype(np.uint8)
    if m.sum() < 10:
        return False
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 180, threshold=max(5, int(0.25 * u)),
                        minLineLength=max(5, int(0.5 * u)), maxLineGap=2)
    if L is None:
        return False
    for xa, ya, xb, yb in np.asarray(L).reshape(-1, 4):
        a = np.degrees(np.arctan2(yb - ya, xb - xa)) % 180
        if 25 < a < 65 or 115 < a < 155:
            return True
    return False


def carve_attached_grounds(comps, S, P):
    """A ground symbol whose stem ends right on another part's pin end (no wire between)
    joins that part's stroke group and is lost.  Find ground-shaped stroke stacks inside
    multi-pin parts and cut them out as one-pin ground symbols."""
    from .sym import Pin
    u, t = P.u, P.t
    glab = S['glab']
    H, W = glab.shape
    gid2key = {}
    for key, c in comps.items():
        if getattr(c, 'forced_flag', False) and GND_HOST_FIX:
            continue                      # flags carved from a group share its gid
        for g in c.gids:
            gid2key[g] = key
    out = dict(comps)
    k = 0
    carved_tips = {}
    for (base_c, blo, bhi), bars, stem, tip, o, sgn in _ground_stacks(S, P):
        b0 = bars[0]
        x, y = (int(round((b0.a + b0.b) / 2)), int(round(b0.c))) if o == 'h' else (int(round(b0.c)), int(round((b0.a + b0.b) / 2)))
        if not (0 <= x < W and 0 <= y < H):
            continue
        win = glab[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        gs = [int(g) for g in np.unique(win) if g > 0]
        key = next((gid2key[g] for g in gs if g in gid2key), None)
        if key is None:
            continue
        c = out.get(key)
        if c is None or getattr(c, 'forced_flag', False):
            continue
        lo, hi = blo + 0.15 * u - t / 2, bhi - 0.15 * u + t / 2
        cs = [b.c for b in bars] + [base_c]
        if o == 'h':
            bb = [lo, min(cs) - t / 2, hi, max(cs) + t / 2]
            pn = Pin(tip, (tip[0], base_c), 'v', 'contact')
        else:
            bb = [min(cs) - t / 2, lo, max(cs) + t / 2, hi]
            pn = Pin(tip, (base_c, tip[1]), 'h', 'contact')
        if CARVE_LONE_HOST and any(abs(tp[0] - tip[0]) + abs(tp[1] - tip[1]) <= 2 * t + 2
                                   for tps in carved_tips.values() for tp in tps):
            continue                                    # the same stack found twice
        if len(c.pins) <= 1:
            # a lone ground (its own group) stays as it is; a ground hanging on another part's
            # lead with no wire between (a switch's contact) is carved out of that part
            ext = [min(bb[0], tip[0]), min(bb[1], tip[1]), max(bb[2], tip[0]), max(bb[3], tip[1])]
            # (beside the stem, across the bars: a long stem alone is still a plain ground)
            side = max(ext[0] - c.bbox[0], c.bbox[2] - ext[2]) if o == 'h' else max(ext[1] - c.bbox[1], c.bbox[3] - ext[3])
            if not (CARVE_LONE_HOST and len(c.pins) == 1 and side > 1.2 * u and bhi - blo >= 1.3 * u):
                continue                                # an ordinary stand-alone ground
        pn.group = c.gids[0]
        if CARVE_NO_ARROWS and _slanted_in(glab, c.gids, bb, u):
            continue                                    # an LED's light arrows: their heads stack like bars
        nc = Comp(c.gids[0], bb, [pn])
        nc.feat = {'w': bb[2] - bb[0], 'h': bb[3] - bb[1], 'area': 0, 'holes': [], 'fill': 0.3}
        nc.type = 'gnd'
        nc.value = 'GND'
        nc.anon = True
        nc.carved_gnd = True
        out[('gnd', key, k)] = nc
        k += 1
        # the host loses the stem's end point as a pin (keep its own pin ending there)
        at = [p for p in c.pins if abs(p.tip[0] - tip[0]) + abs(p.tip[1] - tip[1]) <= 2 * t + 2]
        stem_o = 'v' if o == 'h' else 'h'
        if len(at) >= 2:
            drop = [p for p in at if p.o == stem_o] or at[1:]
            c.pins = [p for p in c.pins if p not in drop[:len(at) - 1]]
        elif len(at) == 1 and GND_HOST_FIX and at[0].o == stem_o:
            # the host's only pin there is the stem itself (it starts at the base bar)
            bx, by = (mid_of(bars[0]), base_c) if o == 'h' else (base_c, mid_of(bars[0]))
            if np.hypot(at[0].base[0] - bx, at[0].base[1] - by) <= 2 * t + 3:
                c.pins = [p for p in c.pins if p is not at[0]]
                nc.pins = [at[0]]             # it keeps its wire terminal
        if GND_HOST_FIX:
            if not hasattr(c, 'carved_boxes'):
                c.carved_boxes = []
            # a small last bar (4th) lies one step past the second one
            far = bars[1].c + sgn * (abs(bars[1].c - bars[0].c) + t)
            cb = [bb[0] - t, bb[1] - t, bb[2] + t, bb[3] + t]
            ea = 0.15 * u                       # bars' ends: the box above stops 0.15u short
            if o == 'h':
                cb[0], cb[2] = cb[0] - ea, cb[2] + ea
            else:
                cb[1], cb[3] = cb[1] - ea, cb[3] + ea
            if o == 'h':
                cb[1], cb[3] = min(cb[1], far), max(cb[3], far)
            else:
                cb[0], cb[2] = min(cb[0], far), max(cb[2], far)
            c.carved_boxes.append(cb)
        carved_tips.setdefault(key, []).append(tip)
    # a 'host' that was nothing but grounds drawn on top of each other: every pin it has
    # ends at a carved ground's stem -> it is gone
    for key, tips in carved_tips.items():
        c = out.get(key)
        if c is None:
            continue
        if all(any(abs(p.tip[0] - q[0]) + abs(p.tip[1] - q[1]) <= 2 * t + 2 for q in tips) for p in c.pins):
            del out[key]
            continue
        cbx = getattr(c, 'carved_boxes', None)
        if GND_HOST_BBOX and cbx and len(c.gids) == 1:
            # the host's box (and shape) without the grounds cut out of it: a net flag
            # with a ground drawn against it is a flag again
            gid = c.gids[0]
            gx0, gy0, gx1, gy1 = S['groups'][gid]['bbox']
            mm = (glab[gy0:gy1 + 1, gx0:gx1 + 1] == gid).copy()
            for bx0, by0, bx1, by1 in cbx:
                mm[max(0, int(by0) - gy0):max(0, int(by1) - gy0 + 1), max(0, int(bx0) - gx0):max(0, int(bx1) - gx0 + 1)] = False
            ys, xs = np.nonzero(mm)
            if len(xs) >= 4:
                nb = [gx0 + int(xs.min()), gy0 + int(ys.min()), gx0 + int(xs.max()), gy0 + int(ys.max())]
                # pins keep their tips inside / on the new box
                if all(nb[0] - 2 * t - 0.6 * u <= p.base[0] <= nb[2] + 2 * t + 0.6 * u and
                       nb[1] - 2 * t - 0.6 * u <= p.base[1] <= nb[3] + 2 * t + 0.6 * u for p in c.pins):
                    c.bbox = nb
                    c.feat = dict(c.feat, w=nb[2] - nb[0] + 1, h=nb[3] - nb[1] + 1)
    return out


def _group_holes(gid, S, bbox):
    """Enclosed regions of a body group: list of (cx, cy, w, h, area, rect) in image coords."""
    x0, y0, x1, y1 = bbox
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == gid).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is not None:
        for cnt, hh in zip(cnts, hier[0]):
            if hh[3] >= 0:
                a = cv2.contourArea(cnt)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if a >= 4:
                    out.append((x0 + bx + bw / 2.0, y0 + by + bh / 2.0, bw, bh, a, a / max(bw * bh, 1)))
    return out


def _rhombus_holes(gid, S, bbox):
    """For each enclosed region of a body group (same order as _group_holes): True when it is
    widest in the middle along both axes (a rhombus or ellipse), False for a triangle (widest
    at its base)."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == gid).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for cnt, hh in zip(cnts, hier[0]):
        if hh[3] < 0 or cv2.contourArea(cnt) < 4:
            continue
        fill = np.zeros_like(m)
        cv2.drawContours(fill, [cnt], -1, 1, -1)
        ok = True
        for prof in (fill.sum(0), fill.sum(1)):
            idx = np.nonzero(prof)[0]
            if len(idx) < 4:
                continue
            mx = prof.max()
            top = np.nonzero(prof >= mx - 1)[0]
            pos = ((top[0] + top[-1]) / 2.0 - idx[0]) / max(idx[-1] - idx[0], 1)
            if not 0.3 <= pos <= 0.7:
                ok = False
        out.append(ok)
    return out


def _hollow_triangle(c, S):
    """One-pin hollow shape: a triangle (widest at the end away from or at the pin, along the
    pin's axis) rather than a diamond (widest in the middle)."""
    x0, y0, x1, y1 = [int(v) for v in S['groups'][c.gids[0]]['bbox']]
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == c.gids[0]).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    hole = [cc for cc, hh in zip(cnts, hier[0]) if hh[3] >= 0]
    if not hole:
        return False
    fill = np.zeros_like(m)
    cv2.drawContours(fill, [max(hole, key=cv2.contourArea)], -1, 1, -1)
    px, py = c.pins[0].tip
    horiz = abs(px - (x0 + x1) / 2.0) >= abs(py - (y0 + y1) / 2.0)
    prof = fill.sum(0) if horiz else fill.sum(1)      # hole width at each position along the pin axis
    idx = np.nonzero(prof)[0]
    if len(idx) < 4:
        return False
    pos = (float(np.argmax(prof)) - idx[0]) / max(idx[-1] - idx[0], 1)
    return not (0.3 <= pos <= 0.7)


def _pinless_diamond(c, S, P):
    """A short symbol drawn pin-to-pin between two other symbols (no wire on either side, so
    no pin was found): a small rhombus with a straight stub on each end along its axis.
    Returns (axis 'h'/'v', centre line, end0, end1, hole box) or None."""
    u, t = P.u, P.t
    if c.pins or len(c.gids) != 1:
        return None
    x0, y0, x1, y1 = c.bbox
    w, h = x1 - x0 + 1, y1 - y0 + 1
    horiz = w >= h
    L, Wd = (w, h) if horiz else (h, w)
    if not (1.4 * u <= L <= 2.6 * u and 0.45 * u <= Wd <= 1.0 * u):
        return None
    holes = _group_holes(c.gids[0], S, c.bbox)
    if len(holes) != 1:
        return None
    hx, hy, hw, hh, a, r = holes[0]
    hL, hS = (hw, hh) if horiz else (hh, hw)
    if not (0.35 <= r <= 0.62 and 0.5 * u <= hL <= 1.1 * u and hS >= 0.3 * u):
        return None
    if not all(_rhombus_holes(c.gids[0], S, c.bbox)):
        return None
    mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if abs((hx if horiz else hy) - (mid[0] if horiz else mid[1])) > 0.2 * u:
        return None
    # the stubs: beyond the diamond only a thin band of ink on the axis
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == c.gids[0])
    if not horiz:
        m = m.T
    lo = int(round((hx - x0 if horiz else hy - y0) - hL / 2.0 - 1.5 * t))
    hi = int(round((hx - x0 if horiz else hy - y0) + hL / 2.0 + 1.5 * t))
    ends = [m[:, :max(lo, 0)], m[:, min(hi, m.shape[1]):]]
    cl = (hy - y0) if horiz else (hx - x0)
    for e in ends:
        if e.shape[1] < 0.2 * u:
            return None
        rows = np.flatnonzero(e.any(1))
        if rows.size == 0 or rows[-1] - rows[0] + 1 > 1.6 * t + 2 or abs((rows[0] + rows[-1]) / 2.0 - cl) > 0.6 * t + 1:
            return None
    if horiz:
        return ('h', hy, x0, x1, (hx - hw / 2.0, hy - hh / 2.0, hx + hw / 2.0, hy + hh / 2.0))
    return ('v', hx, y0, y1, (hx - hw / 2.0, hy - hh / 2.0, hx + hw / 2.0, hy + hh / 2.0))


def _diamond_stack(c, S, P):
    """Short symbols on neighbouring rows touch and form one body group: several rhombus holes
    in a column (or row), nothing else.  Returns [(axis, centre line, end0, end1, hole box)]."""
    u, t = P.u, P.t
    if len(c.gids) != 1:
        return None
    x0, y0, x1, y1 = c.bbox
    w, h = x1 - x0 + 1, y1 - y0 + 1
    holes = _group_holes(c.gids[0], S, c.bbox)
    if len(holes) < 2:
        return None
    rh = _rhombus_holes(c.gids[0], S, c.bbox)
    if len(rh) != len(holes) or not all(rh):
        return None                       # e.g. a bidirectional TVS: two triangles tip to tip
    for horiz in (True, False):
        L = w if horiz else h
        if not (1.4 * u <= L <= 2.6 * u):
            continue
        ok = True
        for hx, hy, hw, hh, a, r in holes:
            hL, hS = (hw, hh) if horiz else (hh, hw)
            if not (0.35 <= r <= 0.62 and 0.5 * u <= hL <= 1.1 * u and hS >= 0.3 * u):
                ok = False
                break
            if abs((hx if horiz else hy) - ((x0 + x1) / 2.0 if horiz else (y0 + y1) / 2.0)) > 0.2 * u:
                ok = False
                break
        if not ok:
            continue
        cs = sorted((hy if horiz else hx) for hx, hy, hw, hh, a, r in holes)
        if min(b - a for a, b in zip(cs, cs[1:])) < 0.6 * u:
            continue
        span = (h if horiz else w)
        dS = max((hh if horiz else hw) for hx, hy, hw, hh, a, r in holes)
        if span > cs[-1] - cs[0] + dS + 4 * t + 2:
            continue                      # something else is part of the group
        out = []
        for hx, hy, hw, hh, a, r in sorted(holes, key=lambda x: x[1] if horiz else x[0]):
            hb = (hx - hw / 2.0, hy - hh / 2.0, hx + hw / 2.0, hy + hh / 2.0)
            out.append(('h', hy, x0, x1, hb) if horiz else ('v', hx, y0, y1, hb))
        return out
    return None


DC_JACK = os.environ.get('PCB_DC_JACK', '1') == '1'
MOUNT_KEYED = os.environ.get('PCB_MOUNT_KEYED', '1') == '1'
FACING_PLATES = os.environ.get('PCB_FACING_PLATES', '1') == '1'


def _merge_facing_plates(comps, S, P):
    """A battery's two plate pairs (long + short bars) are separate stroke groups a few pixels
    apart, each with one pin; alone, either looks like a ground symbol.  Two one-pin bar groups
    thin along their common pin axis, facing each other across a small gap with their pins
    pointing away from each other, are one two-pin part."""
    u, t = P.u, P.t
    cand = []
    for k, c in comps.items():
        if c.type is not None or c.anon or len(c.pins) != 1 or len(c.gids) != 1 or c.feat.get('holes'):
            continue
        pn = c.pins[0]
        x0, y0, x1, y1 = c.bbox
        w, h = x1 - x0 + 1, y1 - y0 + 1
        tx, ty = pn.tip
        if pn.o == 'h' or (pn.o == 'z' and (tx < x0 or tx > x1)):
            horiz = True
            L, Wc = w, h
        elif pn.o == 'v' or (pn.o == 'z' and (ty < y0 or ty > y1)):
            horiz = False
            L, Wc = h, w
        else:
            continue
        if not (L <= 0.7 * u and 0.5 * u <= Wc <= 2.2 * u):
            continue
        cand.append((k, c, horiz))
    used = set()
    for i, (ka, a, ha) in enumerate(cand):
        if ka in used:
            continue
        for kb, b, hb in cand[i + 1:]:
            if kb in used or hb != ha:
                continue
            pa, pb = a.pins[0].tip, b.pins[0].tip
            if ha:
                if abs(pa[1] - pb[1]) > t + 1:
                    continue
                lo, hi = (a, b) if a.bbox[0] < b.bbox[0] else (b, a)
                gap = hi.bbox[0] - lo.bbox[2]
                ov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
                span = min(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1])
                away = lo.pins[0].tip[0] <= lo.bbox[0] + 1 and hi.pins[0].tip[0] >= hi.bbox[2] - 1
            else:
                if abs(pa[0] - pb[0]) > t + 1:
                    continue
                lo, hi = (a, b) if a.bbox[1] < b.bbox[1] else (b, a)
                gap = hi.bbox[1] - lo.bbox[3]
                ov = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
                span = min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
                away = lo.pins[0].tip[1] <= lo.bbox[1] + 1 and hi.pins[0].tip[1] >= hi.bbox[3] - 1
            # capacitor plates (one straight bar each, equal widths) stand further apart
            la = (a.bbox[2] - a.bbox[0]) if ha else (a.bbox[3] - a.bbox[1])
            lb = (b.bbox[2] - b.bbox[0]) if ha else (b.bbox[3] - b.bbox[1])
            wa = (a.bbox[3] - a.bbox[1]) if ha else (a.bbox[2] - a.bbox[0])
            wb = (b.bbox[3] - b.bbox[1]) if ha else (b.bbox[2] - b.bbox[0])
            bars = max(la, lb) <= 2 * t + 3 and abs(wa - wb) <= max(3, 0.1 * max(wa, wb))
            if not (0 <= gap <= (1.1 * u if bars else 0.6 * u) and ov >= 0.5 * span and away):
                continue
            a.gids = a.gids + b.gids
            a.pins = a.pins + b.pins
            a.bbox = [min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]), max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3])]
            a.feat = dict(a.feat, w=a.bbox[2] - a.bbox[0] + 1, h=a.bbox[3] - a.bbox[1] + 1)
            del comps[kb]
            used.update([ka, kb])
            break
ORPHAN_NEAREST = os.environ.get('PCB_ORPHAN_NEAREST', '1') == '1'


def _dc_barrel(c, S, P):
    """The barrel of EasyEDA's DC jack (DC005 ...): a long cell with a rounded far end and a
    narrow collar cell at the pin end; one pin.  Returns (horizontal, pin side +1/-1) or None."""
    u = P.u
    if len(c.gids) != 1 or len(c.pins) != 1:
        return None
    holes = _group_holes(c.gids[0], S, c.bbox)
    if len(holes) != 2:
        return None
    big = max(holes, key=lambda x: x[4])
    oth = min(holes, key=lambda x: x[4])
    hx, hy, hw, hh, a, r = big
    horiz = hw >= hh
    L, Sh = (hw, hh) if horiz else (hh, hw)
    if not (L >= 2.0 * Sh and 1.5 * u <= L <= 4.0 * u and 0.3 * u <= Sh <= 1.2 * u and 0.6 <= r <= 0.95):
        return None                       # (rounded or pointed far end)
    ox, oy, ow, oh, oa, orr = oth
    oL, oS = (ow, oh) if horiz else (oh, ow)
    if not (oL <= max(0.7 * Sh, 0.35 * u) and oS >= 1.1 * Sh and orr >= 0.6):
        return None
    tip = c.pins[0].tip
    ax_t, ax_h, ax_o = (tip[0], hx, ox) if horiz else (tip[1], hy, oy)
    side = 1 if ax_t > ax_h else -1
    if (ax_o - ax_h) * side < 0.3 * L or (ax_t - ax_o) * side < 0:
        return None
    # rounded far end: the barrel cell narrows towards it (a rectangle would not)
    x0, y0, x1, y1 = [int(v) for v in c.bbox]
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == c.gids[0]).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole = max([cc for cc, h_ in zip(cnts, hier[0]) if h_[3] >= 0], key=cv2.contourArea)
    fill = np.zeros_like(m)
    cv2.drawContours(fill, [hole], -1, 1, -1)
    prof = fill.sum(0) if horiz else fill.sum(1)
    idx = np.nonzero(prof)[0]
    far = prof[idx[0]:idx[0] + max(2, len(idx) // 10)] if side > 0 else prof[idx[-1] - max(2, len(idx) // 10) + 1:idx[-1] + 1]
    if far.mean() > 0.8 * prof.max():
        return None
    return horiz, side


def _merge_dc_jacks(comps, S, P):
    """A DC jack is drawn as the barrel (one pin) and, beside it, the switch contact with its
    arrow (two pins): separate stroke groups.  Fold the pieces next to the barrel's long side,
    within the jack's extent, into the barrel: one box with all three pins."""
    u = P.u
    for k, c in list(comps.items()):
        if k not in comps or c.type is not None:
            continue
        r = _dc_barrel(c, S, P)
        if r is None:
            continue
        horiz, side = r
        x0, y0, x1, y1 = c.bbox
        tip = c.pins[0].tip
        best = None
        for sgn in (1, -1):
            if horiz:
                zone = (min(x0, tip[0]) - 0.6 * u, y1 + 0.2 * u, max(x1, tip[0]) + 0.6 * u, y1 + 2.6 * u) if sgn > 0 \
                    else (min(x0, tip[0]) - 0.6 * u, y0 - 2.6 * u, max(x1, tip[0]) + 0.6 * u, y0 - 0.2 * u)
            else:
                zone = (x1 + 0.2 * u, min(y0, tip[1]) - 0.6 * u, x1 + 2.6 * u, max(y1, tip[1]) + 0.6 * u) if sgn > 0 \
                    else (x0 - 2.6 * u, min(y0, tip[1]) - 0.6 * u, x0 - 0.2 * u, max(y1, tip[1]) + 0.6 * u)
            parts = [k2 for k2, o in comps.items() if o is not c and o.type is None and not o.anon
                     and zone[0] <= o.bbox[0] and o.bbox[2] <= zone[2] and zone[1] <= o.bbox[1] and o.bbox[3] <= zone[3]]
            npins = sum(len(comps[k2].pins) for k2 in parts)
            if parts and 1 <= npins <= 3 and (best is None or npins > best[1]):
                best = (parts, npins)
        if best is None:
            continue
        from .sym import Pin
        bar = list(c.bbox)
        ax = 0 if horiz else 1                 # the pins' axis
        lines = []
        for k2 in best[0]:
            o = comps.pop(k2)
            c.gids = c.gids + o.gids
            c.bbox = [min(c.bbox[0], o.bbox[0]), min(c.bbox[1], o.bbox[1]), max(c.bbox[2], o.bbox[2]), max(c.bbox[3], o.bbox[3])]
            # the contact lines of the switch part run out to the pin side: one pin per line,
            # at the barrel pin's distance (all three pins end on one line)
            g = o.gids[0]
            gx0, gy0, gx1, gy1 = S['groups'][g]['bbox']
            mm = (S['glab'][gy0:gy1 + 1, gx0:gx1 + 1] == g)
            if not horiz:
                mm = mm.T
            prof = mm.sum(1)
            rows = np.nonzero(prof >= 0.5 * u)[0]
            for run in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1) if len(rows) else []:
                lines.append((gy0 if horiz else gx0) + float(run.mean()))
            for pn in o.pins:
                lines.append(pn.tip[1] if horiz else pn.tip[0])
        # the box ends a little beyond the barrel's collar; the pin lines are not part of it
        if horiz:
            if side > 0:
                c.bbox[2] = min(c.bbox[2], bar[2] + 0.5 * u)
            else:
                c.bbox[0] = max(c.bbox[0], bar[0] - 0.5 * u)
        else:
            if side > 0:
                c.bbox[3] = min(c.bbox[3], bar[3] + 0.5 * u)
            else:
                c.bbox[1] = max(c.bbox[1], bar[1] - 0.5 * u)
        tip = c.pins[0].tip
        pins = [c.pins[0]]
        own = [pn for k2 in best[0] for pn in []]
        lines.sort()
        merged = []
        for v in lines:
            if not merged or v - merged[-1][-1] > 0.4 * u:
                merged.append([v])
            else:
                merged[-1].append(v)
        for grp in merged:
            v = float(np.mean(grp))
            if abs(v - (tip[1] if horiz else tip[0])) <= 0.4 * u:
                continue
            t_ = (tip[0], v) if horiz else (v, tip[1])
            b_ = (tip[0] - side * u, v) if horiz else (v, tip[1] - side * u)
            pn = Pin(t_, b_, 'h' if horiz else 'v', 'free')
            pn.group = c.gids[-1]
            pins.append(pn)
        if 2 <= len(pins) <= 3:
            c.pins = pins
        c.dc_jack = True


def _is_screw(c, S, P):
    """JLC screw symbol: a domed head over a hatched, pointed shank (about 2u x 2.5u).  The
    head encloses one wide flat region; the hatch lines cut the shank into a column of cells."""
    u = P.u
    if len(c.pins) > 1 or len(c.gids) != 1:
        return False
    x0, y0, x1, y1 = c.bbox
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if not (1.0 * u <= max(w, h) <= 3.6 * u and 0.8 * u <= min(w, h) <= 3.0 * u):
        return False
    holes = _group_holes(c.gids[0], S, c.bbox)
    if len(holes) < 4:
        return False
    vert = h >= w
    L, Wd = (h, w) if vert else (w, h)
    mid = (x0 + x1) / 2.0 if vert else (y0 + y1) / 2.0
    lo = y0 if vert else x0
    for hx, hy, hw, hh, a, r in holes:
        across, along = (hw, hh) if vert else (hh, hw)
        pos = ((hy if vert else hx) - lo) / L
        if not (across >= 0.45 * Wd and across >= 2.5 * along and (pos <= 0.35 or pos >= 0.65)):
            continue
        # the shank cells: on the axis, narrower than the head, on the far side of the head
        cells = []
        for gx, gy, gw, gh, a2, r2 in holes:
            ca = gw if vert else gh
            cpos = ((gy if vert else gx) - lo) / L
            off = abs((gx if vert else gy) - mid)
            if ca <= 0.65 * Wd and off <= 0.2 * Wd and ((cpos > pos + 0.1) if pos <= 0.35 else (cpos < pos - 0.1)):
                cells.append(cpos)
        if len(cells) >= 2:
            return True
    return False


def _is_hole_ring(c, S, P, groups):
    """JLC mounting-hole symbol: a large circle (3-7u) holding an inner circle and a ring of
    small circles (the plated vias), one pin or none."""
    u = P.u
    if len(c.pins) > 1 or len(c.gids) != 1:
        return False
    x0, y0, x1, y1 = c.bbox
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if not (2.0 * u <= w <= 7.5 * u and 0.85 <= h / max(w, 1) <= 1.18):
        return False
    holes = _group_holes(c.gids[0], S, c.bbox)
    big = [hh for hh in holes if hh[2] >= 0.6 * w and 0.65 <= hh[5] <= 0.9]
    if not big:
        return False
    cx, cy, R = (x0 + x1) / 2.0, (y0 + y1) / 2.0, w / 2.0
    # inner circle drawn touching the outer one's group: a second, concentric round hole
    if any(0.3 * w <= hh[2] <= 0.7 * w and abs(hh[2] - hh[3]) <= 0.1 * hh[2] + 2 and 0.65 <= hh[5] <= 0.9
           and np.hypot(hh[0] - cx, hh[1] - cy) <= 0.1 * R for hh in holes):
        return True
    small = inner = 0
    for gid, g in groups.items():
        if gid == c.gids[0]:
            continue
        gx0, gy0, gx1, gy1 = g['bbox']
        gw, gh = gx1 - gx0 + 1, gy1 - gy0 + 1
        d = np.hypot((gx0 + gx1) / 2.0 - cx, (gy0 + gy1) / 2.0 - cy)
        if max(gw, gh) <= 0.7 * u and abs(gw - gh) <= 0.3 * max(gw, gh) + 2:
            if 0.3 * R <= d <= 0.9 * R:
                small += 1
        elif 0.3 * w <= gw <= 0.7 * w and abs(gw - gh) <= 0.1 * gw + 2 and d <= 0.1 * R:
            # a concentric inner circle (the hole itself; the 'GND' text sits inside it)
            hs = _group_holes(gid, S, g['bbox'])
            if any(hh[2] >= 0.6 * gw and 0.65 <= hh[5] <= 0.9 for hh in hs):
                inner += 1
    return small >= 6 or inner >= 1


_MOUNT_DESIG = re.compile(r'^(SCREW|HOLE|MH|M|H|HO|K|TP|U)\d+$', re.I)
_SCREW_SIZE = re.compile(r'^M[2-6](\.\d)?$')
M3_CIRCLE = os.environ.get('PCB_M3_CIRCLE', '1') == '1'


def _is_m3_circle(c, S, P, texts):
    """A mounting hole drawn as a plain circle with its screw size ('M3') written inside,
    on a short stem or without a pin."""
    u = P.u
    if len(c.pins) > 1 or len(c.gids) != 1:
        return False
    x0, y0, x1, y1 = c.bbox
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if not (1.0 * u <= w <= 3.5 * u and 0.85 <= h / max(w, 1) <= 1.6):
        return False
    holes = _group_holes(c.gids[0], S, c.bbox)
    big = [hh for hh in holes if hh[2] >= 0.6 * w and 0.65 <= hh[5] <= 0.9 and abs(hh[2] - hh[3]) <= 0.15 * hh[2] + 2]
    if not big:
        return False
    hx, hy, hw, hh_ = big[0][:4]
    inside = [it for it in texts if abs(it.cx - hx) <= 0.35 * hw and abs(it.cy - hy) <= 0.35 * hh_]
    if any(_SCREW_SIZE.match(it.text or '') for it in inside):
        return True
    # the 'M' of a small red 'M3' is often lost: a lone size digit inside, with the size
    # written out again beside the hole
    return any(re.fullmatch(r'[2-6](\.\d)?', it.text or '') for it in inside) and \
        any(_SCREW_SIZE.match(it.text or '') and box_gap(c.bbox, tbox(it)) <= 1.0 * u for it in texts)
_RING_DESIG = re.compile(r'^(SCREW|HOLE|MH|M|H|HO)\d+$', re.I)


TEXT_SHADE = os.environ.get('PCB_TEXT_SHADE', '1') == '1'
SHADE_WORD_DESIG = os.environ.get('PCB_SHADE_WORD_DESIG', '1') == '1'
WORD_NO_GND = os.environ.get('PCB_WORD_NO_GND', '1') == '1'
SHADE_PWR_PEN = os.environ.get('PCB_SHADE_PWR_PEN', '1') == '1'
WORD_NO_LINE = os.environ.get('PCB_WORD_NO_LINE', '1') == '1'
SHADE_DESIG_PEN = os.environ.get('PCB_SHADE_DESIG_PEN', '1') == '1'
SHADE_NAME_PEN = os.environ.get('PCB_SHADE_NAME_PEN', '1') == '1'
CJK_NAME_PEN = os.environ.get('PCB_CJK_NAME_PEN', '1') == '1'
_CJK_PN = re.compile(r'(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}')


def _text_shades(blue, P):
    """EasyEDA prints designators (and net / supply names) in navy and part names and
    values in bright blue.  Each blue text gets it.shade 'navy' / 'bright' from its darkest
    pixels; P.shade_mode is set when the sheet shows both shades often enough to rely on."""
    img = getattr(P, 'img', None)
    P.shade_mode = False
    if img is None:
        return
    H, W = img.shape[:2]
    nn = nb = 0
    for it in blue:
        it.shade = None
        x0, y0 = max(int(it.x0), 0), max(int(it.y0), 0)
        x1, y1 = min(int(it.x1), W - 1), min(int(it.y1), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        sub = img[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3).astype(np.int32)
        if len(sub) < 6:
            continue
        # the most saturated blue pixels (glyph cores): navy ~(0,0,128), bright ~(0,0,255);
        # pixels mixed with a wire or a symbol stroke under the text are less saturated
        sat = sub[:, 2] - np.maximum(sub[:, 0], sub[:, 1])
        k = max(4, len(sub) // 15)
        idx = np.argpartition(-sat, k)[:k]
        r, g, b = sub[idx].mean(0)
        if r > 70 or g > 70 or sat[idx].mean() < 60:
            continue
        if b < 165:
            it.shade = 'navy'
            nn += 1
        elif b > 205:
            it.shade = 'bright'
            nb += 1
    P.shade_mode = nn >= 4 and nb >= 4


def build(P, items, S, lines, uf, terms, lib=None, img_shape=None):
    global _LIB
    _LIB = lib
    _P_CTX[0] = P
    _S_CTX[0] = S
    NAME_LOG.clear()
    FB_LOG.clear()
    u, t = P.u, P.t
    H = P.H
    clean_pins(S, P)
    groups = S['groups']
    # ---------------------------------------------------------------- texts
    texts = [it for it in items if it.text]
    from .textfix import resolve_k_case
    _KFIX[0] = resolve_k_case(texts)
    blue = [it for it in texts if it.color == 'blue']
    if TEXT_SHADE:
        _text_shades(blue, P)
    # ---------------------------------------------------------------- comps from groups
    comps = {}
    for gid, g in groups.items():
        if g['area'] < max(6, 0.6 * t * t):
            continue
        pins = g['pins']
        if DEDUP_PINS and len(pins) > 1:
            # one stub found twice (split span, both ends on the same wire terminal)
            kept = []
            for pn in pins:
                dup = next((q for q in kept if q.o == pn.o and q.kind == pn.kind and pn.kind in ('wire', 'dot', 'free')
                            and abs(q.tip[0] - pn.tip[0]) <= 1.5 * t and abs(q.tip[1] - pn.tip[1]) <= 1.5 * t), None)
                if dup is None:
                    # the same point found by two routes (zero-length + contact, two contacts)
                    dup = next((q for q in kept if abs(q.tip[0] - pn.tip[0]) <= 1.5 and abs(q.tip[1] - pn.tip[1]) <= 1.5
                                and (q.o == pn.o or 'z' in (q.o, pn.o))), None)
                    if dup is not None and getattr(pn, 'mate', None) is not None and getattr(dup, 'mate', None) is None:
                        kept[kept.index(dup)] = pn
                        continue
                if dup is None:
                    kept.append(pn)
                elif np.hypot(pn.tip[0] - pn.base[0], pn.tip[1] - pn.base[1]) > np.hypot(dup.tip[0] - dup.base[0], dup.tip[1] - dup.base[1]):
                    kept[kept.index(dup)] = pn
            pins = kept
        c = Comp(gid, g['bbox'], pins)
        c.feat = shape_features(gid, S, P)
        comps[gid] = c
    if DROP_FRAMES and LINE_DROP:
        # page-sized line work (section dividers, function boxes drawn in the symbol colour)
        # with at most one stray pin: never a part (it was typed as a power symbol)
        for k in [k for k, c in comps.items() if len(c.pins) <= 1
                  and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) > 15 * u and c.feat.get('fill', 1) < 0.03]:
            del comps[k]
    mount_texts = set()
    if MOUNT_HOLES:
        for c in comps.values():
            screw = _is_screw(c, S, P) or (M3_CIRCLE and _is_m3_circle(c, S, P, texts))
            if not screw and _is_hole_ring(c, S, P, groups):
                # a test pad / connector pin drawn as two circles carries a signal name (RX):
                # the label next to a mounting hole, if any, is its designator (hole1, H3)
                lab = [it for it in blue if box_gap(c.bbox, tbox(it)) <= 1.2 * u]
                if lab and not _RING_DESIG.match(min(lab, key=lambda it: box_gap(c.bbox, tbox(it))).text):
                    continue
                if not lab and (c.bbox[2] - c.bbox[0]) < 3.0 * u:
                    continue
            elif not screw:
                continue
            if True:
                c.type = 'm3螺丝'
                c.anon = True
                c.mount = True
                # its designator (SCREW1, H3, hole2 ...) names no other part
                near = [it for it in blue if _MOUNT_DESIG.match(it.text) and box_gap(c.bbox, tbox(it)) <= 1.5 * u
                        and it.id not in mount_texts]
                if near:
                    # 'M3' printed with the hole is its screw size, not its designator (H13)
                    it = min(near, key=lambda it: (bool(_SCREW_SIZE.match(it.text)), box_gap(c.bbox, tbox(it))))
                    for o in blue:
                        if _SCREW_SIZE.match(o.text) and box_gap(c.bbox, tbox(o)) <= 1.5 * u and o is not it:
                            mount_texts.add(o.id)
                    mount_texts.add(it.id)
                    if MOUNT_KEYED:
                        # the newer answers key parts by designator (only VCC / GND are ∅)
                        c.anon = False
                        c.key = it.text
                        c.desig_item = it
    if FACING_PLATES:
        _merge_facing_plates(comps, S, P)
    if DC_JACK:
        _merge_dc_jacks(comps, S, P)
    if NET_SHORT_OUT:
        from .sym import Pin
        for c in comps.values():
            if c.type is not None:
                continue
            dm = _pinless_diamond(c, S, P)
            if dm is None:
                continue
            o, cl, e0, e1, hb = dm
            c.type, c.anon = 'net_short', True
            m_ = P.t + 1
            if o == 'h':
                c.pins = [Pin((float(e0), float(cl)), (hb[0], float(cl)), 'h', 'free'),
                          Pin((float(e1), float(cl)), (hb[2], float(cl)), 'h', 'free')]
                c.bbox = [hb[0] - m_, c.bbox[1], hb[2] + m_, c.bbox[3]]
            else:
                c.pins = [Pin((float(cl), float(e0)), (float(cl), hb[1]), 'v', 'free'),
                          Pin((float(cl), float(e1)), (float(cl), hb[3]), 'v', 'free')]
                c.bbox = [c.bbox[0], hb[1] - m_, c.bbox[2], hb[3] + m_]
            for pn in c.pins:
                pn.group = c.gids[0]
        for key in [k for k, c in comps.items() if c.type is None and len(c.pins) <= 2 * 4]:
            c = comps[key]
            ds = _diamond_stack(c, S, P)
            if not ds:
                continue
            m_ = P.t + 1
            for i, (o, cl, e0, e1, hb) in enumerate(ds):
                own = [pn for pn in c.pins if abs((pn.tip[1] if o == 'h' else pn.tip[0]) - cl) <= 0.4 * u]
                pins = []
                for e in (e0, e1):
                    tip = (float(e), float(cl)) if o == 'h' else (float(cl), float(e))
                    near = [pn for pn in own if abs((pn.tip[0] if o == 'h' else pn.tip[1]) - e) <= 0.5 * u]
                    if near:
                        pins.append(near[0])
                    else:
                        base = (hb[0] if e == e0 else hb[2], float(cl)) if o == 'h' else (float(cl), hb[1] if e == e0 else hb[3])
                        pn = Pin(tip, base, o, 'free')
                        pn.group = c.gids[0]
                        pins.append(pn)
                bb = [hb[0] - m_, cl - (hb[3] - hb[1]) / 2.0 - m_, hb[2] + m_, cl + (hb[3] - hb[1]) / 2.0 + m_] if o == 'h' \
                    else [cl - (hb[2] - hb[0]) / 2.0 - m_, hb[1] - m_, cl + (hb[2] - hb[0]) / 2.0 + m_, hb[3] + m_]
                nc = Comp(c.gids[0], bb, pins)
                nc.feat = c.feat
                nc.type, nc.anon = 'net_short', True
                comps[('short', key, i)] = nc
            del comps[key]
    comps = drop_red_text_parts(comps, texts, P)
    if DROP_CONTACT_ISLANDS:
        comps = drop_contact_islands(comps, P)
    if CARVE_GROUNDS and GND_FIRST:
        comps = carve_attached_grounds(comps, S, P)
    if SPLIT_FLAG_STACKS:
        comps = split_flag_stacks(comps, S, P, blue)
    if CARVE_FLAGS:
        comps = carve_attached_flags(comps, S, P, blue)
    if CARVE_GROUNDS and not GND_FIRST:
        comps = carve_attached_grounds(comps, S, P)
    # ---------------------------------------------------------------- power / labels (1-pin groups)
    used_text = set(mount_texts)

    def near_texts(c, maxd, pool):
        out = []
        for it in pool:
            d = box_gap(c.bbox, tbox(it))
            if d <= maxd:
                out.append((d, it))
        out.sort(key=lambda x: x[0])
        return out

    if FLAG_DOT_DROP:
        # a junction dot of a passing wire touching a flag's long side gives it a second
        # 'pin': without it the shape is a flag
        for c in comps.values():
            if len(c.pins) != 2 or getattr(c, 'feat', None) is None:
                continue
            dots = [pn for pn in c.pins if pn.kind in ('dot', 'zero')]
            if len(dots) != 1:
                continue
            # the dropped point must lie on a long side, away from both short ends
            x0_, y0_, x1_, y1_ = c.bbox
            dx_, dy_ = dots[0].tip
            if (x1_ - x0_) >= (y1_ - y0_):
                if not (x0_ + 0.2 * (x1_ - x0_) < dx_ < x1_ - 0.2 * (x1_ - x0_)):
                    continue
            elif not (y0_ + 0.2 * (y1_ - y0_) < dy_ < y1_ - 0.2 * (y1_ - y0_)):
                continue
            keep = [pn for pn in c.pins if pn is not dots[0]]
            saved = c.pins
            c.pins = keep
            if not _is_flag(c, P):
                c.pins = saved
    one_pin = [c for c in comps.values() if len(c.pins) == 1]
    blacks = [it for it in texts if it.color == 'black']
    for c in one_pin:
        c.flag = getattr(c, 'forced_flag', False) or _is_flag(c, P)
        if not c.flag and RECT_PORT and _rect_port(c, P, blue):
            c.flag = True                 # a plain rectangular net port with its name inside
        if c.flag:
            # a box with a pin number written inside is a one-pin connector (H2 '1'), not a net flag
            x0, y0, x1, y1 = c.bbox
            if any(it.text.isdigit() and x0 < it.cx < x1 and y0 < it.cy < y1 for it in blacks):
                c.flag = False
    # net labels (flag shaped ports): text next to the flag
    flags_ = [c for c in one_pin if c.flag]

    def _flag_cands(c):
        return [(d + _side_pen(c, it, u) + _align_pen(c, it, u), it) for d, it in near_texts(c, 1.5 * u, blue)
                if it.id not in used_text and (not is_value(it.text)
                                               or (FLAG_VOLT and is_power_name(it.text) and _side_pen(c, it, u) == 0
                                                   and _axis_off(c, it) <= 0.4 * u)
                                               or (IO_ALT and getattr(it, 'io_alt', None))
                                               # a numbered port ('1' ... '12') right on the flag's axis
                                               or (FLAG_DIGITS and re.fullmatch(r'\d{1,3}', it.text or '')
                                                   and _side_pen(c, it, u) == 0 and _axis_off(c, it) <= 0.4 * u))
                and not (FLAG_OWN_AXIS and _better_aligned_flag(c, it, flags_, u))]
    pick = {}
    if FLAG_HUNGARIAN and flags_:
        # a stack of flags with labels set half a step off: one label per flag, the best
        # assignment overall (taking flags one by one shifts the whole stack by a label)
        cl = {id(c): _flag_cands(c) for c in flags_}
        tx = {}
        for c in flags_:
            for _, it in cl[id(c)]:
                tx.setdefault(it.id, it)
        tl = list(tx.values())
        if tl:
            ti = {it.id: j for j, it in enumerate(tl)}
            Cf = np.full((len(flags_), len(tl)), 1e6)
            for i, c in enumerate(flags_):
                for sc, it in cl[id(c)]:
                    Cf[i, ti[it.id]] = sc
            Cf = np.hstack([Cf, np.full((len(flags_), len(flags_)), 1e5)])
            rf, cf = linear_sum_assignment(Cf)
            for i, j in zip(rf, cf):
                if j < len(tl) and Cf[i, j] < 1e5:
                    pick[id(flags_[i])] = tl[j]
    for c in one_pin:
        if not c.flag:
            continue
        if FLAG_HUNGARIAN:
            cand = [(0.0, pick[id(c)])] if id(c) in pick and pick[id(c)].id not in used_text else []
        else:
            cand = _flag_cands(c)
        cand.sort(key=lambda x: x[0])
        if cand:
            d, it = cand[0]
            if IO_ALT and getattr(it, 'io_alt', None) and is_value(it.text):
                it.text = it.io_alt              # a port label IO48 read as 1048
            c.type = _flag_type(c, P)
            c.label_item = it
            c.key = it.text
            c.value = it.text
            c.name = it.text
            used_text.add(it.id)
    # power / ground symbols recognised by SHAPE (T bar / stacked bars); the name text can be anything
    # (VM1, 3V3_MCU, VBAT ...), so it is taken from the far side of the symbol, not matched by a regex
    for c in one_pin:
        carved = CARVED_GND_NAME and getattr(c, 'carved_gnd', False)
        if c.flag or (c.type is not None and not carved):
            continue
        # grounds cut out of another group are grounds already; they still need their name
        ps = 'gnd' if carved else _power_shape(c, S, P)
        if ps is not None and not carved and ARROWS_NOT_GND and _arrow_pair(c, S, P):
            ps = None                       # an LED's light arrows touching its cathode bar
        if ps is None:
            continue
        tip = np.array(c.pins[0].tip, float)
        ctr = np.array([(c.bbox[0] + c.bbox[2]) / 2.0, (c.bbox[1] + c.bbox[3]) / 2.0])
        axis = ctr - tip
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm > 1e-6 else np.array([0.0, -1.0])
        best = None
        for d, it in near_texts(c, (2.0 if FAR_POWER_NAME else 1.3) * u, blue):
            if it.id in used_text or (is_value(it.text) and not (PWR_VOLT_NAME and is_power_name(it.text))):
                continue                        # (a supply named by its voltage, '3V3', is a name)
            if d > 1.3 * u:
                # a name set further out: only straight beyond the bar, on the symbol's axis
                tc_ = np.array([it.cx, it.cy]) - ctr
                al_ = float(tc_ @ axis)
                pe_ = abs(float(tc_[0] * axis[1] - tc_[1] * axis[0]))
                if FAR_PWR_SPAN and pe_ > 0.5 * u:
                    # beside the axis but still level with the symbol (name set flush with
                    # the top of a sideways ground)
                    if abs(axis[0]) >= abs(axis[1]):
                        pe_ = 0.0 if c.bbox[1] <= it.cy <= c.bbox[3] else pe_
                    else:
                        pe_ = 0.0 if c.bbox[0] <= it.cx <= c.bbox[2] else pe_
                if not (is_power_name(it.text) and al_ > 0 and pe_ <= 0.5 * u):
                    continue
            if ps == 'gnd' and not (is_gnd_name(it.text) or (is_power_name(it.text) and not GND_NAMES_ONLY)
                                    or it.text.upper() in ('EARTH', 'PE', 'VSS', '0V', 'COM', 'B-', 'FG')):
                continue
            if ps == 'v' and not is_power_name(it.text) and is_desig(it.text) and prefix_type(it.text) not in (None,):
                # a designator-like name ('DS0') right beyond the bar, on its axis, names the port
                tq = np.array([it.cx, it.cy]) - ctr
                if not (PWR_DESIG_LIKE and float(tq @ axis) > 0 and d <= 0.5 * u
                        and abs(float(tq[0] * axis[1] - tq[1] * axis[0])) <= 0.3 * u):
                    continue
            tc = np.array([it.cx, it.cy]) - ctr
            along = float(tc @ axis)
            beside = 0.0
            if along < -0.2 * u:
                # text on the pin side belongs to something else - unless it sits beside the
                # stem (between the pin tip and the bars), a common place for 'PGND' / 'AGND'
                if along < -nrm - 0.1 * u or not (is_gnd_name(it.text) if ps == 'gnd' else is_power_name(it.text)):
                    continue
                beside = 0.3 * u
            score = d + beside - 0.3 * u * (1 if is_power_name(it.text) else 0)
            if SHADE_PWR_PEN and getattr(P, 'shade_mode', False) and getattr(it, 'shade', None) == 'bright':
                score += 0.5 * u              # supply names are printed in navy, values in bright blue
            if best is None or score < best[0]:
                best = (score, it)
        if ps == 'v' and best is None:
            continue                        # a lone bar without a name: leave it to the generic logic
        c.type = 'gnd' if (ps == 'gnd' or (best is not None and is_gnd_name(best[1].text))) else 'v'
        if best is not None:
            it = best[1]
            c.label_item = it
            used_text.add(it.id)
            c.value = 'GND' if (c.type == 'gnd' and (len(it.text) == 3 and it.text != 'GND' and not it.text.endswith('GND')
                                                     or re.fullmatch(r'(?:GND){2,}', it.text))) else _fix_comma_only(it)
        elif c.type == 'gnd' and _tri_ground(c, P):
            c.value = 'AGND'
        else:
            c.value = None if HIDDEN_GND_NONE else 'GND'     # name not printed: value None (official labels)
        c.anon = True
    # power / ground symbols with a name text
    for c in one_pin:
        if c.flag or c.type is not None:
            continue
        cand = near_texts(c, 1.2 * u, blue)
        pw = [(d + _side_pen(c, it, u), it) for d, it in cand if is_power_name(it.text) and not is_desig(it.text) and it.id not in used_text]
        pw.sort(key=lambda x: x[0])
        if pw:
            d, it = pw[0]
            c.label_item = it
            used_text.add(it.id)
            c.type = 'gnd' if is_gnd_name(it.text) else 'v'
            c.value = 'GND' if (c.type == 'gnd' and (len(it.text) == 3 and it.text != 'GND' and not it.text.endswith('GND')
                                                     or re.fullmatch(r'(?:GND){2,}', it.text))) else _fix_comma_only(it)
            c.anon = True
    if MERGE_BODY_PARTS:
        _merge_body_parts(comps, S, P)
    if DROP_NET_TIES:
        for k in [k for k, c in comps.items() if c.type is None and not c.anon and _is_net_tie(c, P)]:
            c = comps[k]
            if NET_SHORT_OUT and len(c.pins) == 1 and TRI_AGND and SRC_STYLE[0] in ('easyeda', 'off') \
                    and _hollow_triangle(c, S):
                # a lone hollow triangle on one wire is EasyEDA's analogue-ground symbol (the
                # labels have AGND there), not half a short
                c.type, c.value, c.anon = 'gnd', 'AGND', True
                continue
            if NET_SHORT_OUT and len(c.pins) in (1, 2):
                # the official type list has it: 'net_short', anonymous key, bbox = the diamond
                c.type = 'net_short'
                c.anon = True
                hs = _group_holes(c.gids[0], S, S['groups'][c.gids[0]]['bbox'])
                if hs:
                    hx, hy, hw, hh, a, r = max(hs, key=lambda x: x[4])
                    m_ = P.t + 1
                    x0, y0, x1, y1 = c.bbox
                    if (x1 - x0) >= (y1 - y0):
                        c.bbox = [max(x0, hx - hw / 2.0 - m_), y0, min(x1, hx + hw / 2.0 + m_), y1]
                    else:
                        c.bbox = [x0, max(y0, hy - hh / 2.0 - m_), x1, min(y1, hy + hh / 2.0 + m_)]
            else:
                del comps[k]
    if DROP_FRAMES:
        # red section-divider lines of the sheet: a pinless group spanning a large part of
        # the page with almost no ink inside its box
        for k in [k for k, c in comps.items() if c.type is None and not c.anon and not c.pins
                  and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) > 15 * u
                  and (c.feat.get('fill', 1) < 0.02
                       # a long straight pinless stroke: edge of a function box / section divider
                       or (LINE_DROP and min(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) < 1.2 * u))]:
            del comps[k]
    if SPLIT_PAIRS:
        _split_grouped_pairs(comps, S, P, [it for it in blue if it.id not in used_text])
    if SPLIT_STACKS:
        _split_stacked_rows(comps, S, P, [it for it in blue if it.id not in used_text])
    if ARROWS_TO_LED:
        _fold_light_arrows(comps, S, P)
    # ---------------------------------------------------------------- designators
    func_ids = _functional_designators([it for it in blue if it.id not in used_text]) if FUNC_DESIG else set()
    desig_items = [it for it in blue if it.id not in used_text and not getattr(it, 'big', False)
                   and (it.id in func_ids or _fix_desig(it))
                   and not (lib is not None and len(it.text) >= 5 and lib.match_name(it.text, 0.0))]
    word_ids = set()
    word_items = []
    if SHADE_WORD_DESIG and getattr(P, 'shade_mode', False):
        # a word printed in the designator colour off the wires may name a part ('RESET',
        # 'BOOT', '电机A接口'); such words only name parts no regular designator claims
        have = {it.id for it in desig_items}
        for it in blue:
            t_ = (it.text or '').strip()
            if (it.id in used_text or it.id in have or getattr(it, 'big', False) or getattr(it, 'shade', None) != 'navy'
                    or not t_ or len(t_) > 12 or is_value(t_) or is_power_name(t_) or _on_wire(it, P)
                    or not re.fullmatch(r'[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff+\-.]*', t_)
                    or re.fullmatch(r'[A-Z]{1,3}', t_) or (WORD_NO_GND and re.search(r'GN\w{0,2}D', t_))
                    or (WORD_NO_LINE and len(re.sub(r'[^A-Za-z0-9\u4e00-\u9fff]', '', t_)) < 2)):
                continue
            word_items.append(it)
    if WIRE_PIECES:
        # a designator set over a pin lead cuts the red line under its glyphs; the piece left
        # between the glyphs and the wire is a short straight run with a pin at each end: it
        # is part of the lead (a wire), not a part
        P.wire_pieces = []
        for k in [k for k, c in comps.items() if not c.anon and c.type is None and len(c.pins) == 2]:
            c = comps[k]
            w, h = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
            if min(w, h) > max(2.5, 1.5 * t) or max(w, h) >= 0.6 * u:
                continue
            a, b = c.pins
            ax = 0 if w >= h else 1
            if abs(a.tip[ax] - b.tip[ax]) < 0.6 * max(w, h) or abs(a.tip[1 - ax] - b.tip[1 - ax]) > max(2.0, t):
                continue
            c.type = 'wire'
            P.wire_pieces.append(c)
            del comps[k]
    cand_comps = [c for c in comps.values() if not c.anon and c.type is None]
    if DESIG_NO_SPECK:
        # a speck of ink (a wire-corner fragment a few pixels across) never carries a designator
        cand_comps = [c for c in cand_comps
                      if not (len(c.pins) <= 1 and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) < 0.3 * u)]
    if desig_items and cand_comps:
        C = np.full((len(desig_items), len(cand_comps)), 1e6)
        onw = [_on_wire(it, P) for it in desig_items]
        under = [0.0] * len(desig_items)
        if DESIG_STACK_PRIOR:
            # EasyEDA prints the designator first and the part number on the next line: a
            # model-like text (S10M, US1M) right under a clean designator (D1) is the name
            for i, b in enumerate(desig_items):
                if b.vert or desig_implausibility(b.text) <= 0:
                    continue
                for a in desig_items:
                    if a is b or a.vert or desig_implausibility(a.text) > 0:
                        continue
                    h = max(a.h, b.h)
                    if (abs(a.x0 - b.x0) <= 0.3 * u or abs(a.cx - b.cx) <= 0.3 * u) and 0 < b.cy - a.cy <= 1.8 * h:
                        under[i] = 0.4
                        break
        rects = {id(c): rect_hole(c.gids[0], S, P) for c in cand_comps}
        cores = {id(c): _two_pin_core(c, P) for c in cand_comps} if LED_CORE_DESIG else {}
        blocks = {}
        if DESIG_BLOCK:
            # a designator with its part number set on the line right under it is one label
            # block; the part sits next to the block, not necessarily next to the designator
            # (a column of LEDs: 'LED2' / 'FC-A2012...' above LED2, LED1's arrows just above)
            for it in desig_items:
                if it.vert:
                    continue
                for o in blue:
                    if o is it or o.vert or o.id in used_text or is_desig(o.text) or len(o.text) < 4 \
                            or not _partnumber_shape(o.text):
                        continue
                    h = max(it.h, o.h)
                    if (abs(o.x0 - it.x0) <= 0.3 * u or abs(o.cx - it.cx) <= 0.3 * u) and 0 < o.y0 - it.y1 <= 0.6 * h:
                        blocks[it.id] = [min(it.x0, o.x0), it.y0, max(it.x1, o.x1), o.y1]
                        break
        for i, it in enumerate(desig_items):
            for j, c in enumerate(cand_comps):
                d = box_gap(cores.get(id(c)) or c.bbox, tbox(it)) / u
                if it.id in blocks:
                    d = min(d, box_gap(cores.get(id(c)) or c.bbox, blocks[it.id]) / u)
                if d > 3.0:
                    # labels of big ICs sit beyond the pin row: measure from the pin tips too
                    d2 = box_gap(_ext_bbox(c), tbox(it)) / u
                    if d2 > 2.6 or len(c.pins) < 8:
                        continue
                    d = 0.4 + 0.8 * d2    # must stay below the 2.6 cost of leaving the text unassigned
                pt = prefix_type(it.text)
                npins = len(c.pins)
                pen = 0.8 if onw[i] else 0.0
                big = rects[id(c)] is not None and min(c.feat['w'], c.feat['h']) >= 2.0 * u
                if pt == 'box' and not big and max(c.feat['w'], c.feat['h']) < 1.6 * u and npins <= 2:
                    pen += 0.8
                if pt in ('r', 'c', 'l', 'd', 'led') and big and npins >= 3:
                    pen += 1.5
                if big and npins >= 4 and pt not in ('box', 'switch', 'pin', None):
                    pen += 0.8
                if npins == 0:
                    pen += PINLESS_PEN
                if pt in ('r', 'c', 'l') and npins > 2:
                    pen += 1.0
                elif pt == 'battary' and npins > 3:
                    pen += 2.0            # a 4-pin 'BAT0': a net label next to a transistor
                if (pt == 'box' and npins < 2 and c.feat['w'] < 1.5 * u and c.feat['h'] < 1.5 * u
                        and not max(c.feat['w'], c.feat['h']) < 1.6 * u):      # never stack the two small-box penalties
                    pen += 0.8
                if npins == 1 and pt in ('r', 'c', 'l', 'd', 'led'):
                    pen += 0.6
                imp = 0.2 if it.id in func_ids else desig_implausibility(it.text)
                if getattr(c, 'cell_kind', None) and pt != c.cell_kind and not (CELL_SUFFIX and re.search(
                        r'(?<=[A-Z]{2})%s\d{1,4}$' % {'r': 'R', 'c': 'C', 'l': 'L'}.get(c.cell_kind, 'R'), it.text)):
                    pen += 3.0               # a cell of stacked resistor boxes takes a resistor's label
                if PIN_NUM_PEN and npins >= 6 and len(it.text) <= 3 and _on_pin_lead(c, it, u):
                    pen += 3.0               # a pin number written along the pin's lead (KiCad 'A6')
                if SHADE_DESIG_PEN and getattr(P, 'shade_mode', False) and getattr(it, 'shade', None) == 'bright':
                    pen += 1.5               # bright blue is a part name / value, designators are navy
                C[i, j] = d + pen + imp + under[i]
        # dummy columns: leaving a text unassigned costs a fixed amount
        nT, nC = C.shape
        C = np.hstack([C, np.full((nT, nT), 2.6)])
        r, cidx = linear_sum_assignment(C)
        if DESIG_CONSIST:
            C2 = _desig_consistency(C, r, cidx, desig_items, cand_comps, u)
            if C2 is not None:
                r, cidx = linear_sum_assignment(C2)
        if DESIG_CHAIN:
            ch = _desig_chains(C, r, cidx, desig_items, cand_comps, u)
            if ch is not None:
                r, cidx = ch
        for i, j in zip(r, cidx):
            if j < nC and C[i, j] < 1e5:
                c = cand_comps[j]
                c.desig_item = desig_items[i]
                c.key = desig_items[i].text
                used_text.add(desig_items[i].id)
    if CELL_SUFFIX:
        # one cell of a stack of resistor boxes: its designator ran into the net name written
        # just before it on the wire ('SCL' + 'R97' read as 'SCLR97')
        for c in comps.values():
            ck = getattr(c, 'cell_kind', None)
            if c.key and ck in ('r', 'c', 'l') and prefix_type(c.key) != ck:
                sm = re.search(r'(?<=[A-Z]{2})(%s\d{1,4})$' % {'r': 'R', 'c': 'C', 'l': 'L'}[ck], c.key)
                if sm:
                    c.key = sm.group(1)
                    c.word_key = False
    if FAR_DESIG:
        _far_designators(comps, [it for it in desig_items if it.id not in used_text] if desig_items else [], used_text, P)
    if word_items:
        # last pass: parts still without a designator (after the far designators) take a
        # nearby navy word
        left = [c for c in comps.values() if not c.anon and c.type is None and c.key is None and len(c.pins) >= 2
                # (a bare straight stroke is no part: two port bars drawn end to end)
                and not (WORD_NO_LINE and min(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) < 0.3 * u)
                and not (WORD_NO_RES and len(c.pins) == 2 and _resistor_box(c, S, P))]
        words = [it for it in word_items if it.id not in used_text]
        if left and words:
            Cw = np.full((len(words), len(left)), 1e6)
            for i, it in enumerate(words):
                for j, c in enumerate(left):
                    if c.bbox[0] < it.cx < c.bbox[2] and c.bbox[1] < it.cy < c.bbox[3]:
                        continue
                    d = box_gap(_ext_bbox(c), tbox(it)) / u
                    if d <= 1.5:
                        Cw[i, j] = d
            rw, cw = linear_sum_assignment(Cw)
            for i, j in zip(rw, cw):
                if Cw[i, j] < 1e5:
                    it = words[i]
                    it.text = (it.text or '').strip()
                    left[j].desig_item = it
                    dk = _word_as_desig(it, left[j]) if WORD_REDECODE else None
                    if dk:
                        left[j].key = dk          # an IC designator misread ('U6' as 'UE')
                    else:
                        left[j].key = it.text
                        left[j].word_key = True
                    used_text.add(it.id)
    # components that got no designator would be dropped altogether: give multi-pin ones the
    # nearest free label outside their body (non-standard designators: SWD, BUZZER1, SERVO_A)
    if MERGE_TOUCHING:
        _merge_touching_pieces(comps, S, P)
        one_pin = [c for c in one_pin if len(c.pins) == 1]
    if ORPHAN_DESIG:
        # a numbered series of one stem (WS2812B1 ... WS2812B8 over a row of identical
        # parts) names parts even though each looks like a part number: a part number is
        # repeated verbatim, a designator counts up
        series = set()
        if ORPHAN_SERIES:
            stems = {}
            for it in blue:
                m_ = re.fullmatch(r'([A-Z][A-Z0-9]*[A-Z])(\d{1,2})', it.text or '')
                if m_ and it.id not in used_text:
                    stems.setdefault(m_.group(1), set()).add(m_.group(2))
            series = {st + n for st, ns in stems.items() if len(ns) >= 2 for n in ns}
        pairs = []
        for c in comps.values():
            if c.anon or c.type is not None or c.key is not None or len(c.pins) < 3:
                continue
            eb = _ext_bbox(c)
            best = None
            for it in blue:
                t = it.text
                if IO_ALT and getattr(it, 'io_alt', None) and it.id not in used_text:
                    t = it.io_alt                # '104' over a connector: its designator IO4
                if it.id in used_text or not t or len(t) > 12 or is_value(t) or is_power_name(t):
                    continue
                if not _ORPHAN_RE.match(t) and not (ORPHAN_SERIES and t in series):
                    continue                    # designator-shaped: letters, then at most 3 digits (not a part number)
                if c.bbox[0] < it.cx < c.bbox[2] and c.bbox[1] < it.cy < c.bbox[3]:
                    continue
                if _on_wire(it, P):
                    continue
                d = box_gap(eb, tbox(it)) / u
                if d <= 1.5:
                    if ORPHAN_NEAREST:
                        pairs.append((d, id(c), c, it))
                    elif best is None or d < best[0]:
                        best = (d, it)
            if best is not None:
                pairs.append((best[0], id(c), c, best[1]))
        # nearest pairs first: a label goes to the part it sits next to, not to whichever part
        # happened to be visited first
        for d, _, c, it in sorted(pairs, key=lambda x: (x[0], x[1])):
            if c.key is not None or it.id in used_text:
                continue
            if IO_ALT and getattr(it, 'io_alt', None) and not _ORPHAN_RE.match(it.text):
                it.text = it.io_alt
            c.desig_item = it
            c.key = (_word_as_desig(it, c) if WORD_REDECODE else None) or it.text
            c.orphan_key = True
            used_text.add(it.id)
    if HIDDEN_DESIG:
        # an IC whose designator is hidden (the sheet shows U2, U3 but no U1) would be
        # dropped for want of a key: designators count up, so it takes the gap in the series
        hidden = []
        for c in comps.values():
            if c.anon or c.type is not None or c.key is not None or len(c.pins) < 4:
                continue
            r_ = rect_hole(c.gids[0], S, P)
            if r_ is None or min(r_[2] - r_[0], r_[3] - r_[1]) < 1.5 * u:
                continue
            # an IC body: the rectangle is the part (not a gap between two stacked parts)
            # and pins leave it on two sides or more (a display or header has one pin row)
            bw, bh = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
            if (r_[2] - r_[0]) * (r_[3] - r_[1]) < 0.5 * bw * bh:
                continue
            sides = set()
            for pn in c.pins:
                tx, ty = pn.tip
                if tx < r_[0]:
                    sides.add('l')
                elif tx > r_[2]:
                    sides.add('r')
                elif ty < r_[1]:
                    sides.add('t')
                elif ty > r_[3]:
                    sides.add('b')
            if len(sides) < 2 and len(c.pins) < 10:
                continue                    # (a module with one long pin row is an IC too)
            if HIDDEN_NOT_PIECE and any(o is not c and o.key and not o.anon and not getattr(o, 'flag', False)
                                        and o.type not in ('gnd', 'v') and len(o.pins) <= 3 and box_gap(c.bbox, o.bbox) <= 0.4 * u
                                        for o in comps.values()):
                continue                    # the body of a designated part drawn in two pieces (ZX1)
            hidden.append(c)
        # a designator set farther out (above the wire running along the body) is no hidden
        # one: the nearest free designator-shaped label within 3 units names the part
        for c in list(hidden):
            eb = _ext_bbox(c)
            best = None
            for it in blue:
                if it.id in used_text or it.vert or not is_desig(it.text or '') or is_power_name(it.text) \
                        or is_value(it.text) or prefix_type(it.text) != 'box':
                    continue                # (an IC's designator, not a net name like IO15)
                if c.bbox[0] < it.cx < c.bbox[2] and c.bbox[1] < it.cy < c.bbox[3]:
                    continue
                d = box_gap(eb, tbox(it)) / u
                if d <= 3.0 and (best is None or d < best[0]):
                    best = (d, it)
            if best is not None:
                c.desig_item = best[1]
                c.key = best[1].text
                c.orphan_key = True
                used_text.add(best[1].id)
                hidden.remove(c)
        if hidden and any(re.fullmatch(r'U\d{1,3}', it.text or '') and it.id not in used_text for it in blue):
            hidden = []                     # a U label nobody took: the part may be that one's
        if hidden:
            nums = set()
            for o in comps.values():
                m_ = re.fullmatch(r'U(\d{1,3})(?:_\d+)?', o.key or '')
                if m_:
                    nums.add(int(m_.group(1)))
            for it in blue:
                m_ = re.fullmatch(r'U(\d{1,3})', it.text or '')
                if m_:
                    nums.add(int(m_.group(1)))
            gaps = [n for n in range(1, max(nums) + 1) if n not in nums] if nums else [1]
            if len(hidden) == 1 and len(gaps) == 1:
                keys_ = ['U%d' % gaps[0]]       # one gap only: U2, U3 shown -> U1
            elif len(gaps) == len(hidden):
                # as many hidden ICs as gaps: numbered in reading order (rows, then left to right)
                hidden.sort(key=lambda c: (round(c.bbox[1] / (2.0 * u)), c.bbox[0]))
                keys_ = ['U%d' % n for n in gaps]
            else:
                hidden, keys_ = [], []
            for c, k_ in zip(hidden, keys_):
                c.key = k_
                c.hidden_key = True
    # GND symbols without text: detect by shape (parallel bars)
    for c in one_pin:
        if c.type is None and c.key is None:
            ps = _power_shape(c, S, P)
            if ps and ARROWS_NOT_GND and _arrow_pair(c, S, P):
                ps = None                   # an LED's light arrows touching its cathode bar
            if ps:                          # nameless bars are filtered by size at the end
                c.type = ps
                c.value = 'GND' if (ps == 'gnd' and not HIDDEN_GND_NONE) else None
                if ps == 'gnd' and _tri_ground(c, P):
                    c.value = 'AGND'
                c.anon = True
    # ---------------------------------------------------------------- net labels (1-pin, text along axis)
    for c in one_pin:
        if c.type is not None or c.key is not None:
            continue
        if not FALLBACK_LABELS:
            # (measured on the dev set: these one-pin pieces near a text are fragments of a
            # part - body diode, LED arrows, a pad - and the text is that part's name)
            continue
        cand = [(d, it) for d, it in near_texts(c, 1.0 * u, blue) if it.id not in used_text and not is_value(it.text)]
        if cand:
            d, it = cand[0]
            c.type = _flag_type(c, P)
            c.fallback_label = True
            FB_LOG.append(dict(key=it.text, bbox=list(c.bbox), d=round(d / u, 2), side=round(_side_pen(c, it, u) / u, 2),
                               align=round(_align_pen(c, it, u) / u, 2), w=c.bbox[2] - c.bbox[0], h=c.bbox[3] - c.bbox[1],
                               holes=len(c.feat.get('holes') or []), ngid=len(c.gids)))
            c.label_item = it
            c.key = it.text
            c.value = it.text
            c.name = it.text
            used_text.add(it.id)
    # ---------------------------------------------------------------- types
    for c in comps.values():
        if c.type is not None:
            continue
        if c.key:
            if (Q_FROM_O and re.match(r'^[O0o]\d{1,3}$', c.key) and len(c.pins) == 3
                    and (_n_diagonals(c, S, P) >= 2 or _has_gate_gap(c, S, P))):
                c.key = 'Q' + c.key[1:]       # OCR read the Q of a transistor's designator as O
                if c.desig_item is not None:
                    c.desig_item.text = c.key
            pt = prefix_type(c.key) or 'box'
            if getattr(c, 'word_key', False):
                pt = _word_key_type(c.key, len(c.pins))
            elif getattr(c, 'orphan_key', False) and not is_desig(c.key) and pt not in ('buzzer', 'battary', 'led'):
                pt = 'box'                  # SWD / SERVO_A: a connector, not a switch
            if pt == 'd' and _has_arrows(c, S, P):
                pt = 'led'
            elif XTAL_LED and pt == 'crystal_2pin' and len(c.pins) == 2 and _has_arrows(c, S, P) \
                    and _n_diagonals(c, S, P) >= 2:
                pt = 'led'                  # an LED labelled X1: a diode triangle with light arrows
            if pt == 'box' and len(c.pins) == 1 and re.match(r'^(H|J|P|CN|JP|TP|PIN|PAD)\d', c.key or ''):
                pt = 'pin'                  # one-pin headers / pads; a one-pin U stays an IC (box)
            npin = len(c.pins)
            if pt in ('mosfet', 'bjt'):
                pt = 'bjt' if _is_bjt(c, S, P) else 'mosfet'
            elif pt == 'crystal_2pin':
                pt = {3: 'crystal_3pin', 4: 'crystal_4pin'}.get(npin, 'crystal_2pin') if npin <= 4 else 'box'
            elif pt == 'fuse' and npin >= 3:
                pt = 'box'                  # F-prefixed connector (FPC1, FH1): fuses have two pins
            elif pt == 'transfomer' and npin >= 4 and _rect_main(c, S, P):
                pt = 'box'                  # a T-prefixed module drawn as a plain rectangle
            elif (pt in ('buzzer', 'battary', 'motor') and npin >= 5 and _rect_main(c, S, P)
                  and re.match(r'^[A-Za-z]+', c.key).group(0).upper() not in _EXACT_PREFIXES):
                pt = 'box'                  # Board1: a pin row on a rectangle (B1 stays a buzzer by its prefix)
            elif pt == 'box' and npin >= 3 and not rect_hole(c.gids[0], S, P) and (
                    _amp_triangle(c, S, P) if AMP_TRIANGLE else _n_diagonals(c, S, P) >= 2):
                pt = {3: 'amp_3pin', 5: 'amp_5pin', 7: 'amp_7pin'}.get(npin, 'amp')
            if getattr(c, 'dc_jack', False):
                pt = 'dc'                   # official type of a DC power jack (anonymous key)
            c.type = pt
    if SPLIT_DOUBLES:
        _split_double_parts(comps, blue, used_text, P)
    # drop unassigned comps without pins/type
    if MERGE_CONTACT_ORPHANS:
        _merge_orphan_pieces(comps)
    if MERGE_SWITCH_PIECES:
        _merge_switch_pieces(comps, u)

    if LEAD_PIECES:
        _absorb_lead_pieces(comps, P)
    final = [c for c in comps.values() if c.type is not None]
    final = _merge_contact_pieces(final, used_text)
    if DOT_END_PIN:
        _pins_at_end_dots(final, P)
    # ---------------------------------------------------------------- values / names
    free_blue = [it for it in blue if it.id not in used_text]
    # values of passives: one-to-one assignment
    passives = [c for c in final if c.desig_item is not None and c.type in ('r', 'c', 'l', 'fuse', 'crystal_2pin')
                + (('crystal_3pin', 'crystal_4pin') if CRYSTAL_ALL_VALUES else ())]
    _CAP_CTX[0] = {'prefixed': any(re.search(r'\d[pnuμµUNP]F', it.text) for it in blue),
                   'counts': Counter(it.text.replace(' ', '') for it in blue if re.fullmatch(r'\d+(?:\.\d+)?F', it.text.replace(' ', '')))}
    vcands = [it for it in free_blue if not is_desig(it.text) and not is_power_name(it.text)
              and (len(it.text) <= 14 or (len(it.text) <= 22 and re.search(r'[,%\s±]', it.text))
                   or (THERMISTOR_VALUE and len(it.text) <= 24 and re.search(r'[NP]TC', it.text)))]
    if passives and vcands:
        C = np.full((len(passives), len(vcands)), 1e6)
        for i, c in enumerate(passives):
            for j, it in enumerate(vcands):
                d = min(box_gap(tbox(it), tbox(c.desig_item)), box_gap(tbox(it), c.bbox) + 0.3 * u) / u
                if not is_value(it.text) and not (THERMISTOR_VALUE and c.type == 'r' and re.search(r'[NP]TC', it.text)):
                    d += 0.9                 # (a thermistor's value is its type string 'NTC10D-5')
                if SHADE_NAME_PEN and getattr(P, 'shade_mode', False) and getattr(it, 'shade', None) == 'navy':
                    d += 1.5                 # navy is a designator / net name; values are bright blue
                d += _unit_pen(it.text, c.type)
                if d <= 2.2:
                    C[i, j] = d
        r, cc = linear_sum_assignment(C)
        for i, j in zip(r, cc):
            if C[i, j] < 1e5:
                passives[i].value = _fix_value(vcands[j], passives[i].type)
                used_text.add(vcands[j].id)
    named = [c for c in final if (c.desig_item is not None or getattr(c, 'hidden_key', False))
             and c.type not in ('r', 'c', 'l', 'fuse', 'crystal_2pin')
             and not (CRYSTAL_ALL_VALUES and c.type in ('crystal_3pin', 'crystal_4pin'))]
    if NAME_ASSIGN:
        _assign_names(named, free_blue, used_text, u, lib)
    elif NAME_GLOBAL:
        # one pass without claiming; then the closest claims first, and a part whose text
        # was taken by a closer one looks again (a large IC must not take the part number
        # printed right under the diode next to it)
        first = {}
        for c in named:
            res = pick_name(c, free_blue, used_text, u, lib if c.type in ("box", "switch") else None)
            if res:
                first[id(c)] = (getattr(c, 'name_score', 0.0), res)
        for c in sorted(named, key=lambda c: first[id(c)][0] if id(c) in first else 1e9):
            if id(c) not in first:
                continue
            res = first[id(c)][1]
            if any(it.id in used_text for it in res[1]):
                res = pick_name(c, free_blue, used_text, u, lib if c.type in ("box", "switch") else None)
            if res:
                c.name = res[0]
                for it in res[1]:
                    used_text.add(it.id)
    else:
        for c in named:
            res = pick_name(c, free_blue, used_text, u, lib if c.type in ("box", "switch") else None)
            if res:
                c.name = res[0]
                for it in res[1]:
                    used_text.add(it.id)
    # twins split from one body (two copies of a part) carry the same part number: a free
    # copy of the twin's name next to the unnamed one is its name
    for c in final:
        if not getattr(c, 'split_double', False) or c.name:
            continue
        twin = next((o for o in final if o is not c and getattr(o, 'split_double', False) and o.name
                     and o.type == c.type and box_gap(o.bbox, c.bbox) <= 2 * u), None)
        if twin is None:
            continue
        cands = [it for it in free_blue if it.id not in used_text and it.text == twin.name
                 and box_gap(tbox(it), c.bbox) <= 2.5 * u]
        if cands:
            it = min(cands, key=lambda it: box_gap(tbox(it), c.bbox))
            c.name = it.text
            used_text.add(it.id)
    # ---------------------------------------------------------------- LED arrows
    reassign_arrows(final, S, P)
    # ---------------------------------------------------------------- contact tips
    fix_contacts(final, P)
    standard_contact_tips(final, S, P)
    if GT_STYLE_TYPES:
        # (after the merges: a piece's type steers them, the answers' rule names the result)
        for c in final:
            if c.key and not c.anon and not getattr(c, 'word_key', False) and c.type:
                c.type = _gt_style_type(c.key, len(c.pins), c.type,
                                        max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) > 3.0 * u)
    # ---------------------------------------------------------------- bbox refinement
    for c in final:
        if c.type in ('led', 'fuse', 'crystal_2pin', 'l', 'switch', 'buzzer', 'battary', 'other'):
            trim_stems(c, S, P)
        refine_bbox(c, P)
        if c.type in ('box', 'switch', 'pin'):
            r = rect_hole(c.gids[0], S, P)
            if r:
                h = P.t * 0.5
                c.bbox = [r[0] - h, r[1] - h, r[2] + h, r[3] + h]
    # ---------------------------------------------------------------- box pins
    for c in final:
        if c.type in ('box', 'switch') or (AMP_PINS and c.type.startswith('amp')):
            assign_box_pins(c, items, P, S, lib)
            if OPTO_PIN_NAMES and c.type == 'box':
                _opto_pin_names(c, S, P)
    # a power port whose name was not read: keep it only with the standard port-bar size
    # (about 1 x 0.6 units); everything else that ends up here is debris
    def _port_sized(c):
        w, h = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
        M, m = max(w, h) / u, min(w, h) / u
        return 0.9 <= M <= 1.25 and 0.45 <= m <= 0.78
    final = [c for c in final if not (c.type == 'v' and not c.value and not _port_sized(c))]
    if DUP_DEBRIS:
        # one designator on two parts, one of them a sliver (a stray stroke that caught a
        # copy of the text): the sliver is debris, and it would turn the real part's key into
        # SW1_2
        bykey = {}
        for c in final:
            if c.key and not c.anon and not (c.type or '').startswith('net') and c.type not in ('gnd', 'v'):
                bykey.setdefault(c.key, []).append(c)
        drop = set()
        for k_, lst in bykey.items():
            if len(lst) < 2:
                continue
            def _sliver(c):
                w, h = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
                return min(w, h) < 0.25 * u and max(w, h) < 1.5 * u and len(c.pins) <= 1
            sl = [c for c in lst if _sliver(c)]
            if sl and len(sl) < len(lst):
                drop.update(id(c) for c in sl)
        # slivers named by a free text (an orphan label such as 'P沟道' beside a MOSFET's
        # body-diode stroke) are no parts either
        for c in final:
            if getattr(c, 'orphan_key', False) and c.type == 'box' and len(c.pins) <= 1 \
                    and min(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) < 0.4 * u:
                drop.add(id(c))
        # specks that caught a designator (a few pixels of a stroke beside a net name)
        for c in final:
            if c.key and not c.anon and not (c.type or '').startswith('net') and c.type not in ('gnd', 'v') \
                    and len(c.pins) <= 1 and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) < 0.3 * u:
                drop.add(id(c))
        if drop:
            final = [c for c in final if id(c) not in drop]
    if NC_SNAP:
        _snap_tips_to_noconnect(final, P)
    if EQUAL_PINS:
        _equalize_pin_lengths(final, P)
    if POWER_RED_TEXT_TRIM:
        _trim_power_red_text(final, items, P)
    return final


POWER_RED_TEXT_TRIM = os.environ.get('PCB_POWER_RED_TRIM', '1') == '1'


def _trim_power_red_text(final, texts, P):
    """A ground symbol whose name 'GND' is drawn in the symbol colour under its bars: the
    letters are no part of the symbol's box."""
    u = P.u
    reds = [it for it in texts if it.color == 'red' and it.text and sum(ch.isalnum() for ch in it.text) >= 2]
    if not reds:
        return
    for c in final:
        if c.type not in ('gnd', 'v') or len(c.pins) != 1:
            continue
        x0, y0, x1, y1 = c.bbox
        tx, ty = c.pins[0].tip
        for it in reds:
            if it.x1 < x0 or it.x0 > x1 or it.y1 < y0 or it.y0 > y1:
                continue
            if it.h > 1.2 * u or it.w > 3 * u:
                continue
            # the side of the box away from the pin
            if ty <= y0 + 1 and it.cy > (y0 + y1) / 2.0:          # pin on top, text below
                y1 = min(y1, it.y0 - 1)
            elif ty >= y1 - 1 and it.cy < (y0 + y1) / 2.0:        # pin below, text above
                y0 = max(y0, it.y1 + 1)
            elif tx <= x0 + 1 and it.cx > (x0 + x1) / 2.0:
                x1 = min(x1, it.x0 - 1)
            elif tx >= x1 - 1 and it.cx < (x0 + x1) / 2.0:
                x0 = max(x0, it.x1 + 1)
        if (x0, y0, x1, y1) != tuple(c.bbox) and x1 - x0 >= 0.3 * u and y1 - y0 >= 0.3 * u:
            c.bbox = [x0, y0, x1, y1]


EQUAL_PINS = os.environ.get('PCB_EQUAL_PINS', '1') == '1'
DUP_DEBRIS = os.environ.get('PCB_DUP_DEBRIS', '1') == '1'


def _equalize_pin_lengths(final, P):
    """A wire laid over the last few pixels of a pin hides its end (green is drawn over
    red), so the visible stub is short.  Pins on one side of a box share one length: a
    shorter one whose missing part is covered by wire is extended to that length."""
    u, t = P.u, P.t
    H, W = P.green.shape
    for c in final:
        if c.type not in ('box', 'switch') or len(c.pins) < 3:
            continue
        sides = {}
        for pn in c.pins:
            if pn.o not in ('h', 'v') or pn.kind not in ('wire', 'free', 'dot'):
                continue
            k = 0 if pn.o == 'h' else 1
            L = pn.tip[k] - pn.base[k]
            if abs(L) < 0.3 * u or abs(pn.tip[1 - k] - pn.base[1 - k]) > 1.5:
                continue
            sides.setdefault((pn.o, 1 if L > 0 else -1, round(pn.base[k] / 2.0)), []).append((pn, abs(L)))
        for (o, sg, _), lst in sides.items():
            if len(lst) < 3:
                continue
            Ls = np.array([L for _, L in lst])
            med = float(np.median(Ls))
            if np.sum(np.abs(Ls - med) <= 1.5) < max(2, 0.6 * len(lst)):
                continue
            k = 0 if o == 'h' else 1
            for pn, L in lst:
                miss = med - L
                if miss < 3 or miss > 0.6 * u:
                    continue
                # the missing part must be wire-covered
                a = pn.tip[k]
                b = pn.base[k] + sg * med
                lo, hi = int(round(min(a, b))), int(round(max(a, b)))
                c0 = int(round(pn.tip[1 - k]))
                r = max(1, int(np.ceil(t / 2)))
                # wire green, or the dark blend of the wire's end cap over the pin
                ink = (P.green > 0) | np.isin(P.M['lab'], (2, 4))
                if o == 'h':
                    band = ink[max(c0 - r, 0):min(c0 + r + 1, H), max(lo, 0):min(hi + 1, W)]
                    cov = band.any(0).mean() if band.size else 0
                    ex, ey = int(round(pn.base[0] + sg * med)), c0
                else:
                    band = ink[max(lo, 0):min(hi + 1, H), max(c0 - r, 0):min(c0 + r + 1, W)]
                    cov = band.any(1).mean() if band.size else 0
                    ex, ey = c0, int(round(pn.base[1] + sg * med))
                R_ = int(np.ceil(t + 1))
                wire_end = P.green[max(ey - R_, 0):ey + R_ + 1, max(ex - R_, 0):ex + R_ + 1].any()
                if cov >= 0.8 and wire_end:
                    tip = list(pn.tip)
                    tip[k] = pn.base[k] + sg * med
                    pn.tip = tuple(tip)


NC_SNAP = os.environ.get('PCB_NC_SNAP', '1') == '1'
NAME_WORDS = os.environ.get('PCB_NAME_WORDS', '1') == '1'
POWER_STEM = os.environ.get('PCB_POWER_STEM', '1') == '1'
FALLBACK_LABELS = os.environ.get('PCB_FB_LABELS', '0') == '1'
GND_NAMES_ONLY = os.environ.get('PCB_GND_NAMES', '1') == '1'   # a stacked-bar symbol is not named VBUS
FAR_POWER_NAME = os.environ.get('PCB_FAR_PWR', '1') == '1'
CARVED_GND_NAME = os.environ.get('PCB_CARVED_GND_NAME', '1') == '1'
FAR_PWR_SPAN = os.environ.get('PCB_FAR_PWR_SPAN', '1') == '1'
NAME_NO_DIGITS = os.environ.get('PCB_NAME_NO_DIGITS', '1') == '1'
DEDUP_PINS = os.environ.get('PCB_DEDUP_PINS', '1') == '1'
NUM_SIDE_PRIOR = os.environ.get('PCB_NUM_SIDE', '1') == '1'
NUM_ABOVE_BIAS = float(os.environ.get('PCB_NUM_ABOVE', '0.0'))
DESIG_PREFIX_PEN = os.environ.get('PCB_DESIG_PREFIX_PEN', '1') == '1'
GATE_RELAX = os.environ.get('PCB_GATE_RELAX', '1') == '1'
Q_FROM_O = os.environ.get('PCB_Q_FROM_O', '1') == '1'
ROT_VOTE = os.environ.get('PCB_ROT_VOTE', '1') == '1'
INSIDE_MODEL_OK = os.environ.get('PCB_INSIDE_MODEL', '1') == '1'
NAME_GLOBAL = os.environ.get('PCB_NAME_GLOBAL', '1') == '1'


def _snap_tips_to_noconnect(final, P):
    """An unconnected pin ends at the centre of its green 'X'; the X is drawn over the pin
    end, so the visible red line stops short by up to half the X.  The IC pin threshold is
    5 px: move such tips onto the X centre."""
    u, t = P.u, P.t
    xs = list(getattr(P, 'noconn', [])) + [(d[0], d[1]) for d in getattr(P, 'contact_dots', [])]
    if not xs:
        return
    for c in final:
        for pn in c.pins:
            tx, ty = pn.tip
            dx, dy = pn.dir if getattr(pn, 'dir', None) else (0.0, 0.0)
            best = None
            for (cx, cy) in xs:
                d = math.hypot(cx - tx, cy - ty)
                if d > 0.6 * u or d < 0.5:
                    continue
                if pn.o == 'h' and abs(cy - ty) > t + 1:
                    continue
                if pn.o == 'v' and abs(cx - tx) > t + 1:
                    continue
                if (dx or dy) and (cx - tx) * dx + (cy - ty) * dy < -1:      # must lie beyond the tip
                    continue
                if best is None or d < best[0]:
                    best = (d, cx, cy)
            # a wire ends exactly at the pin end; a junction / contact dot drawn over the end
            # hides the last pixels of the red pin line
            tm = getattr(pn, 'term', None)
            if best is None and tm is not None and tm.get('pt') is not None:
                cx, cy = tm['pt']
                d = math.hypot(cx - tx, cy - ty)
                if 2.5 < d <= 0.6 * u and ((pn.o == 'h' and abs(cy - ty) <= t + 1) or (pn.o == 'v' and abs(cx - tx) <= t + 1)):
                    if not (dx or dy) or (cx - tx) * dx + (cy - ty) * dy >= -1:
                        best = (d, cx, cy)
            if best is not None:
                pn.tip = (best[1], best[2])


LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'


FUNC_DESIG = os.environ.get('PCB_FUNC_DESIG', '1') == '1'
HIDDEN_DESIG = os.environ.get('PCB_HIDDEN_DESIG', '1') == '1'


WORD_REDECODE = os.environ.get('PCB_WORD_REDECODE', '1') == '1'


WORD_NO_RES = os.environ.get('PCB_WORD_NO_RES', '1') == '1'


def _resistor_box(c, S, P):
    """A two-pin part whose body is an empty elongated box: a resistor, always labelled R...
    (a word beside it is a net name)."""
    try:
        r = rect_hole(c.gids[0], S, P)
    except Exception:
        return False
    if not r:
        return False
    w, h = r[2] - r[0], r[3] - r[1]
    return max(w, h) >= 1.8 * max(min(w, h), 1) and max(w, h) <= 3.5 * P.u


def _word_as_desig(it, c):
    """A short navy word naming an IC-like part (four pins or more) that is a designator
    with its number misread ('UE' for 'U6'): letters then digits, re-decoded from the
    recogniser's output."""
    t = it.text or ''
    if it.probs is None or it.vert or len(t) != 2 or len(c.pins) < 4 or re.search(r'\d', t):
        return None
    try:
        from .ocr import get_rec
        rec = get_rec()
    except Exception:
        return None
    pr = it.probs
    am = pr.argmax(-1)
    runs = []                                   # (char index, frames) of the emitted characters
    prev = -1
    for f_, j in enumerate(am):
        if j != 0 and j != prev:
            runs.append([j, [f_]])
        elif j != 0 and runs:
            runs[-1][1].append(f_)
        prev = j
    if len(runs) != len(t):
        return None
    di = np.array([i for i, ch in enumerate(rec.chars) if ch in DIGITS and i > 0])
    best = None
    for k in range(1, len(t)):
        if not t[:k].isalpha() or not t[:k].isupper():
            continue
        tail, cf = '', 1.0
        for j, fr in runs[k:]:
            sub = pr[fr][:, di].max(0)
            q = int(np.argmax(sub))
            tail += rec.chars[di[q]]
            cf = min(cf, float(sub[q]))
        d = t[:k] + tail
        if is_desig(d) and prefix_type(d) == 'box' and (best is None or cf > best[1]):
            best = (d, cf)
    return best[0] if best and best[1] >= 0.005 else None


XTAL_LED = os.environ.get('PCB_XTAL_LED', '1') == '1'
GT_STYLE_TYPES = os.environ.get('PCB_GT_STYLE_TYPES', '1') == '1'
# letter prefixes the answers type by name; any other prefix is typed by its pin count
_GT_EXACT = {'D', 'ZD', 'TVS', 'VD', 'R', 'RN', 'RP', 'RV', 'RT', 'VR', 'NTC', 'PTC', 'C', 'CE', 'EC', 'TC',
             'L', 'FB', 'LB', 'Q', 'VT', 'T', 'TR', 'TF', 'Y', 'X', 'XTAL', 'OSC', 'G', 'F', 'FU', 'SW', 'S',
             'K', 'KEY', 'BTN', 'BZ', 'BUZ', 'B', 'LS', 'SPK', 'BT', 'BAT', 'TP', 'H', 'J', 'P', 'CN', 'M',
             'MOT', 'E', 'ANT', 'U', 'IC', 'OP', 'A', 'N', 'USB', 'JP', 'CON', 'MK', 'MIC', 'SIM', 'DC',
             'DCIN', 'DCJ'}


def _gt_style_type(key, npins, pt, big=False):
    """Align the type with the answers' rules where a prefix match alone misleads: an
    unknown prefix (RST1, CGQ1, DIP1, XL1, FH1, BUZZER1) is a resistor / capacitor only with
    two pins, an LED by its 'LED' start, otherwise a box (3+ pins) or 'other'; a T part is a
    transformer only with four pins or more; microphones are boxes."""
    m = re.match(r'[A-Za-z]+', key or '')
    p = m.group(0).upper() if m else ''
    if not p or pt in ('mosfet', 'bjt', 'dc', 'm3螺丝', 'net_short') or pt.startswith('net') or pt.startswith('amp'):
        return pt
    if p in _GT_EXACT:
        if p in ('T', 'TR', 'TF') and pt == 'transfomer' and npins < 4:
            return 'box' if npins >= 3 else 'other'
        if p in ('MK', 'MIC') and pt == 'other':
            return 'box'
        return pt
    if p.startswith('LED'):
        return 'led' if pt in ('led', 'd') or npins <= 2 else pt
    if p.startswith('R') and npins == 2 and p not in ('RLY', 'RJ'):
        return 'r'
    if p.startswith('C') and npins == 2 and p not in ('CN', 'CON', 'CH'):
        return 'c'
    if pt == 'pin' or (pt in ('r', 'c') and p[:1] in 'RC') or (pt == 'box' and (npins >= 3 or big)):
        return pt                           # (pin counts read off a large drawing can fall short)
    return 'box' if npins >= 3 else 'other'


def _word_key_type(key, npins):
    """Type of a part named by a word ('MOTOR_A', 'DBUS', 'red-LED', 'BAT3.3-4.2V'): the
    whole leading letter run decides, as in the answers' generator; an unknown word is a
    box with three pins or more, 'other' below that."""
    m = re.match(r'[A-Za-z]+', key or '')
    p = m.group(0).upper() if m else ''
    if p.startswith('LED'):
        return 'led'
    if p in ('D', 'ZD', 'TVS', 'VD'):
        return 'd'
    if p in ('R', 'RN', 'RP', 'RV', 'RT', 'VR', 'NTC', 'PTC') or (p.startswith('R') and npins == 2 and p not in ('RLY', 'RJ')):
        return 'r'
    if p in ('C', 'CE', 'EC', 'TC') or (p.startswith('C') and npins == 2 and p not in ('CN', 'CON', 'CH')):
        return 'c'
    if p in ('L', 'FB', 'LB'):
        return 'l'
    if p in ('Y', 'X', 'XTAL', 'OSC', 'G'):
        return {3: 'crystal_3pin', 4: 'crystal_4pin'}.get(npins, 'crystal_2pin')
    if p in ('F', 'FU'):
        return 'fuse'
    if p in ('SW', 'S', 'K', 'KEY', 'BTN'):
        return 'switch'
    if p in ('BZ', 'BUZ', 'B', 'LS', 'SPK'):
        return 'buzzer'
    if p in ('BT', 'BAT'):
        return 'battary'
    if p in ('M', 'MOT'):
        return 'motor'
    if p in ('U', 'IC', 'J', 'P', 'H', 'CN', 'USB', 'JP', 'CON', 'A', 'MK', 'MIC', 'SIM'):
        return 'box'
    return 'box' if npins >= 3 else 'other'


def _functional_designators(items):
    """Some sheets name parts by function: RB1..RB6, RBB1..RBB6, RCO1..RCO3, CB1..CB6, then
    one-offs like RDOT1, RESENSE1, CIT1.  Multi-letter prefixes are normally part numbers
    (penalised as designators), but when several such prefixes on the sheet count up as
    series, R / C / L words of that shape are designators.  Returns the ids of those items
    (texts corrected to their series: 'RC01' among RCO2, RCO3 is RCO1)."""
    pat = re.compile(r'([A-Z]{2,7})(\d{1,2})')
    stems = {}
    for it in items:
        m = pat.fullmatch(it.text or '')
        if m:
            stems.setdefault(m.group(1), set()).add(m.group(2))
    series = {st for st, ns in stems.items() if len(ns) >= 2 and st[0] in 'RC'
              and st not in _EXACT_PREFIXES and st not in _KNOWN_LONG_PREFIX}
    if len(series) < 3:
        return set()
    ids = set()
    for it in items:
        t = it.text or ''
        m = re.fullmatch(r'([A-Z]{1,6})0(\d)', t)
        if m and m.group(1) + 'O' in series and m.group(1) not in series:
            t = m.group(1) + 'O' + m.group(2)         # a letter O read as zero
        m = pat.fullmatch(t)
        if m and m.group(1)[0] in 'RCL' and len(m.group(1)) >= 2:
            it.text = t
            ids.add(it.id)
    return ids


def _fix_desig(it):
    s = it.text.replace(' ', '')
    if is_desig(s) and _desig_text(s) == s:
        it.text = s
        return True
    if _desig_text(s) != s:
        it.text = _desig_text(s)               # a body corner read as a trailing dot ('R5.')
        return True
    # constrained re-decode
    s2, conf = redecode(it, UPPER_DIGITS)
    if is_desig(s2) and conf > 0.5 and not (DESIG_KEEP_PUNCT and len(s2) < len(s) and it.conf >= 0.9
                                            and re.search(r'[-./]', s)):
        # (a confident 'XH-W-3A' is a part name: re-read without its hyphens it would pass
        # for a designator 'XHW3A')
        it.text = s2
        return True
    # designator-shaped text (letters then digits) misread: decode letters/digits split
    if it.probs is not None and 2 <= len(s) <= 6 and s[:1].isalpha() and not it.vert or (it.probs is not None and it.vert and 2 <= len(s) <= 6):
        from .ocr import get_rec
        rec = get_rec()
        best = None
        for k in ((1, 2, 3, 4) if DESIG_SPLIT4 else (1, 2, 3)):
            t, c = rec.decode_split(it.probs, LETTERS, DIGITS, k)
            if is_desig(t) and len(t) == len(s) and (best is None or c > best[1]):
                best = (t, c)
        if best and best[1] > 0.4 and best[0][:1] == s[:1].upper() and not (SURE_CHARS and _sure_change(it, s, best[0])):
            it.text = best[0]
            return True
    return False


SURE_CHARS = os.environ.get('PCB_SURE_CHARS', '1') == '1'
DESIG_TRAIL_PUNCT = os.environ.get('PCB_DESIG_TRAIL_PUNCT', '1') == '1'
DESIG_SPLIT4 = os.environ.get('PCB_DESIG_SPLIT4', '1') == '1'   # four-letter prefixes: USBC1
DESIG_KEEP_PUNCT = os.environ.get('PCB_DESIG_KEEP_PUNCT', '1') == '1'


def _sure_change(it, s, t):
    """True when the letter/digit split decode would overwrite a character the recogniser
    read with near certainty (CJ3401 -> CJB401: a part number, not a misread designator)."""
    if it.probs is None or len(s) != len(t):
        return False
    import numpy as np
    pr = np.asarray(it.probs)
    am = pr.argmax(1)
    # greedy CTC alignment: one probability per emitted character
    cp, prev = [], 0
    for f in range(len(am)):
        if am[f] != 0 and am[f] != prev:
            cp.append(float(pr[f, am[f]]))
        prev = am[f]
    if len(cp) != len(s):
        return False
    return any(a.upper() != b and p >= 0.95 for a, b, p in zip(s, t, cp))


_KFIX = [None]


def _fix_comma_only(it):
    from .textfix import fix_comma
    return fix_comma(it)


_RATING_SPACE = [re.compile(r'^(\d+(?:\.\d+)?[pnuµμm]?F)(\d+(?:\.\d+)?V)$'),       # 470uF25V -> 470uF 25V
                 re.compile(r'^(\d+(?:\.\d+)?[kKMmR]?Ω?)(\d+(?:\.\d+)?%)$'),        # 10K1%    -> 10K 1%
                 re.compile(r'^(\d+(?:\.\d+)?[pnuµμm]?[FH])(\d+(?:\.\d+)?%)$')]


def _fix_value(it, ctype=None):
    from .textfix import fix_comma, repair_value, fix_trailing_ohm
    t = fix_comma(it).replace(' ', '')
    if ctype == 'r':
        t = fix_trailing_ohm(it, t)
    t = re.sub(r'^(\d+(?:\.\d+)?)rF$', r'\1nF', t)          # 'n' read as 'r'
    if ctype == 'c' and CAP_UNIT_FIX:
        t = _cap_unit_fix(it, t)
    if ctype == 'c':
        from .textfix import fix_trailing_u
        t = fix_trailing_u(it, t)
    t = repair_value(t)
    if _KFIX[0] is not None:
        t = _KFIX[0](it, t)
    for rx in _RATING_SPACE:                                   # value + rating keep their space
        m = rx.match(t)
        if m:
            t = m.group(1) + ' ' + m.group(2)
            break
    return t


CAP_UNIT_FIX = os.environ.get('PCB_CAP_UNIT', '1') == '1'
_CAP_CTX = [None]


def _cap_unit_fix(it, t):
    """'22uF' read as '22F' / '224F', '10UF' as '100F': a capacitor value in whole farads is
    implausible.  The digit just before F is the lost prefix when the recogniser gave a
    unit letter a fair chance there; otherwise the prefix was dropped and is 'u'."""
    m = re.fullmatch(r'(\d+(?:\.\d+)?)F', t)
    if not m:
        return t
    num = m.group(1)
    try:
        if float(num) < 2:
            return t                       # 1F / 1.5F supercapacitors exist
    except ValueError:
        return t
    # supercapacitor banks print whole farads ('60F' on seven parts): only a lone farad
    # value among micro/nano ones is a misread
    ctx = _CAP_CTX[0] or {}
    if not ctx.get('prefixed') or ctx.get('counts', {}).get(t, 0) > 1:
        return t
    probs = getattr(it, 'probs', None)
    if probs is not None and len(num) >= 2 and num[-1] in '04' and it.text.replace(' ', '') == t:
        from .ocr import get_rec
        R = get_rec()
        txt, _, _, pos = R.decode(probs)
        if txt.replace(' ', '') == t and len(pos) == len(txt):
            k = len(txt) - 2                # the character before 'F'
            ids = [i for i, ch in enumerate(R.chars) if ch in ('u', 'U', 'μ', 'µ')]
            fr = probs[pos[k]]
            j = max(ids, key=lambda i: fr[i])
            if fr[j] >= 0.01:
                ch = R.chars[j]
                return num[:-1] + ('u' if ch in 'μµ' else ch) + 'F'
    return num + 'uF'


def _unit_pen(text, ctype):
    """A capacitance cannot be a resistor's value and so on."""
    t = text.replace(' ', '')
    if re.search(r'\d(?:[pnuµμm])?F', t):
        return 0.0 if ctype == 'c' else 3.0
    if re.search(r'\d(?:[pnuµμm])?H$', t):
        return 0.0 if ctype == 'l' else 3.0
    if 'Ω' in t:
        return 0.0 if ctype == 'r' else 3.0
    return 0.0


def _power_shape(c, S, P):
    """Classify a text-less one-pin symbol: 'gnd' (stacked bars), 'v' (single bar) or None."""
    if len(c.pins) != 1:
        return None
    pn = c.pins[0]
    f = c.feat
    x0, y0, x1, y1 = c.bbox
    rs = [s for s in S['rsegs'] if box_gap(c.bbox, [s.p0()[0], s.p0()[1], s.p1()[0], s.p1()[1]]) <= 1]
    # power symbols are drawn with axis-aligned strokes only; LED arrows and other
    # diagonal debris are not power symbols
    X0, Y0, X1, Y1 = int(x0), int(y0), int(x1), int(y1)
    ink = P.red[Y0:Y1 + 1, X0:X1 + 1] > 0
    if ink.sum() > 0:
        cov = np.zeros_like(ink)
        for s in rs:
            if s.length < 0.35 * P.u and s.t > 1.6 * P.t:   # arrow heads: short THICK stubs
                continue
            hw = int(np.ceil(s.t / 2 + 1))
            if s.o == 'h':
                cov[max(int(s.c) - hw - Y0, 0):int(s.c) + hw + 1 - Y0, max(int(s.a) - X0 - 1, 0):int(s.b) - X0 + 2] = True
            else:
                cov[max(int(s.a) - Y0 - 1, 0):int(s.b) - Y0 + 2, max(int(s.c) - hw - X0, 0):int(s.c) + hw + 1 - X0] = True
        if (ink & ~cov).sum() > 0.35 * ink.sum():
            return None
        diag_frac = (ink & ~cov).sum() / max(ink.sum(), 1)
    else:
        diag_frac = 1.0
    tx, ty = pn.tip
    vert = pn.o == 'v' or (pn.o == 'z' and (ty < y0 or ty > y1))
    r = _bar_rows(rs, vert, tx, ty, P)

    def stack_ok(rows, along):
        # ground bars are centred on the stem and stacked close together (an inductor's
        # humps are a row of short arcs a unit apart)
        return (rows is not None and len(rows) >= 2
                and all(abs((b[1] + b[2]) / 2.0 - along) <= 0.3 * P.u for b in rows)
                and all(b2[0] - b1[0] <= 0.6 * P.u for b1, b2 in zip(rows, rows[1:]))
                and max(b[2] - b[1] for b in rows) >= 0.4 * P.u)
    if GND_BOTH_AXES:
        # lines of a neighbour that run out of the symbol's box (the wire or pin it hangs on)
        # are no bars: a '+5V' bar above a wire passing its stem foot is no ground
        m_ = 2 * P.t
        rs2 = [s for s in rs if (s.o == 'h' and x0 - m_ <= s.a and s.b <= x1 + m_)
               or (s.o == 'v' and y0 - m_ <= s.a and s.b <= y1 + m_)]
        rf = _bar_rows(rs2, vert, tx, ty, P)
        along_ = tx if vert else ty
        if rf and (stack_ok(rf, along_) if len(rf) > 1 else abs((rf[0][1] + rf[0][2]) / 2.0 - along_) <= 0.3 * P.u) \
                and len(rs2) < len(rs) and (r or diag_frac <= 0.1):
            # (a piece not taken for power before must be drawn in straight strokes only:
            # a transistor's body-diode piece has its triangle)
            r = rf
    if r is None and GND_BOTH_AXES:
        # retry without those lines, and with the bars across the other axis (the stem bends
        # before the pin end); only a compact centred stack of bars counts then
        r1 = _bar_rows(rs2, vert, tx, ty, P)
        if r1 is not None and len(r1) >= 3 and stack_ok(r1, tx if vert else ty):
            r = r1                           # (a full stack: a '+5V' bar plus a stray stroke is no ground)
        else:
            r2 = _bar_rows(rs2, not vert, tx, ty, P)
            if r2 is not None and len(r2) >= 3 and stack_ok(r2, ty if vert else tx):
                r = r2
    rows = r or []
    if len(rows) >= 2:
        lens = [r[2] - r[1] for r in rows]
        if all(a >= b - P.t for a, b in zip(lens, lens[1:])) and lens[0] > lens[-1] + P.t:
            return 'gnd'
        return None
    if len(rows) == 1 and f['w'] * f['h'] < 2.5 * P.u * P.u:
        if BAR_MIN_LEN and (rows[0][2] - rows[0][1] < 0.45 * P.u or max(f['w'], f['h']) < 0.45 * P.u):
            return None                     # a speck at a lead's end with a stub across it is no supply bar
        return 'v'
    return None


BAR_MIN_LEN = os.environ.get('PCB_BAR_MIN_LEN', '1') == '1'
PWR_VOLT_NAME = os.environ.get('PCB_PWR_VOLT_NAME', '1') == '1'
FLAG_HUNGARIAN = os.environ.get('PCB_FLAG_HUNGARIAN', '1') == '1'


GND_BOTH_AXES = os.environ.get('PCB_GND_BOTH_AXES', '1') == '1'


def _bar_rows(rs, vert, tx, ty, P):
    """Bars across the stem axis grouped into rows by distance from the pin end."""
    if vert:
        bars = [s for s in rs if s.o == 'h' and s.length >= 0.25 * P.u]
        dist = lambda s: abs(s.c - ty)
    else:
        bars = [s for s in rs if s.o == 'v' and s.length >= 0.25 * P.u]
        dist = lambda s: abs(s.c - tx)
    rows = []
    for s in sorted(bars, key=dist):
        if rows and abs(dist(s) - rows[-1][0]) <= P.t + 1:
            rows[-1][1] = min(rows[-1][1], s.a)
            rows[-1][2] = max(rows[-1][2], s.b)
        else:
            rows.append([dist(s), s.a, s.b])
    if len(rows) >= 2:
        lens = [r[2] - r[1] for r in rows]
        if all(a >= b - P.t for a, b in zip(lens, lens[1:])) and lens[0] > lens[-1] + P.t:
            return rows
        return None
    return rows


def _looks_gnd(c, S, P):
    return _power_shape(c, S, P) == 'gnd'


def _has_arrows(c, S, P):
    x0, y0, x1, y1 = c.bbox
    sub = (S['glab'][y0:y1 + 1, x0:x1 + 1] > 0)
    # arrows make the bbox clearly non-square relative to the triangle
    f = c.feat
    pins = c.pins
    if len(pins) >= 2:
        o = pins[0].o
        across = f['w'] if o == 'v' else f['h']
        along = f['h'] if o == 'v' else f['w']
        if not across > 1.35 * along:
            return False
        return _arrow_pair(c, S, P) if LED_ARROW_PAIR else True
    return False


LED_ARROW_PAIR = os.environ.get('PCB_LED_ARROWS', '1') == '1'
ARROWS_NOT_GND = os.environ.get('PCB_ARROWS_NOT_GND', '1') == '1'
ARROW_NO_BARS = os.environ.get('PCB_ARROW_NO_BARS', '1') == '1'
PWR_DESIG_LIKE = os.environ.get('PCB_PWR_DESIG_LIKE', '1') == '1'


ARROW_FILLED = os.environ.get('PCB_ARROW_FILLED', '1') == '1'


def _arrow_pair_raw(m, u):
    """Two light arrows with filled heads: taking out straight strokes eats the heads, but
    the shafts are long parallel slanted strokes side by side (ground bars never make two
    parallel slanted lines half a unit apart)."""
    L = cv2.HoughLinesP(m.astype(np.uint8) * 255, 1, np.pi / 180, threshold=max(5, int(0.3 * u)),
                        minLineLength=max(6, int(0.6 * u)), maxLineGap=2)
    if L is None:
        return False
    segs = []
    for x1, y1, x2, y2 in np.asarray(L).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 30 < a < 60 or 120 < a < 150:
            segs.append((a, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a1, xa, ya = segs[i]
            a2, xb, yb = segs[j]
            if abs(a1 - a2) > 6:
                continue
            t = np.radians(a1)
            d = np.array([xb - xa, yb - ya])
            perp = abs(-np.sin(t) * d[0] + np.cos(t) * d[1])
            par = abs(np.cos(t) * d[0] + np.sin(t) * d[1])
            if 0.3 * u <= perp <= 0.9 * u and par <= 0.8 * u:
                return True
    return False


def _arrow_pair(c, S, P):
    """LED / photodiode: two short parallel diagonal arrows side by side.  A TVS bar with
    bent ends or a dual diode is also wider than tall but has no such pair."""
    m = _body_mask(c, S)
    if m.sum() < 10:
        return False
    u = P.u
    if ARROW_FILLED and _arrow_pair_raw(m, u):
        return True
    if ARROW_NO_BARS:
        # the ends of a ground symbol's stacked bars line up along two slanted edges; only
        # ink left after taking out horizontal and vertical strokes can be an arrow
        k = max(3, int(round(0.3 * u)))
        m = m.astype(np.uint8)
        hv = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, k), np.uint8)) | \
            cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
        m = (m & (1 - hv)).astype(np.uint8)
        if m.sum() < 10:
            return False
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 180, threshold=max(5, int(0.25 * u)),
                        minLineLength=max(5, int(0.45 * u)), maxLineGap=2)
    if L is None:
        return False
    segs = []
    for x1, y1, x2, y2 in np.asarray(L).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 20 < a < 70 or 110 < a < 160:
            segs.append((a, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a1, xa, ya = segs[i]
            a2, xb, yb = segs[j]
            if abs(a1 - a2) > 8:
                continue
            t = np.radians(a1)
            d = np.array([xb - xa, yb - ya])
            perp = abs(-np.sin(t) * d[0] + np.cos(t) * d[1])
            par = abs(np.cos(t) * d[0] + np.sin(t) * d[1])
            if 0.15 * u <= perp <= 0.9 * u and par <= 0.8 * u:
                return True
    return False


# ------------------------------------------------------------------ box pin numbering
COVERED_PINS = os.environ.get('PCB_COVERED_PINS', '1') == '1'


def _covered_pin(c, pn, P):
    """A wire drawn along a pin hides the pin under it (green over red) and ends at the
    body edge.  Rebuild the pin: from the edge outwards along the wire, as long as the
    other pins on that side."""
    u, t = P.u, P.t
    x0, y0, x1, y1 = c.bbox
    tx, ty = pn.tip
    best = None
    for s_ in P.wires:
        if s_.o == 'h' and abs(s_.c - ty) <= t + 1 and s_.a - t <= tx <= s_.b + t:
            for side, edge, far in (('l', x0, s_.a), ('r', x1, s_.b)):
                if abs(tx - edge) <= 1.5 * t + 2 or (x0 <= tx <= x1 and min(abs(tx - x0), abs(tx - x1)) <= 0.6 * u):
                    out = edge - far if side == 'l' else far - edge
                    if out > 0.3 * u and (best is None or out > best[0]):
                        best = (out, 'h', side, edge)
        elif s_.o == 'v' and abs(s_.c - tx) <= t + 1 and s_.a - t <= ty <= s_.b + t:
            for side, edge, far in (('t', y0, s_.a), ('b', y1, s_.b)):
                if abs(ty - edge) <= 1.5 * t + 2 or (y0 <= ty <= y1 and min(abs(ty - y0), abs(ty - y1)) <= 0.6 * u):
                    out = edge - far if side == 't' else far - edge
                    if out > 0.3 * u and (best is None or out > best[0]):
                        best = (out, 'v', side, edge)
    if best is None:
        return False
    out, o, side, edge = best
    # the length of the visible pins on the same side
    Ls = []
    for q in c.pins:
        if q is pn or q.o != o:
            continue
        if o == 'h' and ((side == 'l' and q.tip[0] < x0) or (side == 'r' and q.tip[0] > x1)):
            Ls.append(abs(q.tip[0] - q.base[0]))
        if o == 'v' and ((side == 't' and q.tip[1] < y0) or (side == 'b' and q.tip[1] > y1)):
            Ls.append(abs(q.tip[1] - q.base[1]))
    L = float(np.median(Ls)) if len(Ls) >= 2 else u
    if out < L - 1.5 * t:
        return False                       # the wire is shorter than a pin: something else
    sg = -1 if side in ('l', 't') else 1
    if o == 'h':
        pn.o, pn.base, pn.tip = 'h', (edge, ty), (edge + sg * L, ty)
    else:
        pn.o, pn.base, pn.tip = 'v', (tx, edge), (tx, edge + sg * L)
    return True


WIRE_THROUGH_PINS = os.environ.get('PCB_WIRE_THROUGH_PINS', '1') == '1'


def _wire_through_pins(c, P):
    """A wire drawn straight across a connector from one pin end to the opposite pin end
    (two pins of a row shorted on the sheet) covers both pins.  When it runs along the pin
    pitch and ends one pin length outside each edge, the two pins are put back."""
    from .sym import Pin
    u, t = P.u, P.t
    x0, y0, x1, y1 = c.bbox
    if len(c.pins) < 6:
        return
    for o in ('h', 'v'):
        side_a = [q for q in c.pins if q.o == o and (q.tip[0] < x0 if o == 'h' else q.tip[1] < y0)]
        side_b = [q for q in c.pins if q.o == o and (q.tip[0] > x1 if o == 'h' else q.tip[1] > y1)]
        if len(side_a) < 3 or len(side_b) < 3:
            continue
        k = 0 if o == 'h' else 1
        La = float(np.median([abs(q.tip[k] - q.base[k]) for q in side_a]))
        Lb = float(np.median([abs(q.tip[k] - q.base[k]) for q in side_b]))
        lo, hi = (x0, x1) if o == 'h' else (y0, y1)
        plo, phi = (y0, y1) if o == 'h' else (x0, x1)
        for s_ in P.wires:
            if s_.o != o or not (plo < s_.c < phi):
                continue
            ta, tb = lo - La, hi + Lb
            if s_.a > ta + 0.3 * u or s_.b < tb - 0.3 * u:
                continue                     # the wire must reach both pin ends
            ends_ok = abs(s_.a - ta) <= 0.3 * u and abs(s_.b - tb) <= 0.3 * u
            if not ends_ok:
                # running on past a pin end: a junction dot must mark each pin end
                def _dot_at(v):
                    pt = (v, s_.c) if o == 'h' else (s_.c, v)
                    return any(np.hypot(x - pt[0], y - pt[1]) <= 0.3 * u for x, y, r in P.dots)
                if not (_dot_at(ta) and _dot_at(tb)):
                    continue
            if any(abs(q.tip[1 - k] - s_.c) <= t + 1 for q in side_a + side_b):
                continue                     # a pin is already there
            for tipv, basev in ((lo - La, lo), (hi + Lb, hi)):
                tip = (tipv, s_.c) if o == 'h' else (s_.c, tipv)
                base = (basev, s_.c) if o == 'h' else (s_.c, basev)
                pn = Pin(tip, base, o, 'dot')
                c.pins.append(pn)


OPTO_PIN_NAMES = os.environ.get('PCB_OPTO_PIN_NAMES', '1') == '1'


def _arrow_ccs(m, u):
    """Two loose light arrows inside a box: separate blobs of arrow size, about square
    (slanted), sparse, alike and side by side."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    H, W = m.shape
    bl = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if not (0.5 * u <= max(w, h) <= 1.2 * u and 0.6 <= w / max(h, 1) <= 1.6 and a < 0.5 * w * h):
            continue
        bl.append((w, h, a, cen[i]))
    for i in range(len(bl)):
        for j in range(i + 1, len(bl)):
            (w1, h1, a1, c1), (w2, h2, a2, c2) = bl[i], bl[j]
            if abs(a1 - a2) > 0.25 * max(a1, a2) or abs(w1 - w2) > 0.25 * max(w1, w2):
                continue
            if 0.3 * u <= float(np.hypot(*(c1 - c2))) <= 1.0 * u:
                return True
    return False


def _opto_pin_names(c, S, P):
    """A 4-pin opto-coupler (LED with light arrows facing a phototransistor in one box) whose
    pin names are not printed: the answers name its pins after the symbol's (hidden) names
    AN, CAT, EM, COL for pins 1-4 (10 of the 12 opto-couplers in the labelled sheets)."""
    if len(c.pins) != 4:
        return
    nums = [str(pn.num) if pn.num is not None else None for pn in c.pins]
    if sorted(n for n in nums if n) != ['1', '2', '3', '4']:
        return
    if any(pn.name is not None and not str(pn.name).isdigit() for pn in c.pins):
        return                              # names printed or known from the library
    x0, y0, x1, y1 = [int(round(v)) for v in c.bbox]
    m = (P.red[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] > 0).astype(np.uint8)
    if m.sum() < 20 or not (_arrow_pair(c, S, P) or _arrow_pair_raw(m, P.u) or _arrow_ccs(m, P.u)):
        return
    c.opto = True                           # pin names given once the part name is known (pipeline)


def assign_box_pins(c, items, P, S, lib):
    u = P.u
    t = P.t
    x0, y0, x1, y1 = c.bbox
    # zero-length pins on a box: infer orientation and base from the body outline
    for pn in c.pins:
        if pn.o == 'z':
            tx, ty = pn.tip
            dx = x0 - tx if tx < x0 else (tx - x1 if tx > x1 else 0)
            dy = y0 - ty if ty < y0 else (ty - y1 if ty > y1 else 0)
            if dx >= dy and dx > 0:
                pn.o = 'h'
                pn.base = (x0 if tx < x0 else x1, ty)
            elif dy > 0:
                pn.o = 'v'
                pn.base = (tx, y0 if ty < y0 else y1)
            elif not (COVERED_PINS and _covered_pin(c, pn, P)):
                continue
            d = np.array(pn.tip, float) - np.array(pn.base, float)
            n = np.linalg.norm(d)
            pn.dir = tuple(d / n) if n > 0 else (0.0, 0.0)
    if WIRE_THROUGH_PINS:
        _wire_through_pins(c, P)
    pool = [it for it in items if it.text and box_gap(tbox(it), c.bbox) <= 1.5 * u]
    pins = [pn for pn in c.pins if pn.o in ('h', 'v')]
    # --- pin names: text inside the body next to the pin base
    amp = c.type.startswith('amp')
    names = [it for it in pool if (x0 - 2 <= it.cx <= x1 + 2) and (y0 - 2 <= it.cy <= y1 + 2)
             and not (it.color == 'red' and len(it.text) <= 2 and not (amp and it.text in ('+', '-')))]
    # (a symbol stroke inside the body read as 'Y' is no pin name; an amplifier's +/- are)
    if ROT_VOTE:
        # rotated pin names of one symbol share one reading direction; a short name that
        # reads well both ways ('SW' / 'MS', 'VIO' / 'OIA') follows the majority
        vn = [it for it in names if it.vert and getattr(it, 'alt', None) is not None and it.rot in ('cw', 'ccw')]
        if len(vn) >= 3:
            sc = {'cw': 0.0, 'ccw': 0.0}
            for it in vn:
                sc[it.rot] += it.conf
                sc[it.alt[3]] += it.alt[1]
            maj = max(sc, key=sc.get)
            n_maj = sum(1 for it in vn if it.rot == maj)
            if n_maj >= 0.7 * len(vn):
                for it in vn:
                    if it.rot != maj and it.alt[3] == maj and it.alt[1] >= 0.5 and it.alt[0]:
                        it.text, it.conf, it.rot = it.alt[0], it.alt[1], maj
    if names and pins:
        C = np.full((len(pins), len(names)), 1e6)
        for i, pn in enumerate(pins):
            bx, by = pn.base
            for j, it in enumerate(names):
                cx, cy = it.cx, it.cy
                if pn.o == 'h':
                    if abs(cy - by) > 0.45 * u:
                        continue
                    edge = it.x0 if bx <= (x0 + x1) / 2 else it.x1
                    d = abs(edge - bx) / u + abs(cy - by) / u
                    if it.vert:
                        d += 1.0
                else:
                    if abs(cx - bx) > 0.45 * u:
                        continue
                    edge = it.y0 if by <= (y0 + y1) / 2 else it.y1
                    d = abs(edge - by) / u + abs(cx - bx) / u
                    if not it.vert:
                        d += 0.5
                if d > 2.5:
                    continue
                C[i, j] = d
        r, cc = linear_sum_assignment(C)
        got = {}
        for i, j in zip(r, cc):
            if C[i, j] < 1e5:
                pins[i].name = names[j].text
                got[j] = i
        if NAME_WORDS and got:
            _join_name_words(pins, names, got, c.bbox)
    if amp and AMP_PINS:
        # the two inputs of an amplifier carry + and -: the minus is a short dash stroke
        # that is not read as text - the input beside a '+' (or '-') is the other one
        for pn in pins:
            if pn.name not in ('+', '-'):
                continue
            mates = [q for q in pins if q is not pn and q.o == pn.o and (q.name is None or str(q.name).isdigit())
                     and abs((q.base[0] - pn.base[0]) if pn.o == 'h' else (q.base[1] - pn.base[1])) <= 2 * t + 2
                     and np.hypot(q.tip[0] - pn.tip[0], q.tip[1] - pn.tip[1]) <= 3 * u]
            if len(mates) == 1:
                mates[0].name = '-' if pn.name == '+' else '+'
    # --- pin numbers: one-to-one assignment of number-like texts to pin stubs
    nums = [it for it in pool if (re.match(r'^[A-Z]{0,2}\d{1,3}[A-Z]?$', it.text)
                                  or (BALL_NUMS and re.match(r'^\d[A-Z]{1,2}\d{1,3}$', it.text))) and it.color != 'blue'
            and not (x0 + 2 < it.cx < x1 - 2 and y0 + 2 < it.cy < y1 - 2)]
    pins = [pn for pn in c.pins if pn.o in ('h', 'v')]
    if nums and pins:
        # group pins by side; for each side try 'number before' / 'number after' the line
        sides = {}
        for pn in pins:
            key = (pn.o, 1 if (pn.dir[0] + pn.dir[1]) > 0 else -1)
            sides.setdefault(key, []).append(pn)
        used_nums = set()
        for key, sp in sides.items():
            best = None
            for hyp in (-1, 1):
                C = np.full((len(sp), len(nums)), 1e6)
                for i, pn in enumerate(sp):
                    tx, ty = pn.tip
                    bx, by = pn.base
                    for j, it in enumerate(nums):
                        if j in used_nums:
                            continue
                        if pn.o == 'h':
                            lo, hi = min(tx, bx), max(tx, bx)
                            along = max(lo - it.cx, 0, it.cx - hi)
                            gap = (by - it.y1) if hyp < 0 else (it.y0 - by)
                        else:
                            lo, hi = min(ty, by), max(ty, by)
                            along = max(lo - it.cy, 0, it.cy - hi)
                            gap = (bx - it.x1) if hyp < 0 else (it.x0 - bx)
                        if gap < -0.2 * u or gap > 0.5 * u or along > 0.5 * u:
                            continue
                        C[i, j] = abs(gap - 0.1 * u) + 0.5 * along
                r, cc = linear_sum_assignment(C)
                pairs = [(i, j) for i, j in zip(r, cc) if C[i, j] < 1e5]
                cost = sum(C[i, j] for i, j in pairs) / u
                score = len(pairs) - 0.1 * cost
                if NUM_SIDE_PRIOR:
                    # connectors print the number again as the name: the hypothesis that
                    # agrees with the names wins; otherwise EasyEDA puts the number above /
                    # left of its pin line
                    score += 0.5 * sum(1 for i, j in pairs if sp[i].name is not None and sp[i].name == nums[j].text)
                    score += NUM_ABOVE_BIAS if hyp < 0 else 0.0
                if best is None or score > best[0]:
                    best = (score, pairs)
            for i, j in best[1]:
                sp[i].num = nums[j].text
                used_nums.add(j)
    if lib is not None:
        lib.correct(c)


def _join_name_words(pins, names, got, bbox):
    """Multi-word pin names ('TMDS Data 2 Shield', 'Hot Plug Detect') are split at the
    spaces into several text items; only the word next to the pin base was assigned.
    Chain the unassigned words of the same row, away from the base, with word-space gaps."""
    x0, y0, x1, y1 = bbox
    taken = set(got)
    for j, i in sorted(got.items(), key=lambda kv: kv[0]):
        pn, a = pins[i], names[j]
        if a.vert != (pn.o == 'v'):
            continue
        chain = [a]
        while True:
            last = chain[-1]
            best = None
            for k, b in enumerate(names):
                if k in taken or b in chain or b.vert != a.vert or b.color != a.color or not b.text:
                    continue
                if not a.vert:
                    hh = min(last.h, b.h)
                    ov = min(last.y1, b.y1) - max(last.y0, b.y0)
                    if ov < 0.5 * hh or abs(b.cy - a.cy) > 0.35 * max(a.h, b.h):
                        continue
                    away_right = pn.base[0] <= (x0 + x1) / 2
                    gap = (b.x0 - last.x1) if away_right else (last.x0 - b.x1)
                    ref = a.h
                else:
                    ww = min(last.w, b.w)
                    ov = min(last.x1, b.x1) - max(last.x0, b.x0)
                    if ov < 0.5 * ww or abs(b.cx - a.cx) > 0.35 * max(a.w, b.w):
                        continue
                    away_down = pn.base[1] <= (y0 + y1) / 2
                    gap = (b.y0 - last.y1) if away_down else (last.y0 - b.y1)
                    ref = a.w
                if gap <= 0 or gap > 1.0 * ref:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, k, b)
            if best is None:
                break
            taken.add(best[1])
            chain.append(best[2])
        if len(chain) > 1:
            # reading order: left-to-right, or bottom-to-top for rotated text
            chain.sort(key=(lambda t: -t.cy) if a.vert else (lambda t: t.cx))
            pn.name = ' '.join(t.text for t in chain)


def trim_stems(c, S, P):
    """Remove leftover straight pin-line portions at both ends of a 2-pin body along its axis."""
    if len(c.pins) != 2:
        return
    a, b = c.pins
    horiz = abs(a.tip[0] - b.tip[0]) > abs(a.tip[1] - b.tip[1])
    x0, y0, x1, y1 = [int(round(v)) for v in c.bbox]
    if x1 <= x0 or y1 <= y0:
        return
    m = np.isin(S['glab'][y0:y1 + 1, x0:x1 + 1], c.gids)
    if m.sum() == 0:
        return
    prof = m.any(0) if False else None
    # perpendicular span of every slice along the axis
    if horiz:
        span = np.array([(np.ptp(np.nonzero(m[:, i])[0]) + 1) if m[:, i].any() else 0 for i in range(m.shape[1])])
    else:
        span = np.array([(np.ptp(np.nonzero(m[i, :])[0]) + 1) if m[i, :].any() else 0 for i in range(m.shape[0])])
    thin = span <= 1.6 * P.t + 1
    lo = 0
    while lo < len(span) and thin[lo]:
        lo += 1
    hi = len(span) - 1
    while hi > lo and thin[hi]:
        hi -= 1
    if hi - lo < 0.3 * P.u:
        return
    if horiz:
        c.bbox[0], c.bbox[2] = x0 + lo, x0 + hi
    else:
        c.bbox[1], c.bbox[3] = y0 + lo, y0 + hi


PIN_LEN_TYPES = {'mosfet': 1.0, 'bjt': 1.0}
PASSIVE_PIN_LEN = os.environ.get('PCB_PASSIVE_PIN_LEN', '1') == '1'
SWITCH_PIN_LEN = os.environ.get('PCB_SWITCH_PIN_LEN', '1') == '1'
TRANS_CLAMP = os.environ.get('PCB_TRANS_CLAMP', '1') == '1'     # transistor boxes 1u inside lead tips
LEAD_PIECES = os.environ.get('PCB_LEAD_PIECES', '1') == '1'


def _absorb_lead_pieces(comps, P):
    """A part's lead cut in two (a glyph of a nearby text touching it): the outer piece is
    a bare straight line from the part's contact to the wire.  It becomes part of that pin."""
    t, u = P.t, P.u
    typed = [c for c in comps.values() if c.type in ('r', 'c', 'l', 'd', 'led', 'fuse') and len(c.pins) == 2]
    drop = []
    for k, c in comps.items():
        if c.type is not None or c.key is not None or len(c.pins) != 2:
            continue
        w, h = c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]
        if min(w, h) > 1.5 * t + 1 or max(w, h) > 3 * u:
            continue
        a, b = c.pins
        for inner, outer in ((a, b), (b, a)):
            if inner.kind != 'contact' or outer.kind == 'contact':
                continue
            host = None
            for o in typed:
                for pn in o.pins:
                    if pn.kind == 'contact' and abs(pn.tip[0] - inner.tip[0]) <= 2 * t + 2 and abs(pn.tip[1] - inner.tip[1]) <= 2 * t + 2:
                        host = (o, pn)
                        break
                if host:
                    break
            if host is None:
                continue
            o, pn = host
            # the piece must continue the pin's line
            if pn.o == 'h' and abs(outer.tip[1] - pn.tip[1]) > t + 1 or pn.o == 'v' and abs(outer.tip[0] - pn.tip[0]) > t + 1:
                continue
            pn.tip = outer.tip
            pn.kind = outer.kind
            if getattr(outer, 'term', None) is not None:
                pn.term = outer.term
            drop.append(k)
            break
    for k in drop:
        del comps[k]
SWITCH_CLAMP = os.environ.get('PCB_SW_CLAMP', '1') == '1'
CROSS_CLAMP = os.environ.get('PCB_CROSS_CLAMP', '1') == '1'


def _extend_to_pin_length(c, u, L, only_beyond=0.0):
    """Symbol bodies end one pin length inside the pin tips (EasyEDA library convention:
    10-unit pins); grow the box where we cut away part of the body as 'pin'.
    only_beyond: grow only where the tip lies more than that many units outside the box."""
    x0, y0, x1, y1 = c.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for pn in c.pins:
        tx, ty = pn.tip
        if pn.o == 'h' or (pn.o == 'z' and abs(tx - cx) > abs(ty - cy)):
            if tx < cx:
                if x0 - tx > only_beyond * u:
                    c.bbox[0] = min(c.bbox[0], tx + L * u)
            elif tx - x1 > only_beyond * u:
                c.bbox[2] = max(c.bbox[2], tx - L * u)
        elif pn.o == 'v' or pn.o == 'z':
            if ty < cy:
                if y0 - ty > only_beyond * u:
                    c.bbox[1] = min(c.bbox[1], ty + L * u)
            elif ty - y1 > only_beyond * u:
                c.bbox[3] = max(c.bbox[3], ty - L * u)


def refine_bbox(c, P):
    u = P.u
    if c.type in PIN_LEN_TYPES:
        _extend_to_pin_length(c, u, PIN_LEN_TYPES[c.type])
    elif PASSIVE_PIN_LEN and c.type in ('r', 'c', 'l', 'd', 'led', 'fuse') and len(c.pins) == 2:
        # a lead drawn longer than the pin is symbol graphics: the box ends 1u inside the tip
        _extend_to_pin_length(c, u, 1.0, only_beyond=1.75)
    elif SWITCH_PIN_LEN and c.type == 'switch':
        # switches too: leads out to the contacts are symbol graphics (holdout: 29 of 65
        # switch boxes were 1.1u short on both sides)
        _extend_to_pin_length(c, u, 1.0, only_beyond=1.75)
    if (c.type == 'switch' and SWITCH_CLAMP) or (TRANS_CLAMP and c.type in ('mosfet', 'bjt', 'bjt_npn', 'bjt_pnp')):
        # pin bars drawn as one stroke with the body: the body ends a pin length (>= 1u)
        # inside the tips
        x0, y0, x1, y1 = c.bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nb = list(c.bbox)
        near = 0.35 * u          # only where the body really runs out to the tip
        for pn in c.pins:
            tx, ty = pn.tip
            o = pn.o
            if o == 'z':                     # zero-length pin: side from the box centre
                o = 'h' if abs(tx - cx) * (y1 - y0 + 1) > abs(ty - cy) * (x1 - x0 + 1) else 'v'
            if o == 'h':
                if tx <= cx and tx <= x0 + near:
                    nb[0] = max(nb[0], tx + 0.95 * u)
                elif tx > cx and tx >= x1 - near:
                    nb[2] = min(nb[2], tx - 0.95 * u)
            elif o == 'v':
                if ty <= cy and ty <= y0 + near:
                    nb[1] = max(nb[1], ty + 0.95 * u)
                elif ty > cy and ty >= y1 - near:
                    nb[3] = min(nb[3], ty - 0.95 * u)
        if nb[2] - nb[0] >= 0.5 * u and nb[3] - nb[1] >= 0.2 * u:
            c.bbox = nb
    if c.type in ('gnd', 'v') and len(c.pins) == 1:
        tx, ty = c.pins[0].tip
        x0, y0, x1, y1 = c.bbox
        if c.type == 'v':
            ext = 0.45 * u      # power port: the bbox covers part of the stub
            if tx < x0:
                c.bbox[0] = max(tx, x0 - ext)
            elif tx > x1:
                c.bbox[2] = min(tx, x1 + ext)
            elif ty < y0:
                c.bbox[1] = max(ty, y0 - ext)
            elif ty > y1:
                c.bbox[3] = min(ty, y1 + ext)
        else:
            ext = 0.3 * u       # ground: the bbox has a margin below the last bar
            if tx < x0:
                c.bbox[2] = x1 + ext
            elif tx > x1:
                c.bbox[0] = x0 - ext
            elif ty < y0:
                c.bbox[3] = y1 + ext
            elif ty > y1:
                c.bbox[1] = y0 - ext
    if c.type in ('r', 'c', 'd') and len(c.pins) == 2:
        a, b = c.pins
        if a.o == b.o == 'h' or (abs(a.tip[0] - b.tip[0]) > abs(a.tip[1] - b.tip[1]) and a.o != 'v'):
            lo, hi = sorted([a.tip[0], b.tip[0]])
            if hi - lo > 2.2 * u:
                c.bbox[0], c.bbox[2] = lo + u, hi - u
            if CROSS_CLAMP and abs(a.tip[1] - b.tip[1]) <= 2 * P.t + 2:
                ax = (a.tip[1] + b.tip[1]) / 2      # a stray stroke (red note text) glued to the body
                c.bbox[1], c.bbox[3] = max(c.bbox[1], ax - 1.3 * u), min(c.bbox[3], ax + 1.3 * u)
        else:
            lo, hi = sorted([a.tip[1], b.tip[1]])
            if hi - lo > 2.2 * u:
                c.bbox[1], c.bbox[3] = lo + u, hi - u
            if CROSS_CLAMP and abs(a.tip[0] - b.tip[0]) <= 2 * P.t + 2:
                ax = (a.tip[0] + b.tip[0]) / 2
                c.bbox[0], c.bbox[2] = max(c.bbox[0], ax - 1.3 * u), min(c.bbox[2], ax + 1.3 * u)


NAME_WIRE_ALIGNED = os.environ.get('PCB_NAME_WIRE_ALIGNED', '1') == '1'
NAME_NO_CJK_SEMI = os.environ.get('PCB_NAME_NO_CJK_SEMI', '0') == '1'   # off: the answers do name LEDs / MOSFETs '充电', 'P沟道'
NAME_MIN_ALNUM = os.environ.get('PCB_NAME_MIN_ALNUM', '1') == '1'
NAME_NO_CAP_VALUE = os.environ.get('PCB_NAME_NO_CAP_VALUE', '1') == '1'
INSIDE_BLUE_NAME = os.environ.get('PCB_INSIDE_BLUE_NAME', '1') == '1'
CJK_NAME_ON_WIRE = os.environ.get('PCB_CJK_NAME_ON_WIRE', '1') == '1'
# the answers give every crystal (2-, 3- and 4-pin) its printed frequency as value
CRYSTAL_ALL_VALUES = os.environ.get('PCB_CRYSTAL_ALL_VALUES', '1') == '1'
THERMISTOR_VALUE = os.environ.get('PCB_THERMISTOR_VALUE', '1') == '1'
FLAG_DIGITS = os.environ.get('PCB_FLAG_DIGITS', '1') == '1'


def _partnumber_shape(t):
    """Letters and digits mixed, at least four characters, no lower-case words (SS12, 1N4148,
    AO3401A): what a semiconductor part number looks like, unlike a net name (VIN, SCL)."""
    return (len(t) >= 4 and re.search(r'[A-Z]', t) is not None and re.search(r'\d', t) is not None
            and re.fullmatch(r'[A-Z0-9][A-Z0-9.\-/]*', t) is not None)


def pick_name(c, pool, used, u, lib, all_cands=False):
    """Choose the part-name text of a component (IC name / part number).
    all_cands: return every acceptable (score, name, items) instead, best first."""
    rect = c.bbox
    cands = []
    dg = c.desig_item

    def _under_desig(it):
        # designator-like text that no component claimed, stacked right under/over the
        # designator and aligned with it, is the part number (SS34, US1M, BAT54C)
        if dg is None or it.vert or dg.vert:
            return False
        h = max(dg.h, it.h)
        return (abs(it.x0 - dg.x0) <= 0.3 * u or abs(it.cx - dg.cx) <= 0.3 * u) and 0 < abs(it.cy - dg.cy) <= 1.8 * h

    def _desig_like_model(it):
        # part numbers that happen to look like designators (SS12, US1M, LM358, CH340,
        # BSS84, XT60): a letter prefix no schematic uses for designators, next to a part
        # that already has its designator
        if not DESIG_LIKE_MODEL or dg is None or len(it.text) < 4:
            return False
        if getattr(c, 'dc_jack', False) and re.match(r'^DC-?0\d\d', it.text):
            return box_gap(tbox(it), rect) <= 1.5 * u     # DC005 / DC-044 under a DC jack
        p = desig_prefix(it.text)
        if p == desig_prefix(c.key or ''):
            return False
        if p in _EXACT_PREFIXES and not (c.type == 'd' and re.match(r'^BAT\d{2}[A-Z]?$', it.text)):
            return False
        return box_gap(tbox(it), rect) <= 1.0 * u or box_gap(tbox(it), tbox(dg)) <= 1.0 * u

    free = [it for it in pool if it.id not in used and not getattr(it, 'big', False) and
            (not is_desig(it.text) or (lib is not None and len(it.text) >= 5 and lib.match_name(it.text, 0.0))
             or _under_desig(it) or _desig_like_model(it))]
    # single items and horizontally adjacent pairs (names with spaces)
    for it in free:
        cands.append(([it], it.text))
    for a in free:
        for b in free:
            if a is b or a.vert or b.vert:
                continue
            ch = max(a.h, b.h)
            if abs(a.cy - b.cy) <= 0.35 * ch and 0 <= b.x0 - a.x1 <= 1.3 * ch:
                cands.append(([a, b], a.text + ' ' + b.text))
    best = None
    acc = []
    for its, txt in cands:
        if NAME_NO_DIGITS and c.type != 'box' and re.fullmatch(r'[\d\s.]+', txt):
            continue                       # a pin number is never a diode's / LED's part number
        if NAME_MIN_ALNUM and len(re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]', '', txt)) < 2:
            continue                       # a fragment ('v.', '-') is nobody's part number
        if NAME_NO_CAP_VALUE and not (c.type == 'box' and len(c.pins) <= 2) and re.fullmatch(r'\d+(?:[.,]\d+)?\s?[pnuμµm]?[FH](?:[\s/±].*)?', txt):
            continue                       # a capacitor's / inductor's value ('100nF') names no IC
        if CJK_NAME_PEN and c.type != 'box' and re.search(r'[\u4e00-\u9fff]', txt) and not _CJK_PN.search(txt):
            sc_cjk = 0.5                     # a description ranks below a part number beside it
        else:
            sc_cjk = 0.0
        if NAME_NO_CJK_SEMI and c.type != 'box' and re.search(r'[\u4e00-\u9fff]', txt) \
                and not re.search(r'(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}', txt):
            continue                       # '红灯' beside an LED describes it; it is no part number
            #                                ('IN4148贴片二极管' starts with one)
        x0 = min(i.x0 for i in its); y0 = min(i.y0 for i in its)
        x1 = max(i.x1 for i in its); y1 = max(i.y1 for i in its)
        bb = [x0, y0, x1, y1]
        # inside the body (pin names) -> skip unless it is a library name
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = rect[0] + 2 < cx < rect[2] - 2 and rect[1] + 2 < cy < rect[3] - 2
        d = box_gap(rect, bb)
        if c.desig_item is not None:
            d = min(d, box_gap(tbox(c.desig_item), bb) * 0.8)
        if d > 3.0 * u:
            continue
        sc = d / u + sc_cjk
        libname = lib.match_name(txt, 0.12) if (lib is not None and len(txt) >= 4 and not is_power_name(txt)
                                                and not (LIB_MIN4 and len(re.sub(r'[\s_\-+]', '', txt)) < 4)) else None
        if libname and LIB_PINCOUNT:
            # a library part with a very different pin count (a 2-pin connector's number next
            # to a 5-pin switch) is just a nearby text for this part
            ln, cn = lib.npins(libname), len(c.pins)
            if ln >= 2 and cn >= 2 and (cn > ln + 2 or ln > 2 * cn + 2):
                libname = None
        conf = min((i.conf for i in its), default=0.0)
        keep_txt = False
        if libname and libname != txt and conf >= 0.97:
            from .library import canon, lev as _lev
            keep_txt = _lev(canon(txt), canon(libname)) > 1     # confident read of a different part
        if libname:
            sc -= 2.0 + 0.05 * len(txt)
        elif inside and (c.type in ('box', 'switch') or not INSIDE_MODEL_OK) and not (
                INSIDE_BLUE_NAME and c.type == 'box' and len(its) == 1 and its[0].color == 'blue'
                and _partnumber_shape(txt) and len(txt) >= 5 and not is_power_name(txt)
                and _in_body_rect(c, cx, cy)):
            continue                         # pin names inside an IC body (black); a blue part number may sit there
        elif inside and c.type == 'box':
            sc += 0.3
        elif inside:
            sc += 0.3                        # a transistor drawn wide (pin comb): its part number may sit within
        if is_power_name(txt) and not libname:
            sc += 1.5
        if not libname and any(_on_wire(i, _P_CTX[0]) for i in its) and not (
                CJK_NAME_ON_WIRE and c.type in ('led', 'd', 'mosfet', 'bjt') and re.search(r'[\u4e00-\u9fff]', txt)) \
                and not (SEMI_PN_ON_WIRE and c.type in ('d', 'mosfet', 'bjt') and len(its) == 1 and _SEMI_PN.fullmatch(txt)):
            # (a transistor's / diode's part number of a common family - AO3401A, 1N5819WS - is
            # no net name even where it overlaps a wire)
            # (a Chinese description - '红色' beside an LED - is no net name even on the wire)
            dg_ = c.desig_item
            if not (NAME_WIRE_ALIGNED and dg_ is not None and sc <= 1.0 and len(its) == 1
                    and (abs(x0 - dg_.x0) <= 0.3 * u or abs(x1 - dg_.x1) <= 0.3 * u)
                    and _partnumber_shape(txt)):
                sc += 1.5                    # (a part number squarely under its designator may touch a wire)
        if c.desig_item is not None:
            dg = c.desig_item
            if abs(x0 - dg.x0) <= 0.3 * u or abs(x1 - dg.x1) <= 0.3 * u:
                sc -= 0.4                     # left/right aligned with the designator
            elif abs((y0 + y1) / 2 - dg.cy) < 0.5 * dg.h:
                sc += 0.4                     # same row but not aligned: usually another label
        if is_value(txt) and not libname:
            sc += 0.8
        if SHADE_NAME_PEN and getattr(_P_CTX[0], 'shade_mode', False) and all(getattr(i, 'shade', None) == 'navy' for i in its):
            sc += 1.5                        # navy is a designator / net name; part names are bright blue
        if len(its) == 2 and not libname:
            sc += 1.0
        if libname and _LIB_STRICT and not _confusable_only(txt, libname):
            keep_txt = True        # a different part of the same family: keep what is printed
        sa = sc - sc_cjk if CJK_PEN_RANK_ONLY else sc     # (the description penalty only ranks)
        if all_cands:
            nm = txt if keep_txt else (libname or txt)
            if (sa <= NAME_MAX_BOX or (NAME_FAR_PN and sa <= NAME_FAR_MAX and len(nm) >= 6
                                       and _partnumber_like(nm, its) and '_' not in nm)) if c.type == 'box' \
                    else (sa <= 1.0 or (sa < 2.0 and _partnumber_like(nm, its))):
                acc.append((sc, nm, its))
        if best is None or sc < best[0]:
            best = (sc, txt if keep_txt else (libname or txt), its, sa)
    if all_cands:
        return sorted(acc, key=lambda x: x[0])
    if best is None:
        return None
    # no printed name: the nearest text is a net label / pin text -> Name falls back to the designator
    if c.type == 'box' and best[0] > NAME_MAX_BOX and not (
            NAME_FAR_PN and best[0] <= NAME_FAR_MAX and len(best[1]) >= 6 and _partnumber_like(best[1], best[2])
            and '_' not in best[1]):     # a far 'BAT_ADC1' is a net name beside the pins
        NAME_LOG.append((c.key, c.type, round(best[0], 2), None, False, len(c.pins), best[1]))
        return None
    if c.type != 'box' and best[3] > 1.0 and not (best[3] < 2.0 and _partnumber_like(best[1], best[2])):
        NAME_LOG.append((c.key, c.type, round(best[0], 2), None, False, len(c.pins), best[1]))
        return None
    NAME_LOG.append((c.key, c.type, round(best[0], 2), best[1], bool(libname_of(best)), len(c.pins), best[1]))
    c.name_conf = min((i.conf for i in best[2]), default=0.0)
    c.name_score = best[0]
    return best[1], best[2]


CJK_PEN_RANK_ONLY = os.environ.get('PCB_CJK_PEN_RANK_ONLY', '1') == '1'
SEMI_PN_ON_WIRE = os.environ.get('PCB_SEMI_PN_ON_WIRE', '1') == '1'
_SEMI_PN = re.compile(r'(?:AO|SI|IRF|IRL|IRLML|BSS|NCE|FDN|FDS|DMG|DMN|DMP|WST|CJ|MMBT|BCX?|2N|2S[ABCDJK]|S80|SS8|'
                      r'1N|1SS|SS|SMF|SMAJ|SMBJ|P6KE|B5|BAT|BAV|MBR|ES|US|RS|M7|LL|BZT|MMSZ|PESD|AP|CRS|SK|HSS)'
                      r'\d[0-9A-Z\-.]{1,14}')


def _assign_names(named, pool, used_text, u, lib):
    """Part numbers as one assignment: every acceptable (part, text) pair costs its score
    minus a bonus, so a part does not lose its only text to a neighbour that has another
    one (four LEDs, four part numbers, the nearest text of two of them is the same)."""
    rows, cols, colid = [], [], {}
    for c in named:
        L = pick_name(c, pool, used_text, u, lib if c.type in ("box", "switch") else None, all_cands=True)
        rows.append((c, L))
        for sc, nm, its in L:
            k = tuple(sorted(i.id for i in its))
            if k not in colid:
                colid[k] = len(cols)
                cols.append(its)
    if not cols:
        return
    n, m = len(rows), len(cols)
    C = np.zeros((n, m + n))
    C[:, :m] = 1e6
    for i, (c, L) in enumerate(rows):
        for sc, nm, its in L:
            j = colid[tuple(sorted(i_.id for i_ in its))]
            C[i, j] = min(C[i, j], sc - NAME_BONUS)
    r, cc = linear_sum_assignment(C)
    taken = set()
    order = sorted(((C[i, j], i, j) for i, j in zip(r, cc) if j < m and C[i, j] < 1e5))
    skipped = []
    for _, i, j in order:
        c, L = rows[i]
        its = cols[j]
        if any(x.id in taken or x.id in used_text for x in its):
            skipped.append(c)         # a pair overlapping a single text: keep the better one
            continue
        sc, nm = next((sc, nm) for sc, nm, it2 in L if tuple(sorted(x.id for x in it2)) == tuple(sorted(x.id for x in its)))
        c.name = nm
        c.name_score = sc
        c.name_conf = min((x.conf for x in its), default=0.0)
        for x in its:
            taken.add(x.id)
            used_text.add(x.id)
    for c in skipped:                 # the loser of an overlap looks again among what is left
        res = pick_name(c, pool, used_text, u, lib if c.type in ("box", "switch") else None)
        if res:
            c.name = res[0]
            for x in res[1]:
                used_text.add(x.id)


NAME_ASSIGN = os.environ.get('PCB_NAME_ASSIGN', '1') == '1'
NAME_BONUS = float(os.environ.get('PCB_NAME_BONUS', '2.5'))
AMP_PINS = os.environ.get('PCB_AMP_PINS', '1') == '1'
NAME_LOG = []
FB_LOG = []
LIB_PINCOUNT = os.environ.get('PCB_LIB_PINCOUNT', '1') == '1'
DESIG_LIKE_MODEL = os.environ.get('PCB_DESIG_MODEL', '1') == '1'
from .textclass import DESIG_PREFIX_TYPE as _DPT  # noqa: E402
_EXACT_PREFIXES = {p for p, _ in _DPT}
from .library import STRICT as _LIB_STRICT, confusable_only as _confusable_only  # noqa: E402
NAME_MAX_BOX = 1.25
# an IC's part number printed farther out (above a section caption, below the pin row):
# accepted when it looks like a part number and nothing closer claims it
NAME_FAR_PN = os.environ.get('PCB_NAME_FAR_PN', '1') == '1'
NAME_FAR_MAX = float(os.environ.get('PCB_NAME_FAR_MAX', '2.6'))
_PN_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\-_.+/()]{5,}$')
_PN_SP_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\-_.+/() ]{5,}$')
PN_SPACES = os.environ.get('PCB_PN_SPACES', '1') == '1'   # connector names: 'TYPE-C 6P LTH6.5'


def _partnumber_like(txt, its):
    ok = bool(_PN_RE.match(txt)) or (PN_SPACES and txt.count(' ') <= 3 and bool(_PN_SP_RE.match(txt))
                                     and bool(_PN_RE.match(txt.split(' ')[0] + 'xx')))
    return (ok and any(ch.isdigit() for ch in txt) and any(ch.isalpha() for ch in txt)
            and not is_value(txt) and not is_power_name(txt) and not any(_on_wire(i, _P_CTX[0]) for i in its))


def libname_of(best):
    return best[1] != ' '.join(i.text for i in best[2])


FLAG_DOT_CLOSE = os.environ.get('PCB_FLAG_DOT_CLOSE', '1') == '1'


def _dot_closed_holes(c, P):
    """A flag whose point sits on a junction dot: the dot was taken out with the point and
    the outline is open there.  Put the dot back as ink and look for the enclosed region.
    Returns (holes in feat format [(area, w, h, rect)], width, height of outline + dot)."""
    S = _S_CTX[0]
    if S is None or len(c.pins) != 1 or len(c.gids) != 1 or not getattr(P, 'dots', None):
        return [], 0, 0
    tx, ty = c.pins[0].tip
    near = [d for d in P.dots if abs(d[0] - tx) <= 1.2 * d[2] + 2 and abs(d[1] - ty) <= 1.2 * d[2] + 2]
    if not near:
        return [], 0, 0
    dx, dy, dr = min(near, key=lambda d: (d[0] - tx) ** 2 + (d[1] - ty) ** 2)
    x0, y0, x1, y1 = [int(v) for v in S['groups'][c.gids[0]]['bbox']]
    r = int(np.ceil(1.6 * dr)) + 1      # the dot's clearing zone is wider than the dot
    X0, Y0 = min(x0, int(dx) - r) - 1, min(y0, int(dy) - r) - 1
    X1, Y1 = max(x1, int(dx) + r) + 1, max(y1, int(dy) + r) + 1
    H, W = S['glab'].shape
    X0, Y0, X1, Y1 = max(X0, 0), max(Y0, 0), min(X1, W - 1), min(Y1, H - 1)
    m = (S['glab'][Y0:Y1 + 1, X0:X1 + 1] == c.gids[0]).astype(np.uint8)
    cv2.circle(m, (int(round(dx - X0)), int(round(dy - Y0))), r, 1, -1)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is not None:
        for cnt, hh in zip(cnts, hier[0]):
            if hh[3] >= 0:
                a = cv2.contourArea(cnt)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if a >= 4:
                    out.append((a, bw, bh, a / max(bw * bh, 1)))
    ys, xs = np.nonzero(m)
    return out, int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


FLAG_TEXT_CLOSE = os.environ.get('PCB_FLAG_TEXT_CLOSE', '1') == '1'
FLAG_ONE_CELL = os.environ.get('PCB_FLAG_ONE_CELL', '1') == '1'


def _text_closed_holes(c, P):
    """A flag with its label written across the outline: the text strokes cut the outline.
    Count text-colour ink inside the part's box as outline and look for the enclosed region
    (feat format)."""
    S = _S_CTX[0]
    M = getattr(P, 'M', None)
    if S is None or M is None or len(c.gids) != 1:
        return []
    x0, y0, x1, y1 = [int(v) for v in S['groups'][c.gids[0]]['bbox']]
    m = (S['glab'][y0:y1 + 1, x0:x1 + 1] == c.gids[0])
    tx = np.isin(M['lab'][y0:y1 + 1, x0:x1 + 1], (3, 4))
    if not tx.any():
        return []
    m = (m | tx).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is not None:
        for cnt, hh in zip(cnts, hier[0]):
            if hh[3] >= 0:
                a = cv2.contourArea(cnt)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if a >= 4:
                    out.append((a, bw, bh, a / max(bw * bh, 1)))
    # only the flag's own cell counts: a big one, not the counters of letters
    return [h for h in out if h[0] >= 0.2 * (x1 - x0 + 1) * (y1 - y0 + 1)]


def _is_flag(c, P):
    """Net-label flag: an elongated closed polygon with the pin on a short side."""
    f = c.feat
    if not f['holes']:
        if not (FLAG_DOT_CLOSE and len(c.pins) == 1):
            return False
        hs, w_, h_ = _dot_closed_holes(c, P)
        if hs:
            f = dict(f, holes=hs, w=w_, h=h_)
            c.dot_closed = True
        elif FLAG_TEXT_CLOSE:
            hs = _text_closed_holes(c, P)
            if not hs:
                return False
            f = dict(f, holes=hs)
            c.text_closed = True
        else:
            return False
    w, h = f['w'], f['h']
    big = max(f['holes'], key=lambda x: x[0])
    a, bw, bh, rect = big
    if a < 0.2 * w * h:
        return False
    if FLAG_ONE_CELL and any(hh is not big and hh[0] >= 0.15 * a for hh in f['holes']):
        return False                      # a second sizeable cell: a DC jack's barrel + collar
    # (closed by putting the dot back, the point is under the dot: the cell is squarer)
    if rect > (0.93 if getattr(c, 'dot_closed', False) or f is not c.feat else 0.88):
        return False
    if max(w, h) < 1.6 * min(w, h):
        return False
    pn = c.pins[0]
    horiz = w > h
    tx, ty = pn.tip
    x0, y0, x1, y1 = c.bbox
    if horiz:
        return abs(ty - (y0 + y1) / 2) <= 0.3 * h + 2 and (tx <= x0 + 0.25 * w or tx >= x1 - 0.25 * w)
    return abs(tx - (x0 + x1) / 2) <= 0.3 * w + 2 and (ty <= y0 + 0.25 * h or ty >= y1 - 0.25 * h)


NET_TYPES_BY_SHAPE = os.environ.get('PCB_NET_SHAPE', '1') == '1'


RECT_PORT = os.environ.get('PCB_RECT_PORT', '1') == '1'
FLAG_DOT_DROP = os.environ.get('PCB_FLAG_DOT_DROP', '1') == '1'


def _rect_port(c, P, blue):
    """A net port drawn as a plain rectangle (no point): elongated, one closed rectangular
    cell holding a label that is not a designator, the pin in the middle of a short side."""
    f = c.feat
    if not f['holes'] or len(c.pins) != 1:
        return False
    w, h = f['w'], f['h']
    a = max(f['holes'], key=lambda x: x[0])[0]
    if a < 0.5 * w * h or max(w, h) < 2.0 * min(w, h) or min(w, h) > 1.5 * P.u:
        return False
    pn = c.pins[0]
    x0, y0, x1, y1 = c.bbox
    tx, ty = pn.tip
    if w > h:
        if not (abs(ty - (y0 + y1) / 2) <= 0.3 * h + 2 and (tx <= x0 + 0.1 * w or tx >= x1 - 0.1 * w)):
            return False
    elif not (abs(tx - (x0 + x1) / 2) <= 0.3 * w + 2 and (ty <= y0 + 0.1 * h or ty >= y1 - 0.1 * h)):
        return False
    return any(x0 < it.cx < x1 and y0 < it.cy < y1 and not is_desig(it.text) and not is_value(it.text)
               for it in blue)


def _flag_type(c, P):
    """Net port type from the flag outline (EasyEDA netport-in / -out / -bi, the official
    labels' net_input / net_output / net_bidirection): the pointed end at the connection
    means input, the pointed end away from it output, both ends pointed bidirectional.
    Verified on the 200 training cases: point-at-pin -> net_input 250:2, point-away ->
    net_output 92:0."""
    if not NET_TYPES_BY_SHAPE:
        return 'net_bidirection'
    if not c.pins:
        return 'net_input'
    x0, y0, x1, y1 = [int(round(v)) for v in c.bbox]
    pad = 2
    H, W = P.red.shape
    sub = (P.red[max(0, y0 - pad):min(H, y1 + pad + 1), max(0, x0 - pad):min(W, x1 + pad + 1)] > 0).astype(np.uint8)
    if getattr(c, 'text_closed', False) and getattr(P, 'M', None) is not None:
        # the label is written across the outline: its strokes stand in for the cut outline
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        sub |= np.isin(P.M['lab'][Y0:Y0 + sub.shape[0], X0:X0 + sub.shape[1]], (3, 4)).astype(np.uint8)
    if getattr(c, 'dot_closed', False):
        # the point lies under a junction dot: put the dot back so the outline is closed
        tx, ty = c.pins[0].tip
        near = [d for d in P.dots if abs(d[0] - tx) <= 1.2 * d[2] + 2 and abs(d[1] - ty) <= 1.2 * d[2] + 2]
        if near:
            dx, dy, dr = min(near, key=lambda d: (d[0] - tx) ** 2 + (d[1] - ty) ** 2)
            X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
            r = int(np.ceil(1.6 * dr)) + 1      # the dot's clearing zone is wider than the dot
            big = np.zeros((sub.shape[0] + 2 * r + 2, sub.shape[1] + 2 * r + 2), np.uint8)
            big[r + 1:r + 1 + sub.shape[0], r + 1:r + 1 + sub.shape[1]] = sub
            cv2.circle(big, (int(round(dx - X0)) + r + 1, int(round(dy - Y0)) + r + 1), r, 1, -1)
            sub = big
            x0, y0, x1, y1 = x0 - r - 1, y0 - r - 1, x1 + r + 1, y1 + r + 1
    if sub.sum() < 5:
        return 'net_input'
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(sub)
    cv2.drawContours(fill, cnts, -1, 1, -1)
    tx, ty = c.pins[0].tip
    horiz = (x1 - x0) >= (y1 - y0)
    if horiz:
        ext = fill.sum(0).astype(float)
        pin_lo = tx < (x0 + x1) / 2
    else:
        ext = fill.sum(1).astype(float)
        pin_lo = ty < (y0 + y1) / 2
    # trim the pin stub / leftover wire: a thin constant run at either end is not part
    # of the flag outline (a real point tapers over several columns)
    thin = 1.8 * P.t + 1
    nz = np.nonzero(ext > thin)[0]
    if len(nz) < 6:
        return 'net_input'
    ext = ext[nz[0]:nz[-1] + 1]
    n = len(ext)
    mx = ext.max()
    k = max(1, n // 8)
    lo_pt = ext[:k].mean() / mx < 0.6
    hi_pt = ext[-k:].mean() / mx < 0.6
    if lo_pt and hi_pt:
        return 'net_bidirection'
    if not lo_pt and not hi_pt:
        return 'net_input'          # rectangle / unknown: the common netport-in
    return 'net_input' if lo_pt == pin_lo else 'net_output'


FLAG_OWN_AXIS = os.environ.get('PCB_FLAG_OWN_AXIS', '1') == '1'


def _axis_off(c, it):
    pn = c.pins[0]
    if pn.o == 'h':
        return abs(it.cy - pn.tip[1])
    if pn.o == 'v':
        return abs(it.cx - pn.tip[0])
    return 0.0


def _better_aligned_flag(c, it, flags, u):
    """In a row of flags a label off this flag's axis that sits on a neighbour's axis is the
    neighbour's (when the neighbour's own label was not read, the next label must not slide
    over: IO10 IO11 IO12 -> the flag of the unread IO11 stays unnamed)."""
    off = _axis_off(c, it)
    if off <= 0.4 * u:
        return False
    for o in flags:
        if o is c or not o.pins or o.pins[0].o != c.pins[0].o:
            continue
        if _axis_off(o, it) < off - 0.25 * u and box_gap(o.bbox, tbox(it)) <= 1.5 * u:
            return True
    return False


def _align_pen(c, it, u):
    """A flag's label is centred on the flag's axis (the pin line); in a stack of flags the
    neighbour's label is just as close by box distance, so misalignment decides."""
    pn = c.pins[0]
    if pn.o == 'h':
        off = abs(it.cy - pn.tip[1])
    elif pn.o == 'v':
        off = abs(it.cx - pn.tip[0])
    else:
        return 0.0
    return 1.5 * off


def _side_pen(c, it, u):
    """Penalty when a label text lies on the pin side of a one-pin symbol."""
    pn = c.pins[0]
    x0, y0, x1, y1 = c.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vx, vy = pn.tip[0] - cx, pn.tip[1] - cy
    tx, ty = it.cx - cx, it.cy - cy
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1e-6:
        return 0.0
    proj = (tx * vx + ty * vy) / n
    return 0.6 * u if proj > 0.3 * u else 0.0


ONE_PIN_TYPES = ('gnd', 'v', 'net_bidirection', 'net_input', 'net_output', 'net_short')


def fix_contacts(comps, P):
    """Place the contact point of two directly touching pins using pin-length
    conventions: power/net symbols have zero-length pins, R/IC pins are 1u long."""
    u = P.u
    owner = {}
    for c in comps:
        for pn in c.pins:
            owner[id(pn)] = c
    # stem length of the power / ground symbols of this sheet, from the ones joined by a
    # wire (stem = their pin): a part touching one directly ends where that stem would start
    stems = {}
    for ty in ('gnd', 'v'):
        ls = [float(np.hypot(c.pins[0].tip[0] - c.pins[0].base[0], c.pins[0].tip[1] - c.pins[0].base[1]))
              for c in comps if c.type == ty and len(c.pins) == 1 and c.pins[0].kind == 'wire'
              and c.pins[0].o in ('h', 'v')]
        ls = [x for x in ls if 0.4 * u <= x <= 1.6 * u]
        if len(ls) >= 2:
            med = float(np.median(ls))
            if sum(abs(x - med) <= 0.12 * u + 1 for x in ls) >= 0.7 * len(ls):
                stems[ty] = med
    done = set()
    for c in comps:
        for pa in c.pins:
            pb = getattr(pa, 'mate', None)
            if pb is None or id(pa) in done or id(pb) not in owner:
                continue
            done.add(id(pa)); done.add(id(pb))
            ca, cb = c, owner[id(pb)]
            if pa.seg is None and pb.seg is None:
                continue
            # link geometry
            if pa.seg is not None and pb.seg is not None and pa.seg is pb.seg:
                A, B = pa.base, pb.base
            else:
                # one side is a zero pin sitting on the other body
                continue
            dx, dy = B[0] - A[0], B[1] - A[1]
            L = (dx * dx + dy * dy) ** 0.5
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            tip = None
            NETS = ('net_bidirection', 'net_input', 'net_output', 'net_short')
            if ca.type in NETS and cb.type not in NETS:
                tip = A
            elif cb.type in NETS and ca.type not in NETS:
                tip = B
            elif POWER_STEM and cb.type in stems and ca.type not in ('gnd', 'v') and L - stems[cb.type] >= 0.8 * u \
                    and L >= 1.6 * u:
                d = stems[cb.type]
                tip = (B[0] - ux * d, B[1] - uy * d)
            elif POWER_STEM and ca.type in stems and cb.type not in ('gnd', 'v') and L - stems[ca.type] >= 0.8 * u \
                    and L >= 1.6 * u:
                d = stems[ca.type]
                tip = (A[0] + ux * d, A[1] + uy * d)
            elif 1.6 * u <= L <= 2.4 * u:
                tip = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
            elif ca.type in ('r', 'box') and cb.type not in ('r', 'box') and L >= 0.8 * u:
                d = min(u, L)
                tip = (A[0] + ux * d, A[1] + uy * d)
            elif cb.type in ('r', 'box') and ca.type not in ('r', 'box') and L >= 0.8 * u:
                d = min(u, L)
                tip = (B[0] - ux * d, B[1] - uy * d)
            if tip is not None:
                pa.tip = tuple(tip)
                pb.tip = tuple(tip)


BODY_ALL_GIDS = os.environ.get('PCB_BODY_ALL', '1') == '1'


def _body_mask(c, S):
    if not BODY_ALL_GIDS or len(c.gids) == 1:
        x0, y0, x1, y1 = [int(round(v)) for v in S['groups'][c.gids[0]]['bbox']]
        return (S['glab'][y0:y1 + 1, x0:x1 + 1] == c.gids[0]).astype(np.uint8)
    # merged pieces (MOSFET + body diode, gate + channel): the whole symbol
    bbs = [S['groups'][g]['bbox'] for g in c.gids if g in S['groups'] or isinstance(S['groups'], list)]
    x0 = int(round(min(b[0] for b in bbs))); y0 = int(round(min(b[1] for b in bbs)))
    x1 = int(round(max(b[2] for b in bbs))); y1 = int(round(max(b[3] for b in bbs)))
    return np.isin(S['glab'][y0:y1 + 1, x0:x1 + 1], list(c.gids)).astype(np.uint8)


BJT_LONG_DIAG = os.environ.get('PCB_BJT_LONG_DIAG', '1') == '1'


def _is_bjt(c, S, P):
    """Bipolar transistor: collector and emitter are slanted strokes a unit long from the
    base bar; a MOSFET's slanted strokes are its body diode's triangle and arrow heads, half
    as long (measured on the official drawings: bipolar >= 1.05u, MOSFET mostly < 1.0u)."""
    if _n_diagonals(c, S, P) < 2 or _has_gate_gap(c, S, P):
        return False
    if BJT_LONG_DIAG:
        md = _max_diagonal(c, S, P)
        # a two-pin piece needs clearly long slanted strokes (a MOSFET's body-diode piece has
        # its triangle); MOSFETs are the rule
        if md < (1.1 if len(c.pins) < 3 else 1.0 if len(c.pins) >= 5 else 0.7) * P.u:
            return False
    return True


def _max_diagonal(c, S, P):
    m = _body_mask(c, S)
    if m.sum() < 10:
        return 0.0
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 90, threshold=max(6, int(0.3 * P.u)),
                        minLineLength=max(6, int(0.45 * P.u)), maxLineGap=2)
    if L is None:
        return 0.0
    best = 0.0
    for x1, y1, x2, y2 in np.asarray(L).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 15 < a < 75 or 105 < a < 165:
            best = max(best, float(np.hypot(x2 - x1, y2 - y1)))
    return best


def _n_diagonals(c, S, P):
    """Number of distinct diagonal strokes (>= 0.45u) in the body."""
    m = _body_mask(c, S)
    if m.sum() < 10:
        return 0
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 90, threshold=max(6, int(0.3 * P.u)),
                        minLineLength=max(6, int(0.45 * P.u)), maxLineGap=2)
    if L is None:
        return 0
    L = np.asarray(L).reshape(-1, 4)
    angs = []
    for x1, y1, x2, y2 in L:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 15 < a < 75 or 105 < a < 165:
            if not any(abs(a - b) < 10 for b in angs):
                angs.append(a)
    return len(angs)


AMP_TRIANGLE = os.environ.get('PCB_AMP_TRI', '1') == '1'


def _amp_triangle(c, S, P):
    """Op-amp body: the outline itself is a triangle - two long slanted sides of opposite
    slope spanning the body.  An opto-coupler / LED driver drawn in a rectangle only has
    short diagonals (LED triangle, transistor, arrows) relative to its body."""
    m = _body_mask(c, S)
    if m.sum() < 10:
        return False
    h, w = m.shape
    ext = max(h, w)
    L = cv2.HoughLinesP(m * 255, 1, np.pi / 180, threshold=max(8, int(0.3 * ext)),
                        minLineLength=max(8, int(0.5 * ext)), maxLineGap=3)
    if L is None:
        return False
    angs = []
    for x1, y1, x2, y2 in np.asarray(L).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 15 < a < 75 or 105 < a < 165:
            angs.append(a)
    for a in angs:
        for b in angs:
            if abs((a + b) - 180) <= 14 and abs(a - b) > 20:
                return True
    return False


def _has_gate_gap(c, S, P):
    """MOSFET gate: two parallel straight strokes separated by a small gap."""
    m = _body_mask(c, S)
    h, w = m.shape
    if GATE_RELAX:
        # the channel of an enhancement MOSFET is three short dashes: its column holds
        # only ~0.5-0.9u of ink, the gate line next to it a full stroke
        u = P.u
        for arr in (m, m.T):
            cols = arr.sum(0)
            weak = np.where(cols >= 0.45 * u)[0]
            if len(weak) < 2:
                continue
            groups = np.split(weak, np.where(np.diff(weak) > 1)[0] + 1)
            info = []
            for g in groups:
                rows = np.where(arr[:, g].any(1))[0]
                info.append((g.mean(), cols[g].max(), rows.min(), rows.max()))
            for i in range(len(info)):
                for j in range(i + 1, len(info)):
                    a, b = info[i], info[j]
                    if not (0.12 * u <= abs(b[0] - a[0]) <= 0.5 * u):
                        continue
                    if max(a[1], b[1]) < 0.8 * u:
                        continue
                    ov = min(a[3], b[3]) - max(a[2], b[2])
                    if ov >= 0.6 * min(a[3] - a[2], b[3] - b[2]):
                        return True
    for arr in (m, m.T):
        cols = arr.sum(0)
        long_cols = np.where(cols >= 0.8 * P.u)[0]
        if len(long_cols) >= 2:
            groups = np.split(long_cols, np.where(np.diff(long_cols) > 1)[0] + 1)
            if len(groups) >= 2:
                centers = [g.mean() for g in groups]
                for a, b in zip(centers, centers[1:]):
                    if 0.15 * P.u <= b - a <= 0.5 * P.u:
                        return True
    return False


def _piece_bbox_map(pieces):
    ys, xs = np.nonzero(pieces)
    labs = pieces[ys, xs]
    n = int(pieces.max()) + 1
    bb = np.zeros((n, 4), np.int64)
    bb[:, 0] = 1 << 30; bb[:, 1] = 1 << 30; bb[:, 2] = -1; bb[:, 3] = -1
    np.minimum.at(bb[:, 0], labs, xs); np.minimum.at(bb[:, 1], labs, ys)
    np.maximum.at(bb[:, 2], labs, xs); np.maximum.at(bb[:, 3], labs, ys)
    area = np.bincount(labs, minlength=n)
    return bb, area


KEEP_FLAGS_ON_D = os.environ.get('PCB_KEEP_FLAGS_ON_D', '1') == '1'


LED_CORE_DESIG = os.environ.get('PCB_LED_CORE_DESIG', '1') == '1'
DESIG_NO_SPECK = os.environ.get('PCB_DESIG_NO_SPECK', '1') == '1'
WIRE_PIECES = os.environ.get('PCB_WIRE_PIECES', '1') == '1'
DESIG_BLOCK = os.environ.get('PCB_DESIG_BLOCK', '1') == '1'


def _two_pin_core(c, P):
    """A two-terminal part with light arrows hanging off its side (an LED, its arrows
    touching it as extra contact pins): the body across its pin axis, without the arrows.
    The arrows of a column of LEDs reach towards the next LED's designator."""
    if len(c.pins) < 3:
        return None
    u, t = P.u, P.t
    if max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) > 3.0 * u:
        return None                         # an IC box with a pin touching a ground symbol
    main = None
    for o in ('h', 'v'):
        same = [pn for pn in c.pins if pn.o == o]
        rest = [pn for pn in c.pins if pn.o != o]
        if len(same) == 2 and rest and all(pn.kind == 'contact' for pn in rest):
            main = same
    if main is None:
        return None
    a, b = main
    x0, y0, x1, y1 = c.bbox
    if a.o == 'h' and b.o == 'h' and abs(a.tip[1] - b.tip[1]) <= t + 1:
        ax = (a.tip[1] + b.tip[1]) / 2.0
        return [x0, max(y0, ax - 0.7 * u), x1, min(y1, ax + 0.7 * u)]
    if a.o == 'v' and b.o == 'v' and abs(a.tip[0] - b.tip[0]) <= t + 1:
        ax = (a.tip[0] + b.tip[0]) / 2.0
        return [max(x0, ax - 0.7 * u), y0, min(x1, ax + 0.7 * u), y1]
    return None


def _merge_contact_pieces(final, used_text):
    """A 1-pin 'part' whose only pin is a direct contact with a diode/LED is a piece of
    that diode (cathode-bar extension touching the light arrows), not a net label."""
    owner = {}
    for c in final:
        for pn in c.pins:
            owner[id(pn)] = c
    keep = []
    drop = set()
    for c in final:
        if len(c.pins) == 1 and c.type not in ('gnd', 'v') and c.pins[0].mate is not None:
            host = owner.get(id(c.pins[0].mate))
            if host is not None and host is not c and host.type in ('led', 'd', 'opt') and len(host.pins) > 2:
                if KEEP_FLAGS_ON_D and getattr(c, 'flag', False) and (
                        getattr(c, 'split_from', None) is not None or getattr(c, 'carved_from', None) is not None
                        or len(host.pins) >= 5):
                    keep.append(c)
                    continue              # a net flag on a pin of a D-designated IC (ESD array)
                host.pins = [q for q in host.pins if q is not c.pins[0].mate]
                host.bbox = [min(host.bbox[0], c.bbox[0]), min(host.bbox[1], c.bbox[1]),
                             max(host.bbox[2], c.bbox[2]), max(host.bbox[3], c.bbox[3])]
                host.gids = host.gids + c.gids
                if c.label_item is not None:
                    used_text.discard(c.label_item.id)
                if c.desig_item is not None:
                    used_text.discard(c.desig_item.id)
                drop.add(id(c))
                continue
    return [c for c in final if id(c) not in drop]


ARROW_LED_TO_LED = os.environ.get('PCB_ARROW_LED', '1') == '1'


def reassign_arrows(comps, S, P):
    """Small arrow pieces belong to the nearest LED / photo device, not to whatever
    anchored piece happened to be closest (dense layouts)."""
    leds = [c for c in comps if c.type in ('led', 'opt')]
    if not leds or 'pieces' not in S:
        return
    u = P.u
    pieces = S['pieces']
    glab = S['glab']
    bb, area = _piece_bbox_map(pieces)
    gid_of = {}
    for c in comps:
        for g in c.gids:
            gid_of[g] = c
    pin_pieces = set()
    for c in comps:
        for pn in c.pins:
            x, y = int(round(pn.base[0])), int(round(pn.base[1]))
            win = pieces[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
            pin_pieces.update(int(v) for v in np.unique(win) if v > 0)
    moved = {}
    for pc in range(1, len(area)):
        if area[pc] == 0 or pc in pin_pieces:
            continue
        w = bb[pc, 2] - bb[pc, 0] + 1
        h = bb[pc, 3] - bb[pc, 1] + 1
        if max(w, h) > 1.5 * u or max(w, h) < 0.25 * u:
            continue
        if area[pc] > 0.45 * w * h:
            continue
        # owner group of this piece
        y0, x0 = bb[pc, 1], bb[pc, 0]
        sub = (pieces[bb[pc, 1]:bb[pc, 3] + 1, bb[pc, 0]:bb[pc, 2] + 1] == pc)
        lab_vals = glab[bb[pc, 1]:bb[pc, 3] + 1, bb[pc, 0]:bb[pc, 2] + 1][sub]
        owner_g = int(np.bincount(lab_vals).argmax()) if lab_vals.size else 0
        owner = gid_of.get(owner_g)
        pbox = [bb[pc, 0], bb[pc, 1], bb[pc, 2], bb[pc, 3]]
        if owner is not None and owner.type in ('led', 'opt'):
            if not ARROW_LED_TO_LED:
                continue
            # arrows grouped with one LED but sitting right against another LED's body
            gx0, gy0, gx1, gy1 = [int(v) for v in S['groups'][owner_g]['bbox']]
            mm = (glab[gy0:gy1 + 1, gx0:gx1 + 1] == owner_g) & (pieces[gy0:gy1 + 1, gx0:gx1 + 1] != pc)
            ys_, xs_ = np.nonzero(mm)
            if not len(ys_):
                continue
            obody = [gx0 + xs_.min(), gy0 + ys_.min(), gx0 + xs_.max(), gy0 + ys_.max()]
            d_own = box_gap(obody, pbox)
            alt = [(box_gap(c.bbox, pbox), c) for c in leds if c is not owner]
            alt = [a for a in alt if a[0] <= 0.5 * u]
            if not alt:
                continue
            d_alt, c_alt = min(alt, key=lambda a: a[0])
            tail = _arrow_tail(sub, bb[pc, 0], bb[pc, 1]) if ARROW_TAIL else None
            if tail is not None:
                # light leaves an LED: the arrow's tail is at its owner
                def _pd(b):
                    return float(np.hypot(max(b[0] - tail[0], 0, tail[0] - b[2]), max(b[1] - tail[1], 0, tail[1] - b[3])))
                ab = c_alt.body_bbox if hasattr(c_alt, 'body_bbox') else c_alt.bbox
                if not (_pd(ab) + 0.25 * u < _pd(obody)):
                    continue
            elif d_alt + 0.3 * u >= d_own:
                continue
            moved.setdefault(id(c_alt), (c_alt, []))[1].append(pbox)
            owner.removed = getattr(owner, 'removed', []) + [pc]
            continue
        best = None
        for c in leds:
            body = c.body_bbox if hasattr(c, 'body_bbox') else c.bbox
            d = box_gap(body, pbox)
            if d <= 1.0 * u and (best is None or d < best[0]):
                best = (d, c)
        if best is None:
            continue
        if owner is not None and box_gap(owner.bbox, pbox) + 0.05 * u < best[0] and owner.type not in ('r', 'c', 'l', 'box'):
            continue
        moved.setdefault(id(best[1]), (best[1], []))[1].append(pbox)
        if owner is not None:
            owner.removed = getattr(owner, 'removed', []) + [pc]
    for c, boxes in moved.values():
        for b in boxes:
            c.bbox = [min(c.bbox[0], b[0]), min(c.bbox[1], b[1]), max(c.bbox[2], b[2]), max(c.bbox[3], b[3])]
    # owners losing pieces: recompute their bbox from the remaining pixels
    for c in comps:
        rem = getattr(c, 'removed', None)
        if not rem:
            continue
        g = c.gids[0]
        x0, y0, x1, y1 = [int(v) for v in S['groups'][g]['bbox']]
        m = (glab[y0:y1 + 1, x0:x1 + 1] == g) & ~np.isin(pieces[y0:y1 + 1, x0:x1 + 1], rem)
        ys, xs = np.nonzero(m)
        if len(ys):
            c.bbox = [x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()]


ARROW_TAIL = os.environ.get('PCB_ARROW_TAIL', '1') == '1'


def _arrow_tail(sub, x0, y0):
    """Tail point of an arrow piece (bool mask at x0, y0): the end of the main axis away
    from the head, the head being the end with the wider cross-section."""
    ys, xs = np.nonzero(sub)
    if len(xs) < 8:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    m = pts.mean(0)
    q = pts - m
    w, v = np.linalg.eigh(q.T @ q)
    ax = v[:, 1]
    pr = q @ ax
    pe = q @ v[:, 0]
    lo, hi = pr.min(), pr.max()
    if hi - lo < 4:
        return None
    a = pr < lo + 0.35 * (hi - lo)
    b = pr > hi - 0.35 * (hi - lo)
    if a.sum() < 2 or b.sum() < 2:
        return None
    sa, sb = np.ptp(pe[a]), np.ptp(pe[b])
    if abs(sa - sb) < 1.5:
        return None
    tail_lo = sa < sb               # the narrow end is the tail
    p = m + ax * (lo if tail_lo else hi)
    return (x0 + p[0], y0 + p[1])


def _on_wire(it, P):
    """Plain net-name text sits on (or right above) a wire."""
    t = P.t
    for w in P.wires:
        if it.vert:
            if w.o != 'v':
                continue
            if it.x0 - 2 * t <= w.c <= it.x1 + 2 * t and min(w.b, it.y1) - max(w.a, it.y0) >= 0.5 * it.h:
                return True
        else:
            if w.o != 'h':
                continue
            if it.y0 - 2 * t <= w.c <= it.y1 + 2 * t and min(w.b, it.x1) - max(w.a, it.x0) >= 0.5 * it.w:
                return True
    return False


def _body_center_along(c, S, P, horiz):
    """Centre of the symbol body along the pin axis, ignoring straight stems."""
    x0, y0, x1, y1 = [int(round(v)) for v in S['groups'][c.gids[0]]['bbox']]
    m = np.isin(S['glab'][y0:y1 + 1, x0:x1 + 1], c.gids)
    if not m.any():
        return None
    if horiz:
        span = np.array([(np.ptp(np.nonzero(m[:, i])[0]) + 1) if m[:, i].any() else 0 for i in range(m.shape[1])])
    else:
        span = np.array([(np.ptp(np.nonzero(m[i, :])[0]) + 1) if m[i, :].any() else 0 for i in range(m.shape[0])])
    wide = np.nonzero(span > 1.6 * P.t + 1)[0]
    if len(wide) == 0:
        return None
    return (x0 if horiz else y0) + (wide[0] + wide[-1]) / 2.0


def standard_contact_tips(comps, S, P):
    """Capacitors / resistors touching another symbol directly: the touching point is
    at the standard pin length from the body (C: centre +- 2u, R: centre +- 2u)."""
    u = P.u
    for c in comps:
        if c.type not in ('c',) or len(c.pins) != 2:
            continue
        a, b = c.pins
        horiz = abs(a.tip[0] - b.tip[0]) > abs(a.tip[1] - b.tip[1])
        cen = _body_center_along(c, S, P, horiz)
        if cen is None:
            continue
        for pn in (a, b):
            if pn.kind not in ('contact', 'zero') or pn.term is not None:
                continue
            k = 0 if horiz else 1
            side = 1 if pn.tip[k] > cen else -1
            new = cen + side * 2 * u
            if abs(new - pn.tip[k]) > 0.6 * u:
                continue
            tip = list(pn.tip)
            tip[k] = new
            pn.tip = tuple(tip)
            m = getattr(pn, 'mate', None)
            if m is not None:
                mt = list(m.tip)
                mt[k] = new
                m.tip = tuple(mt)

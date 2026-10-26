"""Symbol segmentation: pins (straight red stubs) and bodies (everything else)."""
import os
import numpy as np
import cv2
from .geom import extract_segments, Seg


DEBUG = False
TOUCH_MERGE = False


class Pin:
    __slots__ = ('tip', 'base', 'o', 'seg', 'group', 'kind', 'id', 'length', 'dir', 'num', 'name', 'comp', 'net', 'term', 'mate', 'bar')

    def __init__(self, tip, base, o, kind, seg=None):
        self.tip = tip          # (x, y) image coords
        self.base = base        # attachment point on the body
        self.o = o              # 'h' | 'v' | 'z' (zero length)
        self.kind = kind        # 'wire' | 'free' | 'contact' | 'zero'
        self.seg = seg
        self.group = -1
        self.id = -1
        self.length = float(np.hypot(tip[0] - base[0], tip[1] - base[1]))
        d = np.array(tip, float) - np.array(base, float)
        n = np.linalg.norm(d)
        self.dir = tuple(d / n) if n > 0 else (0.0, 0.0)
        self.num = None
        self.name = None
        self.comp = None
        self.net = None
        self.term = None
        self.mate = None
        self.bar = False

    def __repr__(self):
        return f"Pin({self.kind},tip=({self.tip[0]:.0f},{self.tip[1]:.0f}),L={self.length:.0f},g={self.group})"


def _band_mask(shape, s, extra=1.0):
    m = np.zeros(shape, np.uint8)
    hw = s.t / 2 + extra
    if s.o == 'h':
        cv2.rectangle(m, (int(np.floor(s.a)), int(np.floor(s.c - hw))), (int(np.ceil(s.b)), int(np.ceil(s.c + hw))), 1, -1)
    else:
        cv2.rectangle(m, (int(np.floor(s.c - hw)), int(np.floor(s.a))), (int(np.ceil(s.c + hw)), int(np.ceil(s.b))), 1, -1)
    return m


def _end_status(red, s, end, t, H, W):
    """Count red ink attached at the segment end (excluding the segment's own band).
    Only ink 8-connected to the segment end inside a small window counts, so parallel
    neighbours (capacitor plates, GND bars, text) are ignored."""
    hw = s.t / 2 + 0.5
    r = int(np.ceil(t + 2))
    R = int(np.ceil(hw + r + 2))
    if s.o == 'h':
        x = s.b if end else s.a
        cx, cy = x, s.c
    else:
        y = s.b if end else s.a
        cx, cy = s.c, y
    x0, x1 = int(np.floor(cx - R)), int(np.ceil(cx + R))
    y0, y1 = int(np.floor(cy - R)), int(np.ceil(cy + R))
    X0, Y0 = max(x0, 0), max(y0, 0)
    win = red[Y0:min(y1 + 1, H), X0:min(x1 + 1, W)]
    if win.size == 0:
        return 0
    n, lab = cv2.connectedComponents(win.astype(np.uint8), connectivity=8)
    yy = np.arange(Y0, Y0 + win.shape[0])[:, None]
    xx = np.arange(X0, X0 + win.shape[1])[None, :]
    if s.o == 'h':
        sgn = 1 if end else -1
        along = (xx - cx) * sgn
        across = np.abs(yy - cy)
    else:
        sgn = 1 if end else -1
        along = (yy - cy) * sgn
        across = np.abs(xx - cx)
    own = (across <= hw + 0.6) & (along <= 0.5)
    endpix = own & (along >= -2) & (win > 0)
    labs = np.unique(lab[endpix])
    labs = labs[labs > 0]
    if len(labs) == 0:
        return 0
    conn = np.isin(lab, labs)
    valid = ((along > 0.5) & (along <= r)) | ((along <= 0.5) & (across > hw + 0.6) & (along >= -0.5 * r))
    valid &= (across <= hw + r)
    return int((conn & valid & ~own).sum())


def _point_on_seg(pt, s, tol):
    x, y = pt
    if s.o == 'h':
        return abs(y - s.c) <= tol and s.a - tol <= x <= s.b + tol
    return abs(x - s.c) <= tol and s.a - tol <= y <= s.b + tol


def _pin_color_base(P, s, tip, base, u, t):
    """EasyEDA draws pins in #880000 and symbol bodies in #A00000.  Walking from the tip,
    the pin ends where the stroke turns body-coloured (needs a clear colour step)."""
    img = getattr(P, 'img', None)
    if img is None:
        return base
    L = abs(base[0] - tip[0]) + abs(base[1] - tip[1])
    if L < 1.3 * u:
        return base
    sx = np.sign(base[0] - tip[0]); sy = np.sign(base[1] - tip[1])
    c = int(round(s.c))
    H, W = img.shape[:2]
    vals = []
    for k in range(int(L) + 1):
        x = int(round(tip[0] + sx * k)); y = int(round(tip[1] + sy * k))
        if not (0 <= x < W and 0 <= y < H):
            vals.append(None); continue
        win = img[max(y - 1, 0):y + 2, x:x + 1] if s.o == 'h' else img[y:y + 1, max(x - 1, 0):x + 2]
        win = win.reshape(-1, 3).astype(int)
        core = win[(win[:, 1] < 40) & (win[:, 2] < 40)]
        vals.append(int(core[:, 0].max()) if len(core) else None)
    def kind(v):
        return None if v is None else ('pin' if 120 <= v <= 146 else ('body' if 152 <= v <= 175 else None))
    ks = [kind(v) for v in vals]
    start = int(0.25 * u)
    pin_n = sum(1 for k in ks[start:int(0.8 * u)] if k == 'pin')
    if pin_n < 0.3 * u:
        return base
    for k in range(int(0.5 * u), len(ks) - 2):
        run = ks[k:k + max(3, int(0.3 * u))]
        if all(x == 'body' for x in run):
            return (tip[0] + sx * k, tip[1] + sy * k)
    return base


def segment_symbols(P, terms, lines):
    """Find pins and body groups.

    terms: wire terminals from wiregraph (dicts with 'pt','dir','line').
    Returns dict with pins, groups (list of dict with mask bbox), group label image.
    """
    red = P.red
    H, W = red.shape
    t = P.t
    u = P.u
    rs = extract_segments(red, min_len=max(4, int(0.28 * u)), bridge=red, max_gap=1, kind='r')
    rs = [s for s in rs if s.t <= 2.2 * t + 1]
    for i, s in enumerate(rs):
        s.id = i
    # terminal lookup
    term_pts = [(tm['pt'], tm) for tm in terms]
    gray = P.gray_dots + list(getattr(P, 'noconn', []))

    def near_term(pt, tol):
        best = None
        for p, tm in term_pts:
            d = np.hypot(p[0] - pt[0], p[1] - pt[1])
            if d <= tol and (best is None or d < best[0]):
                best = (d, tm)
        return best

    def near_dot(pt, sg):
        for (x, y, r) in P.dots:
            perp = abs(y - sg.c) if sg.o == 'h' else abs(x - sg.c)
            if perp > max(t, 0.5 * r) + 1:
                continue
            if np.hypot(x - pt[0], y - pt[1]) <= r + 1.5 * t + 1:
                # project dot centre onto the segment axis
                return (x, sg.c) if sg.o == 'h' else (sg.c, y)
        return None

    cdots = getattr(P, 'contact_dots', [])

    def near_cdot(pt):
        for (x, y, o) in cdots:
            if np.hypot(x - pt[0], y - pt[1]) <= 1.5 * t + 3:
                return (x, y)
        return None

    free_th = max(3, int(1.5 * t))
    spans = []
    ends = []
    for s0 in rs:
        e0 = _end_status(red, s0, 0, t, H, W)
        e1 = _end_status(red, s0, 1, t, H, W)
        sps = split_spans(red, s0, t, u)
        for sp, a0, a1, ok0, ok1 in sps:
            sp.id = len(spans)
            sp.kind = (ok0, ok1, len(sps) == 1, getattr(s0, 'start_both', False) or getattr(s0, 'end_both', False))
            spans.append(sp)
            ends.append([e0 if not a0 else 10 ** 6, e1 if not a1 else 10 ** 6])
    rs = spans
    pins = []
    pin_seg_ids = set()
    link_ids = []
    for s, (e0, e1) in zip(rs, ends):
        L = s.length
        f0, f1 = e0 <= free_th, e1 <= free_th
        p0, p1 = s.p0(), s.p1()
        w0 = near_term(p0, 1.8 * t + 2)
        w1 = near_term(p1, 1.8 * t + 2)
        d0 = near_dot(p0, s)
        d1 = near_dot(p1, s)
        if w0 is not None or d0 is not None:
            f0 = True
        if w1 is not None or d1 is not None:
            f1 = True
        c0, c1 = near_cdot(p0), near_cdot(p1)
        if c0 and not f1:
            f0 = True
        if c1 and not f0:
            f1 = True
        base_ok = s.kind[1] if f0 else s.kind[0]
        w_any = (w0 if f0 else w1) is not None or (d0 if f0 else d1) is not None
        need = 1 if w_any else 2
        if (c0 if f0 else c1) and f0 != f1:
            need = 0             # explicit pin-contact mark: the free end is certain
        if (FREE_BAR_PINS and f0 != f1 and base_ok == 0 and need == 2 and 1.5 * u <= L <= 4.5 * u
                and s.kind[2] is False):
            # the far half of a switch's pin bar: the lever lead joins the bar from one side
            # in the middle, the bar end is an unconnected pin
            need = -1
        if f0 != f1 and 0.3 * u <= L <= 4.5 * u and base_ok >= need:
            tip, base = (p0, p1) if f0 else (p1, p0)
            w = w0 if f0 else w1
            kind = 'wire' if w is not None else 'free'
            dd = d0 if f0 else d1
            if w is not None:
                tip = w[1]['pt']
            elif dd is not None:
                tip = dd
                kind = 'dot'
            if need <= 0:
                base = _pin_color_base(P, s, tip, base, u, t)
            pn = Pin(tuple(tip), tuple(base), s.o, kind, seg=s)
            pn.bar = need < 0
            if w is not None:
                pn.term = w[1]
            pins.append(pn)
            pin_seg_ids.add(s.id)
        elif (not f0) and (not f1) and 0.8 * u <= L <= 4.0 * u and s.kind[2] and s.kind[3]:
            link_ids.append(s.id)
    # body mask = red minus pin stubs
    body = red.copy()
    for pn in pins:
        s = pn.seg
        # remove the stub but stop ~1 stroke width before the base so the body
        # outline it attaches to stays intact
        bx, by = pn.base
        keep = 1.2 * t + 1
        if s.o == 'h':
            a_, b_ = (s.a, s.b - keep) if bx >= s.b - 1 else (s.a + keep, s.b)
            if s.a + 1 < bx < s.b - 1:          # base inside the segment (colour-derived)
                a_, b_ = (s.a, bx - keep) if pn.tip[0] < bx else (bx + keep, s.b)
        else:
            a_, b_ = (s.a, s.b - keep) if by >= s.b - 1 else (s.a + keep, s.b)
            if s.a + 1 < by < s.b - 1:
                a_, b_ = (s.a, by - keep) if pn.tip[1] < by else (by + keep, s.b)
        if b_ <= a_:
            continue
        m = _band_mask(red.shape, Seg(s.o, s.c, a_, b_, s.t), extra=1.0)
        body[m > 0] = 0
    rad = max(2, int(round(0.2 * u)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))

    def grouping(mask):
        dil = cv2.dilate(mask, k)
        n_, gl = cv2.connectedComponents(dil, connectivity=8)
        return gl * (mask > 0)

    glab = grouping(body)

    def group_at(pt, r=None, lab=None):
        lab = glab if lab is None else lab
        r = r if r is not None else int(np.ceil(t + 2))
        x, y = int(round(pt[0])), int(round(pt[1]))
        win = lab[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        v = win[win > 0]
        if len(v) == 0:
            return 0
        return int(np.bincount(v).argmax())

    # ---- link candidates: straight red lines attached at both ends.
    # They are either body edges (rectangle sides, internal strokes) or pins of two
    # symbols touching each other directly.  Rectangle sides are recognised by a
    # parallel partner closed by perpendicular strokes; the rest are removed together
    # and those whose ends then fall into different groups become contacts.
    def side_cov(x0_, y0_, x1_, y1_):
        """fraction of red along an axis-aligned line from (x0,y0) to (x1,y1)"""
        n = int(max(abs(x1_ - x0_), abs(y1_ - y0_))) + 1
        xs_ = np.linspace(x0_, x1_, n).round().astype(int)
        ys_ = np.linspace(y0_, y1_, n).round().astype(int)
        ok = 0
        for xx, yy in zip(xs_, ys_):
            win = red[max(yy - 2, 0):yy + 3, max(xx - 2, 0):xx + 3]
            ok += win.any()
        return ok / max(n, 1)

    rect_side = set()
    cand = [rs[i] for i in link_ids]
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a_, b_ = cand[i], cand[j]
            if a_.o != b_.o:
                continue
            if abs(a_.a - b_.a) > 2 * t + 2 or abs(a_.b - b_.b) > 2 * t + 2:
                continue
            off = abs(a_.c - b_.c)
            if off < 0.25 * u or off > 60 * u:
                continue
            lo, hi = sorted([a_.c, b_.c])
            if a_.o == 'h':
                c1 = side_cov(a_.a, lo, a_.a, hi)
                c2 = side_cov(a_.b, lo, a_.b, hi)
            else:
                c1 = side_cov(lo, a_.a, hi, a_.a)
                c2 = side_cov(lo, a_.b, hi, a_.b)
            if c1 > 0.85 and c2 > 0.85:
                rect_side.add(a_.id)
                rect_side.add(b_.id)
    links = [rs[i] for i in link_ids if i not in rect_side]
    if DEBUG:
        print('LINKCAND', [(rs[i], round(rs[i].length / u, 2), i in rect_side) for i in link_ids])
    masks_ = {}
    cut = body.copy()
    for s_ in links:
        m = _band_mask(red.shape, s_, extra=0.8)
        for pt in (s_.p0(), s_.p1()):
            cv2.circle(m, (int(round(pt[0])), int(round(pt[1]))), int(np.ceil(t + 1)), 0, -1)
        masks_[s_.id] = m
        cut[m > 0] = 0
    _n2, glab2 = cv2.connectedComponents(cv2.dilate(cut, np.ones((3, 3), np.uint8)), connectivity=8)
    glab2 = glab2 * (cut > 0)
    # tiny remnants (e.g. the corner where two links meet) do not count as bodies
    areas = np.bincount(glab2.ravel(), minlength=_n2)
    tiny = areas < max(12, (2.5 * t) ** 2)
    tiny[0] = False
    glab2[tiny[glab2]] = 0
    contacts = []
    half = []
    for s_ in links:
        e0 = group_at(s_.p0(), lab=glab2)
        e1 = group_at(s_.p1(), lab=glab2)
        if e0 and e1 and e0 != e1:
            contacts.append(s_)
            if DEBUG: print('CONTACT', s_, round(s_.length / u, 2))
        elif bool(e0) != bool(e1):
            half.append((s_, 0 if e0 else 1))
    body_final = body
    if contacts or half:
        cut2 = body.copy()
        for s_ in contacts:
            cut2[masks_[s_.id] > 0] = 0
        for s_, _ in half:
            cut2[masks_[s_.id] > 0] = 0
        glab = grouping(cut2)
        body_final = cut2
        for s_ in contacts:
            g0 = group_at(s_.p0())
            g1 = group_at(s_.p1())
            if not g0 or not g1 or g0 == g1:
                continue
            mid = ((s_.p0()[0] + s_.p1()[0]) / 2, (s_.p0()[1] + s_.p1()[1]) / 2)
            pa = Pin(mid, s_.p0(), s_.o, 'contact', seg=s_)
            pb = Pin(mid, s_.p1(), s_.o, 'contact', seg=s_)
            pa.group, pb.group = g0, g1
            pa.mate = pb
            pb.mate = pa
            pins += [pa, pb]
        for s_, base_end in half:
            base = s_.p0() if base_end == 0 else s_.p1()
            tip = s_.p1() if base_end == 0 else s_.p0()
            g = group_at(base)
            if not g:
                continue
            pn = Pin(tuple(tip), tuple(base), s_.o, 'contact', seg=s_)
            pn.group = g
            pins.append(pn)
    # ---- free pins meeting each other (pin-to-pin contact with a tiny gap) ----
    frees = [pn for pn in pins if pn.kind == 'free']
    for i in range(len(frees)):
        for j in range(i + 1, len(frees)):
            a_, b_ = frees[i], frees[j]
            if a_.o != b_.o:
                continue
            if a_.dir[0] * b_.dir[0] + a_.dir[1] * b_.dir[1] > -0.9:
                continue
            if np.hypot(a_.tip[0] - b_.tip[0], a_.tip[1] - b_.tip[1]) <= 1.5 * t + 3:
                mid = ((a_.tip[0] + b_.tip[0]) / 2, (a_.tip[1] + b_.tip[1]) / 2)
                a_.tip = mid
                b_.tip = mid
                a_.kind = b_.kind = 'contact'
                a_.mate, b_.mate = b_, a_
    # ---- anchored regrouping -------------------------------------------------
    # pieces = connected parts of the body; pieces carrying pins are anchors.
    pieces_n, pieces = cv2.connectedComponents(cv2.dilate(body_final, np.ones((3, 3), np.uint8)), connectivity=8)
    pieces = pieces * (body_final > 0)

    def piece_at(pt, r):
        return group_at(pt, r, lab=pieces)

    for pn in pins:
        pn.group = piece_at(pn.base, int(np.ceil(1.5 * t + 2)))
    used_terms = set(id(pn.term) for pn in pins if pn.term is not None)
    for p, tm in term_pts:
        if id(tm) in used_terms:
            continue
        if any(np.hypot(pn.tip[0] - p[0], pn.tip[1] - p[1]) <= 1.5 * t + 2 for pn in pins):
            continue
        g = piece_at(p, int(np.ceil(1.2 * t + 2)))
        if g:
            pn = Pin(tuple(p), tuple(p), 'z', 'zero')
            pn.group = g
            pn.term = tm
            pins.append(pn)
    def bar_through(g, tip, o):
        # does piece g contain a straight stroke through tip perpendicular to (or along) the pin?
        x, y = int(round(tip[0])), int(round(tip[1]))
        r = int(np.ceil(0.5 * u))
        w = int(np.ceil(t + 1))
        if o == 'h':
            perp = pieces[max(y - r, 0):y + r + 1, max(x - w, 0):x + w + 1] == g
            along = pieces[max(y - w, 0):y + w + 1, max(x - r, 0):x + r + 1] == g
            return perp.any(1).sum() >= 0.6 * r or along.any(0).sum() >= 0.8 * r
        perp = pieces[max(y - w, 0):y + w + 1, max(x - r, 0):x + r + 1] == g
        along = pieces[max(y - r, 0):y + r + 1, max(x - w, 0):x + w + 1] == g
        return perp.any(0).sum() >= 0.6 * r or along.any(1).sum() >= 0.8 * r

    for pn in [p_ for p_ in pins if p_.kind == 'free']:
        g = piece_at(pn.tip, int(np.ceil(1.0 * t + 2)))
        if g and g != pn.group and bar_through(g, pn.tip, pn.o):
            z = Pin(tuple(pn.tip), tuple(pn.tip), 'z', 'contact')
            z.group = g
            pn.kind = 'contact'
            z.mate, pn.mate = pn, z
            pins.append(z)
    # junction dots touching a body directly (no pin stub): zero-length pin at the dot
    for (x, y, r) in P.dots:
        if any(np.hypot(pn.tip[0] - x, pn.tip[1] - y) <= r + t + 1 for pn in pins):
            continue
        R = int(np.ceil(r + 1.5 * t + 1))
        xi, yi = int(round(x)), int(round(y))
        win = pieces[max(yi - R, 0):yi + R + 1, max(xi - R, 0):xi + R + 1]
        labs = [l for l in np.unique(win) if l > 0]
        for l in labs:
            area = int((pieces == l).sum()) if len(labs) <= 3 else 0
            if area < 3 * t * t:
                continue
            pn = Pin((float(x), float(y)), (float(x), float(y)), 'z', 'zero')
            pn.group = int(l)
            pins.append(pn)
    markers = list(P.gray_dots) + list(getattr(P, 'noconn', []))
    strong = [pn for pn in pins if pn.kind != 'free' or any(np.hypot(mx - pn.tip[0], my - pn.tip[1]) <= 1.8 * t + 2 for mx, my in markers)]
    glab = anchored_grouping(pieces, pieces_n, strong, u, t)
    pmap = np.zeros(int(pieces.max()) + 1, np.int64)
    sel = pieces > 0
    pmap[pieces[sel]] = glab[sel]
    for pn in pins:
        if pn.group > 0:
            pn.group = int(pmap[pn.group])
    for i, pn in enumerate(pins):
        pn.id = i
    groups = {}
    ys, xs = np.nonzero(glab)
    labs = glab[ys, xs]
    order = np.argsort(labs)
    labs, ys, xs = labs[order], ys[order], xs[order]
    if len(labs):
        cuts = np.flatnonzero(np.diff(labs)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(labs)]):
            g = int(labs[a])
            groups[g] = {'id': g, 'bbox': [int(xs[a:b].min()), int(ys[a:b].min()), int(xs[a:b].max()), int(ys[a:b].max())],
                         'area': int(b - a), 'pins': []}
    for pn in pins:
        if pn.group in groups:
            groups[pn.group]['pins'].append(pn)
    return {'pins': pins, 'groups': groups, 'glab': glab, 'rsegs': rs, 'body': body, 'pieces': pieces}


def _contact_end(red, s, t):
    """For a ~1u red link between two bodies decide which end is the touching point.
    The tip end is the one meeting a perpendicular bar at (roughly) its middle."""
    H, W = red.shape
    best = 0
    scores = []
    for e, pt in enumerate((s.p0(), s.p1())):
        x, y = int(round(pt[0])), int(round(pt[1]))
        r = int(4 * t + 4)
        if s.o == 'h':
            col = red[max(y - r, 0):y + r + 1, min(max(x + (1 if e else -1) * int(t / 2), 0), W - 1)]
            sc = col.sum()
        else:
            row = red[min(max(y + (1 if e else -1) * int(t / 2), 0), H - 1), max(x - r, 0):x + r + 1]
            sc = row.sum()
        scores.append(sc)
    return int(np.argmax(scores))


def estimate_unit(P, S):
    """Estimate the EasyEDA pin-length unit u (px) from pin pitch and stub lengths."""
    u0 = P.u
    vals = []   # (value, weight, allowed multiples)
    by = {}
    for pn in S['pins']:
        if pn.kind not in ('wire', 'free') or pn.o not in ('h', 'v'):
            continue
        key = (pn.group, pn.o, round(pn.base[0] if pn.o == 'h' else pn.base[1]) // 3)
        by.setdefault(key, []).append(pn.tip[1] if pn.o == 'h' else pn.tip[0])
        vals.append((pn.length - 0.45 * P.t, 0.5, (1, 2)))
    for k, arr in by.items():
        arr = sorted(arr)
        for a, b in zip(arr, arr[1:]):
            d = b - a
            if d > 0.5 * u0:
                vals.append((d, 1.0, (1, 2, 3, 4)))
    if len(vals) < 3:
        return u0
    def score(u):
        sc = 0.0
        for v, w, mult in vals:
            k = round(v / u)
            if k not in mult:
                continue
            err = v - k * u
            sc += w * np.exp(-err * err / (2 * 1.2 ** 2))
        return sc
    cands = np.arange(0.7 * u0, 1.45 * u0, 0.05)
    best = None
    for u in cands:
        sc = score(u)
        if best is None or sc > best[0] + 1e-9:
            best = (sc, u)
    UNIT_LOG[0] = None
    if UNIT_WIDE:
        # thick strokes make the stroke-based first guess u0 about twice too large (official
        # training renders: t ~ 5.8 px at u ~ 20 px); the search above cannot reach u0 / 2.
        # Look there too, and take it only when it explains the pins much better.
        low = None
        for u in np.arange(0.4 * u0, 0.7 * u0, 0.05):
            sc = score(u)
            if low is None or sc > low[0] + 1e-9:
                low = (sc, u)
        UNIT_LOG[0] = (best, low)
        if low is not None and low[0] > UNIT_WIDE_GAIN * best[0] and low[0] >= 20.0:
            best = low
    return float(best[1])


GLYPH_CUTS = os.environ.get('PCB_GLYPH_CUTS', '1') == '1'
UNIT_WIDE = os.environ.get('PCB_UNIT_WIDE', '1') == '1'   # unit search down to 0.4 u0 (thick-stroke images)
UNIT_WIDE_GAIN = float(os.environ.get('PCB_UNIT_WIDE_GAIN', '2.5'))
UNIT_LOG = [None]
FREE_BAR_PINS = os.environ.get('PCB_FREE_BAR', '0') == '1'   # no score effect (switch pins are not scored); kept off


def _glyph_blob(red, s, p0, p1, up_side, u):
    """Is the ink attached to one side of segment s between p0..p1 a small isolated blob
    (a pin-number glyph touching its pin line) rather than a stroke of the symbol?"""
    H, W = red.shape
    R = int(np.ceil(1.2 * u))
    hw = s.t / 2.0 + 1.0
    if s.o == 'h':
        x0, x1 = max(int(p0) - R, 0), min(int(p1) + R + 1, W)
        y0, y1 = max(int(s.c - hw - R), 0), min(int(s.c + hw + R) + 1, H)
        win = (red[y0:y1, x0:x1] > 0).astype(np.uint8)
        yy = np.arange(y0, y1)[:, None]
        band = np.abs(yy - s.c) <= hw
        win[np.broadcast_to(band, win.shape)] = 0
        n, lab = cv2.connectedComponents(win, connectivity=8)
        off = s.t / 2.0 + 1.5
        r = int(np.floor(s.c - off)) - 1 - y0 if up_side else int(np.ceil(s.c + off)) - y0
        r = min(max(r, 0), win.shape[0] - 2)
        seeds = lab[r:r + 2, max(int(p0) - x0 - 1, 0):int(p1) - x0 + 2]
    else:
        y0, y1 = max(int(p0) - R, 0), min(int(p1) + R + 1, H)
        x0, x1 = max(int(s.c - hw - R), 0), min(int(s.c + hw + R) + 1, W)
        win = (red[y0:y1, x0:x1] > 0).astype(np.uint8)
        xx = np.arange(x0, x1)[None, :]
        band = np.abs(xx - s.c) <= hw
        win[np.broadcast_to(band, win.shape)] = 0
        n, lab = cv2.connectedComponents(win, connectivity=8)
        off = s.t / 2.0 + 1.5
        cc = int(np.floor(s.c - off)) - 1 - x0 if up_side else int(np.ceil(s.c + off)) - x0
        cc = min(max(cc, 0), win.shape[1] - 2)
        seeds = lab[max(int(p0) - y0 - 1, 0):int(p1) - y0 + 2, cc:cc + 2]
    labs = [l for l in np.unique(seeds) if l > 0]
    if not labs:
        return False
    m = np.isin(lab, labs)
    ys, xs = np.nonzero(m)
    if ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1:
        return False                       # runs out of the window: a real stroke
    ext = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    if ext <= max(2, 0.6 * s.t):
        return True                        # an anti-aliasing speck of nearby text
    if ext > 0.9 * u:
        return False
    # a glyph is not a straight bar (a short perpendicular stroke would be)
    return min(ys.max() - ys.min(), xs.max() - xs.min()) + 1 >= 0.25 * u


def split_spans(red, s, t, u=None):
    """Split a red segment at points where other strokes attach to its sides.
    Returns [(span_seg, start_attached, end_attached, start_cut_ok, end_cut_ok)] where
    *_cut_ok says whether a cut at that end may serve as a pin base (the attaching
    structure crosses on both sides or is a wide/filled shape)."""
    H, W = red.shape
    hw = s.t / 2.0
    a, b = int(round(s.a)), int(round(s.b))
    if b - a < 3:
        return [(s, False, False, 2, 2)]
    pos = np.arange(a, b + 1)
    off = hw + 1.5
    if s.o == 'h':
        ya = int(np.floor(s.c - off)); yb = int(np.ceil(s.c + off))
        xs = np.clip(pos, 0, W - 1)
        up = np.zeros(len(pos), bool); dn = np.zeros(len(pos), bool)
        for y in (ya - 1, ya):
            if 0 <= y < H:
                up |= red[y, xs] > 0
        for y in (yb, yb + 1):
            if 0 <= y < H:
                dn |= red[y, xs] > 0
    else:
        xa = int(np.floor(s.c - off)); xb = int(np.ceil(s.c + off))
        ys = np.clip(pos, 0, H - 1)
        up = np.zeros(len(pos), bool); dn = np.zeros(len(pos), bool)
        for x in (xa - 1, xa):
            if 0 <= x < W:
                up |= red[ys, x] > 0
        for x in (xb, xb + 1):
            if 0 <= x < W:
                dn |= red[ys, x] > 0
    att = up | dn
    d = np.diff(np.r_[0, att.astype(np.int8), 0])
    st_ = np.where(d == 1)[0]
    en_ = np.where(d == -1)[0] - 1
    edge = max(2, t * 1.2)
    cuts = []
    start_att = end_att = False
    s.start_both = s.end_both = False
    for r0, r1 in zip(st_, en_):
        p0, p1 = pos[r0], pos[r1]
        both = bool(up[r0:r1 + 1].any() and dn[r0:r1 + 1].any())
        wide = (p1 - p0) >= 2.2 * t
        if GLYPH_CUTS and u is not None and not both and not wide and p0 - a > edge and b - p1 > edge \
                and _glyph_blob(red, s, p0, p1, bool(up[r0:r1 + 1].any()), u):
            continue                       # a pin-number glyph touching its pin line
        if p0 - a <= edge:
            s.start_both = s.start_both or both
            start_att = True
            if p1 - a > edge:
                cuts.append((p1, p1, both, wide))
            continue
        if b - p1 <= edge:
            s.end_both = s.end_both or both
            end_att = True
            if b - p0 > edge:
                cuts.append((p0, p0, both, wide))
            continue
        cuts.append((p0, p1, both, wide))
    if not cuts:
        return [(s, start_att, end_att, 2, 2)]
    out = []
    cur = a
    cur_att = start_att
    cur_ok = 2
    for p0, p1, both, wide in cuts:
        narrow = p1 - p0 < 2 * t
        lo = (p0 + p1) / 2.0 if narrow else p0
        hi = (p0 + p1) / 2.0 if narrow else p1
        ok = 2 if wide else (1 if both else 0)
        if lo - cur >= 2:
            out.append((Seg(s.o, s.c, float(cur), float(lo), s.t, s.kind), cur_att, True, cur_ok, ok))
        cur = hi
        cur_att = True
        cur_ok = ok
    if b - cur >= 2:
        out.append((Seg(s.o, s.c, float(cur), float(b), s.t, s.kind), cur_att, end_att, cur_ok, 2))
    return out


def anchored_grouping(pieces, n, pins, u, t):
    """Group body pieces: unanchored pieces join the nearest anchored piece (within
    0.35u); anchored pieces merge only when they form a facing pair (capacitor plates)
    or nearly touch."""
    H, W = pieces.shape
    anchors = set(pn.group for pn in pins if pn.group > 0)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    if n <= 1:
        return pieces
    # bounding boxes of pieces
    ys, xs = np.nonzero(pieces)
    labs = pieces[ys, xs]
    bb = np.zeros((n, 4), np.int32)
    bb[:, 0] = W; bb[:, 1] = H; bb[:, 2] = -1; bb[:, 3] = -1
    np.minimum.at(bb[:, 0], labs, xs); np.minimum.at(bb[:, 1], labs, ys)
    np.maximum.at(bb[:, 2], labs, xs); np.maximum.at(bb[:, 3], labs, ys)
    # anchored-anchored facing pairs (capacitor plates etc.)
    pins_by = {}
    for pn in pins:
        if pn.group > 0:
            pins_by.setdefault(pn.group, []).append(pn)
    al = sorted(anchors)
    for i in range(len(al)):
        for j in range(i + 1, len(al)):
            a, b = al[i], al[j]
            gx = max(bb[a, 0] - bb[b, 2], bb[b, 0] - bb[a, 2], 0)
            gy = max(bb[a, 1] - bb[b, 3], bb[b, 1] - bb[a, 3], 0)
            gap = np.hypot(gx, gy)
            if gap > 0.6 * u:
                continue
            gap = _piece_gap(pieces, a, b, bb, 0.6 * u)
            if gap > 0.6 * u:
                continue
            if gap <= 1.0 and TOUCH_MERGE:
                union(a, b)
                continue
            ok = False
            for pa in pins_by.get(a, []):
                for pb in pins_by.get(b, []):
                    if pa.o != pb.o or pa.o not in ('h', 'v'):
                        continue
                    if pa.dir[0] * pb.dir[0] + pa.dir[1] * pb.dir[1] > -0.9:
                        continue
                    if pa.o == 'h' and abs(pa.tip[1] - pb.tip[1]) <= 1.5 * t and \
                            (pa.tip[0] - pb.tip[0]) * pa.dir[0] > 0:
                        ok = True
                    if pa.o == 'v' and abs(pa.tip[0] - pb.tip[0]) <= 1.5 * t and \
                            (pa.tip[1] - pb.tip[1]) * pa.dir[1] > 0:
                        ok = True
            if ok:
                union(a, b)
    # unanchored -> nearest anchored piece
    amask = np.isin(pieces, list(anchors)) if anchors else np.zeros_like(pieces, bool)
    if amask.any():
        inv = (~amask).astype(np.uint8)
        dist, lbl = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        # map voronoi label -> piece id of the seed pixel
        seed_ys, seed_xs = np.nonzero(amask == True)
        lab_of_seed = np.zeros(int(lbl.max()) + 1, np.int32)
        lab_of_seed[lbl[seed_ys, seed_xs]] = pieces[seed_ys, seed_xs]
        for pc in range(1, n):
            if pc in anchors or bb[pc, 2] < 0:
                continue
            m = pieces == pc
            dd = dist[m]
            if dd.size == 0:
                continue
            k = int(np.argmin(dd))
            if dd[k] <= 0.6 * u:
                target = lab_of_seed[lbl[m][k]]
                if target > 0:
                    union(target, pc)
    # remaining unanchored pieces: cluster by proximity among themselves
    rad = max(1, int(round(0.15 * u)))
    rest = np.zeros_like(pieces, np.uint8)
    for pc in range(1, n):
        if find(pc) == pc and pc not in anchors and bb[pc, 2] >= 0:
            # only pieces not merged to an anchor
            pass
    roots = np.array([find(i) for i in range(n)])
    lab = roots[pieces] * (pieces > 0)
    unanch_roots = set(r for r in range(1, n) if roots[r] == r and r not in anchors and not any(find(a) == r for a in anchors))
    if unanch_roots:
        um = np.isin(lab, list(unanch_roots)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        nn, cl = cv2.connectedComponents(cv2.dilate(um, k), connectivity=8)
        cl = cl * (um > 0)
        off = n + 1
        lab = np.where(um > 0, cl + off, lab)
    return lab


def _piece_gap(pieces, a, b, bb, maxd):
    """Exact minimum pixel distance between two labelled pieces (capped)."""
    H, W = pieces.shape
    m = int(np.ceil(maxd)) + 2
    x0 = max(min(bb[a, 0], bb[b, 0]) - m, 0)
    y0 = max(min(bb[a, 1], bb[b, 1]) - m, 0)
    x1 = min(max(bb[a, 2], bb[b, 2]) + m, W - 1)
    y1 = min(max(bb[a, 3], bb[b, 3]) + m, H - 1)
    roi = pieces[y0:y1 + 1, x0:x1 + 1]
    ma = (roi == a)
    mb = (roi == b)
    if not ma.any() or not mb.any():
        return 1e9
    dt = cv2.distanceTransform((~ma).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dt[mb].min())

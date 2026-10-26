"""Text glyph grouping into text items + OCR."""
import numpy as np
import cv2
import os
import re
from .ocr import get_rec

UNDERSCORE_JOIN = os.environ.get('PCB_USC', '1') == '1'
UNDERSCORE_VERT = os.environ.get('PCB_USC_V', '1') == '1'
TALL_VERT = os.environ.get('PCB_TALL_VERT', '1') == '1'
STRIP_RUNS = os.environ.get('PCB_STRIP_RUNS', '1') == '1'
RED_STRIKE = os.environ.get('PCB_RED_STRIKE', '0') == '1'   # off: fills IC pin lines between stacked labels (case_0049 -22)
GROUP_SUBSTANTIAL = os.environ.get('PCB_GROUP_SUB', '1') == '1'


class TextItem:
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'vert', 'color', 'text', 'conf', 'glyphs', 'id', 'used', 'alts', 'probs', 'rot', 'alt', 'io_alt', 'big',
                 'shade')

    def __init__(self, x0, y0, x1, y1, vert, color, glyphs):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.vert = vert
        self.color = color
        self.glyphs = glyphs
        self.text = ''
        self.conf = 0.0
        self.id = -1
        self.used = False
        self.alts = []
        self.alt = None
        self.io_alt = None
        self.big = False
        self.probs = None
        self.rot = 'h'
        self.shade = None

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def w(self):
        return self.x1 - self.x0 + 1

    @property
    def h(self):
        return self.y1 - self.y0 + 1

    def __repr__(self):
        return f"T({self.text!r},{self.color},{'V' if self.vert else 'H'},[{self.x0},{self.y0},{self.x1},{self.y1}],{self.conf:.2f})"


def _glyphs(mask, max_h, min_area=2):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        if h > max_h and w > max_h:
            continue
        if max(w, h) > 4 * max_h:
            continue
        out.append([int(x), int(y), int(x + w - 1), int(y + h - 1), int(a)])
    return out


def _group(glyphs, ch, vert):
    """Group glyph boxes into lines. ch = typical char height (px)."""
    n = len(glyphs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    g = np.array(glyphs, float) if n else np.zeros((0, 5))
    if vert:
        # swap axes: treat y as reading direction
        g = g[:, [1, 0, 3, 2, 4]]
    order = np.argsort(g[:, 0]) if n else []
    ln_ = g[:, 2] - g[:, 0] + 1 if n else []
    cr_ = g[:, 3] - g[:, 1] + 1 if n else []
    # underscore-like glyphs: thin bars along the reading direction.  They sit below the
    # baseline, so the usual cross-axis centre test would split 'MDI_TP_P0' into 'MDI' +
    # 'TP_P0' (the '_' touches neither neighbour).
    usc = [UNDERSCORE_JOIN and cr_[k] <= max(2.5, 0.3 * ch) and 0.35 * ch <= ln_[k] <= 1.3 * ch
           and ln_[k] >= 2.5 * cr_[k] for k in range(n)]
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if g[j, 0] > g[i, 2] + 0.75 * ch:
                break
            # overlap in cross axis
            ov = min(g[i, 3], g[j, 3]) - max(g[i, 1], g[j, 1])
            hi_ = min(g[i, 3] - g[i, 1], g[j, 3] - g[j, 1]) + 1
            gap = g[j, 0] - g[i, 2]
            c1 = (g[i, 1] + g[i, 3]) / 2
            c2 = (g[j, 1] + g[j, 3]) / 2
            if usc[i] != usc[j]:
                # underscore next to a full glyph: it must hug the glyph's baseline side
                # (below it for horizontal text), touching or within a pixel or two
                us, gg = (i, j) if usc[i] else (j, i)
                full = cr_[gg]
                # an underscore continues the line: it does not lie under its neighbour (the
                # bar of a rotated 'I' does)
                ov_read = min(g[i, 2], g[j, 2]) - max(g[i, 0], g[j, 0]) + 1
                if full >= 0.6 * ch and gap <= 0.35 * ch and ov_read <= 0.35 * min(ln_[i], ln_[j]):
                    below = g[us, 1] >= g[gg, 3] - max(2.0, 0.2 * ch) and g[us, 1] <= g[gg, 3] + max(3.0, 0.45 * ch)
                    above = g[us, 3] <= g[gg, 1] + max(2.0, 0.2 * ch) and g[us, 3] >= g[gg, 1] - max(3.0, 0.45 * ch)
                    # vertical text reads bottom-to-top: the baseline (and '_') is on the right,
                    # i.e. 'below' in swapped axes.  An upright 'I' of horizontal text overlaps its
                    # neighbour along y (ov_read) and is rejected above.
                    if below and (not vert or UNDERSCORE_VERT):
                        parent[find(j)] = find(i)
                        continue
            # allow small glyphs (., -, ') whose cross extent is small but inside the line band
            if ov >= 0.45 * hi_ or (ov >= -1 and hi_ < 0.45 * ch):
                # cross-axis centre consistency
                lim_ = 0.62 * ch
                if vert and VERT_LOCAL_GAP:
                    # a column of short horizontal numbers (pin numbers one pitch apart) is no
                    # vertical word: rotated letters are a cap height across, digits of
                    # horizontal text only a digit width
                    lim_ = min(lim_, max(0.8 * max(ln_[i], ln_[j]), 0.35 * max(cr_[i], cr_[j])))
                if gap <= lim_ and abs(c1 - c2) <= 0.5 * ch:
                    parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _strike_fill(P, lab):
    """A thin red symbol line (a pin row of an overlapping part) running through blue text
    cuts every letter in two.  Relabel red pixels that have blue text right above and
    below (or left and right) as blue, for text extraction only."""
    blue = (lab == 3).astype(np.uint8)
    red = (lab == 1)
    if not blue.any() or not red.any():
        return lab
    k = int(round(P.t + 3))
    up = cv2.dilate(blue, np.ones((k, 1), np.uint8), anchor=(0, k - 1))       # blue somewhere above
    dn = cv2.dilate(blue, np.ones((k, 1), np.uint8), anchor=(0, 0))           # blue somewhere below
    lf = cv2.dilate(blue, np.ones((1, k), np.uint8), anchor=(k - 1, 0))
    rt = cv2.dilate(blue, np.ones((1, k), np.uint8), anchor=(0, 0))
    fill = red & (((up > 0) & (dn > 0)) | ((lf > 0) & (rt > 0)))
    if not fill.any():
        return lab
    out = lab.copy()
    out[fill] = 3
    return out


COLOR_SPLIT = os.environ.get('PCB_COLOR_SPLIT', '1') == '1'
EDGE_COLOR_SPLIT = os.environ.get('PCB_EDGE_COLOR_SPLIT', '1') == '1'
WIRE_BRIDGE = os.environ.get('PCB_WIRE_BRIDGE', '1') == '1'
VERT_LOCAL_GAP = os.environ.get('PCB_VERT_LOCAL_GAP', '0') == '1'   # off: split rotated '3.3V'
WIRE_SPECKS = os.environ.get('PCB_WIRE_SPECKS', '1') == '1'
TITLE_DROP = os.environ.get('PCB_TITLE_DROP', '1') == '1'


def _split_color_runs(its, bluem, m, ch):
    """Glyph grouping ignores colour, so a black pin number set a word-space before a blue
    designator ('23' 'C4') becomes one line '23 C4'.  Split a line where the colour class
    changes across a gap of at least a word space and both sides hold real letters."""
    out = []
    for it in its:
        if len(it.glyphs) < 2:
            out.append(it)
            continue
        def isblue(b):
            sub = bluem[b[1]:b[3] + 1, b[0]:b[2] + 1]
            tot = m[b[1]:b[3] + 1, b[0]:b[2] + 1]
            return (sub & (tot > 0)).sum() >= 0.5 * max(tot.sum(), 1)
        lo, hi = (1, 3) if it.vert else (0, 2)
        gs = sorted(it.glyphs, key=lambda b: b[lo])
        flags = [isblue(b) for b in gs]
        if EDGE_COLOR_SPLIT and len(gs) >= 4:
            # one glyph of the other colour at either end, set apart by a gap: a pin number
            # beside a part name ('2' + 'MT2492')
            def frac(b):
                sub = bluem[b[1]:b[3] + 1, b[0]:b[2] + 1]
                tot = m[b[1]:b[3] + 1, b[0]:b[2] + 1]
                return (sub & (tot > 0)).sum() / max(tot.sum(), 1)
            fr = [frac(b) for b in gs]
            cut = None
            if all(f >= 0.95 for f in fr[1:]) and fr[0] <= 0.05 or all(f <= 0.05 for f in fr[1:]) and fr[0] >= 0.95:
                if gs[1][lo] - gs[0][hi] >= 0.3 * ch:
                    cut = 1
            elif all(f >= 0.95 for f in fr[:-1]) and fr[-1] <= 0.05 or all(f <= 0.05 for f in fr[:-1]) and fr[-1] >= 0.95:
                if gs[-1][lo] - gs[-2][hi] >= 0.3 * ch:
                    cut = len(gs) - 1
            if cut is not None:
                for g in (gs[:cut], gs[cut:]):
                    out.append(TextItem(min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g),
                                        it.vert, it.color, g))
                continue
        big = [max(b[2] - b[0], b[3] - b[1]) + 1 >= 0.55 * ch for b in gs]
        runs = [[0]]
        for i in range(1, len(gs)):
            if flags[i] != flags[runs[-1][-1]] and gs[i][lo] - max(gs[j][hi] for j in runs[-1]) >= 0.45 * ch:
                runs.append([i])
            else:
                runs[-1].append(i)
        # exactly two colour runs of two or more letters each: glyph colours of small dark
        # text flicker (a black pin name 'EN' may have a 'blue' N), which is no boundary
        wide = [b[hi] - b[lo] + 1 >= 1.2 * ch for b in gs]      # touching letters: one blob
        if len(runs) != 2 or not all(sum(big[i] for i in r) >= 2 or any(wide[i] and big[i] for i in r) for r in runs) \
                or not all(len(set(flags[i] for i in r)) == 1 for r in runs):
            out.append(it)
            continue
        for r in runs:
            g = [gs[i] for i in r]
            out.append(TextItem(min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g),
                                it.vert, it.color, g))
    return out


def build_text_items(P, img_rgb):
    lab = P.M['lab']
    if RED_STRIKE:
        lab = _strike_fill(P, lab)
        P.M['lab_txt'] = lab
    items = []
    t = P.t
    blue = (lab == 3).astype(np.uint8)
    blk = (lab == 4).astype(np.uint8)
    near_blue = cv2.dilate(blue, np.ones((5, 5), np.uint8))
    # black pixels touching blue glyphs are blended blue text (e.g. text over a wire)
    blk_cc_n, blk_lab = cv2.connectedComponents(blk, connectivity=8)
    touch = np.unique(blk_lab[(near_blue > 0) & (blk > 0)])
    touch = touch[touch > 0]
    blk_blue = np.isin(blk_lab, touch) & (near_blue > 0)
    anym = (blue | blk).astype(np.uint8)
    if WIRE_BRIDGE:
        # a horizontal wire drawn over a text line cuts every letter into a top and a bottom
        # piece ('5.1kΩ' -> '71' + 'U.LKSZ'); wire pixels with text ink right above and
        # right below them belong to the letters
        wire = (lab == 2)
        k = int(round(t + 2))
        H_, W_ = anym.shape
        # the letter must continue right at the band's edges: walk out of the band through
        # wire / blended pixels only (a label merely standing above a wire, with paper in
        # between, is no letter cut in two)
        pas = wire | ((lab != 0) & (anym == 0))
        a_ = anym.astype(bool)
        up = np.zeros_like(wire); dn = np.zeros_like(wire)
        cu = np.ones_like(wire); cd = np.ones_like(wire)
        for d in range(1, k + 1):
            up[d:] |= cu[d:] & a_[:-d]
            dn[:-d] |= cd[:-d] & a_[d:]
            nu = np.zeros_like(wire); nu[d:] = pas[:-d]
            nd = np.zeros_like(wire); nd[:-d] = pas[d:]
            cu &= nu
            cd &= nd
        horiz = cv2.morphologyEx(wire.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, max(3, int(P.u))), np.uint8)) > 0
        fill = horiz & up & dn
        if fill.any():
            # one letter cut in two, not two text lines sandwiching the wire (Q4 above the
            # wire, CMS4576 below): the ink above + band + ink below spans one letter height
            lim = 0.85 * P.u
            ncc, lcc, scc, _ = cv2.connectedComponentsWithStats(anym, connectivity=8)
            ys, xs = np.nonzero(fill)
            for y, x in zip(ys, xs):
                yy = y
                while yy > 0 and pas[yy - 1, x] and not a_[yy - 1, x]:
                    yy -= 1
                lu = lcc[yy - 1, x] if yy > 0 else 0
                yy = y
                while yy < H_ - 1 and pas[yy + 1, x] and not a_[yy + 1, x]:
                    yy += 1
                ld = lcc[yy + 1, x] if yy < H_ - 1 else 0
                if lu == 0 or ld == 0:
                    fill[y, x] = False
                    continue
                top = scc[lu, cv2.CC_STAT_TOP]
                bot = scc[ld, cv2.CC_STAT_TOP] + scc[ld, cv2.CC_STAT_HEIGHT] - 1
                if bot - top + 1 > lim:
                    fill[y, x] = False
        if fill.any():
            anym = (anym.astype(bool) | fill).astype(np.uint8)
    mask_by_color = {'any': anym}
    bluem = (blue | blk_blue)
    for color in ('any',):
        m = mask_by_color[color]
        if m.sum() == 0:
            continue
        if STRIP_RUNS:
            # long straight runs are never text strokes: coloured wires / buses / frames that a
            # letter touches would otherwise make the letter one huge 'glyph' ('A' + bus wire)
            L = max(8, int(1.6 * P.u))
            runs = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | \
                cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
            if runs.any():
                m = (m & (1 - runs)).astype(np.uint8)
        # character height estimate
        gl = _glyphs(m, max_h=int(2.2 * P.u))
        if WIRE_SPECKS and gl:
            # a few dark pixels where a red pin line turns into a green wire are no text: as
            # dots between stacked pin numbers they chain the numbers into one vertical word
            keep_ = []
            Hl, Wl = lab.shape

            def inline(b):
                # the speck continues a wire: wire right beside it across its whole height
                # (or right above / below across its whole width, the wire running on
                # beyond); a decimal point merely sitting on a wire does not
                x0b, y0b, x1b, y1b = b[0], b[1], b[2], b[3]
                for xx in (x0b - 1, x1b + 1):
                    if 0 <= xx < Wl and (lab[y0b:y1b + 1, xx] == 2).all():
                        xs = range(xx, max(xx - 4, -1), -1) if xx < x0b else range(xx, min(xx + 4, Wl))
                        if all((lab[y0b:y1b + 1, x_] == 2).all() for x_ in xs):
                            return True
                for yy in (y0b - 1, y1b + 1):
                    if 0 <= yy < Hl and (lab[yy, x0b:x1b + 1] == 2).all():
                        ys_ = range(yy, max(yy - 4, -1), -1) if yy < y0b else range(yy, min(yy + 4, Hl))
                        if all((lab[y_, x0b:x1b + 1] == 2).all() for y_ in ys_):
                            return True
                return False
            for b in gl:
                if b[4] <= 4 and inline(b):
                    continue
                keep_.append(b)
            gl = keep_
        if not gl:
            continue
        hs = np.array([b[3] - b[1] + 1 for b in gl])
        ws = np.array([b[2] - b[0] + 1 for b in gl])
        big = hs[(hs >= 0.4 * P.u)]
        ch = float(np.median(big)) if len(big) else 0.75 * P.u
        ch = min(max(ch, 0.45 * P.u), 1.3 * P.u)
        # discard long lines (frames etc.)
        # (thick coloured wires / buses are ~0.3-0.4 ch high: not text either)
        gl = [b for b in gl if not ((b[2] - b[0] > 3 * ch and b[3] - b[1] < 0.42 * ch) or (b[3] - b[1] > 3 * ch and b[2] - b[0] < 0.42 * ch))]
        if SPLIT_STACKED_GLYPHS:
            gl = _split_stacked_glyphs(gl, m, ch)
        if SPLIT_COLOR_GLYPHS:
            gl = _split_color_glyphs(gl, m, lab, ch)
        if SPLIT_SHADE_GLYPHS and img_rgb is not None:
            gl = _split_shade_glyphs(gl, m, img_rgb, ch)
        gcls = None
        if MIXED_COLOR_PEN:
            # pure-blue / pure-black glyphs (blended edges aside): a word is written in one colour
            gcls = []
            for x0, y0, x1, y1, a in gl:
                sub = m[y0:y1 + 1, x0:x1 + 1] > 0
                lb = lab[y0:y1 + 1, x0:x1 + 1]
                n_ = max(int(sub.sum()), 1)
                nb_, nk_ = int((sub & (lb == 3)).sum()), int((sub & (lb == 4)).sum())
                gcls.append('b' if nb_ >= 0.9 * n_ and n_ >= 15 else 'k' if nk_ >= 0.9 * n_ and n_ >= 15 else None)
        its = _make_items(gl, ch, color, gcls)
        if COLOR_SPLIT:
            its = _split_color_runs(its, bluem, m, ch)
        for it in its:
            # drop leading/trailing glyphs of the other colour class (blended line fragments)
            if len(it.glyphs) > 1:
                def isblue(b):
                    sub = bluem[b[1]:b[3] + 1, b[0]:b[2] + 1]
                    tot = m[b[1]:b[3] + 1, b[0]:b[2] + 1]
                    return (sub & (tot > 0)).sum() >= 0.5 * max(tot.sum(), 1)
                key = (lambda b: b[1]) if it.vert else (lambda b: b[0])
                gs = sorted(it.glyphs, key=key)
                flags = [isblue(b) for b in gs]
                maj = sum(flags) >= len(flags) / 2.0
                def frag(b):
                    w_, h_ = b[2] - b[0] + 1, b[3] - b[1] + 1
                    return min(w_, h_) <= max(2, 0.22 * ch)
                while len(gs) > 1 and flags[0] != maj and frag(gs[0]):
                    gs.pop(0); flags.pop(0)
                while len(gs) > 1 and flags[-1] != maj and frag(gs[-1]):
                    gs.pop(); flags.pop()
                if len(gs) != len(it.glyphs):
                    it.glyphs = gs
                    it.x0 = min(b[0] for b in gs); it.y0 = min(b[1] for b in gs)
                    it.x1 = max(b[2] for b in gs); it.y1 = max(b[3] for b in gs)
            nb = 0
            nk = 0
            for b in it.glyphs:
                sub = bluem[b[1]:b[3] + 1, b[0]:b[2] + 1]
                tot = m[b[1]:b[3] + 1, b[0]:b[2] + 1]
                nb += int((sub & (tot > 0)).sum())
                nk += int(tot.sum())
            it.color = 'blue' if nb >= 0.5 * max(nk, 1) else 'black'
            mx = max(max(b[3] - b[1], b[2] - b[0]) + 1 for b in it.glyphs)
            if mx < 0.55 * ch:
                continue
            if sum(b[4] for b in it.glyphs) < 0.12 * ch * ch:
                continue
            items.append(it)
    # red glyph candidates (pin numbers drawn in symbol colour)
    items_red = red_glyph_items(P)
    items += items_red
    for i, it in enumerate(items):
        it.id = i
    return items


SPLIT_STACKED_GLYPHS = os.environ.get('PCB_SPLIT_STACKED_GLYPHS', '1') == '1'


def _split_stacked_glyphs(gl, m, ch):
    """Two characters of neighbouring text lines touching (a value '0' right on top of the
    'R' of 'R36') make one tall blob; read as a rotated word it takes the 'R' away from its
    line.  A blob two characters tall whose lower (upper) part lines up with the characters
    right beside it is cut at its narrowest row there."""
    out = list(gl)
    boxes = [b for b in gl]
    res = []
    for b in out:
        x0, y0, x1, y1, a = b
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if not (1.6 * ch <= h <= 2.6 * ch and 0.3 * ch <= w <= 3.5 * ch):
            res.append(b)
            continue
        if w > 1.3 * ch:
            # several letters of two lines run together ('kΩ' over 'LED'): only when the upper
            # line continues on one side and the lower line on the other
            def _side(top):
                for q in boxes:
                    if q is b or not (0.7 * ch <= q[3] - q[1] + 1 <= 1.3 * ch):
                        continue
                    if 0 <= max(q[0] - x1, x0 - q[2]) <= 0.6 * ch:
                        if top and abs(q[1] - y0) <= max(2, 0.15 * ch):
                            return 'L' if q[2] < x0 else 'R'
                        if not top and abs(q[3] - y1) <= max(2, 0.15 * ch):
                            return 'L' if q[2] < x0 else 'R'
                return None
            st, sb = _side(True), _side(False)
            if st is None or sb is None or st == sb:
                res.append(b)
                continue
        cut = None
        for part in ('lo', 'hi'):
            # neighbours of character size on the same line as the lower / upper part
            for q in boxes:
                if q is b:
                    continue
                qh = q[3] - q[1] + 1
                if not (0.7 * ch <= qh <= 1.3 * ch):
                    continue
                gap = max(q[0] - x1, x0 - q[2])
                if not (0 <= gap <= 0.6 * ch):
                    continue
                if part == 'lo' and abs(q[3] - y1) <= max(2, 0.15 * ch):
                    cut = ('lo', q[1])
                    break
                if part == 'hi' and abs(q[1] - y0) <= max(2, 0.15 * ch):
                    cut = ('hi', q[3])
                    break
            if cut:
                break
        if cut is None:
            res.append(b)
            continue
        # cut where the neighbouring line starts (ends); the characters touch there
        r = (cut[1] - y0) if cut[0] == 'lo' else (cut[1] - y0 + 1)
        if not (1 <= r <= h - 1):
            res.append(b)
            continue
        top, bot = m[y0:y0 + r, x0:x1 + 1], m[y0 + r:y1 + 1, x0:x1 + 1]
        if top.sum() == 0 or bot.sum() == 0:
            res.append(b)
            continue
        ht = np.nonzero(top.any(1))[0]
        hb = np.nonzero(bot.any(1))[0]
        if not (0.6 * ch <= ht[-1] - ht[0] + 1 <= 1.3 * ch and 0.6 * ch <= hb[-1] - hb[0] + 1 <= 1.3 * ch):
            res.append(b)                   # each half must be a whole character
            continue
        for sub, oy in ((top, y0), (bot, y0 + r)):
            ys, xs = np.nonzero(sub)
            res.append([x0 + int(xs.min()), oy + int(ys.min()), x0 + int(xs.max()), oy + int(ys.max()), int(sub.sum())])
    return res


SPLIT_COLOR_GLYPHS = os.environ.get('PCB_SPLIT_COLOR_GLYPHS', '1') == '1'


def _split_color_glyphs(gl, m, lab, ch):
    """A blue designator touching a black pin number ('U3' right on top of a rotated '12')
    makes one blob of two colours.  Split such blobs by colour when each colour holds a
    good share of the ink and the two parts lie side by side rather than one over the other's
    whole extent (blended edges never reach a quarter of a glyph)."""
    out = []
    for b in gl:
        x0, y0, x1, y1, a = b
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if max(w, h) < 0.8 * ch:
            out.append(b)
            continue
        sub = m[y0:y1 + 1, x0:x1 + 1] > 0
        lb = lab[y0:y1 + 1, x0:x1 + 1]
        bl, bk = sub & (lb == 3), sub & (lb == 4)
        nb, nk, n = int(bl.sum()), int(bk.sum()), int(sub.sum())
        if nb < 0.25 * n or nk < 0.25 * n or nb < 12 or nk < 12:
            out.append(b)
            continue
        parts = []
        for pm in (bl, bk):
            ys, xs = np.nonzero(pm)
            parts.append([x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()), y0 + int(ys.max()), int(pm.sum())])
        p, q = parts
        ovx = min(p[2], q[2]) - max(p[0], q[0]) + 1
        ovy = min(p[3], q[3]) - max(p[1], q[1]) + 1
        # the two parts must be separate characters: little overlap along at least one axis
        if ovx > 0.5 * min(p[2] - p[0] + 1, q[2] - q[0] + 1) and ovy > 0.5 * min(p[3] - p[1] + 1, q[3] - q[1] + 1):
            out.append(b)
            continue
        out.extend(parts)
    return out


HEX_NAMES = os.environ.get('PCB_HEX_NAMES', '1') == '1'
SPLIT_SHADE_GLYPHS = os.environ.get('PCB_SPLIT_SHADE_GLYPHS', '1') == '1'


def _split_shade_glyphs(gl, m, img, ch):
    """A navy designator right on top of its bright-blue value ('R22' over '10kΩ', the lines
    touching) makes blobs two characters tall that no neighbour lines up with.  EasyEDA prints
    the two lines in different blues: cut such a blob where the shade changes from one row to
    the next, when each part is a whole character."""
    out = []
    H, W = m.shape[:2]
    for b in gl:
        x0, y0, x1, y1, a = b
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if not (1.5 * ch <= h <= 2.8 * ch) or w > 6 * ch:
            out.append(b)
            continue
        sub = img[y0:y1 + 1, x0:x1 + 1].astype(np.int32)
        msk = m[y0:y1 + 1, x0:x1 + 1] > 0
        sat = sub[:, :, 2] - np.maximum(sub[:, :, 0], sub[:, :, 1])
        core = msk & (sat >= 60)
        cls = []
        for i in range(h):
            n_ = int(core[i].sum())
            if n_ < 2:
                cls.append(0)
                continue
            bm = float(sub[i, :, 2][core[i]].mean())
            cls.append(1 if bm < 180 else 2 if bm > 215 else 0)
        rows = [(i, c_) for i, c_ in enumerate(cls) if c_]
        if len(rows) < 4:
            out.append(b)
            continue
        flips = [k for k in range(1, len(rows)) if rows[k][1] != rows[k - 1][1]]
        if len(flips) != 1:
            out.append(b)
            continue
        r = rows[flips[0]][0]                   # first row of the second shade
        top, bot = msk[:r], msk[r:]
        if not top.any() or not bot.any():
            out.append(b)
            continue
        ht = np.nonzero(top.any(1))[0]
        hb = np.nonzero(bot.any(1))[0]
        if not (0.55 * ch <= ht[-1] - ht[0] + 1 <= 1.35 * ch and 0.55 * ch <= hb[-1] - hb[0] + 1 <= 1.35 * ch):
            out.append(b)                       # each part must be a whole line of characters
            continue
        for sm, oy in ((top, y0), (bot, y0 + r)):
            ys, xs = np.nonzero(sm)
            out.append([x0 + int(xs.min()), oy + int(ys.min()), x0 + int(xs.max()), oy + int(ys.max()), int(sm.sum())])
    return out


MIXED_COLOR_PEN = os.environ.get('PCB_MIXED_COLOR_PEN', '1') == '1'


def _make_items(gl, ch, color, gcls=None, depth=0):
    hgroups = _group(gl, ch, False)
    vgroups = _group(gl, ch, True)
    # decide for each glyph whether it belongs to a horizontal or a vertical group
    # score: groups with more glyphs and consistent glyph sizes win
    assign = {}
    cand = []
    for grp in hgroups:
        cand.append(('h', grp))
    for grp in vgroups:
        cand.append(('v', grp))

    def score(o, grp):
        bs = [gl[i] for i in grp]
        if len(grp) == 1:
            b = bs[0]
            w, h = b[2] - b[0] + 1, b[3] - b[1] + 1
            if TALL_VERT:
                # rotated text whose characters touch: one blob, a cap height wide and
                # several characters long ('3V3' on a power bar)
                if h >= 1.6 * ch and 0.6 * ch <= w <= 1.3 * ch:
                    return 0.5 if o == 'v' else 0.4
                if w >= 1.6 * ch and 0.6 * ch <= h <= 1.3 * ch:
                    return 0.5 if o == 'h' else 0.4
            # single glyph: orientation from aspect (rotated glyphs are usually wider)
            return 0.5 if (o == 'h') == (h >= 0.8 * w or h >= 0.7 * ch) else 0.4
        x0 = min(b[0] for b in bs); x1 = max(b[2] for b in bs)
        y0 = min(b[1] for b in bs); y1 = max(b[3] for b in bs)
        extent_cross = (y1 - y0 + 1) if o == 'h' else (x1 - x0 + 1)
        pen = 0.0
        if extent_cross > 1.35 * ch:
            pen = 1.0
        if gcls is not None:
            cl = {gcls[i] for i in grp} - {None}
            if len(cl) > 1:
                pen += 1.5               # a blue designator chained to a black pin number
        if GROUP_SUBSTANTIAL:
            # specks (anti-aliasing dust, a dot of another colour class) must not outvote a
            # real letter: count glyphs of text size only
            nsub = sum(1 for b in bs if max(b[2] - b[0], b[3] - b[1]) + 1 >= 0.45 * ch)
            return nsub + 0.01 * len(grp) - pen
        return len(grp) - pen
    cand.sort(key=lambda c: -score(*c))
    items = []
    for o, grp in cand:
        free = [i for i in grp if i not in assign]
        if not free:
            continue
        if len(free) < len(grp):
            # partially used: regroup the free ones on their own later
            continue
        for i in grp:
            assign[i] = o
        bs = [gl[i] for i in grp]
        it = TextItem(min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs),
                      o == 'v', color, bs)
        items.append(it)
    left = [i for i in range(len(gl)) if i not in assign]
    if left and gcls is not None and depth == 0 and MIXED_COLOR_PEN:
        # glyphs freed by a better split (the black '12' under the blue 'U3'): group them on
        # their own, in either direction
        items += _make_items([gl[i] for i in left], ch, color, [gcls[i] for i in left], 1)
        left = []
    if left:
        sub = [gl[i] for i in left]
        for grp in _group(sub, ch, False):
            bs = [sub[i] for i in grp]
            items.append(TextItem(min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs),
                                  False, color, bs))
    return items


def red_glyph_items(P):
    """Small isolated red connected components that look like characters."""
    red = P.red
    n, lab, st, cen = cv2.connectedComponentsWithStats(red, connectivity=8)
    gl = []
    u = P.u
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < 0.35 * u or h > 1.0 * u or w > 1.0 * u:
            continue
        if a > 0.6 * w * h and w > 0.25 * u:
            continue
        gl.append([int(x), int(y), int(x + w - 1), int(y + h - 1), int(a)])
    if not gl:
        return []
    hs = [b[3] - b[1] + 1 for b in gl]
    ch = float(np.median(hs))
    return _make_items(gl, ch, 'red')


def crop_for_ocr(P, img_rgb, it, pad=None):
    lab = P.M.get('lab_txt', P.M['lab'])
    H, W = lab.shape
    hh = it.w if it.vert else it.h
    pad = pad if pad is not None else max(2, int(0.25 * hh))
    x0, y0 = max(it.x0 - pad, 0), max(it.y0 - pad, 0)
    x1, y1 = min(it.x1 + pad, W - 1), min(it.y1 + pad, H - 1)
    mag = P.M['mag'][y0:y1 + 1, x0:x1 + 1]
    sub = lab[y0:y1 + 1, x0:x1 + 1]
    cls = {'blue': 3, 'black': 4, 'red': 1}[it.color]
    keep = np.zeros(sub.shape, bool)
    # keep only glyph boxes' pixels of the item class (+ anti-aliased gray near them)
    gm = np.zeros(sub.shape, bool)
    e = 1
    gm[max(it.y0 - y0 - e, 0):it.y1 - y0 + 1 + e, max(it.x0 - x0 - e, 0):it.x1 - x0 + 1 + e] = True
    keep = gm & ((sub == cls) | (sub == 5) | (sub == 4) | ((sub == 6) & (cls == 1)) | ((sub == 1) & (cls == 1)) | (sub == 3))
    if cls != 4:
        keep &= ~(sub == 2)
    val = np.where(keep, np.clip(mag / 600.0, 0, 1), 0)
    if OCR_GAMMA != 1.0:
        val = val ** OCR_GAMMA         # faint thin strokes (a hyphen) stay visible to the recogniser
    # pixels of the text's own colour inside a wire band are text drawn over the wire
    strong = (sub == cls) if (TEXT_OVER_WIRE and cls == 3) else None
    val = _remove_wire_bands(P, val, x0, y0, it, strong)
    g = (255 * (1 - val)).astype(np.uint8)
    crop = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    return crop


TEXT_OVER_WIRE = os.environ.get('PCB_TEXT_OVER_WIRE', '1') == '1'
OCR_GAMMA = float(os.environ.get('PCB_OCR_GAMMA', '1.0'))
BRIDGE_CONT = os.environ.get('PCB_BRIDGE_CONT', '0') == '1'   # measured: no gain


def _cont(v):
    """Ink one step further out from the band, allowing a one-pixel slant."""
    m = v > 0.2
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    return out


def _remove_wire_bands(P, val, x0, y0, it=None, strong=None):
    """Wires crossing text leave a 'strike-through'; replace wire-band pixels by
    vertical (resp. horizontal) interpolation of the glyph strokes around them."""
    h, w = val.shape
    hw = P.t / 2 + 0.8
    for s_ in P.wires:
        if s_.o == 'h':
            if not (y0 - hw <= s_.c <= y0 + h + hw):
                continue
            if s_.b < x0 or s_.a > x0 + w:
                continue
            if it is not None and not it.vert:
                hh = it.y1 - it.y0 + 1
                if s_.c - it.y0 < 0.22 * hh or it.y1 - s_.c < 0.22 * hh:
                    continue      # wire only grazes the text edge
            r0 = int(np.ceil(s_.c - P.t / 2)) - y0
            r1 = int(np.floor(s_.c + P.t / 2)) - y0
            a = max(int(s_.a) - x0 - 1, 0)
            b = min(int(s_.b) - x0 + 2, w)
            if r1 < 0 or r0 >= h:
                continue
            above = val[r0 - 1, a:b] if r0 - 1 >= 0 else np.zeros(b - a)
            below = val[r1 + 1, a:b] if r1 + 1 < h else np.zeros(b - a)
            fill = np.minimum(above, below)
            if BRIDGE_CONT:
                # bridge only strokes that continue through the band (a stroke crossing the
                # wire), not the facing ends of two neighbouring glyphs
                a2 = _cont(val[r0 - 2, a:b] if r0 - 2 >= 0 else np.zeros(b - a))
                b2 = _cont(val[r1 + 2, a:b] if r1 + 2 < h else np.zeros(b - a))
                fill = np.where(a2 & b2, fill, 0)
            for r in range(max(r0, 0), min(r1 + 1, h)):
                val[r, a:b] = fill if strong is None else np.where(strong[r, a:b], np.maximum(fill, val[r, a:b]), fill)
        else:
            if not (x0 - hw <= s_.c <= x0 + w + hw):
                continue
            if s_.b < y0 or s_.a > y0 + h:
                continue
            if it is not None and it.vert:
                ww = it.x1 - it.x0 + 1
                if s_.c - it.x0 < 0.22 * ww or it.x1 - s_.c < 0.22 * ww:
                    continue
            c0 = int(np.ceil(s_.c - P.t / 2)) - x0
            c1 = int(np.floor(s_.c + P.t / 2)) - x0
            a = max(int(s_.a) - y0 - 1, 0)
            b = min(int(s_.b) - y0 + 2, h)
            if c1 < 0 or c0 >= w:
                continue
            left = val[a:b, c0 - 1] if c0 - 1 >= 0 else np.zeros(b - a)
            right = val[a:b, c1 + 1] if c1 + 1 < w else np.zeros(b - a)
            fill = np.minimum(left, right)
            if BRIDGE_CONT:
                l2 = _cont(val[a:b, c0 - 2] if c0 - 2 >= 0 else np.zeros(b - a))
                r2 = _cont(val[a:b, c1 + 2] if c1 + 2 < w else np.zeros(b - a))
                fill = np.where(l2 & r2, fill, 0)
            for c in range(max(c0, 0), min(c1 + 1, w)):
                val[a:b, c] = fill if strong is None else np.where(strong[a:b, c], np.maximum(fill, val[a:b, c]), fill)
    return val


ZO_FIX = True
CASE_FIX = os.environ.get('PCB_CASE', '1') == '1'
DOT_FIX = os.environ.get('PCB_DOT', '1') == '1'


def ocr_items(P, img_rgb, items, rec=None):
    rec = rec or get_rec()
    crops = []
    idx = []
    for it in items:
        c = crop_for_ocr(P, img_rgb, it)
        if it.vert:
            # EasyEDA vertical text reads bottom-to-top (rotated 90deg CCW)
            crops.append(cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE))
            idx.append((it, 'cw'))
            crops.append(cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE))
            idx.append((it, 'ccw'))
        else:
            crops.append(c)
            idx.append((it, 'h'))
    res = rec(crops, return_probs=True) if crops else []
    best = {}
    alt = {}
    for (it, o), (txt, conf, probs) in zip(idx, res):
        b = best.get(it.id)
        if b is None or conf > b[1]:
            if b is not None:
                alt[it.id] = b
            best[it.id] = (txt, conf, probs, o)
        else:
            alt[it.id] = (txt, conf, probs, o)
    for it in items:
        if it.id in best:
            it.text, it.conf, it.probs, it.rot = best[it.id]
            if it.id in alt:
                a = alt[it.id]
                it.alt = (a[0].strip(), a[1], a[2], a[3])
            it.text = it.text.strip()
            if ZO_FIX:
                from .textfix import fix_zero_o, fix_missing_dot
                it.text = fix_zero_o(it)
                if DOT_FIX:
                    it.text = fix_missing_dot(it)
    if HEX_NAMES:
        # a part name that is a 32-digit hex id (EasyEDA device uuid): re-read with hex digits only
        import re as _re
        for it in items:
            t_ = (it.text or '').replace(' ', '')
            if it.id in best and it.probs is not None and 30 <= len(t_) <= 36 \
                    and len(_re.findall(r'[0-9a-f]', t_)) >= len(t_) - 4 and not _re.fullmatch(r'[0-9a-f]{32}', t_):
                h_, cf_ = redecode(it, '0123456789abcdef', rec)
                if len(h_) == 32:
                    it.text = h_
    if ZO_FIX:
        from .textfix import fix_spaces, fix_case_all
        if CASE_FIX:
            fix_case_all(P, [it for it in items if it.id in best])
        for it in items:
            if it.id in best:
                it.text = fix_spaces(it)
    return items


def redecode(it, allowed, rec=None):
    rec = rec or get_rec()
    if it.probs is None:
        return it.text, it.conf
    return rec.constrained(it.probs, allowed)


UNDERSCORE_WIRE = os.environ.get('PCB_UNDERSCORE_WIRE', '1') == '1'


def _SRC_ALTIUM():
    from . import assemble as _A
    return _A.SRC_STYLE[0] == 'altium'


def _wire_underscore(lab, a, b, h):
    """Altium net labels stand on their wire and are written in the wire colour: the '_' of
    VIN_BOARD merges into the wire as a one-character bump on its upper edge.  A row at the
    foot of the text where wire ink covers the gap between the two words but not the
    columns under the letters."""
    gap = b.x0 - a.x1 - 1
    if not (0.3 * h <= gap <= 1.3 * h):
        return False
    H, W = lab.shape
    base = int(round(max(a.y1, b.y1)))
    r0, r1 = max(base - int(round(0.1 * h)), 0), min(base + int(round(0.45 * h)) + 1, H)
    xa0, xa1 = int(round(a.x0)), int(round(a.x1))
    xb0, xb1 = int(round(b.x0)), int(round(b.x1))
    # no wire ink in the gap at letter height (a vertical wire between two labels)
    up = lab[max(int(round(min(a.y0, b.y0))), 0):max(base - int(round(0.3 * h)), 0), xa1 + 1:xb0]
    if up.size == 0 or (up == 2).mean() > 0.1:
        return False
    n = 0
    for r in range(r0, r1):
        row = lab[r] == 2
        g = row[xa1 + 1:xb0]
        if g.size == 0 or g.mean() < 0.6:
            continue
        if row[xa0:xa1 + 1].mean() < 0.3 and row[xb0:xb1 + 1].mean() < 0.3:
            n += 1
    # an underscore is a hairline; a short wire between two labels is a full stroke
    return 1 <= n <= max(2, 0.25 * h)


def merge_fragments(P, img_rgb, items, rec=None):
    """Re-join a text line that glyph grouping split at faint punctuation
    ('VDD' '_' '1' '.' '9V' -> 'VDD_1.9V').  Two same-colour horizontal items on one
    baseline are merged when there is same-colour ink in the gap between them and the
    re-recognised union reads as left + punctuation + right."""
    rec = rec or get_rec()
    lab = P.M.get('lab_txt', P.M['lab'])
    cls_of = {'blue': 3, 'black': 4}
    hor = [it for it in items if not it.vert and it.text and it.color in cls_of]
    hor.sort(key=lambda it: it.x0)
    gone = set()
    new = []
    next_id = max([it.id for it in items] + [0]) + 1
    for i, a in enumerate(hor):
        if a.id in gone:
            continue
        for b in hor[i + 1:]:
            if b.id in gone or b.color != a.color:
                continue
            h = max(a.h, b.h)
            gap = b.x0 - a.x1
            if gap <= 0 or gap > 1.3 * h:
                if b.x0 - a.x1 > 1.3 * h:
                    break
                continue
            if abs(a.y1 - b.y1) > 0.25 * h or abs(a.h - b.h) > 0.35 * h:
                continue
            if UNDERSCORE_WIRE and _SRC_ALTIUM() and _wire_underscore(lab, a, b, h):
                gl = list(a.glyphs) + list(b.glyphs)
                m = TextItem(a.x0, min(a.y0, b.y0), b.x1, max(a.y1, b.y1), False, a.color, gl)
                m.id = next_id
                m.text, m.conf, m.rot = a.text + '_' + b.text, min(a.conf, b.conf), 'h'
                m.probs = None
                next_id += 1
                gone.update([a.id, b.id])
                new.append(m)
                a = m
                continue
            # ink of the same class strictly inside the gap, near the text band
            y0, y1 = int(min(a.y0, b.y0)), int(max(a.y1, b.y1) + 0.35 * h)
            sub = lab[y0:y1 + 1, a.x1 + 1:b.x0]
            ink = (3, 4, 5) if a.color == 'blue' else (4, 5)      # faint punctuation is often grey
            if sub.size == 0 or not np.isin(sub, ink).any():
                continue
            tiny = [t for t in items if t is not a and t is not b and t.id not in gone and t.color == a.color
                    and t.x0 > a.x1 - 1 and t.x1 < b.x0 + 1 and t.y1 >= y0 and t.y0 <= y1 and t.h < 0.6 * h]
            gl = list(a.glyphs) + list(b.glyphs) + [g for t in tiny for g in t.glyphs]
            m = TextItem(a.x0, min(a.y0, b.y0), b.x1, max(a.y1, b.y1), False, a.color, gl)
            m.id = next_id
            ext = TextItem(m.x0, m.y0, m.x1, max([m.y1] + [t.y1 for t in tiny]) + int(round(0.3 * h)), False, a.color, gl)
            crop = crop_for_ocr(P, img_rgb, ext)
            txt, conf, probs = rec([crop], return_probs=True)[0]
            txt = txt.strip()
            # tolerant comparison: the re-read may resolve O/0 and I/l/1 differently
            zo = lambda t: t.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('|', '1')
            if conf < 0.75 or not zo(txt).startswith(zo(a.text)) or not zo(txt).endswith(zo(b.text)):
                continue
            mid = txt[len(a.text):len(txt) - len(b.text)]
            if not mid or any(c.isalnum() or c == ' ' for c in mid) or len(mid) > 3:
                continue
            # keep the glyph-checked spelling of multi-character halves; a lone character
            # ('1' for a bare-bar 'I') is better read in context by the re-read
            left = a.text if len(a.text) > 1 else txt[:len(a.text)]
            right = b.text if len(b.text) > 1 else txt[len(txt) - len(b.text):]
            txt = left + mid + right
            m.text, m.conf, m.probs, m.rot = txt, conf, probs, 'h'
            next_id += 1
            gone.update([a.id, b.id] + [t.id for t in tiny])
            new.append(m)
            a = m   # allow chaining  VDD + _1 + .9V
    if not gone:
        return items
    out = [it for it in items if it.id not in gone]
    # chained merges: drop intermediate merged items that were merged again
    out += [m for m in new if m.id not in gone]
    return out


SPLIT_VALUES = os.environ.get('PCB_SPLIT_VALUES', '1') == '1'
_HALF_VALUE = None


def _value_halves(P, img_rgb, it, rec):
    """Re-read the two halves of an unreadable long item cut at a glyph gap near its middle;
    the cut is kept when both halves read as values with a unit."""
    gl = sorted(it.glyphs or [], key=lambda b: b[0])
    if len(gl) < 4:
        return None
    W = it.x1 - it.x0 + 1
    gaps = [((gl[i][2] + gl[i + 1][0]) / 2.0, gl[i + 1][0] - gl[i][2]) for i in range(len(gl) - 1) if gl[i + 1][0] > gl[i][2]]
    gaps = [g for g in gaps if 0.3 * W <= g[0] - it.x0 <= 0.7 * W]
    gaps.sort(key=lambda g: abs(g[0] - it.x0 - W / 2.0))
    for xs, _ in gaps[:3]:
        parts, reads = [], []
        for lo, hi in ((it.x0, int(np.floor(xs))), (int(np.ceil(xs)), it.x1)):
            g = [b for b in gl if lo <= (b[0] + b[2]) / 2.0 <= hi]
            if not g:
                break
            n = TextItem(min(b[0] for b in g), it.y0, max(b[2] for b in g), it.y1, False, it.color, g)
            txt, conf, _ = rec([crop_for_ocr(P, img_rgb, n)], return_probs=True)[0]
            n.text, n.conf, n.rot = txt.strip(), conf, it.rot
            if HALF_TWIN and not _HALF_VALUE.fullmatch(n.text):
                n.text = _fix_half(n.text)
            reads.append(n)
            if conf >= 0.8 and _HALF_VALUE.fullmatch(n.text):
                parts.append(n)
        if len(parts) == 2:
            return parts
        if HALF_TWIN and len(parts) == 1 and len(reads) == 2:
            # two equal values side by side, one of them crossed by a wire ('100nF' '100rF'):
            # a half one letter off the clean half is its twin
            a_ = parts[0]
            b_ = reads[1] if reads[0] is a_ else reads[0]
            if len(a_.text) == len(b_.text) and sum(x != y for x, y in zip(a_.text, b_.text)) == 1:
                b_.text = a_.text
                return reads
    return None


HALF_TWIN = os.environ.get('PCB_HALF_TWIN', '1') == '1'


def _fix_half(t):
    """Slips a wire through a value causes: ')' / 'O' for a zero between digits, 'r' for
    the 'n' of 'nF' ('1)0nF', '10OnF', '100rF')."""
    t = re.sub(r'(?<=\d)[)O](?=\d)', '0', t)
    t = re.sub(r'(?<=\d)[)O](?=[pnuμµmkKM]?[FHΩ]$)', '0', t)
    t = re.sub(r'(?<=\d)rF$', 'nF', t)
    return t


def split_joined_values(items, P=None, img_rgb=None, rec=None):
    """Two value texts printed edge to edge ('100nF' '100nF' under neighbouring capacitors)
    come out of glyph grouping as one item '100nF100nF', which is no value at all and is
    left unassigned.  Split such an item at the unit/digit boundary when both halves are
    complete values with a unit; the cut goes into the glyph gap nearest the character
    proportion."""
    global _HALF_VALUE
    if not SPLIT_VALUES:
        return items
    import re
    if _HALF_VALUE is None:
        _HALF_VALUE = re.compile(r'\d+(?:[.,]\d+)?(?:[pnuμµ]F|[pnuμµm]H|[kKMm]?Ω|[kKM]Hz|MHz|[kKMR]\d*|[pnuμµ]\d*)')
    out = []
    next_id = max([it.id for it in items] + [0]) + 1
    for it in items:
        t = (it.text or '').strip()
        cut = None
        if not it.vert and it.color == 'blue' and len(t) >= 4:
            for k in range(2, len(t) - 1):
                if (not t[k - 1].isdigit() and t[k - 1] not in '.,') and t[k].isdigit() \
                        and _HALF_VALUE.fullmatch(t[:k]) and _HALF_VALUE.fullmatch(t[k:]):
                    cut = k if cut is None else -1
        if cut is None and P is not None and not it.vert and it.color == 'blue' and len(t) >= 7 \
                and not any(ch.isspace() for ch in t) and it.w >= 5 * it.h \
                and t[0] in '0123456789lIO' and sum(ch.isdigit() for ch in t) >= 3:
            try:
                parts = _value_halves(P, img_rgb, it, rec or get_rec())
            except Exception:
                parts = None
            if parts:
                for n in parts:
                    n.id = next_id
                    next_id += 1
                out.extend(parts)
                continue
        if cut is None or cut < 0:
            out.append(it)
            continue
        gl = sorted(it.glyphs or [], key=lambda b: b[0])
        target = it.x0 + (it.x1 - it.x0 + 1) * cut / len(t)
        xs = None
        if len(gl) >= 2:
            gaps = [((gl[i][2] + gl[i + 1][0]) / 2.0, i) for i in range(len(gl) - 1) if gl[i + 1][0] > gl[i][2]]
            if gaps:
                xs, gi = min(gaps, key=lambda g: abs(g[0] - target))
                if abs(xs - target) > 0.6 * (it.x1 - it.x0 + 1) / len(t):
                    xs = None
        if xs is None:
            xs = target
        halves = []
        for txt, lo, hi in ((t[:cut], it.x0, int(np.floor(xs))), (t[cut:], int(np.ceil(xs)), it.x1)):
            g = [b for b in gl if lo <= (b[0] + b[2]) / 2.0 <= hi]
            n = TextItem(lo if not g else min(b[0] for b in g), it.y0, hi if not g else max(b[2] for b in g), it.y1,
                         False, it.color, g)
            n.text = txt
            n.conf = it.conf
            n.rot = it.rot
            n.id = next_id
            next_id += 1
            halves.append(n)
        out.extend(halves)
    return out


def drop_titles(items, P):
    """Headings of function boxes and sheet notes are set much larger than the component
    labels ("单片机电路", "核心板插座").  They name no component, but their characters were
    picked up as designators / part names of whatever is nearby.  Such items are marked
    (it.big) and kept out of designator and part-name assignment; they stay available as
    net-label text (some drawings use a larger font for their port labels)."""
    if not TITLE_DROP or len(items) < 8:
        return items
    def chh(it):
        # character size from the glyphs (an item can hold two stacked lines: '10nF±10%' over
        # '50V' is twice as high as one line of text)
        gs = it.glyphs or []
        if gs:
            v = [(g[2] - g[0] + 1) if it.vert else (g[3] - g[1] + 1) for g in gs]
            return float(np.percentile(v, 75))
        return it.w if it.vert else it.h
    hs = np.array([chh(it) for it in items if it.text], float)
    if hs.size < 8:
        return items
    med = float(np.median(hs))
    lim = max(1.8 * med, 1.0 * P.u)
    for it in items:
        if it.text and chh(it) >= lim:
            it.big = True
    return items


SPLIT_DESIG_PN = os.environ.get('PCB_SPLIT_DESIG_PN', '1') == '1'


SPLIT_DOUBLED = os.environ.get('PCB_SPLIT_DOUBLED', '1') == '1'


def split_doubled(items):
    """Equal part numbers or values set side by side (the names of identical neighbours,
    'SMCJ36CA' 'SMCJ36CA', or three '22uF' in a row) are read as one line when the gaps
    between them are small.  A line that is one word written two to four times is split
    into its copies."""
    if not SPLIT_DOUBLED:
        return items
    out = []
    nid = max([it.id for it in items] + [0]) + 1
    for it in items:
        txt_ = (it.text or '').replace(' ', '')
        rep = None
        if not it.vert and it.glyphs and len(it.glyphs) >= 4:
            for k in (2, 3, 4):
                n = len(txt_) // k
                if (n >= 4 and len(txt_) == n * k and txt_ == txt_[:n] * k and re.search(r'[A-Z]', txt_[:n])
                        and re.search(r'\d', txt_[:n])):
                    rep = (k, n)
                    break
        if rep is None:
            out.append(it)
            continue
        k, n = rep
        gs = sorted(it.glyphs, key=lambda b: b[0])
        W = float(it.x1 - it.x0)
        cuts = []
        if len(gs) == k * n:
            cuts = [n * i - 1 for i in range(1, k)]
        else:
            for i in range(1, k):
                pos = it.x0 + W * i / k
                cands = [(gs[j + 1][0] - max(b[2] for b in gs[:j + 1]), j) for j in range(len(gs) - 1)
                         if abs((gs[j][2] + gs[j + 1][0]) / 2.0 - pos) <= 0.15 * W / (k / 2.0)]
                if not cands:
                    cuts = None
                    break
                cuts.append(max(cands)[1])
        if not cuts or sorted(set(cuts)) != cuts:
            out.append(it)
            continue
        bounds = [-1] + cuts + [len(gs) - 1]
        parts = []
        for i in range(k):
            g = gs[bounds[i] + 1:bounds[i + 1] + 1]
            if not g:
                parts = []
                break
            p = TextItem(min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g),
                         False, it.color, g)
            p.text, p.conf, p.rot = txt_[:n], it.conf, it.rot
            p.id = nid
            nid += 1
            parts.append(p)
        if not parts:
            out.append(it)
            continue
        out.extend(parts)
    return out


def split_desig_partnumber(items):
    """A designator set a space before a part number on the same line ('R1 UCC27517' - the
    resistor's label beside the IC's name) is read as one line.  Split it at the widest glyph
    gap when the first word is a designator and the rest is not."""
    if not SPLIT_DESIG_PN:
        return items
    from .textclass import is_desig
    out = []
    nid = max([it.id for it in items] + [0]) + 1
    for it in items:
        txt_ = it.text or ''
        if it.vert or not it.glyphs or len(it.glyphs) < 3 or len(txt_) < 5:
            out.append(it)
            continue
        gs = sorted(it.glyphs, key=lambda b: b[0])
        gaps = [(gs[k + 1][0] - max(b[2] for b in gs[:k + 1]), k) for k in range(len(gs) - 1)]
        gap, k = max(gaps)
        n_left = k + 1
        h = it.y1 - it.y0 + 1
        m = re.fullmatch(r'([A-Z]{1,4}\d{1,3})\s+(\S.{2,})', txt_)
        if m:
            left, right = m.group(1), m.group(2)
            ok = 1 <= n_left <= len(left) + 1
        else:
            # no space read: the left word is as many characters as glyphs before the gap
            left, right = txt_[:n_left], txt_[n_left:].strip()
            med = float(np.median([g_ for g_, _ in gaps])) if gaps else 0.0
            ok = (len(right) >= 3 and re.fullmatch(r'[A-Z]{1,4}\d{1,3}', left) is not None
                  and right[:1].isalpha() and gap >= 0.35 * h and gap >= 1.8 * max(med, 1.0))
        if not ok or gap < 0.35 * h or not is_desig(left) or is_desig(right.replace(' ', '')):
            out.append(it)
            continue
        parts = []
        for txt, g in ((left, gs[:n_left]), (right, gs[n_left:])):
            p = TextItem(min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g),
                         False, it.color, g)
            p.text, p.conf, p.rot = txt, it.conf, it.rot
            p.id = nid
            nid += 1
            parts.append(p)
        out.extend(parts)
    return out

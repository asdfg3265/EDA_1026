"""Drawing frame / title block removal.

Schematic exports often include the sheet frame: one or two border rectangles with zone
labels (1..6, A..D) between them and a title block (a table of cells with the title,
version, page, logo ...) in a corner.  None of it is part of the circuit, but its lines,
cells and texts look like wires, box outlines, power names ("V1.0") and net labels.

`remove_frame(img)` finds the border rectangle(s) from long straight lines along the four
image sides and paints over
  * the border lines themselves (keeping pixels where a drawing stroke crosses them),
  * everything that lies entirely in the band between the outer and the inner border,
  * the title block: a table attached to the inner border in a corner, i.e. every ink
    component that lies entirely inside its rectangle.
Ink components that reach outside these areas are kept, so circuit content that touches
or crosses the frame survives.  Returns the cleaned image and a small info dict (None when
no frame is found, in which case the image is returned unchanged)."""
import os
import numpy as np
import cv2

FRAME_MASK = os.environ.get('PCB_FRAME_MASK', '1') == '1'
FRAME_CROP = os.environ.get('PCB_FRAME_CROP', '1') == '1'


def _paper(img):
    # paper = the most common colour (coarsely quantised)
    small = img[::4, ::4].reshape(-1, 3) // 8
    keys = small[:, 0].astype(np.int32) * 1024 + small[:, 1].astype(np.int32) * 32 + small[:, 2]
    k = np.bincount(keys).argmax()
    sel = img[::4, ::4].reshape(-1, 3)[keys == k]
    return np.median(sel, axis=0).astype(np.uint8)


def _runs(v, gap=3):
    """Runs of True in a 1-D bool array, bridging gaps of <= gap pixels: list of (a, b) inclusive."""
    idx = np.flatnonzero(v)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _long_lines(ink, frac, axis):
    """Lines along `axis` (0: horizontal rows, 1: vertical columns) whose longest run is
    >= frac of the image extent: grouped consecutive rows -> (p0, p1, a, b)."""
    A = ink if axis == 0 else ink.T
    n, L = A.shape
    cnt = A.sum(1)
    cand = np.flatnonzero(cnt >= frac * L)
    rows = []
    for r in cand:
        best = None
        for a, b in _runs(A[r]):
            if b - a + 1 >= frac * L and (best is None or b - a > best[1] - best[0]):
                best = (a, b)
        if best:
            rows.append((r, best[0], best[1]))
    lines = []
    for r, a, b in rows:
        if lines and r - lines[-1][1] <= 1 and abs(a - lines[-1][2]) <= 6 and abs(b - lines[-1][3]) <= 6:
            p0, _, a0, b0 = lines[-1]
            lines[-1] = (p0, r, min(a0, a), max(b0, b))
        else:
            lines.append((r, r, a, b))
    return [l for l in lines if l[1] - l[0] <= 12]      # thick bands are fills, not lines


def find_frame(ink):
    H, W = ink.shape
    hl = _long_lines(ink, 0.6, 0)
    vl = _long_lines(ink, 0.6, 1)
    top = [l for l in hl if l[1] < 0.15 * H]
    bot = [l for l in hl if l[0] > 0.85 * H]
    lef = [l for l in vl if l[1] < 0.15 * W]
    rig = [l for l in vl if l[0] > 0.85 * W]
    if not (top and bot and lef and rig):
        return None
    # border lines span the whole frame; the innermost such line on each side is the inner border
    X0, X1 = min(l[0] for l in lef), max(l[1] for l in rig)
    Y0, Y1 = min(l[0] for l in top), max(l[1] for l in bot)
    tol = 0.03 * max(H, W) + 6
    full_h = lambda l: l[2] <= X0 + 0.1 * W and l[3] >= X1 - 0.1 * W
    full_v = lambda l: l[2] <= Y0 + 0.1 * H and l[3] >= Y1 - 0.1 * H
    top, bot = [l for l in top if full_h(l)], [l for l in bot if full_h(l)]
    lef, rig = [l for l in lef if full_v(l)], [l for l in rig if full_v(l)]
    if not (top and bot and lef and rig):
        return None
    # inner border: the innermost line within a zone band (~4 % of the page) of the outermost
    # one -- a title block spanning the full width has full-length row lines further in
    zb = 0.04 * max(H, W)
    t = max([l for l in top if l[0] - Y0 <= zb], key=lambda l: l[1])
    b = min([l for l in bot if Y1 - l[1] <= zb], key=lambda l: l[0])
    l_ = max([l for l in lef if l[0] - X0 <= zb], key=lambda l: l[1])
    r = min([l for l in rig if X1 - l[1] <= zb], key=lambda l: l[0])
    top = [l for l in top if l[0] - Y0 <= zb]
    bot = [l for l in bot if Y1 - l[1] <= zb]
    lef = [l for l in lef if l[0] - X0 <= zb]
    rig = [l for l in rig if X1 - l[1] <= zb]
    for hline in (t, b):
        if hline[2] > l_[1] + tol or hline[3] < r[0] - tol:
            return None
    for vline in (l_, r):
        if vline[2] > t[1] + tol or vline[3] < b[0] - tol:
            return None
    outer = (min(l[0] for l in lef), min(l[0] for l in top), max(l[1] for l in rig), max(l[1] for l in bot))
    inner = (l_[1], t[1], r[0], b[0])
    if inner[2] - inner[0] < 0.5 * W or inner[3] - inner[1] < 0.5 * H:
        return None
    return dict(outer=outer, inner=inner, lines=[('h', l) for l in top + bot] + [('v', l) for l in lef + rig])


def _title_block(ink, inner):
    """Title block: a table in the lower-right corner of the inner border.  Its top edge is a
    horizontal line ending at the right border and its free side a vertical line standing on
    the bottom border (or the left border, for a block spanning the full width).  It must be a
    table of short rows on both its left and right side, with inner vertical cell lines; a
    function box in the corner (tall cells holding circuit) fails this.  Returns
    (x0, y0, x1, y1) or None.  Erasing circuit by mistake is far worse than leaving a title
    block in place, so every test errs on the side of 'no'."""
    x0i, y0i, x1i, y1i = inner
    Wi, Hi = x1i - x0i, y1i - y0i

    def hruns(y):
        return [(a + x0i + 2, b + x0i + 2) for a, b in _runs(ink[y, x0i + 2:x1i - 1])]

    def vline(x, ya, yb):
        # a vertical edge from ya to yb near column x
        for dx in range(-3, 4):
            xx = x + dx
            if x0i < xx < x1i and ink[ya:yb + 1, xx].mean() >= 0.9:
                return True
        return False

    def rows_ok(box, xa, xb, maxgap):
        """horizontal cell lines across the strip [xa, xb] of the box at most maxgap apart"""
        sw = xb - xa
        seps = [box[1] + 3]
        yy = box[1] + 8
        while yy < box[3] - 6:
            if any(min(bb, xb) - max(aa, xa) >= 0.6 * sw for aa, bb in hruns(yy)):
                seps.append(yy)
                yy += 4
            yy += 1
        seps.append(box[3])
        return len(seps) >= 3 and np.diff(seps).max() <= maxgap

    def is_table(box):
        bw = box[2] - box[0]
        if not rows_ok(box, box[0] + 4, box[2] - 4, 0.12 * Hi) and not (
                rows_ok(box, box[0] + 4, box[0] + 0.5 * bw, 0.12 * Hi) and rows_ok(box, box[0] + 0.5 * bw, box[2] - 4, 0.12 * Hi)):
            return False
        # both outer quarters are tables of short rows (no tall cell holding circuit)
        if not (rows_ok(box, box[0] + 4, box[0] + 0.25 * bw, 0.13 * Hi) and rows_ok(box, box[2] - 0.25 * bw, box[2] - 4, 0.13 * Hi)):
            return False
        sub = ink[box[1] + 6:box[3] - 4, box[0] + 8:box[2] - 6]
        return sub.size > 0 and len(_runs(sub.mean(0) >= 0.3, gap=2)) >= 1

    cands = []
    for y in range(int(y0i + 0.5 * Hi), int(y1i - 0.03 * Hi)):
        for a, b in hruns(y):
            if b - a < 0.12 * Wi or b < x1i - 6:
                continue
            if a > x0i + 0.05 * Wi:
                cands.append((y, a))
            elif a <= x0i + 6:
                cands.append((y, None))        # full-width title block
    last_y = -10
    for y, xe in sorted(cands, key=lambda c: c[0]):
        if y - last_y <= 2:          # same (thick) line
            continue
        last_y = y
        if xe is None:
            box = (x0i, y - 3, x1i, y1i)
        elif not vline(xe, y, y1i):
            continue
        else:
            box = (xe - 3, y - 3, x1i, y1i)
        bh = box[3] - box[1]
        if bh > 0.42 * Hi or bh < 0.04 * Hi or not is_table(box):
            continue
        # a top cell closed by a line of the same width above: a single text row, or an
        # empty cell (KiCad comments area)
        while True:
            up = None
            for yy in range(box[1] - 6, max(int(y0i + 0.4 * Hi), int(box[1] - 0.15 * Hi)), -1):
                if any(aa <= box[0] + 6 and bb >= box[2] - 6 for aa, bb in hruns(yy)):
                    if vline(box[0] + 3, yy, box[1] + 3):
                        up = yy
                    break
            if up is None or (box[3] - up) > 0.42 * Hi:
                break
            cell = ink[up + 4:box[1] - 1, box[0] + 6:box[2] - 6]
            if cell.size == 0 or (cell.mean() > 0.015 and box[1] - up > 0.08 * Hi):
                break
            box = (box[0], up - 3, box[2], box[3])
        return box
    return None


def remove_frame(img):
    if not FRAME_MASK:
        return img, None
    H, W = img.shape[:2]
    paper = _paper(img)
    d = np.abs(img.astype(np.int16) - paper.astype(np.int16)).sum(2)
    ink = d > 90
    fr = find_frame(ink)
    if fr is None:
        return img, None
    out = img.copy()
    erase = np.zeros((H, W), bool)
    # 1. border lines (keep pixels where a stroke crosses the line)
    for kind, (p0, p1, a, b) in fr['lines']:
        if kind == 'h':
            above = ink[max(0, p0 - 3), a:b + 1]
            below = ink[min(H - 1, p1 + 3), a:b + 1]
            keep = above & below
            erase[p0:p1 + 1, a:b + 1] |= ~keep[None, :]
        else:
            le = ink[a:b + 1, max(0, p0 - 3)]
            ri = ink[a:b + 1, min(W - 1, p1 + 3)]
            keep = le & ri
            erase[a:b + 1, p0:p1 + 1] |= ~keep[:, None]
    ink2 = ink & ~erase
    # 2. band between the outer and inner borders, 3. title block
    xo0, yo0, xo1, yo1 = fr['outer']
    xi0, yi0, xi1, yi1 = fr['inner']
    tb = _title_block(ink, fr['inner'])
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink2.astype(np.uint8), connectivity=8)
    x, y, w, h = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    X1, Y1 = x + w - 1, y + h - 1
    inside_outer = (x >= xo0 - 2) & (y >= yo0 - 2) & (X1 <= xo1 + 2) & (Y1 <= yo1 + 2)
    off_inner = (X1 <= xi0 + 2) | (x >= xi1 - 2) | (Y1 <= yi0 + 2) | (y >= yi1 - 2)
    # zone labels and ticks are small; anything bigger in the band is left alone
    band = max(xi0 - xo0, yi0 - yo0, xo1 - xi1, yo1 - yi1)
    drop = inside_outer & off_inner & (np.maximum(w, h) <= max(1.3 * band, 30))
    if tb is not None:
        drop |= (x >= tb[0] - 2) & (y >= tb[1] - 2) & (X1 <= tb[2] + 2) & (Y1 <= tb[3] + 2)
    drop[0] = False
    erase |= drop[lab]
    # paint with paper, a 1-px halo so that anti-aliased edges go too
    halo = cv2.dilate(erase.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~ink
    out[erase | halo] = paper
    return out, dict(outer=fr['outer'], inner=fr['inner'], title=tb)


def content_crop(img, margin=None):
    """Bounding box (x0, y0, x1, y1) of the drawing on a page whose frame was removed, with a
    margin; None when it would not make the page meaningfully smaller.  A full page is much
    larger than the circuit on it, and image-size dependent steps (the 2x upscale of small
    text is skipped above 9 MP) should see the circuit, not the empty sheet."""
    H, W = img.shape[:2]
    paper = _paper(img)
    d = np.abs(img.astype(np.int16) - paper.astype(np.int16)).sum(2)
    ink = d > 90
    rows = np.flatnonzero(ink.any(1))
    cols = np.flatnonzero(ink.any(0))
    if rows.size == 0 or cols.size == 0:
        return None
    m = int(margin if margin is not None else max(30, 0.015 * max(H, W)))
    x0, x1 = max(0, cols[0] - m), min(W, cols[-1] + 1 + m)
    y0, y1 = max(0, rows[0] - m), min(H, rows[-1] + 1 + m)
    if (x1 - x0) * (y1 - y0) > 0.9 * W * H:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def shift_output(out, dx, dy):
    """Move a parse result (bottom-left origin) by (dx, dy) pixels."""
    for c in out.get('components', {}).values():
        b = c.get('bbox')
        if b and len(b) == 4:
            c['bbox'] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
    for pins in out.get('pins', {}).values():
        for p in pins.values():
            pt = p.get('point')
            if pt:
                pt['x'] += dx
                pt['y'] += dy
    for n in out.get('nets', {}).values():
        for e in n.get('edges', {}).values():
            for pt in e:
                pt['x'] += dx
                pt['y'] += dy
    return out


DASH_MASK = os.environ.get('PCB_DASH_MASK', '1') == '1'


def _thin_runs(ink, maxw):
    """Per pixel: True where the ink run across (vertical, for horizontal strokes) is at most maxw px."""
    H, W = ink.shape
    up = np.zeros((H, W), np.int32)
    dn = np.zeros((H, W), np.int32)
    for y in range(1, H):
        up[y] = np.where(ink[y - 1], up[y - 1] + 1, 0)
    for y in range(H - 2, -1, -1):
        dn[y] = np.where(ink[y + 1], dn[y + 1] + 1, 0)
    return ink & (up + dn + 1 <= maxw), up, dn


def _clear_sides(ink, up, dn, y, a, b, k=4):
    """Fraction of columns a..b whose k px beyond the stroke (above and below) are paper."""
    H = ink.shape[0]
    cols = np.arange(a, b + 1)
    top = y - up[y, cols] - 1
    bot = y + dn[y, cols] + 1
    ok = np.ones(cols.size, bool)
    for j in range(k):
        t = top - j
        bb = bot + j
        ok &= ~((t >= 0) & ink[np.clip(t, 0, H - 1), cols])
        ok &= ~((bb < H) & ink[np.clip(bb, 0, H - 1), cols])
    return ok.mean()


def _clean_ends(ink, up, dn, y, a, b, g=2):
    """Nothing continues past either end of the stroke (a dash ends square; a letter stroke
    or an arc runs on into a bowl or a curve)."""
    H, W = ink.shape
    for x, step in ((a, -1), (b, 1)):
        y0 = max(0, y - up[y, x] - 1)
        y1 = min(H - 1, y + dn[y, x] + 1)
        cap = (up[y, x] + dn[y, x] + 1) // 2 + 1     # round cap: the rows beside run a little further
        for j in range(cap, cap + g + 1):
            xx = x + step * j
            # rows just outside the stroke band (a dash-dot's dot sits on the band itself)
            if 0 <= xx < W and (ink[y0, xx] or ink[y1, xx]):
                return False
    return True


def _dash_rows(ink, tud, side_ink=None, min_n=5, min_len=4, max_len=70, min_ext=120):
    """Rows of dashed horizontal lines: chains of >= min_n short thin runs with regular gaps.
    Returns a bool mask of the dash pixels."""
    H, W = ink.shape
    thin, up, dn = tud
    out = np.zeros((H, W), bool)
    min_ext = max(min_ext, 0.05 * max(H, W))
    cand = np.flatnonzero(thin.sum(1) >= min_n * min_len)
    for y in cand:
        # thin runs only: a stroke crossing the dashed line is not a dash (and a dash it
        # crosses still counts, only its thin pixels are removed)
        runs = [(a, b) for a, b in _runs(ink[y], gap=0)
                if min_len <= b - a + 1 <= max_len and thin[y, a:b + 1].mean() >= 0.6]
        if len(runs) < min_n:
            continue
        # chains of runs separated by gaps of 2 .. max_len px
        chain = [runs[0]]
        chains = []
        for r in runs[1:]:
            g = r[0] - chain[-1][1] - 1
            if 2 <= g <= max_len:
                chain.append(r)
            else:
                chains.append(chain)
                chain = [r]
        chains.append(chain)
        for ch in chains:
            if len(ch) < min_n or ch[-1][1] - ch[0][0] < min_ext:
                continue
            lens = np.array([b - a + 1 for a, b in ch], float)
            gaps = np.array([ch[i + 1][0] - ch[i][1] - 1 for i in range(len(ch) - 1)], float)
            # drawn dash patterns: gaps regular and comparable to the dashes (a row of equal
            # symbols - e.g. ground bars of a capacitor bank - is spaced much wider)
            if gaps.std() > 0.15 * gaps.mean() + 1.5 or not 0.3 * np.median(lens) <= gaps.mean() <= 1.8 * np.median(lens):
                continue
            # dash lengths: one or two values (dash-dot), each fairly constant
            ls = np.sort(lens[1:-1]) if len(lens) > 4 else np.sort(lens)
            if ls[-1] > 3 * ls[0]:
                lo, hi = ls[ls <= (ls[0] + ls[-1]) / 2], ls[ls > (ls[0] + ls[-1]) / 2]
                if lo.std() > 0.15 * lo.mean() + 1.5 or hi.std() > 0.15 * hi.mean() + 1.5:
                    continue
            elif ls.std() > 0.15 * ls.mean() + 1.5:
                continue
            # each dash: stands free (nothing attached - a capacitor plate or a ground bar
            # carries its lead), ends square (a letter stroke or an arc runs on), straight,
            # with paper on both sides (letters stacked in a column of labels have their
            # neighbours a few px away)
            def straight(a, b):
                cols = np.arange(a, b + 1)
                c = (dn[y, cols] - up[y, cols]) / 2.0
                return c.max() - c.min() <= 1.0
            ok = [bool(thin[y, a:b + 1].all()) and _clean_ends(ink, up, dn, y, a, b) and straight(a, b)
                  and (side_ink is None or _clear_sides(side_ink, up, dn, y, a, b, 6) >= 0.95) for a, b in ch]
            # accept runs of good dashes (single bad ones bridged: a dash crossed by a wire,
            # or one beside a label); each accepted run must be a line on its own
            i = 0
            while i < len(ch):
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(ch) and (ok[j + 1] or (j + 2 < len(ch) and ok[j + 2])):
                    j += 1
                seg = ch[i:j + 1]
                if sum(ok[i:j + 1]) >= min_n and seg[-1][1] - seg[0][0] >= min_ext:
                    for a, b in seg:
                        out[y, a:b + 1] = thin[y, a:b + 1]
                i = j + 1
    return out


def remove_dashes(img):
    """Dashed / dash-dot lines (function boxes, section dividers) are never circuit: paint
    their dashes over, keeping pixels where a stroke crosses them."""
    if not DASH_MASK:
        return img, 0
    paper = _paper(img)
    d = np.abs(img.astype(np.int16) - paper.astype(np.int16)).sum(2)
    ink = d > 90
    inkT = np.ascontiguousarray(ink.T)
    tud, tudT = _thin_runs(ink, 6), _thin_runs(inkT, 6)
    # pass 1 without the side-clearance test: candidate dashed lines; pass 2 requires paper
    # beside each dash, except where the neighbour is itself a candidate dashed line (two
    # adjacent function boxes draw their shared edge twice)
    # (widened along the line so that the dots of a dash-dot neighbour are covered too)
    k = np.ones((1, 9), np.uint8)
    c = cv2.dilate(_dash_rows(ink, tud).astype(np.uint8), k).astype(bool)
    cT = cv2.dilate(_dash_rows(inkT, tudT).astype(np.uint8), k).astype(bool)
    m = _dash_rows(ink, tud, side_ink=ink & ~c)
    m |= _dash_rows(inkT, tudT, side_ink=inkT & ~cT).T
    n = int(m.sum())
    if n == 0:
        return img, 0
    halo = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~ink
    out = img.copy()
    out[m | halo] = paper
    return out, n

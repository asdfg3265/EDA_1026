"""Style normalisation for schematics that are not drawn in the EasyEDA palette.

The parser reads colour semantics: dark red = symbol strokes and pins, green = wires, blue /
navy = designators, values and net names, black = pin numbers and pin names, bright red =
junction dots.  Altium, KiCad and other tools use other palettes (navy wires and black pins,
teal reference text, wire-coloured junction dots, yellow body fills, sheet grids).  This
module recognises such an image, gives every ink pixel a role from its colour family AND its
shape (long straight runs, glyph-sized blobs, filled dots), and repaints the image in the
EasyEDA palette so that the rest of the pipeline runs unchanged.

EasyEDA-coloured images are returned untouched.
"""
import os
import numpy as np
import cv2

ENABLED = os.environ.get('PCB_STYLE_NORM', '1') == '1'
_LAST = {}
DEBUG = os.environ.get('PCB_STYLE_DEBUG', '0') == '1'     # keep role maps of the last call

BG = np.array([255, 252, 248], np.float32)          # EasyEDA Pro paper (RGB)
T_WIRE = np.array([0, 136, 0], np.float32)
T_SYM = np.array([160, 0, 0], np.float32)
T_BLUE = np.array([0, 0, 255], np.float32)
T_BLACK = np.array([0, 0, 0], np.float32)
T_DOT = np.array([204, 0, 0], np.float32)

FAMILIES = ('red', 'green', 'navy', 'blue', 'teal', 'purple', 'yellow', 'black', 'gray')
MAIN_FAMS = (0, 1, 2, 3, 4, 7)                       # red green navy blue teal black


def _bg(img):
    small = img[1::3, 2::7].reshape(-1, 3).astype(np.int32)     # odd strides: not grid-aligned
    q = small // 4
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, cnt = np.unique(keys, return_counts=True)
    sel = small[keys == vals[np.argmax(cnt)]]
    return sel.mean(0).astype(np.float32)


def _paper_map(img, bg):
    """Per-pixel paper colour: the sheet colour, or the body fill colour (Altium / KiCad
    yellow) inside filled symbol bodies, so that ink on a fill keeps its colour family."""
    small = img[1::3, 2::7].reshape(-1, 3).astype(np.int32)
    q = small // 8
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, cnt = np.unique(keys, return_counts=True)
    fills = []
    for v, c in zip(vals[np.argsort(-cnt)], np.sort(cnt)[::-1]):
        if c < 0.003 * len(small):
            break
        col = small[keys == v].mean(0)
        dm = np.clip(bg - col, 0, None).sum()
        if 20 <= dm <= 160 and np.abs(col - bg).max() >= 15:
            d = np.clip(bg - col, 0, None)
            # coloured (not a gray grid): one channel clearly lower than the others
            if d.max() - d.min() >= 0.6 * d.max():
                fills.append(col.astype(np.float32))
    if not fills:
        return bg, None
    H, W = img.shape[:2]
    bgmap = np.empty((H, W, 3), np.float32)
    bgmap[:] = bg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    f = img.astype(np.int16)
    real = []
    ring = np.zeros((H, W), bool)
    for col in fills:
        near = (np.abs(f - col[None, None].astype(np.int16)).max(2) <= 12).astype(np.uint8)
        # a fill is solid (body interiors); anti-aliased stroke edges of the same tint are thin
        core = cv2.morphologyEx(near, cv2.MORPH_OPEN, k)
        if core.sum() < 0.002 * H * W:
            continue
        # every pixel with plenty of fill around it (text and strokes on the fill, and the
        # paper just outside a body, which is lighter than the fill and stays paper anyway)
        # the fill paper covers the bounding rectangle of each solid fill area (bodies are
        # rectangles; an opening keeps a large area's full extent), text and strokes on it
        # included; the ring around it (the outline and its anti-aliased edges) decides per pixel
        n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
        for j in range(1, n):
            x, y, w, h, a = st[j]
            bgmap[y:y + h, x:x + w] = col
            ring[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6] = True
        real.append(col)
    if not real:
        return bg, None
    inner = (bgmap != bg[None, None]).any(2)
    # any pixel of a fill colour is paper, also outside the rectangles (narrow strips between
    # pin names and the outline)
    fillpx = np.zeros((H, W), bool)
    for col in real:
        fillpx |= np.abs(f - col[None, None].astype(np.int16)).max(2) <= 12
    return bgmap, (inner, ring & ~inner, fillpx)


def _family(img, bg):
    """Colour family of every pixel from the direction of its deficit against the paper
    (bg - pixel), which does not depend on how strongly the pixel is inked."""
    f = img.astype(np.float32)
    d = np.clip((bg if bg.ndim == 3 else bg[None, None]) - f, 0, None)
    mag = d.sum(2)
    eps = 1e-3
    fr, fg, fb = d[..., 0] / (mag + eps), d[..., 1] / (mag + eps), d[..., 2] / (mag + eps)
    hi = np.maximum(np.maximum(fr, fg), fb)
    lo = np.minimum(np.minimum(fr, fg), fb)
    spread = hi - lo
    fam = np.full(mag.shape, 8, np.uint8)            # gray
    ink = mag > 60
    colored = ink & (spread >= 0.12)
    neutral = ink & ~colored
    fam[neutral & (mag > 360)] = 7                    # black
    low_r = colored & (fr <= fg - 0.08) & (fr <= fb - 0.08)
    low_g = colored & (fg <= fr - 0.08) & (fg <= fb - 0.08)
    low_b = colored & (fb <= fr - 0.08) & (fb <= fg - 0.08)
    hi_r = colored & ~low_r & ~low_g & ~low_b & (fr >= fg + 0.08) & (fr >= fb + 0.08)
    hi_g = colored & ~low_r & ~low_g & ~low_b & (fg >= fr + 0.08) & (fg >= fb + 0.08)
    hi_b = colored & ~low_r & ~low_g & ~low_b & (fb >= fr + 0.08) & (fb >= fg + 0.08)
    fam[low_r] = 0                                    # red (ink lacks red deficit)
    fam[low_g] = 1                                    # green
    fam[low_b & (fb >= 0.1)] = 2                      # navy
    fam[low_b & (fb < 0.1)] = 3                       # bright blue
    fam[hi_r] = 4                                     # teal / cyan
    fam[hi_g] = 5                                     # purple / magenta
    fam[hi_b] = 6                                     # yellow (fills, highlights)
    fam[~ink] = 255
    return fam, mag


def _extrapolate_ink(img, paper):
    """The saturated ink each pixel is a blend of, seen from the given paper: extend the line
    from the paper through the pixel to the edge of the RGB cube."""
    f = img.astype(np.float32)
    paper = np.asarray(paper, np.float32)
    d = f - paper
    with np.errstate(divide='ignore', invalid='ignore'):
        lim_lo = np.where(d < -1e-3, paper / np.maximum(-d, 1e-3), np.inf)
        lim_hi = np.where(d > 1e-3, (255.0 - paper) / np.maximum(d, 1e-3), np.inf)
    t = np.minimum(lim_lo.min(2), lim_hi.min(2))
    t = np.where(np.isfinite(t), np.maximum(t, 1.0), 1.0)
    return np.clip(paper + t[..., None] * d, 0, 255)


def _family_unmixed(img, paper, bg):
    """Families of pixels drawn on a coloured paper (a body fill): the ink colour is
    extrapolated from the paper through the pixel (anti-aliased and dark inks are blends of
    paper and a saturated ink) and classified against the sheet colour like any other ink.
    The ink threshold uses the deficit against the paper."""
    ink = _extrapolate_ink(img, paper)
    fam, _ = _family(ink.astype(np.uint8), bg)
    mag = np.clip(np.asarray(paper, np.float32) - img.astype(np.float32), 0, None).sum(2)
    fam[mag <= 60] = 255
    return fam, mag, ink


def _long_lines(m, L):
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    h = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    v = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)
    return (h | v) > 0


def _segment_contacts(fam, i, L):
    """(number of long straight segment ends, fraction of them touching ink of ANOTHER
    colour family).  Wires end on pins, labels and junctions drawn in other colours;
    the edges of a symbol outline end in its own corners."""
    m = (fam == i).astype(np.uint8)
    H, W = fam.shape
    tot = oth = 0
    for k in ((L, 1), (1, L)):
        op = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        n, lab, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
        for j in range(1, n):
            x, y, w, h = st[j, 0], st[j, 1], st[j, 2], st[j, 3]
            if k[0] > 1:
                cy = y + h // 2
                ends = ((x - 1, cy, -1, 0), (x + w, cy, 1, 0))
            else:
                cx = x + w // 2
                ends = ((cx, y - 1, 0, -1), (cx, y + h, 0, 1))
            for ex, ey, dx, dy in ends:
                tot += 1
                for s_ in range(4):
                    xx, yy = ex + dx * s_, ey + dy * s_
                    if 0 <= xx < W and 0 <= yy < H:
                        v = fam[yy, xx]
                        if v != 255 and v != i and v != 8 and v != 6:
                            oth += 1
                            break
    return tot, (oth / tot if tot else 0.0)


WIRE_FAMS = ('green', 'navy', 'blue', 'red', 'teal', 'purple', 'black')


def analyse(img):
    """Return (style, info) - style 'easyeda' when no repaint is needed.  info['wire'] is the
    colour family that carries the wires."""
    bg = _bg(img)
    bgmap, areas = _paper_map(img, bg)
    fam, mag = _family(img, bg)
    if areas is not None:
        # on and around body fills: the family against the fill where that gives a real ink
        # colour and the sheet does not (inside), the reverse just outside
        inner, ring, fillpx = areas
        fam1, mag1, ink1 = _family_unmixed(img, bgmap, bg)
        main = np.isin(fam, MAIN_FAMS)
        main1 = np.isin(fam1, MAIN_FAMS)
        # around the fill an edge pixel belongs to the paper that makes its ink the darker
        # (outline edges on the sheet side come out dark from the sheet, purple from the fill)
        ink0 = _extrapolate_ink(img, bg)
        darker1 = ink1.sum(2) < ink0.sum(2)
        use1 = (inner & (main1 | ~main)) | (ring & ~main & main1 & darker1)
        fam = np.where(use1, fam1, fam)
        mag = np.where(use1, mag1, mag)
        fam[fillpx] = 255
        mag[fillpx] = 0
    inked = fam != 255
    n_ink = max(1, int(inked.sum()))
    stats = {}
    L = max(16, int(round(min(img.shape[:2]) / 60)))
    for i, name in enumerate(FAMILIES):
        m = (fam == i).astype(np.uint8)
        n = int(m.sum())
        if n == 0:
            continue
        ll = _long_lines(m, L)
        st_ = dict(n=n, frac=n / n_ink, line=float(ll.sum()) / n, line_px=int(ll.sum()))
        if name in WIRE_FAMS and st_['line_px'] > 0:
            st_['ends'], st_['contact'] = _segment_contacts(fam, i, L)
        stats[name] = st_
    info = dict(bg=bg, fam=fam, mag=mag, stats=stats, L=L)
    # the wire family.  Green wires (EasyEDA, KiCad) win whenever green has a few dozen
    # straight runs ending on other colours; otherwise the coloured family with most such
    # ends (Altium navy), then black, then red (symbol pins always end on wires, so red and
    # black only as fallbacks)
    def _sc(k):
        v = stats.get(k, {})
        return v.get('ends', 0) * v.get('contact', 0.0)
    g = stats.get('green', {})
    if _sc('green') >= 8 and g.get('line', 0.0) >= 0.4:
        wire = 'green'
    else:
        wire = None
        for group in (('navy', 'blue', 'teal', 'purple'), ('black',), ('red',)):
            cand = {k: _sc(k) for k in group}
            best = max(cand, key=cand.get)
            if cand[best] >= 8:
                wire = best
                break
        # the wire colour must carry a fair share of the straight runs (a few coloured
        # junction marks on black wires do not make a wire colour)
        tot = sum(v.get('line_px', 0) for k, v in stats.items() if k not in ('gray', 'yellow'))
        if wire is not None and stats.get(wire, {}).get('line_px', 0) < 0.1 * max(tot, 1):
            wire = None
        if wire is None:
            info['wire'] = None
            return 'mono', info             # no wire colour found: leave the image alone
    info['wire'] = wire
    # text colours: EasyEDA writes designators, values and names in blue / navy
    th = _text_height([(fam == i).astype(np.uint8) for i in range(8)])
    info['th'] = th
    allink = ((fam != 255) & (fam != 6) & (fam != 8)).astype(np.uint8)     # no fills, no grid
    lines_all = _long_lines(allink, L)
    info['lines_all'] = lines_all
    gl = {}
    for i, name in enumerate(FAMILIES[:8]):
        if name == 'yellow' or name not in stats:
            continue
        g = _glyph_mask((fam == i).astype(np.uint8), 0.45 * th, 1.5 * th, lines_all)
        gl[name] = int(g.sum())
    info['glyph'] = gl
    bn = gl.get('blue', 0) + gl.get('navy', 0)
    text_tot = max(1, bn + gl.get('teal', 0) + gl.get('black', 0) + gl.get('purple', 0))
    r = stats.get('red', {})
    # EasyEDA never writes in teal; KiCad writes references, values and pin names in teal
    if wire == 'green' and r.get('n', 0) > 0 and (bn >= 0.15 * text_tot or bn >= 2000) \
            and gl.get('teal', 0) < max(300, 0.2 * bn):
        return 'easyeda', info
    if wire == 'green':
        return 'kicad', info
    if wire in ('navy', 'blue'):
        return 'altium', info
    return 'unknown', info


def _glyph_mask(m, h_lo, h_hi, lines):
    """Connected components of m (minus long lines) of text-glyph size."""
    rest = (m > 0) & ~lines
    n, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        a = st[i, cv2.CC_STAT_AREA]
        if h_lo <= h <= h_hi and w <= 1.6 * h_hi and a >= 4:
            keep[i] = True
        elif h < h_lo and w < h_lo and a >= 2:
            keep[i] = True                      # dots of i / j, punctuation
    return keep[lab]


def _text_mask(m, h_lo, h_hi, lines, t, grid=None, words=True, digits_only=False):
    """Text glyphs in a mask that also holds wires: words (several strokes per row / column),
    single glyphs, and thin glyphs ('1', 'l', '-') only next to other glyphs.  One-pixel gaps
    (a grid line crossing the text) are bridged."""
    if lines is None:
        lines = np.zeros(m.shape, bool)
    mm = m > 0
    if grid is not None:
        # grid pixels that cut a stroke (ink on both sides ACROSS the grid line) belong to it;
        # along the line nothing is bridged (the gap between the feet of an omega)
        g_h, g_v = grid if isinstance(grid, tuple) else (grid, grid)
        up = np.zeros_like(mm); up[1:] = mm[:-1]
        dn = np.zeros_like(mm); dn[:-1] = mm[1:]
        lf = np.zeros_like(mm); lf[:, 1:] = mm[:, :-1]
        rt = np.zeros_like(mm); rt[:, :-1] = mm[:, 1:]
        mm = mm | (g_h & up & dn) | (g_v & lf & rt)
    rest = (mm & ~lines).astype(np.uint8)
    cl = rest
    n, lab, st, _ = cv2.connectedComponentsWithStats(cl, connectivity=8)
    kind = np.zeros(n, np.uint8)                 # 1 glyph / word, 2 thin candidate

    def classify(w, h, sub):
        """Glyph tests for text running along x (h = glyph height); 0 none, 1 text, 2 thin."""
        if not (h_lo <= h <= h_hi):
            return 0
        if w <= max(2, 0.3 * h):                # thin: '1', 'l', 'I' (or a stroke)
            return 2 if h <= 0.8 * h_hi else 0  # '1', 'l', '[', '|' about cap height
        if w <= 1.6 * h_hi:
            return 1
        if w > 10 * h_hi or not words:
            return 0
        runs_c = (np.diff(sub.astype(np.int8), axis=0) == 1).sum(0) + sub[0]
        runs_r = (np.diff(sub.astype(np.int8), axis=1) == 1).sum(1) + sub[:, 0]
        # words: several strokes across most columns, three somewhere (E, B, 8, S, a, e);
        # a rectangle outline never has three
        # (a zigzag resistor crosses rows three times but its columns once: not a word)
        c2 = (runs_c >= 2).mean()
        if ((runs_r >= 3).mean() >= 0.25 and c2 >= 0.3) or (c2 >= 0.25 and (runs_c >= 3).mean() >= 0.08):
            return 1
        return 0

    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h < h_lo and w < h_lo and a >= 2:
            kind[i] = 2                         # dot of i / j, punctuation
            continue
        sub = (lab[y:y + h, x:x + w] == i)
        k1 = classify(w, h, sub)
        # vertical text (rotated 90 degrees): the same tests on the transposed glyph
        # (only as full glyphs: a short wire stub would pass as a rotated '1')
        k2 = classify(h, w, sub.T) if k1 != 1 else 0
        kind[i] = 1 if 1 in (k1, k2) else (2 if k1 == 2 else 0)
        if kind[i] and (_diag_stroke(sub, t) or (digits_only and _chevron(sub, t))):
            kind[i] = 0                         # a lone slanted stroke: the edge of a flag
    good = (kind == 1)[lab]
    near = cv2.dilate(good.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=max(2, int(round(0.4 * h_hi)))) > 0
    for i in np.nonzero(kind == 2)[0]:
        x, y, w, h, a = st[i]
        if near[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].any():
            kind[i] = 1
    return (kind == 1)[lab] & (m > 0)


def _diag_stroke(sub, t):
    """One straight slanted stroke (20-70 degrees): the pointed end of a net flag or a
    transistor lead, never a character (only '/' is drawn so)."""
    ys, xs = np.nonzero(sub)
    if len(xs) < 6:
        return False
    pts = np.stack([xs, ys], 1).astype(np.float32)
    pts -= pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov(pts.T) + 1e-6 * np.eye(2))
    major = evec[:, 1]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180
    ang = min(ang, 180 - ang)
    if not 20 <= ang <= 70:
        return False
    length = 4.0 * np.sqrt(ev[1])                 # ~ full length of a uniform stroke
    width = 4.0 * np.sqrt(ev[0])
    return length >= 6 and width <= max(2.5, 1.2 * t + 1)


def _chevron(sub, t):
    """Two slanted strokes meeting in a point ('>' of a flag tip): no digit looks so."""
    ys, xs = np.nonzero(sub)
    if len(xs) < 10:
        return False
    for ax in (ys, xs):
        mid = np.median(ax)
        a, b = ax < mid, ax > mid
        if a.sum() < 5 or b.sum() < 5:
            continue
        pa = np.zeros_like(sub); pa[ys[a], xs[a]] = True
        pb = np.zeros_like(sub); pb[ys[b], xs[b]] = True
        if _diag_stroke(pa, t) and _diag_stroke(pb, t):
            sa = np.sign(np.cov(xs[a], ys[a])[0, 1]); sb = np.sign(np.cov(xs[b], ys[b])[0, 1])
            if sa != sb:
                return True
    return False


def _line_width(m, L):
    """Most common thickness of the long horizontal / vertical runs in mask m."""
    hs = []
    for k, ax in (((L, 1), 0), ((1, L), 1)):
        op = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, k))
        if not op.any():
            continue
        a = op.astype(np.int8) if ax == 0 else op.T.astype(np.int8)
        # run lengths across the line direction
        d = np.diff(np.pad(a, ((1, 1), (0, 0))), axis=0)
        st = np.argwhere(d.T == 1)
        en = np.argwhere(d.T == -1)
        if len(st) and len(st) == len(en):
            hs.extend((en[:, 1] - st[:, 1]).tolist())
    if not hs:
        return 2
    hs = np.array(hs)
    hs = hs[hs <= 12]
    return int(np.bincount(hs).argmax()) if len(hs) else 2


def _dots(m, t):
    """Filled round blobs (junction dots) in mask m: pixels within r of a point that is at
    least r deep inside m, r = line width (a crossing of two lines has no such point; a
    morphological opening would also leak its kernel's tips along the wires)."""
    r = max(2, int(t))
    m8 = (m > 0).astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    core = (dt >= r + 0.5).astype(np.uint8)
    if not core.any():
        return np.zeros(m.shape, bool)
    d2 = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
    return (d2 <= r + 0.5) & (m8 > 0)


def _stroke_width(m):
    if m.sum() < 20:
        return 2.0
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    sk = dt[(dt > 0) & (cv2.dilate(dt, np.ones((3, 3), np.uint8)) == dt)]
    return float(max(1.0, 2 * np.median(sk))) if len(sk) else 2.0


def _text_height(masks):
    hs = []
    for m in masks:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if 6 <= h <= 60 and 0.2 * h <= w <= 1.2 * h:
                hs.append(h)
    if len(hs) < 10:
        return 12.0
    hist = np.bincount(np.array(hs))
    # the most common glyph height (capitals / digits)
    return float(np.argmax(hist[6:]) + 6)


def _box_interiors(sym, L):
    """Interiors of closed symbol outlines at least L wide and tall (IC bodies)."""
    m = (sym > 0).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    H, W = m.shape
    # flood the paper from the border; what stays unfilled and is not ink is enclosed
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if not (w >= L and h >= L and a >= 0.5 * w * h and w * h <= 0.4 * H * W):
            continue
        # a body is a rectangle drawn in one piece: every side of the hole is outline along
        # (nearly) its whole length; loops of wires through other symbols are not
        sides = (m[y - 1, x:x + w], m[y + h, x:x + w], m[y:y + h, x - 1], m[y:y + h, x + w])
        if min(float(sd.mean()) for sd in sides) >= 0.9:
            keep[j] = True
    out = np.zeros((H, W), bool)
    for j in np.nonzero(keep)[0]:
        x, y, w, h, a = st[j]
        out[y:y + h, x:x + w] = True             # bodies are rectangles: text holes included
    return out


def _pin_number_zone(sym, inside, th):
    """Where pin numbers stand: beside straight symbol strokes that leave an IC body (pins),
    up to 1.4 text heights off the pin line, over the pin's length."""
    H, W = sym.shape
    zone = np.zeros((H, W), bool)
    pins = np.zeros((H, W), bool)
    if not inside.any():
        return zone, pins
    k = max(4, int(0.8 * th))
    # (a pin may start a little off the outline: Altium draws inversion dots / clock marks)
    kn = max(7, int(round(2 * th)) + 1)
    near = cv2.dilate(inside.astype(np.uint8), np.ones((kn, kn), np.uint8)) > 0
    edge = cv2.dilate(inside.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    s8 = sym.astype(np.uint8)
    off = int(round(1.4 * th))
    for kern, horiz in (((k, 1), True), ((1, k), False)):
        op = cv2.morphologyEx(s8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kern))
        n, lab, st, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
        for j in range(1, n):
            x, y, w, h, a = st[j]
            # the line itself: the run of rows (columns) with the most pixels; text touching a
            # pin in a small image is part of the opened blob but not of the line
            sub = lab[y:y + h, x:x + w] == j
            prof = sub.sum(1) if horiz else sub.sum(0)
            r0 = int(np.argmax(prof))
            ra = r0
            while ra > 0 and prof[ra - 1] >= 0.7 * prof[r0]:
                ra -= 1
            rb = r0
            while rb + 1 < len(prof) and prof[rb + 1] >= 0.7 * prof[r0]:
                rb += 1
            if rb - ra + 1 > max(3, 0.4 * th):
                continue                         # a thick bar, not a pin line
            line = sub[ra:rb + 1] if horiz else sub[:, ra:rb + 1]
            cols = np.nonzero(line.any(0) if horiz else line.any(1))[0]
            if horiz:
                y, h, x, w = y + ra, rb - ra + 1, x + cols.min(), cols.max() - cols.min() + 1
            else:
                x, w, y, h = x + ra, rb - ra + 1, y + cols.min(), cols.max() - cols.min() + 1
            if (w if horiz else h) < 1.4 * th:
                continue                         # no longer than a glyph: a '1' or an 'l'
            nb = near[y:y + h, x:x + w]
            if not nb.any() or edge[y:y + h, x:x + w].mean() > 0.5:
                continue                         # not at a body, or the body outline itself
            pins[y:y + h, x:x + w] |= sym[y:y + h, x:x + w] > 0
            ext = int(round(1.0 * th))           # a number may run past the pin's end
            if horiz:
                zone[max(0, y - off):min(H, y + h + 2), max(0, x - ext):x + w + ext] = True
            else:
                zone[max(0, y - ext):y + h + ext, max(0, x - off):min(W, x + w + off)] = True
    return zone & ~inside, pins


def _short_lines(m, th):
    """Straight runs longer than a glyph (1.3 text heights): pins, not digits."""
    k = max(6, int(round(1.3 * th)))
    m8 = (m > 0).astype(np.uint8)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    # a digit standing on a pin makes a run through both: take lines out in both orders and
    # keep what both orders call a line
    h1 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kh)
    v1 = cv2.morphologyEx(m8 & (1 - h1), cv2.MORPH_OPEN, kv)
    v2 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kv)
    h2 = cv2.morphologyEx(m8 & (1 - v2), cv2.MORPH_OPEN, kh)
    return ((h1 | v1) & (v2 | h2)) > 0


def _touching(mask, other):
    """Connected components of mask that touch other."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    hit = np.zeros(n, bool)
    hit[np.unique(lab[(lab > 0) & other])] = True
    return hit[lab]


def _centres_in(mask, zone):
    """Connected components of mask whose centre lies in zone."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    H, W = mask.shape
    cy = np.clip(st[:, 1] + st[:, 3] // 2, 0, H - 1)
    cx = np.clip(st[:, 0] + st[:, 2] // 2, 0, W - 1)
    ok = zone[cy, cx]
    ok[0] = False
    return ok[lab] & (mask > 0)


def _nearest_role(role, mag, sel):
    """For pixels in sel: role and position offset of the strongest-ink 8-neighbour that has a
    role (cv2.dilate would take the largest role number instead)."""
    H, W = role.shape
    best_m = np.zeros((H, W), np.float32)
    best_r = np.zeros((H, W), np.uint8)
    best_f = np.full((H, W), -1, np.int32)
    rp = np.pad(role, 1)
    mp = np.pad(mag, 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            r_ = rp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            m_ = mp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            upd = sel & (r_ > 0) & (m_ > best_m)
            best_m[upd] = m_[upd]
            best_r[upd] = r_[upd]
            best_f[upd] = (dy + 1) * 3 + (dx + 1)
    return best_r, best_f


UNDERSCORE = os.environ.get('PCB_STYLE_UNDERSCORE', '1') == '1'


def _underscores(role, th):
    """Altium writes net names in its wire colour: the '_' of VIN_BOARD is a short straight
    run of wire colour and was given to the wires, splitting the name in two.  A short, thin,
    horizontal piece of 'wire' that touches no symbol stroke, wire or dot and has letters
    right before and after it, at its height or just above, is text."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((role == 1).astype(np.uint8), connectivity=8)
    if n <= 1:
        return
    txt = (role == 3) | (role == 4)
    H, W = role.shape
    gap = max(2, int(round(0.6 * th)))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (h <= max(3, 0.3 * th) and 0.3 * th <= w <= 1.6 * th):
            continue
        y0, y1 = max(y - int(round(1.2 * th)), 0), min(y + h + 1, H)
        x0, x1 = max(x - 2, 0), min(x + w + 2, W)
        ring = role[max(y - 2, 0):min(y + h + 2, H), x0:x1]
        if ((ring == 2) | (ring == 5)).any():
            continue                        # a wire stub on a pin or a dot
        if not (txt[y0:y1, max(x - gap, 0):x + 1].any() and txt[y0:y1, x + w - 1:min(x + w + gap, W)].any()):
            continue
        # the letters stand on it: their ink sits above, not below
        if txt[y + h:min(y + h + int(round(0.5 * th)), H), x:x + w].sum() > txt[y0:y, x:x + w].sum():
            continue
        role[lab == i] = 3


def normalise(img):
    """img RGB uint8 -> (repainted RGB uint8, style)."""
    if not ENABLED:
        return img, 'off'
    style, info = analyse(img)
    if style in ('easyeda', 'mono'):
        return img, style
    fam, mag = info['fam'], info['mag']
    H, W = fam.shape
    L = info['L']
    stats = info['stats']
    # ink strength per colour family: a pixel as strong as the family's typical stroke
    # becomes a full-strength stroke of its target colour (AA edges stay partial)
    full = np.full(256, 400.0, np.float32)
    for i in range(len(FAMILIES)):
        sel = fam == i
        if int(sel.sum()) >= 50:
            full[i] = max(60.0, float(np.percentile(mag[sel], 90)))
    masks = {name: (fam == i).astype(np.uint8) for i, name in enumerate(FAMILIES)}
    # solid yellow areas are body fills the paper map missed: paper, not ink
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(info['th'])), max(3, int(info['th']))))
    solid = cv2.dilate(cv2.morphologyEx(masks['yellow'], cv2.MORPH_OPEN, ky), np.ones((3, 3), np.uint8)) > 0
    masks['yellow'][solid] = 0
    # a light-gray sheet grid is paper
    # (rows / columns where most pixels are light gray)
    gm = masks['gray'] > 0
    grow = gm.mean(1) >= 0.4
    gcol = gm.mean(0) >= 0.4
    grid = gm & (grow[:, None] | gcol[None, :])
    grid_hv = (gm & grow[:, None], gm & gcol[None, :])     # horizontal / vertical grid lines
    has_grid = bool(grow.any() or gcol.any())
    # role targets per family
    wire_f = (info['wire'],)
    if info['wire'] == 'green':
        # KiCad-like: symbols and pins dark red, pin numbers in the symbol colour
        # (olive global labels: 'yellow' ink once body fills are taken as paper)
        sym_f = tuple(f for f in ('red', 'black', 'navy', 'blue', 'teal', 'yellow') if f not in wire_f)
        text_blue = ('teal', 'navy', 'blue', 'black', 'purple', 'yellow')
        text_black = ('red',)
    else:
        # Altium-like: navy wires, black pins with black pin numbers / names, dark red
        # outlines with dark red net / power names
        sym_f = tuple(f for f in ('red', 'black', 'blue', 'navy', 'green', 'teal', 'purple', 'yellow')
                      if f not in wire_f)
        text_blue = ('navy', 'blue', 'red', 'teal', 'green', 'purple', 'yellow')
        text_black = ('black',)
    t = _stroke_width(np.isin(fam, [FAMILIES.index(f) for f in wire_f + sym_f]).astype(np.uint8))
    th = info['th']
    lines_all = info['lines_all']
    role = np.zeros((H, W), np.uint8)        # 0 none 1 wire 2 symbol 3 blue text 4 black text 5 dot
    # IC bodies and their pins, from all symbol colours: text roles go by position (tools
    # differ in which colour they use for what) - pin names inside a body and pin numbers
    # along its pins are black, everything else (designators, values, net and power names)
    # blue, as EasyEDA draws them
    sym_union = np.zeros((H, W), bool)
    for f in sym_f:
        if not (info['wire'] == 'green' and f in ('teal', 'black')):   # KiCad: text only
            sym_union |= masks[f] > 0
    # (outlines are straight runs longer than a glyph; text inside a body must not break it up)
    sym_struct = _long_lines(sym_union.astype(np.uint8), max(8, int(round(1.5 * th))))
    inside = _box_interiors(sym_struct, L)
    pz, pin_lines = _pin_number_zone(sym_union, inside, th)
    zone = inside | pz
    kicad = info['wire'] == 'green'
    for f in wire_f:
        m = masks[f]
        g = _text_mask(m, 0.45 * th, 1.8 * th, lines_all, t, grid_hv) if f in text_blue \
            else np.zeros(m.shape, bool)
        lw = _line_width(m, L)
        dots = _dots(m, lw)
        # a dot's anti-aliased rim stays with the dot even where text touches it
        dots = (cv2.dilate(dots.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (m > 0)
        g &= ~dots
        role[(m > 0) & g] = 3
        role[(m > 0) & ~g] = 1
        role[dots] = 5
    bus = np.zeros((H, W), bool)
    if kicad:
        # KiCad buses: long navy runs; they are not wires and not symbols - leave them out
        bus = _long_lines(masks['navy'], L)
        bus = cv2.dilate(bus.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        bus &= masks['navy'] > 0
        masks['navy'][bus] = 0
    for f in sym_f:
        m = masks[f]
        if kicad and f in ('teal', 'black'):
            g = m > 0                            # KiCad draws only text in teal and black
        elif f in text_black and kicad:
            # pin numbers in the symbol colour: free of the pin lines, and only along pins
            g = _text_mask(m, 0.45 * th, 1.8 * th, _short_lines(m, th), t, grid_hv, words=False, digits_only=True)
            g = _centres_in(g, zone)
        elif f in text_black:
            # Altium black: pins, pin numbers / names, designators and values
            g = _text_mask(m, 0.45 * th, 1.8 * th, _short_lines(m, th) | lines_all, t, grid_hv)
        else:
            # symbol colours: whole connected strokes (an arrow or diode triangle hangs on
            # its pin line, a label stands free); Altium writes net and power names in its
            # symbol colour, often as joined-up words in small images.  A glyph-sized piece
            # that touches strokes of another symbol colour (a blue capacitor on black pins)
            # is part of a symbol, not text
            g = _text_mask(m, 0.45 * th, 1.8 * th, None, t, grid_hv, words=not kicad)
            others = sym_union & (m == 0)
            if f != 'black' and others.any():
                od = cv2.dilate(others.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                g &= ~_touching(g, od)
        if kicad and f == 'yellow':
            # hierarchical / global label markers sit on the wire end: shape, not text
            wd = cv2.dilate(((role == 1) | bus).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            g &= ~_touching(g, wd)
        role[(m > 0) & g & (role == 0)] = 3
        role[(m > 0) & (role == 0)] = 2
    for f in set(text_blue + text_black) - set(wire_f + sym_f):
        m = masks[f]
        role[(m > 0) & (role == 0)] = 3
    text = role == 3
    role[_centres_in(text, zone)] = 4
    role[pin_lines & ((role == 3) | (role == 4))] = 2     # a pin under its number stays a pin
    if UNDERSCORE and not kicad:
        _underscores(role, th)          # Altium-like only: KiCad text is never wire-coloured
    # specks of wire colour (anti-aliased edges of other strokes over a grid line) are not
    # wires: they go back to the fringe step and take their neighbour's role
    # ... and so are short blobs of it that are not lines (a blue arrow over a black pin
    # blends to navy, Altium's wire colour)
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats((role == 1).astype(np.uint8), connectivity=8)
    if n_ > 1:
        lw_w = _line_width(masks[info['wire']], L)
        wl = _long_lines((role == 1).astype(np.uint8), L)
        has_line = np.zeros(n_, bool)
        has_line[np.unique(lab_[wl & (lab_ > 0)])] = True
        w_, h_ = st_[:, cv2.CC_STAT_WIDTH], st_[:, cv2.CC_STAT_HEIGHT]
        small = (st_[:, cv2.CC_STAT_AREA] <= 6) | (~has_line & (np.minimum(w_, h_) > 2 * lw_w + 1))
        small[0] = False
        role[small[lab_]] = 0
    alpha = np.clip(mag / full[fam], 0, 1)
    # anti-aliased fringe (below the ink threshold, or dropped gray) takes the role and the
    # strength scale of its strongest neighbour
    fr = (role == 0) & (mag > 25) & ~grid & ~solid & ~bus
    if fr.any():
        nr, nf = _nearest_role(role, mag, fr)
        fr &= nr > 0
        role[fr] = nr[fr]
        fp = np.pad(fam, 1)
        nfam = np.full((H, W), 255, np.uint8)
        for k in range(9):
            if k == 4:
                continue
            dy, dx = k // 3 - 1, k % 3 - 1
            s_ = fr & (nf == k)
            nfam[s_] = fp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W][s_]
        alpha[fr] = np.clip(mag[fr] / full[nfam[fr]], 0, 1)
    # a grid line crossing a stroke leaves a notch: grid pixels between two pixels of the same
    # role (across the grid line) take that role at their neighbours' strength
    if grid.any():
        for ax in (0, 1):
            a_ = np.roll(role, 1, axis=ax)
            b_ = np.roll(role, -1, axis=ax)
            fa = np.roll(alpha, 1, axis=ax)
            fb = np.roll(alpha, -1, axis=ax)
            # strokes only (wire, symbol): a one-pixel gap inside a letter (the feet of an
            # omega) can sit on a grid line too
            hole = grid_hv[ax] & (role == 0) & ((a_ == 1) | (a_ == 2)) & (a_ == b_)
            role[hole] = a_[hole]
            alpha[hole] = np.minimum(fa, fb)[hole]
    if DEBUG:
        global _LAST
        _LAST = dict(role=role, fam=fam, alpha=alpha, pins=pin_lines, zone=zone, inside=inside)
    # symbol red blended at 59-84 % looks like junction-dot red to the parser: strong edges
    # go to full strength, weak ones stay below that band
    sy = role == 2
    alpha[sy] = np.where(alpha[sy] >= 0.45, 1.0, np.minimum(alpha[sy], 0.55))
    tgt = np.zeros((H, W, 3), np.float32)
    tgt[role == 1] = T_WIRE
    tgt[role == 2] = T_SYM
    tgt[role == 3] = T_BLUE
    tgt[role == 4] = T_BLACK
    tgt[role == 5] = T_DOT
    out = np.empty((H, W, 3), np.float32)
    out[:] = BG
    sel = role > 0
    out[sel] = BG - alpha[sel][:, None] * (BG - tgt[sel])
    return np.clip(out, 0, 255).astype(np.uint8), style

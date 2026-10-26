"""Primitive extraction: scale, wires, junction dots, unconnected markers, red strokes."""
import os
import numpy as np
import cv2
from .colors import masks
from .geom import extract_segments, Seg


class Prims:
    pass


STROKE_MIN_RED = os.environ.get('PCB_STROKE_RED', '1') == '1'


def estimate_stroke(mask, axis_samples=9):
    m = mask.astype(np.int8)
    runs = []
    for x in range(0, m.shape[1], axis_samples):
        col = m[:, x]
        d = np.diff(np.r_[0, col, 0])
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        runs += list(e - s)
    for y in range(0, m.shape[0], axis_samples):
        row = m[y, :]
        d = np.diff(np.r_[0, row, 0])
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        runs += list(e - s)
    runs = np.array([r for r in runs if 1 <= r <= 12])
    if len(runs) == 0:
        return 3.0
    c = np.bincount(runs, minlength=13).astype(float)
    k = int(np.argmax(c))
    # sub-pixel estimate around mode
    lo, hi = max(k - 1, 1), min(k + 1, 12)
    w = c[lo:hi + 1]
    return float((np.arange(lo, hi + 1) * w).sum() / w.sum())


def find_junctions(red, green, t):
    """Junction dots: filled red discs sitting on wires."""
    dt = cv2.distanceTransform((red > 0).astype(np.uint8), cv2.DIST_L2, 5)
    thr = max(1.5, 0.95 * t)
    core = (dt >= thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(core, connectivity=8)
    dots = []
    gd = cv2.dilate(green, np.ones((3, 3), np.uint8), iterations=max(1, int(round(t))))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w > 6 * t + 6 or h > 6 * t + 6:
            continue
        cx, cy = cen[i]
        r = float(dt[lab == i].max())
        if r > 3.5 * t + 2:
            continue
        # must touch a wire
        yy, xx = int(round(cy)), int(round(cx))
        R = int(r + 2 * t + 2)
        win = gd[max(yy - R, 0):yy + R + 1, max(xx - R, 0):xx + R + 1]
        if win.sum() == 0:
            continue
        # roundness: area of red disc around
        dots.append((float(cx), float(cy), r))
    return dots


def remove_noconnect(green, t):
    """Remove EasyEDA no-connect 'X' flags (small green diagonal crosses)."""
    g = green.copy()
    n, lab, st, cen = cv2.connectedComponentsWithStats(green, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w > 12 * t or h > 12 * t or w < 2 * t or h < 2 * t:
            continue
        if 0.6 < w / h < 1.6:
            sub = (lab[y:y + h, x:x + w] == i)
            # diagonal content: the two diagonals are well covered, centre row not full
            k = min(w, h)
            d1 = np.mean([sub[int(j * h / k), int(j * w / k)] or sub[min(int(j * h / k) + 1, h - 1), int(j * w / k)] for j in range(k)])
            d2 = np.mean([sub[int(j * h / k), w - 1 - int(j * w / k)] or sub[min(int(j * h / k) + 1, h - 1), w - 1 - int(j * w / k)] for j in range(k)])
            if d1 > 0.7 and d2 > 0.7 and a < 0.6 * w * h:
                g[lab == i] = 0
                out.append((float(cen[i][0]), float(cen[i][1])))
    return g, out


def fill_dots(green, dots, t):
    """Wires are partially hidden under junction dots: paint the wire through the dot
    in every axis direction where a green wire leaves the dot."""
    g = green.copy()
    H, W = g.shape
    hw = max(1, int(round(t / 2)))
    for (cx, cy, r) in dots:
        x0, y0 = int(round(cx)), int(round(cy))
        reach = int(r + 3 * t + 3)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            hit = None
            for k in range(int(r * 0.5), reach):
                x, y = x0 + dx * k, y0 + dy * k
                if not (0 <= x < W and 0 <= y < H):
                    break
                # probe across the stroke width
                if dx:
                    win = green[max(y - hw - 1, 0):y + hw + 2, x]
                else:
                    win = green[y, max(x - hw - 1, 0):x + hw + 2]
                if win.sum() >= max(1, hw):
                    hit = k
                    break
            if hit is None:
                continue
            # find the actual centre line of the wire leaving the dot
            x, y = x0 + dx * (hit + 2), y0 + dy * (hit + 2)
            if dx:
                col = green[max(y - 3 * hw - 2, 0):y + 3 * hw + 3, min(max(x, 0), W - 1)]
                idx = np.where(col > 0)[0]
                yc = int(round(idx.mean())) + max(y - 3 * hw - 2, 0) if len(idx) else y0
                cv2.line(g, (x0, yc), (x0 + dx * (hit + 2), yc), 1, thickness=max(1, int(round(t))))
            else:
                row = green[min(max(y, 0), H - 1), max(x - 3 * hw - 2, 0):x + 3 * hw + 3]
                idx = np.where(row > 0)[0]
                xc = int(round(idx.mean())) + max(x - 3 * hw - 2, 0) if len(idx) else x0
                cv2.line(g, (xc, y0), (xc, y0 + dy * (hit + 2)), 1, thickness=max(1, int(round(t))))
    return g


def find_gray_dots(gray, t):
    n, lab, st, cen = cv2.connectedComponentsWithStats(gray, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 3 and w <= 4 * t + 4 and h <= 4 * t + 4 and 0.5 < w / max(h, 1) < 2.0 and a >= 0.3 * w * h:
            out.append((float(cen[i][0]), float(cen[i][1])))
    return out


def extract(img_rgb):
    P = Prims()
    P.img = img_rgb
    M = masks(img_rgb)
    P.M = M
    H, W = img_rgb.shape[:2]
    P.H, P.W = H, W
    green = M['green']
    red = M['red']
    t = estimate_stroke(green) if green.sum() > 200 else estimate_stroke(red)
    if STROKE_MIN_RED and green.sum() > 200 and red.sum() > 200:
        # thick coloured frames around sheet blocks (teal, classed as wire colour) outvote
        # the thin wires; symbol strokes are drawn with the wire width
        tr = estimate_stroke(red)
        if t > 1.3 * tr:
            t = tr
    P.t = t
    P.s = t / 1.5             # px per EasyEDA unit (approx)
    P.u = 10 * P.s            # one standard pin length
    fg = (M['mag'] > 60).astype(np.uint8)
    P.fg = fg
    # junction dots
    P.dots = find_junctions(red, green, t)
    # wires
    bridge = (M['lab'] > 0).astype(np.uint8)
    for (cx, cy, r) in P.dots:
        cv2.circle(bridge, (int(round(cx)), int(round(cy))), int(r + 2), 1, -1)
    green, P.noconn = remove_noconnect(green, t)
    green = fill_dots(green, P.dots, t)
    P.green = green
    textm = ((M['lab'] == 3) | (M['lab'] == 4) | (M['lab'] == 5) | (green > 0)).astype(np.uint8)
    segs = extract_segments(green, min_len=max(4, int(1.8 * t)), kind='w',
                            bridges=[(bridge, int(max(6, 0.75 * P.u))), (textm, int(4 * P.u))])
    # drop tiny wires that are just anti-aliasing debris
    P.wires = [s for s in segs if s.length >= max(4, 2.2 * t)]
    # residual diagonal wires
    P.diag = find_diagonals(green, P.wires, t)
    # red without dots
    red2 = red.copy()
    for (cx, cy, r) in P.dots:
        cv2.circle(red2, (int(round(cx)), int(round(cy))), int(np.ceil(r + 1.5)), 0, -1)
    if BRIDGE_WIRE_CROSSINGS:
        red2 = bridge_wire_crossings(red2, M['green'], t)
    red2 = remove_partition_frames(red2, green, P)
    if BRIDGE_TEXT_CROSSINGS:
        # (after the region frames are gone: a box title written across a frame line must
        # not tie the frame to the parts inside)
        red2 = bridge_text_crossings(red2, M['lab'], P.u)
    if CUT_CONTACT_DOTS:
        red2, P.contact_dots = cut_at_contact_dots(red2, M['green'], t)
    else:
        P.contact_dots = []
    P.red = red2
    P.gray_dots = find_gray_dots(M['gray'], t)
    return P


def find_diagonals(green, wires, t):
    g = green.copy()
    for s in wires:
        w = int(np.ceil(s.t / 2 + 1.5))
        if s.o == 'h':
            cv2.rectangle(g, (int(s.a) - 1, int(s.c - w)), (int(s.b) + 1, int(s.c + w)), 0, -1)
        else:
            cv2.rectangle(g, (int(s.c - w), int(s.a) - 1), (int(s.c + w), int(s.b) + 1), 0, -1)
    if g.sum() < 30:
        return []
    lines = cv2.HoughLinesP(g * 255, 1, np.pi / 180, threshold=int(8 * t), minLineLength=int(8 * t), maxLineGap=int(2 * t))
    out = []
    if lines is None:
        return out
    lines = np.asarray(lines).reshape(-1, 4)
    for l in lines:
        x1, y1, x2, y2 = map(float, l)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if min(ang, 180 - ang) < 8 or abs(ang - 90) < 8:
            continue
        out.append((x1, y1, x2, y2))
    # merge near-duplicates
    res = []
    for l in out:
        dup = False
        for r in res:
            if max(abs(l[0] - r[0]), abs(l[1] - r[1]), abs(l[2] - r[2]), abs(l[3] - r[3])) < 3 * t or \
               max(abs(l[0] - r[2]), abs(l[1] - r[3]), abs(l[2] - r[0]), abs(l[3] - r[1])) < 3 * t:
                dup = True
                break
        if not dup:
            res.append(l)
    return res


CUT_CONTACT_DOTS = True
BRIDGE_WIRE_CROSSINGS = True


def bridge_wire_crossings(red, green_raw, t):
    """A wire drawn across a symbol paints over the red outline and cuts the body in two.
    Restore red where a green stroke interrupts a straight red line (red on both sides)."""
    k = int(round(2 * t + 3)) | 1
    g = green_raw > 0
    out = red.copy()
    for ker in (np.ones((k, 1), np.uint8), np.ones((1, k), np.uint8)):
        closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ker)
        add = (closed > 0) & g & (red == 0)
        out[add] = 1
    return out


BRIDGE_TEXT_CROSSINGS = os.environ.get('PCB_BRIDGE_TEXT', '1') == '1'


def bridge_text_crossings(red, lab, u):
    """A designator or value set over a symbol (the label of a capacitor in a tight row
    written across the next capacitor's lead and plate) paints over the red strokes under
    its glyphs and cuts the symbol apart.  Restore red where text ink interrupts a straight
    red line that continues on both sides within about a text height."""
    k = int(round(0.6 * u)) | 1
    # blue text (designators, values) and the dark anti-aliased rim of its glyphs; black /
    # gray ink alone is pin numbers and no-connect dots, which sit on pin ends
    blue = (lab == 3)
    # glyph strokes, not the odd bluish pixel where a magenta bus line meets a red stroke
    n_, cl_, st_, _ = cv2.connectedComponentsWithStats(blue.astype(np.uint8), connectivity=8)
    keep_ = np.zeros(n_, bool)
    # (and not a blue bus line: a glyph spans at most about a text height)
    w_, h_, a_ = st_[1:, 2], st_[1:, 3], st_[1:, 4]
    bar = (a_ >= 0.85 * w_ * h_) & (np.minimum(w_, h_) >= 0.25 * u)     # a solid thick bus piece
    keep_[1:] = (a_ >= 8) & (np.maximum(w_, h_) <= 1.5 * u) & ~bar
    blue = keep_[cl_]
    txt = blue | (((lab == 4) | (lab == 5)) & (cv2.dilate(blue.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0))
    # never next to green: the anti-aliased rim of a contact dot or wire end is no text
    txt &= cv2.dilate((lab == 2).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    if not txt.any():
        return red
    out = red.copy()
    k2 = max(3, int(round(0.5 * u)))
    for ker, run, ext in ((np.ones((k, 1), np.uint8), np.ones((k2, 1), np.uint8), np.ones((2 * k + 1, 1), np.uint8)),
                          (np.ones((1, k), np.uint8), np.ones((1, k2), np.uint8), np.ones((1, 2 * k + 1), np.uint8))):
        closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, ker)
        # a stroke running in the bridging direction must end at the gap (a lead cut by a
        # label); text squeezed between two strokes that run across it (a frame line above a
        # box edge, two flag outlines side by side) is not bridged
        along = cv2.dilate(cv2.morphologyEx(red, cv2.MORPH_OPEN, run), ext)
        add = (closed > 0) & txt & (red == 0) & (along > 0)
        out[add] = 1
    return out


def cut_at_contact_dots(red, green_raw, t):
    """EasyEDA marks a direct pin-to-pin contact with a tiny green dot drawn on the red
    stroke.  Cut the red stroke there so both pins end (and get mated) at the dot."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(green_raw.astype(np.uint8), connectivity=8)
    H, W = red.shape
    lim = int(round(2 * t + 3))
    r = int(np.ceil(t / 2 + 1))
    out = red
    dots = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w > lim or h > lim or a > lim * lim:
            continue
        cx, cy = int(round(cen[i][0])), int(round(cen[i][1]))
        d = int(np.ceil(t + 2))
        if not (d <= cx < W - d and d <= cy < H - d):
            continue
        def redwin(xx, yy):
            return red[max(yy - 1, 0):yy + 2, max(xx - 1, 0):xx + 2].any()
        vert = redwin(cx, cy - d) and redwin(cx, cy + d)
        horz = redwin(cx - d, cy) and redwin(cx + d, cy)
        if vert == horz:          # needs exactly one straight stroke through the dot
            continue
        # the stroke must be thin there (not inside a filled body)
        if vert and (redwin(cx - d - 1, cy - d) or redwin(cx + d + 1, cy - d)):
            continue
        if horz and (redwin(cx - d, cy - d - 1) or redwin(cx - d, cy + d + 1)):
            continue
        if out is red:
            out = red.copy()
        if vert:
            out[cy - 1:cy + 2, cx - r - 1:cx + r + 2] = 0
        else:
            out[cy - r - 1:cy + r + 2, cx - 1:cx + 2] = 0
        dots.append((cx, cy, 'v' if vert else 'h'))
    return out, dots


def remove_partition_frames(red, green, P):
    """Hand-drawn region frames (rectangles/dividers in the symbol colour around groups of
    circuits) would glue the whole page into one symbol.  A red connected component whose
    box encloses wires is such a frame (an IC body never encloses wires); from it we erase
    the long straight runs that have (almost) nothing attached along their length - IC
    edges carry pins every unit, frame lines do not."""
    u, t = P.u, P.t
    n, lab, st, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    out = red
    L = int(round(15 * u))
    band = int(round(t + 3))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 15 * u or h < 8 * u:
            continue
        m = int(u)
        inner = green[y + m:y + h - m, x + m:x + w - m]
        if inner.size == 0 or inner.sum() < 3 * u * t:
            continue
        cm = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        hl = cv2.morphologyEx(cm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        vl = cv2.morphologyEx(cm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        erase = np.zeros_like(cm)
        runs = []          # (label map, label, horiz, x0, y0, x1, y1, attachments)
        for lm, horiz in ((hl, True), (vl, False)):
            k, ll, ls, _ = cv2.connectedComponentsWithStats(lm, connectivity=8)
            for j in range(1, k):
                lx, ly, lw, lh, la = ls[j]
                if horiz:
                    y0, y1 = ly, ly + lh
                    prof = []
                    for yy0, yy1 in ((max(0, y0 - band), max(0, y0 - 1)), (y1 + 1, min(h, y1 + band))):
                        if yy1 > yy0:
                            prof.append(cm[yy0:yy1, lx:lx + lw].any(0))
                else:
                    x0, x1 = lx, lx + lw
                    prof = []
                    for xx0, xx1 in ((max(0, x0 - band), max(0, x0 - 1)), (x1 + 1, min(w, x1 + band))):
                        if xx1 > xx0:
                            prof.append(cm[ly:ly + lh, xx0:xx1].any(1))
                att = 0
                for pr in prof:
                    pr = pr.astype(np.int8)
                    edge = int(2 * t + 3)
                    if len(pr) > 2 * edge:
                        pr[:edge] = 0
                        pr[-edge:] = 0
                    att += int((np.diff(np.concatenate([[0], pr, [0]])) == 1).sum())
                runs.append((ll, j, horiz, lx, ly, lx + lw, ly + lh, att))
        tol = 2 * t + 3
        for r in runs:
            ll, j, horiz, x0, y0, x1, y1, att = r
            if att > 1:
                continue
            # a pin-less side of an IC body: its corner neighbour carries pins -> keep
            body = False
            for q in runs:
                if q[2] == horiz or q[7] < 3:
                    continue
                qx0, qy0, qx1, qy1 = q[3:7]
                for ex, ey in (((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)) if horiz else
                               (((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1))):
                    if qx0 - tol <= ex <= qx1 + tol and qy0 - tol <= ey <= qy1 + tol:
                        body = True
            if not body:
                erase[ll == j] = 1
        if erase.any():
            erase = cv2.dilate(erase, np.ones((3, 3), np.uint8))
            if out is red:
                out = red.copy()
            sub = out[y:y + h, x:x + w]
            sub[(erase > 0) & (cm > 0)] = 0
    return out

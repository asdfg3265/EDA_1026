"""Ground symbols drawn in the wire colour.

Some sheets colour their ground symbols like the wires (green bars under a green stem).  The
parse sees only wire then: the bars are short dangling strokes and the symbol is lost (30
grounds in one sheet).  A ground symbol is a stack of two or more short parallel bars of
decreasing length, centred on one axis, under a longer bar that the stem ends on.  Such
stacks found in the wire colour are repainted in the symbol colour before the parse, so the
ordinary symbol / wire logic sees a ground symbol hanging on a wire.
"""
import os
import numpy as np
import cv2

GREEN_GND = os.environ.get('PCB_GREEN_GND', '1') == '1'


def _bars(green, u, t):
    """Isolated short straight strokes of the green mask: (axis, centre_along, centre_across,
    length, x0, y0, x1, y1) with axis 'h' for horizontal bars."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(green.astype(np.uint8), connectivity=8)
    out = []
    thick = max(3, int(round(2 * t + 1)))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h <= thick and 0.12 * u <= w <= 2.6 * u and a >= 0.6 * w * min(h, t):
            out.append(('h', x + w / 2.0, y + h / 2.0, w, x, y, x + w - 1, y + h - 1))
        elif w <= thick and 0.12 * u <= h <= 2.6 * u and a >= 0.6 * h * min(w, t):
            out.append(('v', y + h / 2.0, x + w / 2.0, h, x, y, x + w - 1, y + h - 1))
    return out


def find_green_grounds(green, u, t):
    """Pixel boxes (x0, y0, x1, y1) of green ground symbols' bars (the top bar included)."""
    bars = _bars(green, u, t)
    H, W = green.shape
    used = set()
    found = []
    for i, b in enumerate(bars):
        if i in used:
            continue
        ax = b[0]
        # bars of the same stack: same axis, same centre along the bar, a small step across
        col = [j for j, c in enumerate(bars) if c[0] == ax and abs(c[1] - b[1]) <= 0.15 * u
               and abs(c[2] - b[2]) <= 0.9 * u]
        if len(col) < 2:
            continue
        col.sort(key=lambda j: bars[j][2])
        across = [bars[j][2] for j in col]
        steps = np.diff(across)
        if len(steps) == 0 or steps.min() < 0.1 * u or steps.max() > 0.45 * u:
            continue
        lens = [bars[j][3] for j in col]
        # lengths shrink away from the stem: increasing or decreasing along the stack
        inc = all(a <= b_ + 1 for a, b_ in zip(lens, lens[1:]))
        dec = all(a + 1 >= b_ for a, b_ in zip(lens, lens[1:]))
        if not (inc or dec) or max(lens) - min(lens) < 0.15 * u:
            continue
        step = float(np.median(steps))
        # the longest bar sits next to the top bar (on the stem): one step further out
        if dec:
            top_c = across[0] - step
            big = bars[col[0]]
        else:
            top_c = across[-1] + step
            big = bars[col[-1]]
        half = big[3] / 2.0 + 0.35 * u
        c0 = big[1]
        if ax == 'h':
            y = int(round(top_c))
            x0, x1 = int(round(c0 - half)), int(round(c0 + half))
            band = green[max(y - int(t) - 1, 0):min(y + int(t) + 2, H), max(x0, 0):min(x1 + 1, W)]
            if band.size == 0:
                continue
            cols = np.nonzero(band.any(0))[0]
            tb = None
            if len(cols) >= big[3]:
                tb = (max(x0, 0) + int(cols.min()), max(y - int(t) - 1, 0), max(x0, 0) + int(cols.max()),
                      min(y + int(t) + 1, H - 1))
                if tb[2] - tb[0] > 2.8 * u:
                    tb = None
        else:
            x = int(round(top_c))
            y0, y1 = int(round(c0 - half)), int(round(c0 + half))
            band = green[max(y0, 0):min(y1 + 1, H), max(x - int(t) - 1, 0):min(x + int(t) + 2, W)]
            if band.size == 0:
                continue
            rows = np.nonzero(band.any(1))[0]
            tb = None
            if len(rows) >= big[3]:
                tb = (max(x - int(t) - 1, 0), max(y0, 0) + int(rows.min()), min(x + int(t) + 1, W - 1),
                      max(y0, 0) + int(rows.max()))
                if tb[3] - tb[1] > 2.8 * u:
                    tb = None
        if tb is None and len(col) < 3:
            continue                         # (the longest bar is the top bar itself: 3 bars needed)
        used.update(col)
        found.append(([tb] if tb else []) + [tuple(bars[j][4:8]) for j in col])
    return found


def repaint(img, boxes_list, green_mask, red_rgb=(160, 0, 0)):
    """Repaint the green pixels inside the given boxes in the symbol colour, keeping their
    anti-aliasing against the paper."""
    out = img.copy()
    H, W = green_mask.shape
    for boxes in boxes_list:
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(int(x0) - 1, 0), max(int(y0) - 1, 0)
            x1, y1 = min(int(x1) + 1, W - 1), min(int(y1) + 1, H - 1)
            sub = out[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
            r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
            gm = (g > r + 25) & (g > b + 25)          # green-ish ink (incl. blended edges)
            a = np.clip((255.0 - r) / 255.0, 0, 1)
            for ch in range(3):
                v = a * red_rgb[ch] + (1 - a) * 252.0
                sub[..., ch] = np.where(gm, v, sub[..., ch])
            out[y0:y1 + 1, x0:x1 + 1] = sub.astype(np.uint8)
    return out

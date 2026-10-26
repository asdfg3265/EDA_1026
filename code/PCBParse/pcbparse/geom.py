"""Axis-aligned stroke extraction from binary masks."""
import numpy as np
import cv2


class Seg:
    __slots__ = ('o', 'c', 'a', 'b', 't', 'id', 'kind', 'start_both', 'end_both')

    def __init__(self, o, c, a, b, t=1.0, kind=''):
        self.o = o      # 'h' or 'v'
        self.c = c      # constant coordinate (y for h, x for v) in image pixel coords (float)
        self.a = a      # start (min) along axis
        self.b = b      # end (max) along axis
        self.t = t      # thickness
        self.kind = kind
        self.id = -1

    @property
    def length(self):
        return self.b - self.a

    def p0(self):
        return (self.a, self.c) if self.o == 'h' else (self.c, self.a)

    def p1(self):
        return (self.b, self.c) if self.o == 'h' else (self.c, self.b)

    def __repr__(self):
        return f"Seg({self.o},c={self.c:.1f},{self.a:.1f}-{self.b:.1f},t={self.t:.1f})"


def _bands(mask, min_len, orient):
    """Return list of (c, a, b, thickness) for straight runs in given orientation."""
    m = mask if orient == 'h' else mask.T
    m = np.ascontiguousarray(m)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(min_len), 1))
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, st, cen = cv2.connectedComponentsWithStats(op, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < min_len:
            continue
        if h <= 8:
            ys = np.arange(y, y + h)
            sub = (lab[y:y + h, x:x + w] == i)
            rowsum = sub.sum(1).astype(np.float64)
            c = float((ys * rowsum).sum() / max(rowsum.sum(), 1))
            thick = float(area) / w
            out.append((c, float(x), float(x + w - 1), thick))
        else:
            # thick blob: split row-wise into sub-bands by row occupancy
            sub = (lab[y:y + h, x:x + w] == i)
            # longest contiguous run per row
            longest = np.zeros(h, int)
            for r in range(h):
                row = sub[r].astype(np.int8)
                d = np.diff(np.r_[0, row, 0])
                s_ = np.where(d == 1)[0]
                e_ = np.where(d == -1)[0]
                if len(s_):
                    longest[r] = (e_ - s_).max()
            good = longest >= max(min_len, min(3 * min_len, 0.5 * w))
            r = 0
            while r < h:
                if not good[r]:
                    r += 1
                    continue
                r0 = r
                while r < h and good[r]:
                    r += 1
                seg = sub[r0:r]
                cols = np.where(seg.any(0))[0]
                if r - r0 <= 8 and len(cols):
                    rs = seg.sum(1).astype(np.float64)
                    c = float(((np.arange(r0, r) + y) * rs).sum() / rs.sum())
                    out.append((c, float(x + cols.min()), float(x + cols.max()), float(rs.sum()) / max(len(cols), 1)))
    return out


def extract_segments(mask, min_len=8, bridge=None, max_gap=6, kind='', bridges=None):
    """Extract horizontal & vertical segments.  `bridge` (optional bool mask) marks
    pixels through which collinear segments may be joined (gap filling)."""
    segs = []
    if bridges is None:
        bridges = [(bridge, max_gap)] if bridge is not None else []
    gmax = max([g for _, g in bridges] + [max_gap if bridge is None else 0, 2])
    for o in ('h', 'v'):
        bands = _bands(mask, min_len, o)
        bands.sort(key=lambda t: (round(t[0]), t[1]))
        # merge collinear
        merged = []
        for c, a, b, t in sorted(bands, key=lambda t: (t[0], t[1])):
            done = False
            for m in merged:
                if abs(m[0] - c) <= 1.6 and a <= m[2] + gmax + 1 and b >= m[1] - gmax - 1:
                    gap_ok = True
                    lo, hi = (m[2], a) if a > m[2] else ((b, m[1]) if b < m[1] else (0, 0))
                    if hi - lo > 1.5:
                        gap_ok = False
                        cc = int(round((m[0] + c) / 2))
                        xs = np.arange(int(lo), int(hi) + 1)
                        for bm, bg in bridges:
                            if hi - lo > bg:
                                continue
                            if o == 'h':
                                vals = bm[max(cc - 1, 0):cc + 2, xs]
                            else:
                                vals = bm[xs, max(cc - 1, 0):cc + 2].T
                            if vals.any(0).mean() > 0.8:
                                gap_ok = True
                                break
                    if gap_ok:
                        L1 = m[2] - m[1]
                        L2 = b - a
                        m[0] = (m[0] * L1 + c * L2) / max(L1 + L2, 1)
                        m[1] = min(m[1], a)
                        m[2] = max(m[2], b)
                        m[3] = max(m[3], t)
                        done = True
                        break
            if not done:
                merged.append([c, a, b, t])
        for c, a, b, t in merged:
            segs.append(Seg(o, c, a, b, t, kind))
    return segs


def dist_point_seg(p, s):
    x, y = p
    if s.o == 'h':
        dx = 0 if s.a <= x <= s.b else min(abs(x - s.a), abs(x - s.b))
        return float(np.hypot(dx, y - s.c))
    dy = 0 if s.a <= y <= s.b else min(abs(y - s.a), abs(y - s.b))
    return float(np.hypot(dy, x - s.c))

"""Wire graph: connectivity of wire segments, terminals and edge decomposition."""
import numpy as np
from .geom import Seg


class UF:
    def __init__(self, n=0):
        self.p = list(range(n))

    def add(self):
        self.p.append(len(self.p))
        return len(self.p) - 1

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


class Line:
    """Generic wire line (axis aligned or diagonal) with endpoints p, q."""
    __slots__ = ('p', 'q', 'o', 't', 'id', 'splits', 'p_att', 'q_att')

    def __init__(self, p, q, o, t, id):
        self.p = list(p)
        self.q = list(q)
        self.o = o
        self.t = t
        self.id = id
        self.splits = []        # parameter positions (points) where something attaches
        self.p_att = False
        self.q_att = False

    def length(self):
        return float(np.hypot(self.q[0] - self.p[0], self.q[1] - self.p[1]))

    def proj(self, pt):
        """Return (param 0..1, perpendicular distance)."""
        p = np.array(self.p, float)
        q = np.array(self.q, float)
        d = q - p
        L2 = float(d @ d)
        if L2 < 1e-6:
            return 0.0, float(np.hypot(pt[0] - p[0], pt[1] - p[1]))
        u = float((np.array(pt, float) - p) @ d / L2)
        foot = p + np.clip(u, 0, 1) * d
        return u, float(np.hypot(pt[0] - foot[0], pt[1] - foot[1]))

    def point_at(self, u):
        return (self.p[0] + u * (self.q[0] - self.p[0]), self.p[1] + u * (self.q[1] - self.p[1]))


def build_lines(P):
    lines = []
    for s in P.wires:
        lines.append(Line(s.p0(), s.p1(), s.o, s.t, len(lines)))
    for (x1, y1, x2, y2) in P.diag:
        lines.append(Line((x1, y1), (x2, y2), 'd', P.t, len(lines)))
    return lines


def snap_and_connect(P, lines):
    """Snap endpoints onto touching lines; compute connectivity union-find over lines.

    Returns uf (over line ids), list of terminals [(line_id, end ('p'|'q'), point, direction)].
    """
    t = P.t
    tol = max(2.0, 0.9 * t + 1.0)
    uf = UF(len(lines))
    dots = [(x, y, r) for (x, y, r) in P.dots]
    # 1. endpoint-on-line attachments (T and L)
    for a in lines:
        for end in ('p', 'q'):
            pt = getattr(a, end)
            best = None
            for b in lines:
                if b is a:
                    continue
                u, d = b.proj(pt)
                Lb = b.length()
                ext = (tol + 1.0) / max(Lb, 1)
                if -ext <= u <= 1 + ext and d <= tol + (1.0 if b.o != a.o else 0):
                    if a.o != 'd' and b.o == a.o:
                        # collinear overlap/continuation: only connect if truly collinear
                        if a.o == 'h' and abs(a.p[1] - b.p[1]) > 2:
                            continue
                        if a.o == 'v' and abs(a.p[0] - b.p[0]) > 2:
                            continue
                    if best is None or d < best[1]:
                        best = (b, d, u)
            if best is not None:
                b, d, u = best
                uf.union(a.id, b.id)
                setattr(a, end + '_att', True)
                # snap point onto b
                if b.o == 'h' and a.o == 'v':
                    pt[1] = b.p[1]
                    u = (pt[0] - b.p[0]) / max(b.q[0] - b.p[0], 1e-6)
                elif b.o == 'v' and a.o == 'h':
                    pt[0] = b.p[0]
                    u = (pt[1] - b.p[1]) / max(b.q[1] - b.p[1], 1e-6)
                u = min(max(u, 0.0), 1.0)
                b.splits.append(tuple(b.point_at(u)))
    # 2. crossings with junction dots
    for (x, y, r) in dots:
        on = []
        for b in lines:
            u, d = b.proj((x, y))
            if d <= r + tol and -0.02 <= u <= 1.02:
                on.append(b)
        for b in on:
            uf.union(on[0].id, b.id)
            u, d = b.proj((x, y))
            b.splits.append(tuple(b.point_at(min(max(u, 0), 1))))
    # also: crossing points of H and V lines that both have a dot -> handled above.
    terms = []
    for a in lines:
        for end in ('p', 'q'):
            if getattr(a, end + '_att'):
                continue
            pt = getattr(a, end)
            other = a.q if end == 'p' else a.p
            dvec = np.array(pt, float) - np.array(other, float)
            n = np.linalg.norm(dvec)
            dvec = dvec / n if n > 0 else np.zeros(2)
            # is endpoint on a dot?  then it's still a junction-ish endpoint, keep as terminal anyway
            terms.append({'line': a.id, 'end': end, 'pt': tuple(pt), 'dir': tuple(dvec)})
    return uf, terms


def decompose_edges(lines):
    """Split every line at its attachment points; returns dict line_id -> list of (p,q)."""
    out = {}
    for a in lines:
        pts = [tuple(a.p), tuple(a.q)] + list(a.splits)
        # param sort
        us = []
        for pt in pts:
            u, d = a.proj(pt)
            us.append((min(max(u, 0), 1), pt))
        us.sort()
        L = a.length()
        res = []
        cur = us[0]
        for u, pt in us[1:]:
            if (u - cur[0]) * L < 2.0:
                continue
            res.append((a.point_at(cur[0]), a.point_at(u)))
            cur = (u, pt)
        if not res:
            res = [(tuple(a.p), tuple(a.q))]
        out[a.id] = res
    return out

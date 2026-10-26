"""Render an EasyEDA Pro sheet to SVG in the editor's default (light) theme.

The output mimics the editor's own SVG/PNG export (colours, stroke widths, font sizes,
text anchoring) so that rasterised images look like the competition images, which are
EasyEDA Pro PNG exports (background #FFFCF8, 2 px per schematic unit at '1x').
"""
import math
from xml.sax.saxutils import escape

BG = '#FFFCF8'
C_WIRE = '#008800'
C_JUNC = '#CC0000'
C_BODY = '#A00000'
C_PIN = '#880000'
C_PINTEXT = '#000000'
C_DESIG = '#000080'
C_VALUE = '#0000FF'
C_NET = '#000080'
FONT_K = 0.875775        # rendered font-size / nominal size (from the editor export)
DEFAULT_SIZE = 8.0
JUNC_R = 2.5


def _fmt(v):
    return ('%.3f' % v).rstrip('0').rstrip('.')


# line styles (EasyEDA Pro strokeStyle 1 dashed, 2 dotted, 3 dash-dot), in sheet units; drawn
# with round caps, so the visible dash is one stroke width longer.  Off by default (the
# released datasets were rendered with solid lines); tools/build_* may switch it on.
DASHES = [False]
DASH_ARRAYS = {'1': '8.5 10.5', '2': '0.01 3', '3': '8.5 4 0.01 4'}


class SVG:
    def __init__(self):
        self.parts = []
        self.texts = []      # (id, text, x, y, anchor, size, rot, color) in sheet coords, for tests / GT

    def line(self, pts, color, width=1.0, eid=None, fill='none', closed=False, dash=None):
        p = ' '.join(f'{_fmt(x)} {_fmt(-y)}' for x, y in pts)
        tag = 'polygon' if closed else 'polyline'
        i = f' id="{eid}"' if eid else ''
        da = ''
        if DASHES[0] and dash is not None and str(dash) in DASH_ARRAYS:
            da = f';stroke-dasharray:{DASH_ARRAYS[str(dash)]}'
        self.parts.append(f'<{tag}{i} points="{p}" style="stroke-width:{_fmt(width)};stroke:{color};fill:{fill}{da}"/>')

    def circle(self, cx, cy, r, color, fill='none', width=1.0, eid=None):
        i = f' id="{eid}"' if eid else ''
        self.parts.append(f'<circle{i} cx="{_fmt(cx)}" cy="{_fmt(-cy)}" r="{_fmt(r)}" style="stroke-width:{_fmt(width)};stroke:{color};fill:{fill}"/>')

    def path(self, d, color, width=1.0, fill='none', eid=None):
        i = f' id="{eid}"' if eid else ''
        self.parts.append(f'<path{i} d="{d}" style="stroke-width:{_fmt(width)};stroke:{color};fill:{fill}"/>')

    def text(self, value, x, y, size, rot, h, v, color, family=None, bold=False, italic=False, eid=None):
        if value is None or value == '':
            return
        size = size or DEFAULT_SIZE
        anchor = ('start', 'middle', 'end')[h if h in (0, 1, 2) else 0]
        dy = size * {0: 1.0, 1: 0.5, 2: 0.0}.get(v if v in (0, 1, 2) else 2, 0.0)
        fam = family or 'Arial'
        style = (f"font-family:'{fam}',Arial,'Liberation Sans','Noto Sans CJK SC',sans-serif;font-size:{_fmt(size * FONT_K)}px;"
                 f"fill:{color};stroke:none;font-weight:{'bold' if bold else 'normal'};font-style:{'italic' if italic else 'normal'}")
        i = f' id="{eid}"' if eid else ''
        lines = str(value).split('\n')
        tsp = ''.join(f'<tspan x="{_fmt(x)}" dy="{_fmt(dy if k == 0 else size)}">{escape(t)}</tspan>' for k, t in enumerate(lines))
        self.parts.append(f'<text{i} x="{_fmt(x)}" y="{_fmt(-y)}" text-anchor="{anchor}" dominant-baseline="ideographic" '
                          f'transform="rotate({_fmt(-rot)} {_fmt(x)} {_fmt(-y)})" style="{style}" xml:space="preserve">{tsp}</text>')
        self.texts.append(dict(id=eid, text=str(value), x=x, y=y, anchor=anchor, h=h, v=v, size=size, rot=rot, color=color))


def _arc_points(p, n=24):
    """Three-point arc (start, through, end) -> polyline points."""
    (x1, y1, x2, y2, x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return [(x1, y1), (x3, y3)]
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    a3 = math.atan2(y3 - uy, x3 - ux)

    def norm(a):
        return a % (2 * math.pi)
    # choose direction so that a2 lies between a1 and a3
    ccw = norm(a3 - a1)
    if norm(a2 - a1) <= ccw:
        sweep = ccw
    else:
        sweep = ccw - 2 * math.pi
    return [(ux + r * math.cos(a1 + sweep * k / n), uy + r * math.sin(a1 + sweep * k / n)) for k in range(n + 1)]


def _color(style, default):
    if style and style.get('stroke') and style.get('stroke') not in ('none',):
        return style['stroke']
    return default


def _fill(style):
    f = style.get('fill') if style else None
    if f and f not in ('none', 'transparent'):
        return f
    return 'none'


def _width(style):
    w = style.get('width') if style else None
    return w if w else 1.0


def _text_rot(base_rot, comp):
    """Final text rotation, kept readable (the editor never draws text upside down)."""
    r = (base_rot + (comp.rot if comp else 0)) % 360
    return r


def _readable(rot, h, v):
    """Normalise an absolute rotation to 0/90 keeping the anchor point fixed."""
    r = rot % 360
    if 90 < r <= 270:
        r = (r + 180) % 360
        h = {0: 2, 1: 1, 2: 0}[h]
        v = {0: 2, 1: 1, 2: 0}[v]
    return r, h, v


def pin_tips(sheet):
    out = []
    for c in sheet.components:
        if c.symbol is None or c.symbol.stype == 20:
            continue
        part = c.symbol.part(c.part_name)
        if part is None:
            continue
        for pn in part.pins:
            out.append(c.xf(pn.x, pn.y))
    return out


def junctions(sheet, pins_xy=None):
    """Points where 3+ connections meet (wire ends, pin tips), or a wire end lands on another wire's interior."""
    ends = {}
    wire_pts = set()
    for w in sheet.wires:
        for (x1, y1, x2, y2) in w.segs:
            wire_pts.add((round(x1, 3), round(y1, 3)))
            wire_pts.add((round(x2, 3), round(y2, 3)))
    for p in (pins_xy if pins_xy is not None else pin_tips(sheet)):
        k = (round(p[0], 3), round(p[1], 3))
        if k in wire_pts:
            ends[k] = ends.get(k, 0) + 1
            ends[('pin',) + k] = ends.get(('pin',) + k, 0) + 1
    for w in sheet.wires:
        for (x1, y1, x2, y2) in w.segs:
            for p in ((x1, y1), (x2, y2)):
                k = (round(p[0], 3), round(p[1], 3))
                ends[k] = ends.get(k, 0) + 1
    segs = [s for w in sheet.wires for s in w.segs]
    out = []
    for key, n in list(ends.items()):
        if key and key[0] == 'pin':
            continue
        (x, y) = key
        cnt = n
        for (x1, y1, x2, y2) in segs:
            if _on_interior(x, y, x1, y1, x2, y2):
                cnt += 2
        npin = ends.get(('pin',) + (x, y), 0)
        if cnt >= 3 and not (npin >= 2 and cnt - npin >= 2):
            out.append((x, y))
    return out


def _on_interior(x, y, x1, y1, x2, y2, eps=1e-3):
    if (abs(x - x1) < eps and abs(y - y1) < eps) or (abs(x - x2) < eps and abs(y - y2) < eps):
        return False
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return False
    t = ((x - x1) * dx + (y - y1) * dy) / L2
    if t <= 0 or t >= 1:
        return False
    px, py = x1 + t * dx, y1 + t * dy
    return abs(px - x) < eps and abs(py - y) < eps


def render_sheet(sheet, skip_frame=True, extra=None):
    """Return (SVG object with parts, bbox of drawn content in sheet coords)."""
    from .model import resolve
    S = SVG()
    # ---------------------------------------------------------------- free graphics
    # free graphics: keep annotations, drop sheet frames / title blocks (very large boxes and lines)
    ext = _sheet_extent(sheet)
    for kind, d, st in sheet.prims:
        if _is_frame_like(kind, d, ext):
            continue
        n0 = len(S.texts)
        _prim(S, kind, d, st, None, C_BODY)
        for t in S.texts[n0:]:
            t['sheet'] = True
    # ---------------------------------------------------------------- components
    for c in sheet.components:
        sym = c.symbol
        if sym is None:
            continue
        if skip_frame and (sym.stype == 20 or str(sym.title).lower().startswith('sheet-symbol')):
            continue
        part = sym.part(c.part_name)
        if part is None:
            continue
        if skip_frame and _is_frame_symbol(part):
            continue
        for kind, d, st in part.prims:
            _prim(S, kind, d, st, c, C_BODY, eid=f'{c.id}-{d.get("id")}')
        for pn in part.pins:
            if pn.display is False or pn.display == 0:
                continue
            tip = c.xf(pn.x, pn.y)
            base = c.xf(*pn.base())
            S.line([tip, base], C_PIN, 1.0, eid=f'{c.id}-{pn.id}')
            if pn.shape and str(pn.shape).upper() in ('INVERTED', 'INVERTED_CLOCK', 'DOT'):
                a = math.atan2(base[1] - tip[1], base[0] - tip[0])
                S.circle(base[0] - 1.5 * math.cos(a), base[1] - 1.5 * math.sin(a), 1.5, C_PIN)
            for at, default_color in ((pn.name_attr, C_PINTEXT), (pn.num_attr, C_PINTEXT)):
                if at is None or not at.val_vis or at.value in (None, ''):
                    continue
                if at.x is None or at.y is None:
                    x, y, rot, h, v = _auto_pin_text(pn, at is pn.num_attr)
                else:
                    x, y, rot, h, v = at.x, at.y, at.rot, at.font.get('h', 0), at.font.get('v', 2)
                X, Y = c.xf(x, y)
                R = _text_rot(rot, c)
                if c.mirror:
                    h = {0: 2, 1: 1, 2: 0}[h]
                    R = (-R) % 360
                R, h, v = _readable(R, h, v)
                f = at.font
                S.text(at.value, X, Y, f.get('size'), R, h, v, f.get('color') or default_color, f.get('family'),
                       f.get('bold'), f.get('italic'), eid=f'{c.id}-{at.id}')
        for key, val, X, Y, R, h, v, f, color in component_label_texts(c, part):
            S.text(val, X, Y, f.get('size'), R, h, v, color, f.get('family'), f.get('bold'), f.get('italic'),
                   eid=f'{c.id}-{key}')
    # ---------------------------------------------------------------- wires
    for w in sheet.wires:
        col = _color(w.style, C_WIRE)
        for k, (x1, y1, x2, y2) in enumerate(w.segs):
            S.line([(x1, y1), (x2, y2)], col, _width(w.style), eid=f'{w.id}-{k}')
        for key, at in w.attrs.items():
            if at.val_vis and at.x is not None and at.value:
                R, h, v = _readable(at.rot, at.font.get('h', 0), at.font.get('v', 2))
                S.text(at.value, at.x, at.y, at.font.get('size'), R, h, v, at.font.get('color') or C_NET,
                       at.font.get('family'), eid=at.id)
    for (x, y) in junctions(sheet):
        S.circle(x, y, JUNC_R, C_JUNC, fill=C_JUNC)
    return S


SKIP_KEYS = {'Symbol', 'Device', 'Footprint', 'Unique ID', '3D Model', '3D Model Title', '3D Model Transform',
             'Add into BOM', 'Convert to PCB', 'Supplier Footprint', 'Description', 'Datasheet', 'Pin Type'}


def component_label_texts(c, part):
    """Visible component labels: (key, value, x, y, rot, h, v, font, color) in sheet coordinates."""
    from .model import resolve
    out = []
    keys = list(c.attrs.keys()) + [a.key for a in part.attrs if a.key not in c.attrs]
    pattrs = {a.key: a for a in part.attrs}
    for key in keys:
        if key in SKIP_KEYS or key is None or key.startswith('@'):
            continue
        sa, pa = c.attrs.get(key), pattrs.get(key)
        vis = (sa is not None and sa.val_vis) or (pa is not None and pa.val_vis)
        if not vis:
            continue
        if sa is not None and sa.x is not None and sa.y is not None:
            X, Y, rot, f = sa.x, sa.y, sa.rot, sa.font
            h, v = f.get('h', 0), f.get('v', 2)
        elif pa is not None and pa.x is not None and pa.y is not None:
            X, Y = c.xf(pa.x, pa.y)
            f = pa.font
            h, v = f.get('h', 0), f.get('v', 2)
            rot = (pa.rot + c.rot) % 360
            if c.mirror:
                rot = (-rot) % 360
                h = {0: 2, 1: 1, 2: 0}[h]
        else:
            continue
        val = sa.value if (sa is not None and sa.value not in (None, '')) else None
        if val is None:
            val = (c.device or {}).get(key)
        if key == 'Global Net Name' and not val:
            val = c.attr('Name') or (c.device or {}).get('Name') or (pa.value if pa is not None else None)
        if val in (None, '') and pa is not None:
            val = pa.value
        val = resolve(c, val)
        if val in (None, '') or str(val).endswith('?'):
            continue
        if (sa is not None and sa.key_vis) and key not in ('Designator', 'Name', 'Global Net Name'):
            val = f'{key}: {val}'
        color = f.get('color') or (C_DESIG if key in ('Designator', 'Global Net Name') else C_VALUE)
        R, h, v = _readable(rot, h, v)
        out.append((key, val, X, Y, R, h, v, f, color))
    return out


def _is_frame_symbol(part):
    if part.pins:
        return False
    b = part.bbox
    if b and max(abs(b[2] - b[0]), abs(b[3] - b[1])) > 300:
        return True
    return False


def _sheet_extent(sheet):
    xs, ys = [], []
    for w in sheet.wires:
        for (x1, y1, x2, y2) in w.segs:
            xs += [x1, x2]
            ys += [y1, y2]
    for c in sheet.components:
        if c.symbol is not None and c.symbol.stype != 20:
            xs.append(c.x)
            ys.append(c.y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _is_frame_like(kind, d, ext):
    if ext is None:
        return False
    W, H = max(1.0, ext[2] - ext[0]), max(1.0, ext[3] - ext[1])
    if kind == 'rect':
        w, h = abs(d['x2'] - d['x1']), abs(d['y2'] - d['y1'])
        return w > 0.7 * W and h > 0.7 * H
    if kind == 'poly':
        pts = d['pts']
        xs, ys = pts[0::2], pts[1::2]
        if not xs:
            return False
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        outside = cx < ext[0] - 20 or cx > ext[2] + 20 or cy < ext[1] - 20 or cy > ext[3] + 20
        return w > 0.8 * W or h > 0.8 * H or (outside and len(xs) <= 2)
    if kind == 'text':
        x, y = d.get('x'), d.get('y')
        if x is None or y is None:
            return True
        v = str(d.get('value') or '')
        outside = x < ext[0] - 30 or x > ext[2] + 30 or y < ext[1] - 30 or y > ext[3] + 30
        return outside or (len(v) == 1 and v.isalnum())
    return False


def _auto_pin_text(pn, is_number):
    """Default placement when the symbol stores no coordinates for a pin label."""
    a = math.radians(pn.rot)
    ux, uy = math.cos(a), math.sin(a)
    if is_number:
        mx, my = pn.x + ux * pn.length / 2, pn.y + uy * pn.length / 2
        # above a horizontal pin, left of a vertical one
        if abs(ux) > 0.5:
            return mx, my + 1, 0, 1, 2
        return mx - 1, my, 90, 1, 2
    bx, by = pn.x + ux * (pn.length + 2), pn.y + uy * (pn.length + 2)
    if abs(ux) > 0.5:
        return bx, by, 0, (0 if ux > 0 else 2), 1
    return bx, by, 90, (0 if uy > 0 else 2), 1


def _prim(S, kind, d, st, comp, default_color, eid=None):
    xf = comp.xf if comp is not None else (lambda x, y: (x, y))
    col = _color(st, default_color)
    w = _width(st)
    if kind == 'poly':
        pts = d['pts']
        P = [xf(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        if len(P) >= 2:
            dash = (st or {}).get('dash')
            S.line(P, col, w, eid=eid, fill=_fill(st), closed=d.get('closed') and _fill(st) != 'none', dash=dash)
            if d.get('closed') and _fill(st) == 'none' and P[0] != P[-1]:
                S.line([P[-1], P[0]], col, w, dash=dash)
    elif kind == 'rect':
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        P = [xf(x1, y1), xf(x2, y1), xf(x2, y2), xf(x1, y2), xf(x1, y1)]
        S.line(P, col, w, eid=eid, fill=_fill(st), closed=_fill(st) != 'none', dash=(st or {}).get('dash'))
        if _fill(st) != 'none':
            S.line(P, col, w, dash=(st or {}).get('dash'))
    elif kind == 'circle':
        cx, cy = xf(d['cx'], d['cy'])
        S.circle(cx, cy, d['r'], col, fill=_fill(st), width=w, eid=eid)
    elif kind == 'arc':
        P = [xf(x, y) for x, y in _arc_points(d['p'])]
        S.line(P, col, w, eid=eid)
    elif kind == 'ellipse':
        n = 36
        P = []
        rot = math.radians(d.get('rot') or 0)
        for k in range(n + 1):
            t = 2 * math.pi * k / n
            ex, ey = d['rx'] * math.cos(t), d['ry'] * math.sin(t)
            P.append(xf(d['cx'] + ex * math.cos(rot) - ey * math.sin(rot), d['cy'] + ex * math.sin(rot) + ey * math.cos(rot)))
        S.line(P, col, w, eid=eid)
    elif kind == 'bezier':
        pts = d['pts']
        P = [xf(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        if len(P) >= 2:
            S.line(P, col, w, eid=eid)
    elif kind == 'text':
        x, y = d['x'], d['y']
        if x is None or y is None:
            return
        X, Y = xf(x, y)
        R = _text_rot(d.get('rot') or 0, comp)
        R, h, v = _readable(R, st.get('h', 0), st.get('v', 2))
        S.text(d.get('value'), X, Y, st.get('size'), R, h, v, st.get('color') or default_color, st.get('family'),
               st.get('bold'), st.get('italic'), eid=eid)


def to_svg_document(S, x0, y0, x1, y1, scale):
    """Wrap rendered parts into a standalone SVG covering sheet box [x0,x1]x[y0,y1] at `scale` px/unit."""
    W = int(round((x1 - x0) * scale))
    H = int(round((y1 - y0) * scale))
    vb = f'{_fmt(x0)} {_fmt(-y1)} {_fmt(x1 - x0)} {_fmt(y1 - y0)}'
    body = '\n'.join(S.parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="{vb}" '
            f'style="background:{BG}"><style>*{{stroke-linejoin:round;stroke-linecap:round}}</style>'
            f'<rect x="{_fmt(x0)}" y="{_fmt(-y1)}" width="{_fmt(x1 - x0)}" height="{_fmt(y1 - y0)}" fill="{BG}"/>\n{body}\n</svg>'), W, H

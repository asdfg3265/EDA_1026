"""EasyEDA Pro schematic document model.

Two input formats are supported and normalised into the same in-memory model:
  * API "dataStr" (one JSON array per line: ["COMPONENT","e36","MT2492.1",315,370,0,0,{},0])
    as returned by pro.lceda.cn/api/v2/documents/lists and /api/v5/components/searchByIds;
  * exported project files (.epro2 = zip of *.epru, one record per line:
    {"type":"COMPONENT","id":"e36"}||{"partId":...,"x":...}|).
Internal convention: sheet / symbol coordinates are y-UP (as in dataStr); the epru format
stores y-DOWN and is flipped on load.
"""
import json
import math
import re
import zipfile

DEFAULT_FONT = {'color': None, 'fill': None, 'family': None, 'size': None, 'italic': None,
                'bold': None, 'h': 0, 'v': 2}
H_ALIGN = {'LEFT': 0, 'CENTER': 1, 'RIGHT': 2}
V_ALIGN = {'TOP': 0, 'MIDDLE': 1, 'BOTTOM': 2}


class Attr:
    __slots__ = ('id', 'parent', 'key', 'value', 'key_vis', 'val_vis', 'x', 'y', 'rot', 'font')

    def __init__(self, id, parent, key, value, key_vis, val_vis, x, y, rot, font):
        self.id, self.parent, self.key, self.value = id, parent, key, value
        self.key_vis, self.val_vis = bool(key_vis), bool(val_vis)
        self.x, self.y, self.rot, self.font = x, y, rot or 0, font

    def __repr__(self):
        return f'Attr({self.key}={self.value!r} vis={self.val_vis} @{self.x},{self.y})'


class Pin:
    __slots__ = ('id', 'display', 'x', 'y', 'length', 'rot', 'shape', 'name', 'number', 'name_attr', 'num_attr')

    def __init__(self, id, display, x, y, length, rot, shape):
        self.id, self.display, self.x, self.y = id, display, x, y
        self.length, self.rot, self.shape = length or 0, rot or 0, shape
        self.name = self.number = None
        self.name_attr = self.num_attr = None

    def base(self):
        a = math.radians(self.rot)
        return self.x + self.length * math.cos(a), self.y + self.length * math.sin(a)


class Part:
    def __init__(self, name, bbox):
        self.name = name
        self.bbox = bbox        # [x1, y1, x2, y2] (any corner order)
        self.prims = []         # (kind, data, style)
        self.pins = []
        self.attrs = []         # part-level attrs (parent '')


class Symbol:
    def __init__(self, uuid, title, stype):
        self.uuid, self.title, self.stype = uuid, title, stype
        self.parts = {}

    @property
    def has_image(self):
        return any(getattr(p, 'has_image', False) for p in self.parts.values())

    def part(self, name):
        if name in self.parts:
            return self.parts[name]
        if len(self.parts) == 1:
            return next(iter(self.parts.values()))
        # 'X.1' style names: fall back by suffix
        for k, p in self.parts.items():
            if k.split('.')[-1] == str(name).split('.')[-1]:
                return p
        return next(iter(self.parts.values())) if self.parts else None


class Component:
    def __init__(self, id, part_name, x, y, rot, mirror):
        self.id, self.part_name, self.x, self.y = id, part_name, x, y
        self.rot, self.mirror = rot or 0, bool(mirror)
        self.attrs = {}          # key -> Attr (sheet level)
        self.symbol = None
        self.device = None       # dict of device attributes

    def attr(self, key):
        a = self.attrs.get(key)
        return a.value if a is not None else None

    def xf(self, px, py):
        """Symbol coordinates -> sheet coordinates."""
        a = math.radians(self.rot)
        c, s = math.cos(a), math.sin(a)
        dx, dy = px * c - py * s, px * s + py * c
        if self.mirror:
            dx = -dx
        return self.x + dx, self.y + dy


def _wire_segs(raw):
    """WIRE geometry: a list of [x1,y1,x2,y2] segments, but some files store polylines
    [x1,y1,x2,y2,x3,y3,...] (or a flat list) -> split into 2-point segments."""
    out = []
    if raw and isinstance(raw[0], (int, float)):
        raw = [raw]
    for s in raw or []:
        pts = [float(v) for v in s if v is not None]
        for i in range(0, len(pts) - 3, 2):
            out.append(tuple(pts[i:i + 4]))
    return out


class Wire:
    def __init__(self, id, segs, style):
        self.id, self.segs, self.style = id, segs, style
        self.attrs = {}


class Sheet:
    def __init__(self):
        self.components = []
        self.wires = []
        self.prims = []          # free graphics / texts on the sheet
        self.attrs = []          # all sheet attrs
        self.unknown = set()


# --------------------------------------------------------------------------- helpers
def _font_from_array(r):
    # ["FONTSTYLE", id, color, fillColor, fontFamily, fontSize, strikeout, underline, italic, fontWeight, vAlign, hAlign]
    f = dict(DEFAULT_FONT)
    f.update(color=r[2], fill=r[3], family=r[4], size=_num(r[5]), italic=r[8], bold=r[9])
    if len(r) > 10 and r[10] is not None:
        f['v'] = int(r[10])
    if len(r) > 11 and r[11] is not None:
        f['h'] = int(r[11])
    return f


def _font_from_obj(o):
    f = dict(DEFAULT_FONT)
    f.update(color=o.get('color'), fill=o.get('fillColor'), family=o.get('fontFamily'),
             size=_num(o.get('fontSize')), italic=o.get('italic'), bold=o.get('fontWeight'))
    al = o.get('align')
    if al:
        h, _, v = al.partition('_')
        f['h'] = H_ALIGN.get(h, 0)
        f['v'] = V_ALIGN.get(v, 2)
    return f


def _num(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _style_from_array(r):
    # ["LINESTYLE", id, strokeColor, strokeStyle, fillColor, strokeWidth]
    return {'stroke': r[2], 'dash': r[3], 'fill': r[4] if len(r) > 4 else None, 'width': _num(r[5]) if len(r) > 5 else None}


def _style_from_obj(o):
    return {'stroke': o.get('strokeColor'), 'dash': o.get('strokeStyle'), 'fill': o.get('fillColor'),
            'width': _num(o.get('strokeWidth')), 'fillStyle': o.get('fillStyle')}


# --------------------------------------------------------------------------- array (API) format
def parse_lines(datastr):
    out = []
    for line in datastr.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def symbol_from_array(uuid, title, stype, datastr):
    recs = parse_lines(datastr)
    sym = Symbol(uuid, title, stype)
    fonts, styles = {}, {}
    part = None
    pins = {}
    for r in recs:
        k = r[0]
        if k == 'HEAD' and isinstance(r[1], dict) and r[1].get('symbolType') is not None and stype is None:
            sym.stype = r[1].get('symbolType')
        elif k == 'FONTSTYLE':
            fonts[r[1]] = _font_from_array(r)
        elif k == 'LINESTYLE':
            styles[r[1]] = _style_from_array(r)
        elif k == 'PART':
            part = Part(r[1], (r[2] or {}).get('BBOX'))
            sym.parts[r[1]] = part
        elif part is None and k in ('PIN', 'POLY', 'RECT', 'ARC', 'CIRCLE', 'TEXT', 'ATTR', 'ELLIPSE', 'BEZIER', 'POLYGON'):
            part = Part('', None)
            sym.parts[''] = part
        if part is None:
            continue
        if k == 'PIN':
            p = Pin(r[1], r[2], r[4], r[5], r[6], r[7], r[9] if len(r) > 9 else None)
            pins[p.id] = p
            part.pins.append(p)
        elif k == 'POLY':
            part.prims.append(('poly', {'id': r[1], 'pts': r[2], 'closed': bool(r[3])}, styles.get(r[4])))
        elif k == 'RECT':
            part.prims.append(('rect', {'id': r[1], 'x1': r[2], 'y1': r[3], 'x2': r[4], 'y2': r[5], 'rx': r[6], 'ry': r[7], 'rot': r[8]}, styles.get(r[9])))
        elif k == 'CIRCLE':
            part.prims.append(('circle', {'id': r[1], 'cx': r[2], 'cy': r[3], 'r': r[4]}, styles.get(r[5])))
        elif k == 'ARC':
            part.prims.append(('arc', {'id': r[1], 'p': r[2:8]}, styles.get(r[8])))
        elif k == 'ELLIPSE':
            part.prims.append(('ellipse', {'id': r[1], 'cx': r[2], 'cy': r[3], 'rx': r[4], 'ry': r[5], 'rot': r[6] if len(r) > 6 else 0},
                               styles.get(r[7]) if len(r) > 7 else None))
        elif k == 'OBJ':
            if len(r) > 9 and isinstance(r[9], str) and r[9].startswith('data:image'):
                part.has_image = True      # embedded picture (module photo, logo)
        elif k == 'BEZIER':
            part.prims.append(('bezier', {'id': r[1], 'pts': r[2]}, styles.get(r[3]) if len(r) > 3 else None))
        elif k == 'TEXT':
            part.prims.append(('text', {'id': r[1], 'x': r[2], 'y': r[3], 'rot': r[4], 'value': r[5]}, fonts.get(r[6], DEFAULT_FONT)))
        elif k == 'ATTR':
            a = Attr(r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], fonts.get(r[10], DEFAULT_FONT))
            if a.parent and a.parent in pins:
                pn = pins[a.parent]
                if a.key in ('NAME', 'Pin Name'):
                    pn.name, pn.name_attr = a.value, a
                elif a.key in ('NUMBER', 'Pin Number'):
                    pn.number, pn.num_attr = a.value, a
            elif not a.parent:
                part.attrs.append(a)
    return sym


def sheet_from_array(datastr):
    recs = parse_lines(datastr)
    sh = Sheet()
    fonts, styles = {}, {}
    comps, wires = {}, {}
    for r in recs:
        k = r[0]
        if k == 'FONTSTYLE':
            fonts[r[1]] = _font_from_array(r)
        elif k == 'LINESTYLE':
            styles[r[1]] = _style_from_array(r)
        elif k == 'COMPONENT':
            c = Component(r[1], r[2], r[3], r[4], r[5], r[6])
            comps[c.id] = c
            sh.components.append(c)
        elif k == 'WIRE':
            w = Wire(r[1], _wire_segs(r[2]), styles.get(r[3]))
            wires[w.id] = w
            sh.wires.append(w)
        elif k == 'ATTR':
            a = Attr(r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], fonts.get(r[10], DEFAULT_FONT))
            sh.attrs.append(a)
            if a.parent in comps:
                comps[a.parent].attrs[a.key] = a
            elif a.parent in wires:
                wires[a.parent].attrs[a.key] = a
        elif k == 'TEXT':
            sh.prims.append(('text', {'id': r[1], 'x': r[2], 'y': r[3], 'rot': r[4], 'value': r[5]}, fonts.get(r[6], DEFAULT_FONT)))
        elif k == 'POLY':
            sh.prims.append(('poly', {'id': r[1], 'pts': r[2], 'closed': bool(r[3])}, styles.get(r[4])))
        elif k == 'RECT':
            sh.prims.append(('rect', {'id': r[1], 'x1': r[2], 'y1': r[3], 'x2': r[4], 'y2': r[5], 'rx': r[6], 'ry': r[7], 'rot': r[8]}, styles.get(r[9])))
        elif k == 'CIRCLE':
            sh.prims.append(('circle', {'id': r[1], 'cx': r[2], 'cy': r[3], 'r': r[4]}, styles.get(r[5])))
        elif k == 'ARC':
            sh.prims.append(('arc', {'id': r[1], 'p': r[2:8]}, styles.get(r[8])))
        elif k == 'OBJ':
            pass                           # sheet-level objects: frame logos etc. (frame is not rendered)
        elif k not in ('DOCTYPE', 'HEAD', 'GROUP', 'OBJ', 'PART'):
            sh.unknown.add(k)
    return sh


# --------------------------------------------------------------------------- epru (export) format
def _iter_epru(text):
    for line in text.splitlines():
        line = line.rstrip()
        if not line or '||' not in line:
            continue
        h, b = line.split('||', 1)
        if b.endswith('|'):
            b = b[:-1]
        try:
            yield json.loads(h), (json.loads(b) if b else {})
        except json.JSONDecodeError:
            continue


def _fy(v):
    return None if v is None else -v


def load_epro(path):
    """Read an .epro2/.epro zip; return (sheets, symbols, devices, meta).

    sheets: list of (title, Sheet); symbols: uuid -> Symbol; devices: uuid -> attributes dict."""
    z = zipfile.ZipFile(path)
    sheets, symbols, devices = [], {}, {}
    for name in z.namelist():
        if not name.endswith('.epru'):
            continue
        text = z.read(name).decode('utf-8', 'replace')
        _load_epru_text(text, sheets, symbols, devices)
    return sheets, symbols, devices


def _load_epru_text(text, sheets, symbols, devices):
    doc = None           # current document: dict(type, uuid, title, records)
    docs = []
    for h, b in _iter_epru(text):
        t = h.get('type')
        if t == 'DOCHEAD':
            doc = {'type': b.get('docType'), 'uuid': b.get('uuid'), 'title': '', 'meta': {}, 'recs': []}
            docs.append(doc)
            continue
        if doc is None:
            continue
        if t == 'META':
            doc['title'] = b.get('title', '')
            doc['meta'] = b
            continue
        doc['recs'].append((t, h.get('id'), b))
    for d in docs:
        if d['type'] == 'DEVICE':
            devices[d['uuid']] = dict(d['meta'].get('attributes') or {}, _title=d['title'])
        elif d['type'] == 'SYMBOL':
            symbols[d['uuid']] = _symbol_from_epru(d)
        elif d['type'] in ('SCH_PAGE', 'SCH'):
            sh = _sheet_from_epru(d)
            if sh.components or sh.wires:
                sheets.append((d['title'], sh))


def _symbol_from_epru(d):
    meta = d['meta']
    sym = Symbol(d['uuid'], d['title'], (meta.get('attributes') or {}).get('symbolType'))
    parts = {}
    pins = {}
    for t, i, b in d['recs']:
        if t == 'PART':
            bb = b.get('BBOX')
            if bb:
                bb = [bb[0], -bb[1], bb[2], -bb[3]]
            parts[i] = Part(i, bb)
            sym.parts[i] = parts[i]
    for t, i, b in d['recs']:
        pid = b.get('partId')
        part = parts.get(pid)
        if part is None:
            if t not in ('PIN', 'POLY', 'RECT', 'ARC', 'CIRCLE', 'TEXT', 'ATTR'):
                continue
            part = parts.setdefault(pid or '', Part(pid or '', None))
            sym.parts.setdefault(pid or '', part)
        if t == 'PIN':
            p = Pin(i, b.get('display', True), b.get('x'), _fy(b.get('y')), b.get('length'), (b.get('rotation') or 0) % 360,
                    b.get('pinShape'))
            pins[i] = p
            part.pins.append(p)
        elif t in ('POLY', 'RECT', 'ARC', 'CIRCLE', 'TEXT'):
            part.prims.append(_prim_from_obj(t, b, i))
    for t, i, b in d['recs']:
        if t != 'ATTR':
            continue
        a = _attr_from_obj(i, b)
        if a.parent in pins:
            pn = pins[a.parent]
            if a.key in ('NAME', 'Pin Name'):
                pn.name, pn.name_attr = a.value, a
            elif a.key in ('NUMBER', 'Pin Number'):
                pn.number, pn.num_attr = a.value, a
        elif not a.parent:
            part = parts.get(b.get('partId'))
            if part is not None:
                part.attrs.append(a)
    sym.stype = _guess_stype(sym) if sym.stype in (None, '', 2) else sym.stype
    return sym


def _guess_stype(sym):
    t = (sym.title or '').lower()
    for p in sym.parts.values():
        keys = {a.key for a in p.attrs}
        if 'Global Net Name' in keys and len(p.pins) <= 1:
            return 18
        if t.startswith('netport') or ('port' in t and len(p.pins) == 1 and 'Name' in keys):
            return 19
    if t.startswith('short-symbol') or t.startswith('short symbol'):
        return 22
    if t.startswith('off-page'):
        return 25
    if t.startswith(('ground', 'power')) or t in ('gnd', 'vcc'):
        return 18
    if t.startswith('netport'):
        return 19
    if t.startswith('sheet-symbol'):
        return 20
    return 2


def _attr_from_obj(i, b):
    return Attr(i, b.get('parentId') or '', b.get('key'), b.get('value'), b.get('keyVisible'), b.get('valueVisible'),
                b.get('x'), _fy(b.get('y')), (b.get('rotation') or 0) % 360, _font_from_obj(b))


def _prim_from_obj(t, b, i=None):
    st = _style_from_obj(b)
    pr = _prim_from_obj0(t, b, st)
    if pr is not None:
        pr[1]['id'] = i
    return pr


def _prim_from_obj0(t, b, st):
    if t == 'POLY':
        pts = []
        for p in b.get('points') or []:
            pts += [p['x'], -p['y']]
        return ('poly', {'pts': pts, 'closed': bool(b.get('closed'))}, st)
    if t == 'RECT':
        return ('rect', {'x1': b['dotX1'], 'y1': -b['dotY1'], 'x2': b['dotX2'], 'y2': -b['dotY2'],
                         'rx': b.get('radiusX'), 'ry': b.get('radiusY'), 'rot': (b.get('rotation') or 0)}, st)
    if t == 'ARC':
        return ('arc', {'p': [b['startX'], -b['startY'], b['referX'], -b['referY'], b['endX'], -b['endY']]}, st)
    if t == 'CIRCLE':
        return ('circle', {'cx': b['centerX'], 'cy': -b['centerY'], 'r': b['radius']}, st)
    if t == 'TEXT':
        return ('text', {'x': b.get('x'), 'y': _fy(b.get('y')), 'rot': (b.get('rotation') or 0) % 360, 'value': b.get('value')},
                _font_from_obj(b))


def _sheet_from_epru(d):
    sh = Sheet()
    comps, wires = {}, {}
    for t, i, b in d['recs']:
        if t == 'COMPONENT':
            c = Component(i, b.get('partId'), b.get('x'), -b.get('y', 0), (b.get('rotation') or 0) % 360, b.get('isMirror'))
            comps[i] = c
            sh.components.append(c)
        elif t == 'WIRE':
            w = Wire(i, [], None)
            wires[i] = w
            sh.wires.append(w)
    for t, i, b in d['recs']:
        if t == 'LINE' and b.get('lineGroup') in wires:
            w = wires[b['lineGroup']]
            w.segs.append((b['startX'], -b['startY'], b['endX'], -b['endY']))
            if w.style is None:
                w.style = _style_from_obj(b)
        elif t == 'ATTR':
            a = _attr_from_obj(i, b)
            sh.attrs.append(a)
            if a.parent in comps:
                comps[a.parent].attrs[a.key] = a
            elif a.parent in wires:
                wires[a.parent].attrs[a.key] = a
        elif t in ('POLY', 'RECT', 'ARC', 'CIRCLE', 'TEXT') and not b.get('partId'):
            sh.prims.append(_prim_from_obj(t, b, i))
    return sh


# --------------------------------------------------------------------------- linking
def link(sheet, symbols, devices):
    """Attach symbol + device attributes to every component."""
    for c in sheet.components:
        dev_uuid = c.attr('Device')
        dev = devices.get(dev_uuid) if dev_uuid else None
        c.device = dev or {}
        sym_uuid = c.attr('Symbol') or (dev or {}).get('Symbol')
        sym = symbols.get(sym_uuid) if sym_uuid else None
        if sym is None and sym_uuid and '_' in str(sym_uuid):
            sym = symbols.get(str(sym_uuid).split('_')[0])
        c.symbol = sym
    return sheet


def resolve(c, text, depth=0):
    """Resolve '={Key}' templates against the component's sheet attrs then device attrs."""
    if text is None or depth > 4:
        return text
    m = re.fullmatch(r'=\{(.+?)\}', str(text).strip())
    if not m:
        return text
    key = m.group(1)
    v = c.attr(key)
    if v is None:
        v = (c.device or {}).get(key)
    return resolve(c, v, depth + 1) if v is not None else ''


# --------------------------------------------------------------------------- downloaded bundle (API format)
def load_bundle(bundle):
    """bundle: dict(sheets=[{uuid,title,dataStr}], symbols={uuid:{title,docType,dataStr}}, devices={uuid:{attributes}})
    -> list of (title, Sheet) already linked."""
    symbols = {}
    for u, s in (bundle.get('symbols') or {}).items():
        if s.get('dataStr'):
            symbols[u] = symbol_from_array(u, s.get('title') or s.get('display_title') or '', s.get('docType'), s['dataStr'])
    devices = {}
    for u, d in (bundle.get('devices') or {}).items():
        attrs = dict(d.get('attributes') or {})
        attrs['_title'] = d.get('title') or d.get('display_title') or ''
        devices[u] = attrs
    out = []
    for sh in bundle.get('sheets') or []:
        s = sheet_from_array(sh['dataStr'])
        link(s, symbols, devices)
        out.append((sh.get('title') or '', s))
    return out

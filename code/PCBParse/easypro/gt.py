"""EasyEDA Pro sheet -> competition ground truth (official JSON conventions).

Conventions (verified against the 13 official open_GT cases, which are themselves
generated from EasyEDA Pro sources):
  * key: designator; power / ground symbols -> '∅' + numeric part of the element id;
    net ports -> the net name, repeated names get _2, _3 ... in document order.
  * Name: printed part number for box / d / led (falls back to the designator);
    net labels: key; others: designator. value: printed value (r/c/l), net name
    (power, ground, net labels).
  * bbox: bounding box of the symbol graphics (pins and texts excluded) + half a stroke.
  * pins: 'pin_<number>' with the source pin name; point = pin tip.
  * nets: purely geometric connectivity (wires + pin tips); same-name labels are NOT
    merged; lone pins without wires are dropped; edges = wire segments split at every
    point where another segment end or a pin tip lands on them.
  * coordinates: integer pixels, bottom-left origin.
"""
import os
import math
import re
from .model import resolve

POWER, NETPORT, FRAME = 18, 19, 20


def _prefix(d):
    m = re.match(r'^([A-Za-z]+)', d or '')
    return m.group(1).upper() if m else ''


MOS_HINT = re.compile(r'MOS|NMOS|PMOS|AO3\d|SI2\d|IRF|IRLM|2N7002|BSS1|DMG|AON|FDN|NCE|SL2\d|AP\d{4}|CJ2\d|LP\d{4}N', re.I)
PIN_NAME_TRANSISTOR = os.environ.get('GT_PIN_NAME_TRANSISTOR', '1') == '1'
BJT_HINT = re.compile(r'NPN|PNP|S80\d\d|S90\d\d|2N390|2N222|BC8\d\d|MMBT|SS8050|SS8550|MJE|TIP1', re.I)


def _is_triangle_symbol(part):
    """Op-amp / comparator symbols: a closed triangle covering most of the symbol."""
    if part is None:
        return False
    for kind, d, st in part.prims:
        if kind != 'poly':
            continue
        pts = d.get('pts') or []
        xy0 = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        xy = [q for k, q in enumerate(xy0) if k == 0 or q != xy0[k - 1]]     # drop repeated vertices
        if len(xy) >= 4 and abs(xy[0][0] - xy[-1][0]) < 1e-6 and abs(xy[0][1] - xy[-1][1]) < 1e-6:
            xy = xy[:-1]
        if len(xy) != 3 and not (len(xy) == 4 and d.get('closed')):
            continue
        xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if len(xy) == 3 and min(w, h) >= 20:
            return True
    return False


AMP_HINT = re.compile(r'LM358|LM324|LM393|LM339|LM2904|LM2902|OPA\d|TL0[678]\d|MCP60\d|SGM8\d|GS8\d|TLV\d{3}|AD8\d{2}|OP\d{2}|NE5532|运放|比较器|op-?amp|comparator', re.I)


def _diag_segments(part, min_len=4):
    n = 0
    for kind, d, st in (part.prims if part else []):
        if kind != 'poly':
            continue
        pts = d.get('pts') or []
        xy = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
        for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
            if abs(x1 - x0) >= 1 and abs(y1 - y0) >= 1 and ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 >= min_len:
                n += 1
    return n


def _n_arcs(part):
    return sum(1 for kind, d, st in (part.prims if part else []) if kind in ('arc', 'bezier'))


MOUNT_HINT = re.compile(r'screw|螺丝|螺孔|安装孔|定位孔|mounting.?hole|mount.?hole|M\d(\.\d)?.{0,3}(hole|孔)', re.I)
MOUNT_TITLE = re.compile(r'^\s*(m\d(\.\d)?[\s_-]*)?(hole|孔)?\s*$|^\s*m\d(\.\d)?\s*$', re.I)


def is_mount_hole(p, title, hint):
    t = (title or '').strip()
    return (p in ('SCREW', 'HOLE', 'MH', 'HOLES') or bool(MOUNT_HINT.search(hint))
            or (t != '' and bool(MOUNT_TITLE.match(t)) and re.search(r'hole|孔|m\d', t, re.I) is not None))


def comp_type(c, desig, npins, title, part=None):
    st = c.symbol.stype if c.symbol else None
    t = (title or '').lower()
    if st == POWER:
        return None
    if st == NETPORT:
        return netport_type(title)
    p = _prefix(desig)
    hint = f"{title} {c.device.get('Manufacturer Part', '')} {c.device.get('_title', '')}"
    if p.startswith('LED') or (p == 'D' and 'led' in t):
        return 'led'
    if p in ('D', 'ZD', 'TVS', 'VD'):
        return 'led' if re.search(r'LED', hint, re.I) else 'd'
    if p in ('R', 'RN', 'RP', 'RV', 'RT', 'VR', 'NTC', 'PTC') or (p.startswith('R') and npins == 2 and p not in ('RLY', 'RJ')):
        return 'r'                      # RS1, RSET1, RF1 ... are resistors too
    if p in ('C', 'CE', 'EC', 'TC') or (p.startswith('C') and npins == 2 and p not in ('CN', 'CON', 'CH')):
        return 'c'
    if p in ('L', 'FB', 'LB'):
        return 'l'
    # transistors keep their device type whatever the pin count (8-pin DFN MOSFETs, dual BJTs):
    # the official training answers type them mosfet / bjt, never box
    if p in ('Q', 'VT') or (p == 'T' and npins == 3):
        if BJT_HINT.search(hint):
            return 'bjt'
        if MOS_HINT.search(hint):
            return 'mosfet'
        if PIN_NAME_TRANSISTOR and part is not None:
            # the symbol's own pin names: B/C/E is a bipolar transistor, G/D/S a MOSFET
            nm = {str(getattr(q, 'name', '') or '').strip().upper() for q in part.pins}
            if {'B', 'C', 'E'} <= nm or {'BASE', 'COLLECTOR', 'EMITTER'} <= nm:
                return 'bjt'
            if {'G', 'D', 'S'} <= nm or {'GATE', 'DRAIN', 'SOURCE'} <= nm:
                return 'mosfet'
        return 'mosfet' if p == 'Q' else 'bjt'
    if p in ('Y', 'X', 'XTAL', 'OSC', 'G'):
        return {2: 'crystal_2pin', 3: 'crystal_3pin', 4: 'crystal_4pin'}.get(npins, 'crystal_2pin') if p != 'X' or npins <= 4 else 'box'
    if p in ('F', 'FU', 'PTC'):
        return 'fuse'
    # mounting holes / screws: official type 'm3螺丝' (anonymous key, like power symbols)
    if is_mount_hole(p, title, hint):
        return 'm3螺丝'
    if DC_TYPE and (p in ('DC', 'DCIN', 'DCJ') or (p in ('J', 'CN', 'P', 'JP', 'U', 'PWR', 'POWER') and DC_HINT.search(hint))) \
            and 2 <= npins <= 6:
        return 'dc'                     # DC power jack: official type 'dc' (anonymous key)
    if p in ('SW', 'S', 'K', 'KEY', 'BTN'):
        return 'switch'
    if p in ('BZ', 'BUZ', 'B', 'LS', 'SPK'):
        return 'buzzer'
    if p in ('BT', 'BAT'):
        return 'battary'
    if p in ('TP',) or (p in ('H', 'J', 'P', 'CN') and npins == 1):
        return 'pin'
    if p in ('M', 'MOT'):
        return 'motor'
    if p in ('E', 'ANT'):
        return 'ant' if p == 'ANT' else 'other'
    if p in ('T', 'TR', 'TF') and npins >= 4 and _n_arcs(part) >= 4:
        return 'transfomer'             # (sic) spelling of the official type list
    if p in ('U', 'IC', 'OP', 'A', 'N') and (_is_triangle_symbol(part) or (AMP_HINT.search(hint) and npins <= 8)):
        return {3: 'amp_3pin', 5: 'amp_5pin', 7: 'amp_7pin'}.get(npins, 'amp')
    if p in ('U', 'IC', 'J', 'P', 'H', 'CN', 'USB', 'JP', 'CON', 'A', 'MK', 'MIC', 'SIM'):
        return 'box'
    return 'box' if npins >= 3 else 'other'


DC_TYPE = os.environ.get('GT_DC_TYPE', '1') == '1'
DC_HINT = re.compile(r'DC-?0\d\d|DC.?jack|DC.?socket|DC插座|DC座|DC电源座|电源座|圆孔电源|DC-?IN', re.I)
# types the official training answers always key anonymously (∅ + id), like power symbols:
# 100 % of battary / buzzer / motor / transfomer / opt / esd / dc / jumper / ant / block /
# circle_header / seg in the 200 training cases ('other' too, but ours is a catch-all for unknown
# prefixes, not the official category: it stays keyed)
ANON_TYPES = {'battary', 'buzzer', 'motor', 'transfomer', 'opt', 'esd', 'dc', 'jumper', 'ant', 'block',
              'circle_header', 'seg'}
ANON_SPECIAL = os.environ.get('GT_ANON_SPECIAL', '0') == '1'   # off: see pcbparse.pipeline.ANON_SPECIAL
MOUNT_ANON = os.environ.get('GT_MOUNT_ANON', '0') == '1'        # off: mounting holes keyed by designator
CRYSTAL_VALUE = os.environ.get('GT_CRYSTAL_VALUE', '1') == '1'


def _prims_bbox(c, part):
    xs, ys = [], []

    def add(x, y):
        X, Y = c.xf(x, y)
        xs.append(X)
        ys.append(Y)
    for kind, d, st in part.prims:
        if kind == 'poly':
            pts = d['pts']
            for i in range(0, len(pts) - 1, 2):
                add(pts[i], pts[i + 1])
        elif kind == 'rect':
            for x in (d['x1'], d['x2']):
                for y in (d['y1'], d['y2']):
                    add(x, y)
        elif kind == 'circle':
            r = d['r']
            for dx, dy in ((-r, -r), (r, r), (-r, r), (r, -r)):
                add(d['cx'] + dx, d['cy'] + dy)
        elif kind == 'arc':
            from .render import _arc_points
            for x, y in _arc_points(d['p'], 12):
                add(x, y)
        elif kind == 'ellipse':
            for dx, dy in ((-d['rx'], -d['ry']), (d['rx'], d['ry'])):
                add(d['cx'] + dx, d['cy'] + dy)
        elif kind == 'bezier':
            pts = d['pts']
            for i in range(0, len(pts) - 1, 2):
                add(pts[i], pts[i + 1])
    if not xs and part.bbox:
        b = part.bbox
        add(b[0], b[1])
        add(b[2], b[3])
    if not xs:
        return None
    h = 0.5
    return [min(xs) - h, min(ys) - h, max(xs) + h, max(ys) + h]


def _num_id(eid):
    m = re.search(r'(\d+)$', str(eid))
    return m.group(1) if m else str(eid)


def displayed(c, part, key):
    from .render import component_label_texts
    for k, val, *_ in component_label_texts(c, part):
        if k == key:
            return val
    return None


def netport_type(title):
    """EasyEDA netport symbols -> official net types (the platform compares them exactly):
    netport-in -> net_input, netport-out -> net_output, netport-bi -> net_bidirection.
    Custom port symbols default to net_input, by far the most common kind."""
    t = (title or '').lower()
    if not NETPORT_TYPES:
        return 'net_bidirection'
    if t.endswith('-out') or t == 'out' or 'output' in t:
        return 'net_output'
    if t.endswith('-bi') or t == 'bi' or 'bidir' in t:
        return 'net_bidirection'
    return 'net_input'


NETPORT_TYPES = os.environ.get('GT_NETPORT_TYPES', '1') == '1'
DUP_SUFFIX = os.environ.get('GT_DUP_SUFFIX', '1') == '1'
POWER_SHOWN_ONLY = os.environ.get('GT_POWER_SHOWN', '0') == '1'   # public GT keeps the (hidden) net name
NET_SHORT = os.environ.get('GT_NET_SHORT', '1') == '1'     # short symbols labelled 'net_short' (official type list)


def build_gt(sheet, x0, y0, scale, render_texts=None):
    """Return the competition JSON for `sheet` drawn with sheet point (x0, y0) at the
    image's bottom-left corner and `scale` px per unit."""
    def P(x, y):
        return int(round((x - x0) * scale)), int(round((y - y0) * scale))

    comps, pins_out = {}, {}
    pin_index = []          # (x, y, key, number)
    label_count = {}
    recs = []               # [key, desig, comp-dict, [(num, name, X, Y)], is_part]
    for c in sheet.components:
        sym = c.symbol
        if sym is None or sym.stype == FRAME or str(sym.title).lower().startswith('sheet-symbol'):
            continue
        part = sym.part(c.part_name)
        if part is None:
            continue
        if not part.pins and not (sym.stype not in (POWER, NETPORT) and
                                  is_mount_hole(_prefix(c.attr('Designator')), sym.title or '',
                                                f"{sym.title} {(c.device or {}).get('Manufacturer Part', '')} {(c.device or {}).get('_title', '')}")):
            continue
        if sym.stype in (25, 31) or (sym.stype == 22 and not NET_SHORT):   # off-page connector, diff-pair flag
            continue
        desig = c.attr('Designator')
        title = sym.title or ''
        is_part = False
        if sym.stype == 22:
            # short symbol (a diamond joining two differently named nets): official type
            # 'net_short', anonymous key, two pins
            key = '∅' + _num_id(c.id)
            typ, name, value = 'net_short', key, None
        elif sym.stype == POWER:
            net = displayed(c, part, 'Global Net Name') or c.attr('Global Net Name') or c.attr('Name') or \
                (c.device or {}).get('Global Net Name') or (c.device or {}).get('Name') or ''
            gnd = 'gnd' in title.lower() or 'GND' in str(net).upper()
            key = '∅' + _num_id(c.id)
            typ = 'gnd' if gnd else 'v'
            if POWER_SHOWN_ONLY:
                # official labels: the value is the name printed next to the symbol; a ground /
                # supply symbol whose name is hidden has value None (training data: 1046 of 2200
                # grounds), not the hidden net name
                shown = displayed(c, part, 'Global Net Name') or displayed(c, part, 'Name')
                name, value = key, (shown or None)
            else:
                name, value = key, (net or ('GND' if gnd else None))
        elif sym.stype == NETPORT:
            net = displayed(c, part, 'Name') or displayed(c, part, 'Global Net Name') or c.attr('Name') or ''
            if not net:
                continue
            k = label_count.get(net, 0) + 1
            label_count[net] = k
            key = net if k == 1 else f'{net}_{k}'
            typ, name, value = netport_type(title), key, net
        else:
            if not desig or desig.endswith('?'):
                continue
            key = desig
            is_part = True
            typ = comp_type(c, desig, len(part.pins), title, part)
            if (typ == 'm3螺丝' and MOUNT_ANON) or (ANON_SPECIAL and typ in ANON_TYPES):
                key = '∅' + _num_id(c.id)        # official labels key these anonymously
                is_part = False
            name_txt = displayed(c, part, 'Name')
            val_txt = displayed(c, part, 'Value')
            if typ in ('r', 'c', 'l') or (CRYSTAL_VALUE and typ.startswith('crystal')):
                # (crystals: the printed frequency, as in the official training answers)
                name, value = key, (name_txt or val_txt)
            elif typ in ('box', 'd', 'led', 'bjt', 'mosfet') or typ.startswith('amp'):
                model = name_txt or displayed(c, part, 'Manufacturer Part') or val_txt
                name, value = (model or key), None
            else:
                name, value = key, None
        bb = _prims_bbox(c, part)
        if bb is None:
            continue
        (bx0, by0), (bx1, by1) = P(bb[0], bb[1]), P(bb[2], bb[3])
        cd = {'Name': name, 'type': typ, 'value': value, 'bbox': [min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1)]}
        pl = []
        for k, pn in enumerate(part.pins, 1):
            num = str(pn.number) if pn.number not in (None, '') else str(k)
            nm = pn.name if pn.name not in (None, '') else num
            X, Y = c.xf(pn.x, pn.y)
            pl.append((num, nm, X, Y))
        recs.append([key, desig, cd, pl, is_part])
    # a designator used by several parts on the sheet (units of a multi-part IC, or two
    # symbols sharing a designator): every one is kept, keyed <desig>_1, _2 ... from left to
    # right, as in the official training data (U3_1 / U3_2, C6_1 / C6_2)
    if DUP_SUFFIX:
        groups = {}
        for r in recs:
            if r[4]:
                groups.setdefault(r[0], []).append(r)
        for key, lst in groups.items():
            if len(lst) < 2:
                continue
            lst.sort(key=lambda r: (r[2]['bbox'][0], -r[2]['bbox'][3]))
            for i, r in enumerate(lst, 1):
                r[0] = f'{key}_{i}'
    for key, desig, cd, pl, is_part in recs:
        if key in comps:          # duplicated key (DUP_SUFFIX off): keep the first
            continue
        comps[key] = cd
        pp = {}
        for num, nm, X, Y in pl:
            px, py = P(X, Y)
            pk = f'pin_{num}'
            if pk in pp:
                continue
            pp[pk] = {'pinname': nm, 'point': {'x': px, 'y': py}}
            pin_index.append((X, Y, key, num))
        pins_out[key] = pp
    nets = _nets(sheet, pin_index, P)
    return {'components': comps, 'pins': pins_out, 'nets': nets}


def _nets(sheet, pin_index, P, eps=1e-3):
    segs = [s for w in sheet.wires for s in w.segs if (abs(s[0] - s[2]) + abs(s[1] - s[3])) > eps]
    n = len(segs)
    par = list(range(n + len(pin_index)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    def key(x, y):
        return (round(x, 2), round(y, 2))
    ends = {}
    for i, (x1, y1, x2, y2) in enumerate(segs):
        for p in ((x1, y1), (x2, y2)):
            ends.setdefault(key(*p), []).append(i)
    for lst in ends.values():
        for j in lst[1:]:
            uni(lst[0], j)
    from .render import _on_interior
    # T-connections: an end lying on another segment's interior
    for (x, y), lst in ends.items():
        for j, s in enumerate(segs):
            if _on_interior(x, y, *s, eps=1e-2):
                uni(lst[0], j)
    # pins
    split_pts = {j: [] for j in range(n)}
    for (x, y), lst in ends.items():
        for j, s in enumerate(segs):
            if _on_interior(x, y, *s, eps=1e-2):
                split_pts[j].append((x, y))
    tip_map = {}
    for k, (x, y, comp, num) in enumerate(pin_index):
        idx = n + k
        kk = key(x, y)
        tip_map.setdefault(kk, []).append(idx)
        for j in ends.get(kk, []):
            uni(idx, j)
        for j, s in enumerate(segs):
            if _on_interior(x, y, *s, eps=1e-2):
                uni(idx, j)
                split_pts[j].append((x, y))
    for lst in tip_map.values():          # pin-to-pin contact
        for j in lst[1:]:
            uni(lst[0], j)
    groups = {}
    for i in range(n + len(pin_index)):
        groups.setdefault(find(i), []).append(i)
    nets = {}
    k = 0
    for root, members in sorted(groups.items(), key=lambda kv: min(kv[1])):
        pins = [pin_index[i - n] for i in members if i >= n]
        wires = [i for i in members if i < n]
        if not pins or (len(pins) < 2 and not wires):
            continue
        k += 1
        toks = []
        for (_, _, comp, num) in pins:
            t = f'{comp}.{num}'
            if t not in toks:
                toks.append(t)
        edges = {}
        e = 0
        for j in wires:
            x1, y1, x2, y2 = segs[j]
            pts = [(x1, y1)] + sorted(set(split_pts[j]), key=lambda p: (p[0] - x1) ** 2 + (p[1] - y1) ** 2) + [(x2, y2)]
            for a, b in zip(pts, pts[1:]):
                pa, pb = P(*a), P(*b)
                if pa == pb:
                    continue
                e += 1
                edges[f'edge_{e}'] = [{'x': pa[0], 'y': pa[1]}, {'x': pb[0], 'y': pb[1]}]
        nets[f'net_{k}'] = {'hyperGraph': '(' + ','.join(toks) + ')', 'edges': edges}
    return nets

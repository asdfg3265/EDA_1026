"""End-to-end schematic image -> competition JSON."""
import os
import re
import json
import time
import math
import numpy as np
import cv2

from .prims import extract
from .wiregraph import build_lines, snap_and_connect, decompose_edges, UF
from .sym import segment_symbols, estimate_unit
from .text import merge_fragments, build_text_items, ocr_items, split_joined_values
from .assemble import build
from .library import Library

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIBS = [os.path.join(os.path.dirname(_HERE), 'data', 'BoxICLibrary.json'),
                os.path.join(os.path.dirname(_HERE), 'data', 'cpntLibrary.json'),
                # IC pin maps of the official 200 training cases not in the two libraries above
                # (Q&A: every IC of the test set appears in the 200 cases or in cpntLibrary)
                os.path.join(os.path.dirname(_HERE), 'data', 'trainLibrary.json')]
_LIB = None
MERGE_POWER = os.environ.get('PCB_MERGE_POWER', '0') == '1'   # off: each ground / supply branch is its own net, as in the labels (Q&A 10-23: the scorer merges rails itself)
DUP_SUFFIX = os.environ.get('PCB_DUP_SUFFIX', '1') == '1'


def get_lib():
    global _LIB
    if _LIB is None:
        _LIB = Library([p for p in os.environ.get('PCB_LIBS', '').split(os.pathsep) if p] + DEFAULT_LIBS)
    return _LIB


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(path)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        alpha = im[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.array([247, 252, 255], np.float32)  # BGR paper colour
        im = (im[:, :, :3].astype(np.float32) * alpha + bg * (1 - alpha)).astype(np.uint8)
    return im[:, :, ::-1].copy()  # RGB


SAFE_FLAGS = [('text', 'UNDERSCORE_JOIN'), ('text', 'STRIP_RUNS'), ('text', 'CASE_FIX'), ('text', 'DOT_FIX'),
              ('assemble', 'MERGE_TOUCHING'), ('assemble', 'MERGE_SWITCH_PIECES'), ('assemble', 'CARVE_FLAGS'),
              ('assemble', 'DESIG_CONSIST'), ('assemble', 'RECT_COMPARTMENTS'), ('assemble', 'LED_ARROW_PAIR'),
              ('assemble', 'AMP_TRIANGLE'), ('assemble', 'BODY_ALL_GIDS'), ('assemble', 'SWITCH_CLAMP'),
              ('assemble', 'MERGE_CONTACT_ORPHANS'), ('assemble', 'ORPHAN_DESIG'),
              # refinements added in v10
              ('text', 'UNDERSCORE_VERT'), ('text', 'TALL_VERT'), ('sym', 'GLYPH_CUTS'),
              ('assemble', 'NAME_WORDS'), ('assemble', 'POWER_STEM'), ('assemble', 'EQUAL_PINS'),
              ('assemble', 'MERGE_BODY_PARTS'), ('assemble', 'BODY_STUBS'), ('assemble', 'DROP_NET_TIES'),
              ('assemble', 'SPLIT_PAIRS'), ('assemble', 'SPLIT_STACKS'), ('assemble', 'FAR_DESIG'),
              ('assemble', 'DESIG_LIKE_MODEL'), ('assemble', 'NAME_GLOBAL'), ('assemble', 'ROT_VOTE'),
              ('assemble', 'NUM_SIDE_PRIOR'), ('assemble', 'RECT_CLOSE'), ('assemble', 'RECT_RED_FALLBACK'),
              ('assemble', 'GATE_RELAX'), ('assemble', 'Q_FROM_O'), ('assemble', 'DEDUP_PINS'),
              ('assemble', 'FAR_POWER_NAME'), ('assemble', 'ARROW_TAIL'), ('library', 'LAST_PIN'),
              ('assemble', 'CARVE_FLAGS_SMALL'), ('assemble', 'SMALL_EDGE_PINS'), ('assemble', 'DROP_FRAMES'),
              # refinements added in v11
              ('assemble', 'NAME_ASSIGN'), ('assemble', 'GND_PASSING'), ('assemble', 'GND_BASE_PIX'),
              ('assemble', 'GND_FIRST'), ('assemble', 'GND_HOST_FIX'), ('assemble', 'GND_HOST_BBOX'),
              ('assemble', 'AMP_PINS'), ('assemble', 'STACK_DROP_Z'), ('assemble', 'STACK_Z_ORIENT'),
              ('assemble', 'STACK_PARTIAL'), ('assemble', 'DESIG_STACK_PRIOR'), ('assemble', 'DROP_CONTACT_ISLANDS'),
              ('assemble', 'RECT_PORT'), ('assemble', 'FLAG_DOT_DROP'), ('assemble', 'STACK_PITCH'),
              ('assemble', 'CARVED_GND_NAME'), ('assemble', 'FAR_PWR_SPAN'), ('assemble', 'CAP_UNIT_FIX'),
              ('assemble', 'PASSIVE_PIN_LEN'), ('assemble', 'LEAD_PIECES'), ('assemble', 'MERGE_SKIP_ARROWS'),
              ('textfix', 'OHM_PROB'),
              # refinements added in v13
              ('assemble', 'BODY_LEADS'), ('assemble', 'TRI_AGND'), ('assemble', 'BALL_NUMS'), ('assemble', 'ORPHAN_SERIES'), ('assemble', 'GND_BOTH_AXES'), ('assemble', 'FAR_FROM_TIPS'), ('assemble', 'FLAG_VOLT'), ('assemble', 'SURE_CHARS'), ('assemble', 'LIB_MIN4'), ('assemble', 'IO_ALT'), ('assemble', 'FLAG_OWN_AXIS'), ('text', 'SPLIT_VALUES'), ('text', 'TEXT_OVER_WIRE'), ('text', 'COLOR_SPLIT'), ('text', 'WIRE_BRIDGE'), ('text', 'VERT_LOCAL_GAP'), ('text', 'WIRE_SPECKS'), ('textfix', 'NAME_ZERO'), ('textfix', 'CONTEXT_IO'), ('text', 'BRIDGE_CONT'), ('assemble', 'SWITCH_PIN_LEN'), ('assemble', 'TRANS_CLAMP'),
              ('assemble', 'NAME_FAR_PN'), ('text', 'TITLE_DROP'), ('assemble', 'LINE_DROP')]


def parse_image(path, debug=False):
    """Parse one image.  If the full pipeline raises, retry once in 'safe mode' with the
    optional refinements switched off, so that a corner case never costs a whole case."""
    try:
        return _parse_image(path, debug)
    except Exception:
        import importlib
        saved = []
        for mod, name in SAFE_FLAGS:
            m = importlib.import_module('.' + mod, __package__)
            if hasattr(m, name):
                saved.append((m, name, getattr(m, name)))
                setattr(m, name, False)
        try:
            return _parse_image(path, debug)
        finally:
            for m, name, v in saved:
                setattr(m, name, v)


def _parse_image(path, debug=False):
    img = load_image(path)
    # sheet frame + title block (zone labels, cells, logo) are not part of the circuit; on a
    # framed page the parse then works on the drawing's bounding box (coordinates mapped back)
    H0 = img.shape[0]
    crop = None
    try:
        from . import frame as _F
        img, finfo = _F.remove_frame(img)
        # dashed / dash-dot lines (function boxes, section dividers)
        img, _ = _F.remove_dashes(img)
        if finfo is not None and _F.FRAME_CROP:
            crop = _F.content_crop(img)
            if crop is not None:
                img = img[crop[1]:crop[3], crop[0]:crop[2]].copy()
    except Exception:
        crop = None
    # drawings in other tools' palettes (Altium, KiCad ...) are repainted in the EasyEDA
    # palette first; EasyEDA images come back untouched.  Any failure keeps the original.
    _style = 'unknown'
    try:
        from .style import normalise
        img, _style = normalise(img)
    except Exception:
        pass
    from . import assemble as _A
    _A.SRC_STYLE[0] = _style
    P = extract(img)
    try:
        from . import greengnd as _GG
        if _GG.GREEN_GND:
            _gg = _GG.find_green_grounds(P.green > 0, P.u, P.t)
            if _gg:
                # ground symbols drawn in the wire colour: repaint them as symbols, parse again
                img = _GG.repaint(img, _gg, P.green > 0)
                P = extract(img)
    except Exception:
        pass
    f = 1.0
    if P.t < 2.4 and img.shape[0] * img.shape[1] <= 9e6:   # >9 MP: 2x copy would need several GB
        # thin strokes / tiny text: work on a 2x upscaled copy, map coordinates back
        f = 2.0
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        P = extract(img)
    lines = build_lines(P)
    uf, terms = snap_and_connect(P, lines)
    S = segment_symbols(P, terms, lines)
    u = estimate_unit(P, S)
    if abs(u - P.u) > 0.03 * P.u:
        P.u = u
        P.s = u / 10.0
        S = segment_symbols(P, terms, lines)
    items = build_text_items(P, img)
    ocr_items(P, img, items)
    items = merge_fragments(P, img, items)
    items = split_joined_values(items, P, img)
    from .text import split_desig_partnumber, split_doubled
    items = split_desig_partnumber(items)
    items = split_doubled(items)
    from .textfix import context_fix
    items = context_fix(items)
    from .text import drop_titles
    items = drop_titles(items, P)
    comps = build(P, items, S, lines, uf, terms, lib=get_lib())
    out = to_json(P, comps, lines, uf, S, f)
    if crop is not None:
        _F.shift_output(out, crop[0], H0 - crop[3])
    if os.environ.get('PCB_NAME_LOG'):
        from . import assemble as _A
        # key -> final key mapping is not needed: NAME_LOG keys are designators before de-dup
        out['_name_log'] = [list(r) for r in _A.NAME_LOG]
    if os.environ.get('PCB_FB_LOG'):
        from . import assemble as _A
        out['_fb_log'] = dict(f=f, H=int(P.H), rows=list(_A.FB_LOG))
    if debug:
        return out, dict(P=P, lines=lines, uf=uf, terms=terms, S=S, items=items, comps=comps)
    return out


def _r(v):
    return int(round(v))


VALUE_CLEAN = os.environ.get('PCB_VALUE_CLEAN', '1') == '1'


def _clean_value(typ, v):
    """OCR slips in passive values that a wire or a pin line through the text causes: the
    'F' of a capacitor read as 'l=' ('2.2ul='), a '0' half hidden by a line read as ')' or 'J'
    ('1)uF'), an 'O' between digits and the unit ('1OuF')."""
    s = v
    if typ == 'c':
        s = re.sub(r'^(\d+(?:\.\d+)?)([pnuμµm])[lI1|]=$', r'\1\2F', s)
    s = re.sub(r'^(\d+)[)J\]]([pnuμµmkKM]?[FHΩ])$', r'\g<1>0\2', s)
    s = re.sub(r'^(\d+)O([pnuμµmkKM]?[FHΩ](?:\d.*)?)$', r'\g<1>0\2', s)
    return s


MODEL_NAME_TYPES = ('d', 'led', 'opt', 'zener', 'schottky', 'tvs')
# evaluation spec 2026-10-22, type appendix: 'crystal' and 'amp' do not distinguish pin counts
TYPE_FAMILY_OUT = os.environ.get('PCB_TYPE_FAMILY_OUT', '1') == '1'
OUT_TYPE = {'crystal_2pin': 'crystal', 'crystal_3pin': 'crystal', 'crystal_4pin': 'crystal',
            'amp_3pin': 'amp', 'amp_5pin': 'amp', 'amp_7pin': 'amp'}


# types the official TRAINING answers key anonymously (100 % of 200 cases).  Off: the newer
# answers (10GT, public) key every part by its designator - 'only VCC and GND are ∅' (Q&A A24) -
# and also moved net labels from ∅ to their names; kept as a switch
ANON_SPECIAL = os.environ.get('PCB_ANON_SPECIAL', '0') == '1'
FUSE_NO_VALUE = os.environ.get('PCB_FUSE_NO_VALUE', '1') == '1'
VALUE_PM_NONE = os.environ.get('PCB_VALUE_PM_NONE', '1') == '1'
NAME_CHAR_FIX = os.environ.get('PCB_NAME_CHAR_FIX', '1') == '1'
UNIT_NAME_SHARE = os.environ.get('PCB_UNIT_NAME_SHARE', '1') == '1'
PORT_UPPER = os.environ.get('PCB_PORT_UPPER', '1') == '1'
PORT_SNAP = os.environ.get('PCB_PORT_SNAP', '1') == '1'
HEX_NAME_SNAP = os.environ.get('PCB_HEX_NAME_SNAP', '1') == '1'
PIN_COMMA_UNIFY = os.environ.get('PCB_PIN_COMMA_UNIFY', '1') == '1'
PORT_SNAP_STEM = os.environ.get('PCB_PORT_SNAP_STEM', '1') == '1'
POWER_SPACE_SPLIT = os.environ.get('PCB_POWER_SPACE_SPLIT', '1') == '1'
ANON_OUT_TYPES = {'battary', 'buzzer', 'motor', 'transfomer', 'opt', 'esd', 'dc', 'jumper', 'ant', 'block',
                  'circle_header', 'seg'}
MICRO_EDGES = os.environ.get('PCB_MICRO_EDGES', '0') == '1'   # experimental, off
MICRO_MAX = float(os.environ.get('PCB_MICRO_MAX', '10'))


def _seg_dist(pt, p, q):
    px, py = pt
    ax, ay = p
    bx, by = q
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - ax - u * vx, py - ay - u * vy)


def to_json(P, comps, lines, uf, S, f=1.0):
    H = P.H

    def _r(v):
        return int(round(v / f))
    t = P.t
    # ------------------------------------------------ keys
    if PORT_UPPER:
        # net names and supply names are upper case in the answers (9570 of 9573): a
        # lower-case letter is a misread ('vcC', '+5v', 'vo')
        def _up(t):
            return ''.join(ch.upper() if 'a' <= ch <= 'z' else ch for ch in t) if t else t
        for c in comps:
            if c.type in ('v', 'gnd') or (c.type or '').startswith('net'):
                c.value = _up(c.value)
                if (c.type or '').startswith('net'):
                    c.key = _up(c.key)
                    c.name = _up(c.name)
    if PORT_SNAP:
        # a supply name read once, one letter off a name the sheet uses several times, and
        # not a well-formed name itself ('+EV', 'V1N', 'PIGND', '+55V' beside four '+5V')
        from .library import lev as _lev
        from collections import Counter as _Ctr
        vals = _Ctr(c.value for c in comps if c.type in ('v', 'gnd') and c.value)
        good = re.compile(r'.*(?:VCC|VDD|VIN|VBUS|VBAT|VSYS|VOUT|VREF|VEE|VSS|GND|AVDD|DVDD).*'
                          r'|[+-]?[A-Z_]*\d+(?:[.,]\d+)?V\d*[A-Z_0-9]*')
        stem = re.compile(r'.*(?:VCC|VDD|VIN|VBUS|VBAT|VSYS|VOUT|VREF|VEE|VSS|GND).*')
        for c in comps:
            v = c.value
            if c.type not in ('v', 'gnd') or not v or vals[v] != 1:
                continue
            tg = [w for w, n in vals.items() if n >= 2 and w != v and _lev(w, v) <= 1]
            if len(tg) != 1:
                continue
            w = tg[0]
            dup = len(v) == len(w) + 1 and any(v[i] == v[i + 1] and v[:i] + v[i + 1:] == w for i in range(len(v) - 1))
            # (a speck read as a decimal point: '1.2VSB' beside two '12VSB')
            dot = len(v) == len(w) + 1 and any(v[i] in '.,' and v[:i] + v[i + 1:] == w for i in range(len(v)))
            # (a supply stem misread: 'YDD_3V3' beside many 'VDD_3V3' still looks like a
            # voltage name, but it lost the stem the frequent one has)
            lost = PORT_SNAP_STEM and stem.fullmatch(w) is not None and stem.fullmatch(v) is None
            # (a stroke beside the name read as a leading letter: 'ZVCC25' beside two 'VCC25';
            # A/D/P/S/+/- before a stem make real names: AVDD, DVDD, PVCC, +VCC)
            lead = PORT_SNAP_STEM and len(v) == len(w) + 1 and v[1:] == w and v[0] not in 'ADPSIO+-' \
                and stem.match(w) is not None and re.match(r'[A-Z]', w) is not None and 'GND' not in w
            if not good.fullmatch(v) or dup or dot or lost or lead:
                c.value = w
    seen = {}
    anon_n = 0
    order = sorted(comps, key=lambda c: (c.bbox[1], c.bbox[0]))
    groups = {}
    for c in order:
        if c.anon or c.type in ('gnd', 'v') and not c.key or (ANON_SPECIAL and c.type in ANON_OUT_TYPES):
            anon_n += 1
            c.key = f'∅{anon_n}'
            c.base_key = c.key
            continue
        k = c.key or f'X{len(order) + len(groups) + 1}'
        c.base_key = k
        groups.setdefault((k, c.type.startswith('net')), []).append(c)
    for (k, is_net), lst in groups.items():
        if len(lst) == 1:
            lst[0].key = k
        elif is_net or not DUP_SUFFIX:
            # repeated net-label names: X, X_2, X_3 ... (reading order)
            for i, c in enumerate(lst, 1):
                c.key = k if i == 1 else f'{k}_{i}'
        else:
            # one designator on several parts (units of a multi-part IC, or two symbols sharing
            # a designator): the official labels key them <desig>_1, _2 ... from left to right
            lst.sort(key=lambda c: (c.bbox[0], c.bbox[1]))
            for i, c in enumerate(lst, 1):
                c.key = f'{k}_{i}'
    keys = [c.key for c in comps]
    if len(set(keys)) != len(keys):        # a suffixed key colliding with a printed one
        used = set()
        for c in order:
            k = c.key
            j = 2
            while k in used:
                k = f'{c.key}_{j}'
                j += 1
            c.key = k
            used.add(k)
    for c in comps:
        # an opto-coupler with hidden pin names: the common symbol (unnamed, or PC817 A-D)
        # names them AN/CAT/EM/COL; other parts' symbols (PC817X2CSP9F) keep the numbers
        if getattr(c, 'opto', False) and (not c.name or c.name == c.key
                                           or re.fullmatch(r'PC817[A-D]?', str(c.name)) is not None):
            nm_ = {'1': 'AN', '2': 'CAT', '3': 'EM', '4': 'COL'}
            for pn in c.pins:
                if str(pn.num) in nm_:
                    pn.name = nm_[str(pn.num)]
    if HEX_NAME_SNAP:
        # 32-digit hex part ids read slightly wrong on some copies: take the clean id another
        # copy on the sheet shows (two characters off at most)
        from .library import lev as _lev
        hexes = {c.name for c in comps if c.name and re.fullmatch(r'[0-9a-f]{32}', str(c.name))}
        for c in comps:
            n_ = str(c.name or '')
            if hexes and 28 <= len(n_) <= 36 and n_ not in hexes:
                best_ = min(hexes, key=lambda h: _lev(h, n_))
                if _lev(best_, n_) <= 3:
                    c.name = best_
    if PIN_COMMA_UNIFY:
        # long pin names of one IC ('IO, DIFFIO_L1p'): the small comma after the first word is
        # often lost ('IO DIFFIO_L1p'); when the part's other names keep it, put it back
        for c in comps:
            if c.type != 'box' or len(c.pins) < 4:
                continue
            nm = [str(pn.name) for pn in c.pins if pn.name]
            heads = {m_.group(1) for m_ in (re.match(r'([A-Za-z0-9]{1,4}), \S', x) for x in nm) if m_}
            for pn in c.pins:
                if not pn.name:
                    continue
                m_ = re.match(r'([A-Za-z0-9]{1,4}) (\S.*)$', str(pn.name))
                if m_ and m_.group(1) in heads:
                    pn.name = m_.group(1) + ', ' + m_.group(2)
    # ------------------------------------------------ pin numbering
    from .assemble import AMP_PINS
    for c in comps:
        if c.type in ('box', 'switch') or (AMP_PINS and c.type.startswith('amp')):
            used = set()
            for pn in c.pins:
                if pn.num and pn.num not in used:
                    used.add(pn.num)
                else:
                    pn.num = None
            # pin numbers hidden but the pin names are numbers (connectors / headers): the name
            # is the number
            for pn in c.pins:
                if pn.num is None and pn.name and str(pn.name).isdigit() and pn.name not in used:
                    pn.num = str(pn.name)
                    used.add(pn.num)
            k = 1
            for pn in sorted(c.pins, key=lambda p: (p.tip[0], p.tip[1])):
                if pn.num is None:
                    while str(k) in used:
                        k += 1
                    pn.num = str(k)
                    used.add(pn.num)
                if pn.name is None:
                    pn.name = pn.num
        elif len(c.pins) == 2:
            a, b = c.pins
            # pin_1: left (or bottom), pin_2: right (or top)
            dx = abs(a.tip[0] - b.tip[0])
            dy = abs(a.tip[1] - b.tip[1])
            if dx >= dy:
                first, second = (a, b) if a.tip[0] <= b.tip[0] else (b, a)
            else:
                first, second = (a, b) if a.tip[1] >= b.tip[1] else (b, a)
            first.num, second.num = '1', '2'
            if c.type in ('d', 'led'):
                first.name, second.name = 'A', 'K'
            else:
                first.name, second.name = '1', '2'
        else:
            for k, pn in enumerate(sorted(c.pins, key=lambda p: (p.tip[0], -p.tip[1])), 1):
                pn.num = str(k)
                pn.name = str(k)
        for pn in c.pins:
            pn.comp = c
    # ------------------------------------------------ nets
    L = len(lines)
    wpieces = list(getattr(P, 'wire_pieces', None) or [])
    for c in wpieces:
        for pn in c.pins:
            pn.comp = c
    allpins = [pn for c in comps for pn in c.pins] + [pn for c in wpieces for pn in c.pins]
    U = UF(L + len(allpins))
    # a short lead piece left between a designator's glyphs and the wire joins its two ends
    pidx = {id(pn): i for i, pn in enumerate(allpins)}
    for c in wpieces:
        U.union(L + pidx[id(c.pins[0])], L + pidx[id(c.pins[1])])
    for a in range(L):
        U.union(a, uf.find(a))
    for i, pn in enumerate(allpins):
        if pn.term is not None:
            U.union(L + i, pn.term['line'])
    # pin-pin contacts
    for i in range(len(allpins)):
        for j in range(i + 1, len(allpins)):
            a, b = allpins[i], allpins[j]
            if a.comp is b.comp:
                continue
            if math.hypot(a.tip[0] - b.tip[0], a.tip[1] - b.tip[1]) <= 1.5 * t + 1.5:
                U.union(L + i, L + j)
    # pins whose tip lies on a junction dot of a wire line (not at a terminal)
    for i, pn in enumerate(allpins):
        if pn.term is not None:
            continue
        for (x, y, r) in P.dots:
            if math.hypot(pn.tip[0] - x, pn.tip[1] - y) <= r + t:
                for ln in lines:
                    uu, d = ln.proj((x, y))
                    if d <= r + t and -0.02 <= uu <= 1.02:
                        U.union(L + i, ln.id)
                        break
    # same-name power / ground symbols are one electrical net (the official scorer
    # merges them as well; merging here is idempotent)
    if MERGE_POWER:
        first = {}
        for i, pn in enumerate(allpins):
            c = pn.comp
            if c.type in ('gnd', 'v'):
                key = (c.type, (c.value or ('GND' if c.type == 'gnd' else '')).upper())
                if key in first:
                    U.union(first[key], L + i)
                else:
                    first[key] = L + i
    groups = {}
    for i, pn in enumerate(allpins):
        if pn.comp.type == 'wire':
            continue
        groups.setdefault(U.find(L + i), []).append(pn)
    E = decompose_edges(lines)
    line_root = {a: U.find(a) for a in range(L)}
    nets = {}
    k = 0
    for root, pins in sorted(groups.items(), key=lambda kv: min((p.tip[1], p.tip[0]) for p in kv[1])):
        wired = [a for a in range(L) if line_root[a] == root]
        if len(pins) < 2 and not wired:
            continue
        if len(pins) < 2:
            # a lone pin with only a tiny wire remnant (e.g. left over from a no-connect 'X')
            wl = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for a in wired for p, q in E[a])
            if wl < 0.8 * P.u:
                continue
        k += 1
        toks = []
        for pn in pins:
            tok = f'{pn.comp.key}.{pn.num}'
            if tok not in toks:
                toks.append(tok)
        edges = {}
        e = 0
        for a in wired:
            for p, q in E[a]:
                e += 1
                edges[f'edge_{e}'] = [{'x': _r(p[0]), 'y': _r(H - p[1])}, {'x': _r(q[0]), 'y': _r(H - q[1])}]
        if MICRO_EDGES and wired:
            # the drawn wire stops a few pixels short of (or past a corner from) the pin end it
            # is snapped to: the labels keep that short piece as a segment of its own
            segs = [(p, q) for a in wired for p, q in E[a]]
            ends = [p for pq in segs for p in pq]
            for pn in pins:
                tx, ty = pn.tip
                if any(_seg_dist((tx, ty), p, q) <= 1.0 * f for p, q in segs):
                    continue
                bp = min(ends, key=lambda p: math.hypot(p[0] - tx, p[1] - ty))
                dx, dy = abs(bp[0] - tx), abs(bp[1] - ty)
                d = math.hypot(dx, dy) / f
                if 1.5 <= d <= MICRO_MAX and min(dx, dy) <= 1.0 * f:
                    e += 1
                    edges[f'edge_{e}'] = [{'x': _r(bp[0]), 'y': _r(H - bp[1])}, {'x': _r(tx), 'y': _r(H - ty)}]
        nets[f'net_{k}'] = {'hyperGraph': '(' + ','.join(toks) + ')', 'edges': edges}
    # ------------------------------------------------ components & pins
    if UNIT_NAME_SHARE:
        # the units of one multi-part IC (U2_1, U2_2) carry one part number; it is usually
        # printed beside one unit only
        byk = {}
        for c in comps:
            if c.type in ('box',) or c.type.startswith('amp'):
                byk.setdefault(getattr(c, 'base_key', c.key), []).append(c)
        for k, lst in byk.items():
            if len(lst) < 2:
                continue
            names = {c.name for c in lst if c.name and c.name != k}
            if len(names) == 1:
                nm = names.pop()
                for c in lst:
                    if not c.name or c.name == k:
                        c.name = nm
    jc, jp = {}, {}
    for c in comps:
        x0, y0, x1, y1 = c.bbox
        bbox = [_r(x0), _r(H - y1), _r(x1), _r(H - y0)]
        # Name convention (verified against the platform scorer, which compares Name and
        # value for every component): part number for ICs / diodes / LEDs (and other
        # semiconductors) when one is printed, otherwise the designator; net labels use
        # their key (incl. _2 suffix); everything else uses the designator.
        base = getattr(c, 'base_key', c.key)
        if ANON_SPECIAL and c.type in ANON_OUT_TYPES:
            name, value = c.key, None
        elif c.type == 'box':
            name = c.name if c.name else base
            value = None
        elif c.type in ('gnd', 'v'):
            name = c.key
            value = c.value
            if POWER_SPACE_SPLIT and value and ' ' in value.strip():
                # a power name read together with a designator set a space away ('+5V1 H2')
                from .textclass import is_desig as _isd
                from .textclass import is_power_name as _ispw
                toks = value.split()
                if len(toks) >= 2 and _ispw(toks[0]) and all(_isd(t) for t in toks[1:]):
                    value = toks[0]
                elif len(toks) >= 2 and _ispw(toks[-1]) and all(_isd(t) for t in toks[:-1]):
                    value = toks[-1]
        elif c.type.startswith('net'):
            name = c.key
            value = c.value
        elif c.type in MODEL_NAME_TYPES or c.type.startswith(('bjt', 'mosfet', 'amp')):
            name = c.name if c.name else base
            value = c.value
        else:
            name = base
            if VALUE_CLEAN and c.type in ('r', 'c', 'l') and c.value:
                c.value = _clean_value(c.type, c.value)
            # official answers: switches and fuses never carry a value (20 of 20 fuses in the
            # training answers), crystals the printed frequency
            value = None if c.type == 'switch' or (FUSE_NO_VALUE and c.type == 'fuse') else c.value
            if VALUE_PM_NONE and value and c.type in ('r', 'c', 'l') and '±' in value:
                # '1uF ±10% 50V' is the part's description line (a supplier field), not its
                # value: the answers leave the value empty (25 of 25 such parts)
                value = None
        if NAME_CHAR_FIX and name and name != c.key and not c.type.startswith('net') and c.type not in ('gnd', 'v'):
            # OCR slips in printed part numbers: a letter O between digits is a zero
            # ('1O0uF'), a raised dot between letters/digits is a hyphen ('LCH10.5·A')
            name = re.sub(r'(?<=\d)O(?=\d)', '0', name)
            name = re.sub(r'(?<=[A-Za-z0-9])·(?=[A-Za-z0-9])', '-', name)
            name = re.sub(r'(?<=\d):(?=\d)', '', name)      # a speck read as ':' ('5:32610471')
        jc[c.key] = {'Name': name, 'type': OUT_TYPE.get(c.type, c.type) if TYPE_FAMILY_OUT else c.type,
                     'value': value, 'bbox': bbox}
        pp = {}
        for pn in c.pins:
            pp[f'pin_{pn.num}'] = {'pinname': pn.name if pn.name is not None else pn.num,
                                   'point': {'x': _r(pn.tip[0]), 'y': _r(H - pn.tip[1])}}
        jp[c.key] = pp
    return {'components': jc, 'pins': jp, 'nets': nets}


def _process_one(args):
    path, out_dir = args
    case = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()
    try:
        out = parse_image(path)
        err = None
    except Exception as e:  # never lose a case
        import traceback
        err = traceback.format_exc()
        out = {'components': {}, 'pins': {}, 'nets': {}}
    d = os.path.join(out_dir, case)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, case + '.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return case, len(out['components']), len(out['nets']), time.time() - t0, err


def list_images(in_dir):
    cases = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not f.startswith('.'):
                cases.append(os.path.join(root, f))
    cases.sort()
    return cases


def run_dir(in_dir, out_dir, verbose=True, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    cases = list_images(in_dir)
    if workers is None:
        workers = int(os.environ.get('PCB_WORKERS', '0')) or max(1, min(len(cases), (os.cpu_count() or 2) // 2, 16))
    jobs = [(p, out_dir) for p in cases]
    if workers <= 1 or len(jobs) <= 1:
        results = map(_process_one, jobs)
    else:
        import multiprocessing as mp
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(workers)
        results = pool.imap_unordered(_process_one, jobs)
    for case, nc, nn, dt, err in results:
        if verbose:
            print(f'{case}: {nc} comps, {nn} nets, {dt:.1f}s' + (' ERROR' if err else ''), flush=True)
            if err:
                print(err, flush=True)

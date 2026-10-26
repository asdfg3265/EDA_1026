"""OCR post-processing that uses geometry and image-level consistency.

* comma vs period: PP-OCR almost always emits '.', but EasyEDA texts such as '3,3V'
  use a comma, which descends below the baseline.  Decide from the glyph box.
* k vs K in values: when the recogniser is unsure, use the case used by the confident
  values of the same drawing (default upper-case K, the majority in the golden data).
* invalid values ('1nK' from a '10K' crossed by a wire): repair with a small
  confusion map while keeping the glyph count, accept only grammar-valid results.
"""
import os
import re
import numpy as np
from .textclass import VALUE_RE
from .ocr import get_rec

CONFUSE = {'n': '0', '∩': '0', 'Ω': '0', 'O': '0', 'o': '0', 'D': '0', 'U': '0', 'Q': '0',
           'l': '1', 'I': '1', 'i': '1', '|': '1', '!': '1', 'S': '5', 's': '5', 'B': '8',
           'Z': '2', 'z': '2', 'G': '6', 'q': '9', 'g': '9', 'T': '7', 'A': '4', '?': '2'}
R_VALUE_RE = re.compile(r'^\d+(?:[.,]\d+)?[RrKkMm]?\d*(?:Ω|R)?$|^\d+[RrKkMm]\d+$')


def _char_probs(it, ch_a, ch_b):
    """Max probability of two characters over frames where either is the argmax."""
    if it.probs is None:
        return None
    rec = get_rec()
    try:
        ia, ib = rec.chars.index(ch_a), rec.chars.index(ch_b)
    except ValueError:
        return None
    pr = np.asarray(it.probs)
    am = pr.argmax(1)
    sel = (am == ia) | (am == ib)
    if not sel.any():
        return None
    return float(pr[sel, ia].max()), float(pr[sel, ib].max())


def fix_comma(it):
    """Turn '.' into ',' when the glyph between digits descends below the baseline."""
    t = it.text
    if it.vert or '.' not in t or not it.glyphs:
        return t
    gl = sorted(it.glyphs, key=lambda g: g[0])
    if len(gl) != len(t):
        return t
    tall = [g for g, c in zip(gl, t) if c.isalnum()]
    if not tall:
        return t
    base = float(np.median([g[3] for g in tall]))
    hgt = float(np.median([g[3] - g[1] + 1 for g in tall]))
    out = list(t)
    for i, (g, c) in enumerate(zip(gl, t)):
        if c != '.' or i == 0 or i == len(t) - 1:
            continue
        gh, gw = g[3] - g[1] + 1, g[2] - g[0] + 1
        if g[3] >= base + max(1.0, 0.08 * hgt) or gh >= 1.7 * gw:
            out[i] = ','
    return ''.join(out)


def repair_value(text):
    if not text or VALUE_RE.match(text):
        return text
    cand = []
    for i, c in enumerate(text):
        if c in CONFUSE:
            v = text[:i] + CONFUSE[c] + text[i + 1:]
            if VALUE_RE.match(v):
                cand.append(v)
    return cand[0] if len(cand) == 1 else text


def resolve_k_case(items):
    """Return a function mapping a value text to its image-consistent k/K spelling."""
    upper = lower = 0
    amb = set()
    for it in items:
        if it.color != 'blue' or not it.text or not R_VALUE_RE.match(it.text) or not re.search('[kK]', it.text):
            continue
        p = _char_probs(it, 'K', 'k')
        if p is None:
            continue
        if min(p) > 0.08:
            amb.add(it.id)
        elif p[0] > p[1]:
            upper += 1
        else:
            lower += 1
    want = 'k' if lower > upper else 'K'

    def fix(it, text):
        if it is not None and it.id in amb and re.search('[kK]', text):
            return re.sub('[kK]', want, text)
        return text
    return fix


DIGIT_O = os.environ.get('PCB_DIGIT_O', '1') == '1'


NAME_ZERO = os.environ.get('PCB_NAME_ZERO', '1') == '1'


def fix_zero_o(it, thr=0.8):
    """'O' vs '0' from the glyph shape: in Arial-like fonts the digit is narrow
    (width ~0.65-0.7 of the cap height), the letter is round (~0.9-1.0).
    Only applied when every character has its own glyph box."""
    t = it.text or ''
    if DIGIT_O and 'O' in t:
        # a letter O between two digits is always a zero (NCE3O80K, 1O0nF)
        t = re.sub(r'(?<=\d)O(?=\d)', '0', t)
        if NAME_ZERO:
            # voltage spelt 5V0 / 1V0, and the unit index of a bus name: UART0, SPI0, I2C0,
            # TX0P / RX0N (USB3 lanes) - the letter O does not occur there
            t = re.sub(r'(?<=\d)VO(?![A-Za-z])', 'V0', t)
            t = re.sub(r'(UART|USART|SPI|I2C|IIC|TIM|ADC|DAC|PWM|CAN|SDIO)O(?![A-Za-z])', r'\g<1>0', t)
            t = re.sub(r'(?<=[TR]X)O(?=[PN]?(?![A-Za-z]))', '0', t)
    if not any(c in t for c in 'O01I') or not it.glyphs:
        return t
    core = t.replace(' ', '')
    if len(it.glyphs) != len(core):
        return t
    if it.vert:
        gl = sorted(it.glyphs, key=lambda g: -g[1] if it.rot == 'cw' else g[1])
        W = lambda g: g[3] - g[1] + 1
        Hh = lambda g: g[2] - g[0] + 1
    else:
        gl = sorted(it.glyphs, key=lambda g: g[0])
        W = lambda g: g[2] - g[0] + 1
        Hh = lambda g: g[3] - g[1] + 1
    hs = [Hh(g) for g, c in zip(gl, core) if c.isdigit() or (c.isupper() and c not in 'QJ')]
    if len(hs) < 2:
        return t
    H = float(np.median(hs))
    if H < 6:
        return t
    out = list(core)
    for i, (g, c) in enumerate(zip(gl, core)):
        if c not in 'O01I' or abs(Hh(g) - H) > 0.2 * H:
            continue
        r = W(g) / H
        if c == 'O' and r < thr - 0.06:
            out[i] = '0'
        elif c == '0' and r > thr + 0.06:
            out[i] = 'O'
        elif c == 'I' and r > 0.38:
            out[i] = '1'
    # '1O' with a bare-bar '1' is the letter pair IO (IO1, GPIO2) - only in that context:
    # in small fonts a real '1' can also be a bare bar (R1, LED1)
    for i, (g, c) in enumerate(zip(gl, core)):
        if (c == '1' and i + 1 < len(out) and out[i + 1] == 'O' and (i == 0 or not out[i - 1].isdigit())
                and W(g) / H < 0.3 and abs(Hh(g) - H) <= 0.2 * H):
            out[i] = 'I'
    new = ''.join(out)
    if new == core:
        return t
    # put the spaces back
    res, j = [], 0
    for c in t:
        if c == ' ':
            res.append(c)
        else:
            res.append(new[j]); j += 1
    return ''.join(res)


# Arial/Liberation Sans advance widths (em) for locating a dropped space inside a text
_AW = {c: 0.556 for c in '0123456789'}
_AW.update({c: 0.667 for c in 'ABEKPSVXY'}); _AW.update({c: 0.722 for c in 'CDHNRUw'})
_AW.update({c: 0.778 for c in 'GOQ'}); _AW.update({'F': 0.611, 'T': 0.611, 'Z': 0.611, 'L': 0.556, 'J': 0.5,
            'I': 0.278, 'M': 0.833, 'W': 0.944, 'm': 0.833})
_AW.update({c: 0.556 for c in 'abdeghnopqu'}); _AW.update({c: 0.5 for c in 'cksvxyz'})
_AW.update({c: 0.222 for c in 'ijl'}); _AW.update({'f': 0.278, 't': 0.278, 'r': 0.333, ' ': 0.278})
_AW.update({c: 0.278 for c in '.,:;!|/'}); _AW.update({c: 0.333 for c in '()-[]'}); _AW.update({'_': 0.556, '+': 0.584})


_NOSPACE = set('1Il|.,:;/\\-_()[]\'"+')


def fix_spaces(it, thr=0.42):
    """PP-OCR drops spaces ('5V2A in' -> '5V2Ain').  Where the glyph boxes leave an ink-free
    gap wider than a space (~0.4 cap height), insert a space at the character boundary
    that Arial advance widths put closest to that gap."""
    t = it.text or ''
    if it.vert or ' ' in t or len(t) < 3 or not it.glyphs:
        return t
    gl = sorted(it.glyphs, key=lambda g: g[0])
    H = float(np.median([g[3] - g[1] + 1 for g in gl]))
    if H < 6:
        return t
    # union of glyph column extents -> ink-free gaps
    gaps = []
    right = gl[0][2]
    for g in gl[1:]:
        if g[0] - right - 1 > thr * H:
            gaps.append((right + 1 + g[0] - 1) / 2.0)
        right = max(right, g[2])
    if not gaps or len(gaps) > 2:
        return t
    x0 = gl[0][0]
    x1 = max(g[2] for g in gl)
    widths = [_AW.get(c, 0.6) for c in t]
    total = sum(widths) + 0.278 * len(gaps)
    scale = (x1 - x0 + 1) / total
    out = t
    for gx in sorted(gaps, reverse=True):
        # boundary positions (before inserting further spaces to the left)
        best = None
        acc = 0.0
        for i in range(1, len(t)):
            acc += widths[i - 1]
            bx = x0 + (acc + 0.139) * scale
            d = abs(bx - gx)
            if best is None or d < best[0]:
                best = (d, i)
        if best and best[0] < 0.6 * H:
            i = best[1]
            # a bare-bar '1' or punctuation leaves wide gaps by itself; keep words of >= 2 chars
            if t[i - 1] in _NOSPACE or t[i] in _NOSPACE or i < 2 or len(t) - i < 2:
                continue
            out = out[:i] + ' ' + out[i:]
    return out


def fix_trailing_ohm(it, t):
    """'330Ω' read as '3302' / '0Ω' as '02': the last glyph of a resistor value is the wide
    omega (width about the cap height) when the OCR says a narrow '2' or '0'."""
    m = re.match(r'^(.*\dΩ)\d{1,2}$', t or '') or re.match(r'^(.*\d[kKmMG]Ω)\d{1,2}$', t or '')
    if m:
        return m.group(1)                # '200Ω2': the omega's right foot read as a digit
    m = re.match(r'^(\d{2,}(?:\.\d+)?[kKmMG])9$', t or '') or re.match(r'^(\d+(?:\.\d+)?[kKmMG]?)g$', t or '')
    if m:
        return m.group(1) + 'Ω'          # '10k9' / '10kg': omega read as 9 or g (4K7-style values never end in 9 here)
    if not t or t[-1] not in '20OQo9' or len(t) < 2 or not (t[-2].isdigit() or t[-2] in 'kKmMR') or not it.glyphs:
        return t
    if it.vert:
        # vertical text reads bottom-to-top ('cw' crop) or top-to-bottom: the last glyph is at the far end
        gl = sorted(it.glyphs, key=lambda g: -g[1] if it.rot == 'cw' else g[1])
        along = lambda g: g[3] - g[1] + 1
        across = lambda g: g[2] - g[0] + 1
    else:
        gl = sorted(it.glyphs, key=lambda g: g[0])
        along = lambda g: g[2] - g[0] + 1
        across = lambda g: g[3] - g[1] + 1
    if len(gl) == len(t):
        last = gl[-1]
        H = float(np.median([across(g) for g in gl]))
        r = along(last) / H if H else 0
        # omega is ~1.0-1.1 cap heights wide; two touching digits ('00', '22') merge into ~1.4+
        if H >= 6 and 0.85 < r < 1.25 and abs(across(last) - H) <= 0.2 * H:
            return t[:-1] + 'Ω'
    # touching glyphs / unclear width: the recogniser's own second choice for the last glyph
    if OHM_PROB and t[-1] in '20Q' and _last_char_prob(it, t, ('Ω',)) >= 0.02:
        return t[:-1] + 'Ω'
    return t


OHM_PROB = os.environ.get('PCB_OHM_PROB', '1') == '1'
CAP_U_PROB = os.environ.get('PCB_CAP_U_PROB', '1') == '1'


def fix_trailing_u(it, t):
    """'47U' / '0.1U' read as '470' / '0.10': a capacitor value printed with a bare U whose
    last glyph the recogniser took for a zero, giving the U a fair second chance."""
    if not CAP_U_PROB or not re.fullmatch(r'\d+(?:\.\d+)?0', t or ''):
        return t
    pu = _last_char_prob(it, t, ('U',))
    pl = _last_char_prob(it, t, ('u',))
    if max(pu, pl) >= 0.05:
        return t[:-1] + ('U' if pu >= pl else 'u')
    return t


def _last_char_prob(it, t, chars):
    """Probability the recogniser gave any of chars at the frame of t's last character."""
    probs = getattr(it, 'probs', None)
    if probs is None or it.text.replace(' ', '') != t.replace(' ', ''):
        return 0.0
    from .ocr import get_rec
    R = get_rec()
    txt, _, _, pos = R.decode(probs)
    if txt.replace(' ', '') != t.replace(' ', '') or not pos:
        return 0.0
    ids = [i for i, ch in enumerate(R.chars) if ch in chars]
    if not ids:
        return 0.0
    fr = probs[pos[-1]]
    return float(max(fr[i] for i in ids))


def _char_glyphs(it):
    """Glyph boxes merged per character (the dot of i/j, the two parts of ':' ...), in
    reading order, with the number of components merged into each; plus accessors."""
    if it.vert:
        gl = sorted(it.glyphs, key=lambda g: -g[1] if it.rot == 'cw' else g[1])
        A0, A1 = (lambda g: g[1]), (lambda g: g[3])
        Hh = lambda g: g[2] - g[0] + 1
    else:
        gl = sorted(it.glyphs, key=lambda g: g[0])
        A0, A1 = (lambda g: g[0]), (lambda g: g[2])
        Hh = lambda g: g[3] - g[1] + 1
    out = []
    for g in gl:
        g = list(g)
        if out:
            p = out[-1][0]
            ov = min(A1(p), A1(g)) - max(A0(p), A0(g)) + 1
            wmin = min(A1(p) - A0(p), A1(g) - A0(g)) + 1
            if ov >= 0.6 * wmin:
                out[-1][0] = [min(p[0], g[0]), min(p[1], g[1]), max(p[2], g[2]), max(p[3], g[3]), p[4] + g[4]]
                out[-1][1] += 1
                continue
        out.append([g, 1])
    return out, Hh


X_HEIGHT_LETTERS = 'acemnorsuvwxz'      # lower-case forms without ascender / descender
SAME_SHAPE = 'cosvwxz'                  # upper and lower case differ only in size
CAP_REF = set('ABDEFGHKLMNPRTUY0123456789') | set('bdhk')


def cap_height(it):
    """Median glyph height of the characters that are always cap-height (None if none)."""
    t = it.text or ''
    if not it.glyphs or not t:
        return None
    core = t.replace(' ', '')
    cg, Hh = _char_glyphs(it)
    if len(cg) != len(core):
        return None
    ref = [Hh(g) for (g, k), c in zip(cg, core) if c in CAP_REF]
    return float(np.median(ref)) if ref else None


def fix_case(it, H_ext=None):
    """Upper / lower case from glyph height: lower-case x-height letters are ~0.72 of the
    cap height in Arial.  'sO' -> 'SO', 'cO' -> 'CO', 'VIN' read for 'Vin' -> 'Vin'.
    Only when every character has its own glyph.  The cap height comes from the item
    itself or, failing that, from nearby texts of the same size (H_ext)."""
    t = it.text or ''
    if not it.glyphs or not any(c.lower() in X_HEIGHT_LETTERS for c in t):
        return t
    core = t.replace(' ', '')
    cg, Hh = _char_glyphs(it)
    if len(cg) != len(core):
        return t
    ref = [Hh(g) for (g, k), c in zip(cg, core) if c in CAP_REF]
    if ref:
        H = float(np.median(ref))
    elif H_ext:
        H = float(H_ext)
        # the item must be of that font size: its tallest glyph is cap height or x-height
        top = max(Hh(g) for g, k in cg)
        if not (0.66 * H <= top <= 1.12 * H):
            return t
    else:
        return t
    if H < 8:
        return t
    out = list(core)
    for i, ((g, k), c) in enumerate(zip(cg, core)):
        lo = c.lower()
        if lo not in X_HEIGHT_LETTERS or k != 1:
            continue
        r = Hh(g) / H
        if c.isupper() and r <= 0.82:
            out[i] = lo
        elif c.islower() and lo in SAME_SHAPE and r >= 0.92:
            out[i] = c.upper()
    new = ''.join(out)
    if new == core:
        return t
    res, j = [], 0
    for ch in t:
        if ch == ' ':
            res.append(' ')
        else:
            res.append(new[j]); j += 1
    return ''.join(res)


def fix_case_all(P, items):
    """fix_case for every item; items without a cap-height character borrow the cap
    height of the nearest texts of the same colour and orientation."""
    refs = []
    for it in items:
        if it.text:
            h = cap_height(it)
            if h:
                refs.append((it.cx, it.cy, it.vert, it.color, h))
    u = P.u
    for it in items:
        if not it.text:
            continue
        H_ext = None
        if cap_height(it) is None and any(c.lower() in SAME_SHAPE for c in it.text):
            near = sorted((abs(x - it.cx) + abs(y - it.cy), h) for x, y, v, col, h in refs
                          if v == it.vert and col == it.color and abs(x - it.cx) + abs(y - it.cy) <= 8 * u)
            if len(near) >= 2:
                H_ext = float(np.median([h for d, h in near[:5]]))
        it.text = fix_case(it, H_ext)


def fix_missing_dot(it):
    """The recogniser sometimes drops a small '.' between digits ('U8.1' -> 'U81',
    '4.7k' -> '47k').  Re-insert it when there is exactly one extra, dot-sized glyph on the
    baseline between two digits."""
    t = it.text or ''
    if it.vert or not it.glyphs or '.' in t or ',' in t or ' ' in t or len(t) < 2:
        return t
    cg, Hh = _char_glyphs(it)
    if len(cg) != len(t) + 1:
        return t
    tall = [g for g, k in cg if Hh(g) >= 0.5 * max(Hh(x) for x, _ in cg)]
    H = float(np.median([Hh(g) for g in tall])) if tall else 0
    if H < 7:
        return t
    base = float(np.median([g[3] for g in tall]))
    dots = [i for i, (g, k) in enumerate(cg)
            if k == 1 and (g[3] - g[1] + 1) <= 0.32 * H and (g[2] - g[0] + 1) <= 0.32 * H and abs(g[3] - base) <= max(1.5, 0.15 * H)]
    if len(dots) != 1:
        return t
    i = dots[0]
    if i == 0 or i >= len(t) or not (t[i - 1].isdigit() and t[i].isdigit()):
        return t
    return t[:i] + '.' + t[i:]


CONTEXT_IO = os.environ.get('PCB_CONTEXT_IO', '1') == '1'


def context_fix(items):
    """Spellings settled by the other labels of the same sheet: a row of port labels
    IO3, IO9, IO10 ... makes a lone '1011' the label IO11 (the recogniser reads I/O as 1/0
    in small vertical text).  GPIO is never spelt GP1O."""
    if not CONTEXT_IO:
        return items
    blue = [it for it in items if it.color == 'blue' and it.text]
    ios = [it for it in blue if re.fullmatch(r'IO\d{1,2}', it.text)]

    def in_row(it):
        # one of a row / column of IO port labels: same orientation, flush with a
        # neighbouring IO label (a capacitor code '104' stands alone)
        for o in ios:
            if o is it or o.vert != it.vert:
                continue
            if it.vert:
                if (abs(o.y0 - it.y0) <= 3 or abs(o.y1 - it.y1) <= 3) and abs(o.cx - it.cx) <= 3 * max(it.w, o.w):
                    return True
            elif (abs(o.x0 - it.x0) <= 3 or abs(o.x1 - it.x1) <= 3) and abs(o.cy - it.cy) <= 3 * max(it.h, o.h):
                return True
        return False
    for it in blue:
        t = it.text
        t2 = re.sub(r'GP[1l|]O(?=\d)', 'GPIO', t)
        # a supply name in capitals with its first V read in lower case (vCC, vIO)
        if re.fullmatch(r'v[A-Z][A-Z0-9_]*', t2):
            t2 = 'V' + t2[1:]
        if re.fullmatch(r'[1lI|][0O]\d{1,2}', t2):
            if (len(ios) >= 3 and in_row(it)) or _io_shapes(it, t2):
                t2 = 'IO' + t2[2:]
            else:
                # '104' is a capacitor code or the label IO4 drawn alike: the parts decide
                it.io_alt = 'IO' + t2[2:]
        if t2 != t:
            it.text = t2
    return items


def _io_shapes(it, t):
    """'IO4' rather than '104' by the glyph shapes: a bare bar (I, narrower than a '1' with
    its flag) followed by a round letter O (wider than the digit 0)."""
    gl = list(it.glyphs or [])
    if len(gl) != len(t):
        return False
    if it.vert:
        gl.sort(key=lambda g: -g[1] if it.rot != 'cw' else g[1])
        W = lambda g: g[3] - g[1] + 1
        Hh = lambda g: g[2] - g[0] + 1
    else:
        gl.sort(key=lambda g: g[0])
        W = lambda g: g[2] - g[0] + 1
        Hh = lambda g: g[3] - g[1] + 1
    H = float(np.median([Hh(g) for g in gl]))
    if H < 7:
        return False
    return W(gl[0]) / H <= 0.25 and W(gl[1]) / H >= 0.85

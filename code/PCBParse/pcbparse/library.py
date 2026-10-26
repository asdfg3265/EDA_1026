"""IC library (BoxICLibrary.json / cpntLibrary) for name & pin correction."""
import json
import os
import re
from collections import defaultdict


def lev(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


CONFUSE = str.maketrans({'O': '0', 'o': '0', 'I': '1', 'l': '1', '|': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6', 'D': '0', 'Q': '0'})


def canon(s):
    return re.sub(r'[\s_\-]', '', (s or '').upper()).translate(CONFUSE)


SNAP_CONF = 0.97
# OCR-confusable character pairs: the only substitutions the library may correct.  A
# digit-for-digit or letter-for-letter difference (PZ254V-11-02P vs -04P, TPS5450 vs
# TPS5430, INA240A2 vs A1) is a different part, never an OCR error.
CONF_PAIRS = {frozenset(p) for p in [('O', '0'), ('o', '0'), ('D', '0'), ('Q', '0'), ('O', 'D'), ('O', 'Q'),
                                      ('I', '1'), ('l', '1'), ('L', '1'), ('I', 'l'), ('S', '5'), ('s', '5'),
                                      ('B', '8'), ('Z', '2'), ('z', '2'), ('G', '6'),
                                      # letters whose upper and lower case differ only in size
                                      ('c', 'C'), ('o', 'O'), ('s', 'S'), ('v', 'V'), ('w', 'W'), ('x', 'X'), ('z', 'Z')]}
STRICT = os.environ.get('PCB_LIB_STRICT', '1') == '1'


def _space_pos(s):
    out, k = set(), 0
    for ch in s:
        if ch == ' ':
            out.add(k)
        else:
            k += 1
    return out


def confusable_only(a, b):
    """a (OCR) can be a misread of b (library): same characters up to OCR-confusable pairs.
    Spaces: the recogniser drops spaces but does not invent them, so a space in a must also
    be in b, while b may have spaces that a lacks (AF90°WJDG -> AF 90° WJDG)."""
    if a is None or b is None:
        return False
    a2, b2 = a.replace(' ', ''), b.replace(' ', '')
    if len(a2) != len(b2) or not _space_pos(a) <= _space_pos(b):
        return False
    for x, y in zip(a2, b2):
        if x != y and frozenset((x, y)) not in CONF_PAIRS:
            return False
    return True


LAST_PIN = os.environ.get('PCB_LIB_LAST_PIN', '1') == '1'


class Library:
    def __init__(self, paths):
        self.parts = {}
        for p in paths:
            if not p or not os.path.exists(p):
                continue
            try:
                d = json.load(open(p, encoding='utf-8'))
            except Exception:
                continue
            for name, rec in d.items():
                pins = rec.get('pins', []) if isinstance(rec, dict) else []
                m = self.parts.setdefault(name, defaultdict(list))
                for pn in pins:
                    num = str(pn.get('number', '')).strip()
                    nm = str(pn.get('name', '')).strip()
                    if num and nm and nm not in m[num]:
                        m[num].append(nm)
        self.canon_index = defaultdict(list)
        for name in self.parts:
            self.canon_index[canon(name)].append(name)
        self.names = list(self.parts)

    def match_name(self, text, max_rel=0.2):  # noqa: C901
        if not text:
            return None
        if text in self.parts:
            return text
        ct = canon(text)
        if ct in self.canon_index:
            return self.canon_index[ct][0]
        best = None
        for name in self.names:
            cn = canon(name)
            if abs(len(cn) - len(ct)) > max(2, int(max_rel * len(ct)) + 1):
                continue
            d = lev(ct, cn)
            rel = d / max(len(cn), 1)
            if rel <= max_rel and (best is None or rel < best[0]):
                best = (rel, name)
        return best[1] if best else None

    def npins(self, name):
        return len(self.parts.get(name, ()))

    def match_strict(self, text):
        if not text:
            return None
        if text in self.parts:
            return text
        cands = [n for n in self.canon_index.get(canon(text), []) if confusable_only(text, n)]
        return cands[0] if len(cands) == 1 else None

    def correct(self, comp):
        if not comp.name or len(comp.name) < 4:
            return
        pins_only = False
        if STRICT:
            name = self.match_strict(comp.name)
            if name is None:
                # a variant of a library part (CP2102-GMR vs CP2102-GM): keep the printed name but
                # borrow the pin map - only when the pin count says it is the same package
                # (STM32F103RBT6 has 64 pins, the C8T6 in the library 48)
                near = self.match_name(comp.name)
                if near is not None and self.parts[near]:
                    nl = len(self.parts[near])
                    if abs(nl - len(comp.pins)) <= max(1, int(0.05 * nl)) and nl >= 3:
                        name, pins_only = near, True
        else:
            name = self.match_name(comp.name) if comp.name else None
        if name is None:
            return
        # a confidently read part number that differs by 2+ characters is a different part
        # (STM32F103RBT6 is not STM32F103C8T6): do not snap it to the library
        if (not pins_only and name != comp.name and getattr(comp, 'name_conf', 0.0) >= SNAP_CONF
                and lev(canon(comp.name), canon(name)) > 1):
            return
        if len(canon(comp.name)) >= 4 and not pins_only:
            comp.name = name
        pm = self.parts[name]
        if not pm:
            return
        numbers = set(pm)
        # pin numbers: fix via names when the number is missing or unknown
        name_to_nums = defaultdict(set)
        for num, nms in pm.items():
            for nm in nms:
                name_to_nums[nm].add(num)
        used = set(pn.num for pn in comp.pins if pn.num in numbers)
        for pn in comp.pins:
            if pn.num in numbers:
                continue
            if pn.name and pn.name in name_to_nums:
                # (library numbers from noisy answers - '3_1', 'R', 'C147 0.047uF' - are never used)
                cands = [n for n in name_to_nums[pn.name] if n not in used and re.fullmatch(r'(?=.*\d)[A-Z0-9]{1,6}', str(n))]
                if len(cands) == 1:
                    pn.num = cands[0]
                    used.add(pn.num)
        # one pin left with neither a number nor a readable name (hidden 'SHLD') and one
        # library pin left: that is the one
        left = [pn for pn in comp.pins if pn.num not in numbers]
        free_nums = [n for n in pm if n not in used]
        if LAST_PIN and len(left) == 1 and len(free_nums) == 1 and left[0].name is None and len(comp.pins) == len(pm) \
                and re.fullmatch(r'(?=.*\d)[A-Z0-9]{1,6}', str(free_nums[0])):
            left[0].num = free_nums[0]
            used.add(free_nums[0])
        for pn in comp.pins:
            if pn.num in pm:
                names = pm[pn.num]
                if pn.name in names:
                    continue
                if pn.name is None:
                    pn.name = names[0]
                    continue
                # choose closest library name if OCR looks like a misread
                best = min(names, key=lambda nm: lev(canon(nm), canon(pn.name)))
                d = lev(canon(best), canon(pn.name))
                if STRICT:
                    # the part is identified exactly: fix OCR confusions and one dropped
                    # character ('MOS' -> 'MOSI'), never a substitution (PA1 vs PA5)
                    if confusable_only(pn.name, best) or (
                            len(best) >= 3 and len(best) == len(pn.name) + 1 and pn.name in best):
                        pn.name = best
                elif d <= max(1, len(best) // 3):
                    pn.name = best

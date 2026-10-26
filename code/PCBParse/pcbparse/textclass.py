"""Text categorisation helpers."""
import re

DESIG_RE = re.compile(r'^(?:[A-Z]{1,4}[0-9]{1,3}[A-Z]?|[A-Z]{1,4}[0-9]{1,3}_[0-9A-Z]{1,3}|(?:U|IC)[0-9]{1,3}\.[0-9]{1,2})$')
POWER_GND_RE = re.compile(r'^[A-Z_0-9]*GND[A-Z_0-9]*$')
POWER_V_RE = re.compile(r'^(?:[+\-]?\d+(?:[.Vv]\d+)?V\d*[A-Z]*|[+\-]?\d+V\d*|V[A-Z0-9_.+\-]*|[A-Z0-9_]*VCC[A-Z0-9_.]*|[A-Z0-9_]*VDD[A-Z0-9_.]*|[A-Z0-9_]*VBUS[A-Z0-9_]*|[A-Z0-9_]*VBAT[A-Z0-9_]*|[+\-]\d+(?:\.\d+)?V?)$')
VALUE_RE = re.compile(
    r'^(?:\d+(?:[.,]\d+)?\s?(?:[pnuμµmkKMGRUNP]|[pnuμµmUNP]?[FfHh]|[kKMmR]?Ω|[kKM]?ohm|[mM]?[Aa]|[mkM]?[Hh]z|V)?'
    r'(?:\d+)?(?:[FfHhΩ])?(?:/\d+(?:\.\d+)?[VWA]?)?(?:\s?\d+%)?|\d+[pnuμµmkKMR]\d*[FfHhΩ]?|\d+-\d+[pnuμµm]?[FfHh]?)$')
DESIG_PREFIX_TYPE = [
    ('DS', 'box'), ('LCD', 'box'), ('OLED', 'box'), ('RLY', 'box'), ('BUZZER', 'buzzer'), ('BUZ', 'buzzer'),
    ('KEY', 'switch'), ('BTN', 'switch'), ('PWR', 'box'), ('FAN', 'box'), ('MOTOR', 'box'), ('MOT', 'box'),
    ('FPC', 'box'), ('FFC', 'box'), ('DC', 'box'), ('LDO', 'box'), ('RELAY', 'box'), ('NTC', 'r'),
    ('SCREW', 'other'), ('HOLE', 'other'), ('BQ', 'box'), ('EC', 'c'), ('CARD', 'box'), ('SWD', 'box'),
    ('SERVO', 'box'), ('SMA', 'box'), ('SPK', 'buzzer'), ('TF', 'box'), ('SD', 'box'), ('SIM', 'box'),
    ('LED', 'led'), ('USB', 'box'), ('SW', 'switch'), ('TP', 'pin'), ('CN', 'box'), ('JP', 'box'),
    ('IC', 'box'), ('BZ', 'buzzer'), ('BT', 'battary'), ('BAT', 'battary'), ('FB', 'l'), ('ANT', 'ant'),
    ('RV', 'r'), ('RP', 'r'), ('RN', 'r'), ('RT', 'r'), ('ESD', 'esd'), ('MK', 'other'), ('MIC', 'other'),
    ('R', 'r'), ('C', 'c'), ('L', 'l'), ('D', 'd'), ('U', 'box'), ('Q', 'mosfet'), ('Y', 'crystal_2pin'),
    ('X', 'crystal_2pin'), ('F', 'fuse'), ('S', 'switch'), ('K', 'switch'), ('J', 'box'), ('P', 'box'),
    ('H', 'box'), ('T', 'transfomer'), ('M', 'motor'), ('B', 'buzzer'), ('A', 'box'), ('E', 'other'),
]


def desig_prefix(s):
    m = re.match(r'^([A-Z]+)', s or '')
    return m.group(1) if m else ''


def prefix_type(s):
    p = desig_prefix(s)
    for pre, ty in DESIG_PREFIX_TYPE:
        if p == pre:
            return ty
    for pre, ty in DESIG_PREFIX_TYPE:
        if p.startswith(pre):
            return ty
    return None


def is_desig(s):
    if not s or len(s) > 9:
        return False
    if not DESIG_RE.match(s):
        return False
    if POWER_GND_RE.match(s):
        return False
    if re.match(r'^V(?:CC|DD|IN|OUT|BAT|BUS)', s):
        return False
    if re.match(r'^\d', s):
        return False
    return True


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_gnd_name(s):
    if not s:
        return False
    u = s.upper()
    if POWER_GND_RE.match(u):
        return True
    if u in ('EARTH', 'PE', 'CHASSIS'):         # protective-earth symbols
        return True
    return len(u) == 3 and _lev(u, 'GND') <= 1 and u[0] in 'GC6' and u[-1] in 'DO0'


def is_power_name(s):
    if not s:
        return False
    s2 = s.strip()
    if is_gnd_name(s2):
        return True
    return bool(POWER_V_RE.match(s2))


def is_value(s):
    if not s:
        return False
    return bool(VALUE_RE.match(s.strip()))


def is_number(s):
    return bool(s) and bool(re.match(r'^[A-Z]{0,2}\d{1,3}$', s))

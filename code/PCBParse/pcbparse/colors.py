"""Colour segmentation for EasyEDA / JLC style schematics.

Every pixel p is modelled as an alpha blend between the paper background and one ink
colour.  The deficit vector d = bg - p points in the direction of (bg - ink) so the
ink class is identified by the *direction* of d while its magnitude gives coverage.
"""
import os
import numpy as np
import cv2

ORANGE_DROP = os.environ.get('PCB_ORANGE_DROP', '1') == '1'

# ink prototypes in RGB
INKS = {
    'red':   (150, 15, 15),     # symbol strokes / pins (dark red 0x880000 .. 0xA00000)
    'bred':  (204, 0, 0),       # junction dots (bright red)
    'green': (0, 136, 0),       # wires
    'blue':  (40, 40, 250),     # bright blue text
    'navy':  (40, 40, 120),     # dark blue text
    'black': (0, 0, 0),         # pin names / numbers / misc text
}


def estimate_bg(img):
    small = img[::4, ::4].reshape(-1, 3).astype(np.int32)
    q = (small // 4)
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, cnt = np.unique(keys, return_counts=True)
    k = vals[np.argmax(cnt)]
    sel = small[keys == k]
    return sel.mean(0)


def classify(img_rgb):
    """Return dict of float coverage maps (0..1) and a hard label map.

    labels: 0 bg, 1 red, 2 green, 3 blue(text), 4 black(text), 5 gray, 6 bright-red
    """
    img = img_rgb.astype(np.float32)
    bg = estimate_bg(img_rgb).astype(np.float32)
    d = np.clip(bg[None, None, :] - img, 0, None)            # deficit
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    dr, dg, db = d[..., 0], d[..., 1], d[..., 2]
    mag = dr + dg + db
    eps = 1e-3
    # chromaticity of deficit
    fr, fg, fb = dr / (mag + eps), dg / (mag + eps), db / (mag + eps)
    lab = np.zeros(R.shape, np.uint8)
    fore = mag > 60
    # ratio tests on the deficit vector
    green = fore & (dg < 0.8 * np.minimum(dr, db))
    red = fore & ~green & (dr < 0.75 * np.minimum(dg, db))
    blue = fore & ~green & ~red & (db < 0.8 * np.minimum(dr, dg))
    navy = np.zeros_like(blue)
    neutral = fore & ~green & ~red & ~blue & ~navy
    black = neutral & (mag > 330)
    gray = neutral & ~black
    lab[gray] = 5
    lab[black] = 4
    lab[blue | navy] = 3
    lab[green] = 2
    lab[red] = 1
    if ORANGE_DROP:
        # orange ink (annotation text and function-box outlines, 255,153,0): red in the
        # ratio test, but the green deficit is far below the blue one (dark red ink and
        # its anti-aliasing keep them equal)
        lab[red & (dg < 0.6 * db) & (dr < 0.3 * db)] = 0
        red = red & ~((dg < 0.6 * db) & (dr < 0.3 * db))
    # bright red (junction): R high
    bred = red & (R > 175) & (mag > 350)
    lab[bred] = 6
    return lab, bg, mag


def masks(img_rgb):
    lab, bg, mag = classify(img_rgb)
    return {
        'red': ((lab == 1) | (lab == 6)).astype(np.uint8),
        'bred': (lab == 6).astype(np.uint8),
        'green': (lab == 2).astype(np.uint8),
        'blue': (lab == 3).astype(np.uint8),
        'black': (lab == 4).astype(np.uint8),
        'gray': (lab == 5).astype(np.uint8),
        'lab': lab, 'bg': bg, 'mag': mag,
    }


def label_vis(lab):
    pal = np.array([[255, 255, 255], [150, 0, 0], [0, 160, 0], [0, 0, 255], [0, 0, 0], [170, 170, 170], [255, 0, 255]], np.uint8)
    return pal[lab]

#!/usr/bin/env python3
"""Pre-annotate new schematic images with the parser (for the data-contribution bonus).

Usage:
    python tools/preannotate.py raw_images/ dataset/ --prefix case_ --start 1

For every image it writes dataset/<case_id>.png (copy), dataset/<case_id>.json
(competition format, integer pixel coordinates, bottom-left origin) and
review/<case_id>_overlay.png showing components (blue), pins (red) and nets
(one colour per net) so a human can check and correct the annotation with
tools/annotator.html before submission.
"""
import os
import sys
import json
import shutil
import colorsys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pcbparse.pipeline import parse_image, list_images, load_image  # noqa: E402


def overlay(img_rgb, J):
    H = img_rgb.shape[0]
    v = (img_rgb[:, :, ::-1] * 0.45 + 255 * 0.55).astype(np.uint8)
    for i, (nk, n) in enumerate(J['nets'].items()):
        h = (i * 0.137) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.85)
        col = (int(b * 255), int(g * 255), int(r * 255))
        for e in n['edges'].values():
            a = (int(e[0]['x']), int(H - e[0]['y']))
            bb = (int(e[1]['x']), int(H - e[1]['y']))
            cv2.line(v, a, bb, col, 2)
    for k, c in J['components'].items():
        x0, y0, x1, y1 = c['bbox']
        cv2.rectangle(v, (int(x0), int(H - y1)), (int(x1), int(H - y0)), (200, 60, 0), 1)
        cv2.putText(v, f"{k}:{c['type']}", (int(x0), int(H - y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 60, 0), 1)
    for k, ps in J['pins'].items():
        for pk, p in ps.items():
            q = (int(p['point']['x']), int(H - p['point']['y']))
            cv2.circle(v, q, 3, (0, 0, 230), -1)
            cv2.putText(v, pk[4:], (q[0] + 3, q[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 230), 1)
    return v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--prefix', default='case_')
    ap.add_argument('--start', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    rev = os.path.join(os.path.dirname(os.path.abspath(a.dst)), 'review')
    os.makedirs(rev, exist_ok=True)
    k = a.start
    for path in list_images(a.src):
        cid = f'{a.prefix}{k:04d}'
        k += 1
        J = parse_image(path)
        img = load_image(path)
        cv2.imwrite(os.path.join(a.dst, cid + '.png'), img[:, :, ::-1])
        with open(os.path.join(a.dst, cid + '.json'), 'w', encoding='utf-8') as f:
            json.dump(J, f, ensure_ascii=False, indent=1)
        cv2.imwrite(os.path.join(rev, cid + '_overlay.png'), overlay(img, J))
        print(cid, '<-', path, len(J['components']), 'components', len(J['nets']), 'nets')


if __name__ == '__main__':
    main()

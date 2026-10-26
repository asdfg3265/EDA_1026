"""Make a competition-style case (PNG + GT JSON) from an EasyEDA Pro sheet."""
import json
import random
from .render import render_sheet, to_svg_document, SVG
from .gt import build_gt, _prims_bbox


def content_box(sheet, S):
    xs, ys = [], []
    for w in sheet.wires:
        for (x1, y1, x2, y2) in w.segs:
            xs += [x1, x2]
            ys += [y1, y2]
    for c in sheet.components:
        if c.symbol is None or c.symbol.stype == 20 or str(c.symbol.title).lower().startswith('sheet-symbol'):
            continue
        part = c.symbol.part(c.part_name)
        if part is None or not part.pins:
            continue
        bb = _prims_bbox(c, part)
        if bb:
            xs += [bb[0], bb[2]]
            ys += [bb[1], bb[3]]
        for pn in part.pins:
            x, y = c.xf(pn.x, pn.y)
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    core = (min(xs), min(ys), max(xs), max(ys))
    span = max(core[2] - core[0], core[3] - core[1])
    for t in S.texts:
        if not t.get('id') or t.get('sheet'):
            continue            # free sheet texts (title block, frame letters) do not extend the crop
        L = len(t['text']) * t['size'] * 0.6
        if t['x'] < core[0] - 0.2 * span or t['x'] > core[2] + 0.2 * span or t['y'] < core[1] - 0.2 * span or t['y'] > core[3] + 0.2 * span:
            continue
        xs += [t['x'] - L, t['x'] + L]
        ys += [t['y'] - t['size'] * 1.5, t['y'] + t['size'] * 1.5]
    return min(xs), min(ys), max(xs), max(ys)


def make_case(sheet, rasterizer, out_png, out_json, scale=1.966, margin=(20, 60), seed=None):
    rng = random.Random(seed)
    S = render_sheet(sheet, skip_frame=True)
    box = content_box(sheet, S)
    if box is None:
        return None
    m = [rng.uniform(*margin) for _ in range(4)]
    x0, y0, x1, y1 = box[0] - m[0], box[1] - m[1], box[2] + m[2], box[3] + m[3]
    svg, W, H = to_svg_document(S, x0, y0, x1, y1, scale)
    # the PNG covers exactly W x H pixels starting at (x0, y1) -> recompute y0 from H for exact mapping
    y0 = y1 - H / scale
    rasterizer.render(svg, W, H, out_png)
    gt = build_gt(sheet, x0, y0, scale)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(gt, f, ensure_ascii=False, indent=1)
    return gt, (W, H)


def make_gt_only(sheet, scale=1.966, margin=(20, 60), seed=None):
    """Same geometry as make_case (same seed -> same crop), without rasterising: used to
    regenerate the labels of an existing dataset after a GT-convention fix."""
    rng = random.Random(seed)
    S = render_sheet(sheet, skip_frame=True)
    box = content_box(sheet, S)
    if box is None:
        return None
    m = [rng.uniform(*margin) for _ in range(4)]
    x0, y0, x1, y1 = box[0] - m[0], box[1] - m[1], box[2] + m[2], box[3] + m[3]
    svg, W, H = to_svg_document(S, x0, y0, x1, y1, scale)
    y0 = y1 - H / scale
    return build_gt(sheet, x0, y0, scale), (W, H)

#!/usr/bin/env python3
"""Dump the raw API responses for a few projects so the fetcher can be fixed.

    python tools/oshw_debug.py carele/powerbox_v2 flyn/flpowerpro sifli/sf32-keyboard --out debug_api
Writes <out>/<slug>.json (no cookies, no personal data).
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oshw_fetch import Client, OSHW, PRO, pro_headers  # noqa: E402


def trunc(o, n=3000):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + '...<truncated>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('projects', nargs='+')
    ap.add_argument('--out', default='debug_api')
    ap.add_argument('--cookie')
    ap.add_argument('--index', default=None, help='raw/index.jsonl from oshw_fetch (maps project path -> uuid)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cl = Client(open(a.cookie).read().strip() if a.cookie else None, 1.5)
    path2uuid = {}
    if a.index and os.path.exists(a.index):
        for line in open(a.index, encoding='utf-8'):
            try:
                r = json.loads(line)
                path2uuid[r.get('path')] = r.get('uuid')
            except ValueError:
                pass
    for key in a.projects:
        key = key.split('oshwhub.com/')[-1].strip('/')
        rec = {'key': key}
        uid = path2uuid.get(key, key)
        info = cl.get(f'{OSHW}/api/project/{uid}')
        rec['project'] = trunc(info, 2000)
        if not info or not info.get('success'):
            json.dump(rec, open(os.path.join(a.out, key.replace('/', '__') + '.json'), 'w'), ensure_ascii=False, indent=1)
            continue
        pu = info['result']['uuid']
        rec['origin'] = info['result'].get('origin')
        pi = cl.get(f'{OSHW}/api/project/{pu}/proImages')
        rec['proImages'] = trunc(pi, 6000)
        ids = []
        try:
            for b in (pi['result'] or {}).get('boards') or []:
                for d in ((b.get('schematic') or {}).get('documents') or []):
                    ids.append((d.get('uuid'), d.get('docType')))
        except Exception as e:
            rec['proImages_error'] = repr(e)
        rec['doc_ids'] = ids
        # also try the editor's own project endpoints
        rec['v4_project'] = trunc(cl.get(f'{PRO}/api/v4/projects/{pu}', headers=pro_headers(pu)), 3000)
        sheet_ids = [i for i, t in ids if t == 1]
        if sheet_ids:
            dl = cl.post(f'{PRO}/api/v2/documents/lists', json={'uuids': sheet_ids[:2], 'docType': 1}, headers=pro_headers(pu))
            res = (dl or {}).get('result') or []
            rec['documents_lists_keys'] = [sorted(d.keys()) for d in res]
            rec['documents_lists'] = trunc([{k: (v[:300] if isinstance(v, str) else v) for k, v in d.items()} for d in res], 4000)
            for d in res:
                url = d.get('dataStrId')
                if url and str(url).startswith('http'):
                    r = cl.s.get(url, timeout=30)
                    rec.setdefault('datastr_fetch', []).append({
                        'url': url, 'status': r.status_code, 'headers': {k: v for k, v in r.headers.items() if 'cookie' not in k.lower()},
                        'len': len(r.content), 'head_b64': base64.b64encode(r.content[:600]).decode(),
                        'iv': d.get('iv'), 'other_keys': {k: d.get(k) for k in d if k not in ('dataStr',)}})
        json.dump(rec, open(os.path.join(a.out, key.replace('/', '__') + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('saved', key, 'docs', ids[:4], rec.get('documents_lists_keys'))


if __name__ == '__main__':
    main()

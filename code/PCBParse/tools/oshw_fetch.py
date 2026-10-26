#!/usr/bin/env python3
"""Download open-source EasyEDA Pro schematics from oshwhub.com (run on your own machine).

Only public projects are fetched, one at a time, with a delay between requests.
For every project a self-contained bundle is written:
    <out>/bundles/<project_uuid>.json = {project, sheets:[{uuid,title,dataStr}], devices:{}, symbols:{}}
and a line is appended to <out>/index.jsonl.

Usage
  # discover projects related to a seed project (pages of 10), then download them
  python tools/oshw_fetch.py --out F:/EDA比赛六/raw --seed d54b0d8e884945fb8d875979510b5211 --max 3000
  # or download an explicit list (one project uuid or oshwhub path per line)
  python tools/oshw_fetch.py --out F:/EDA比赛六/raw --list projects.txt
Options
  --cookie cookie.txt   file containing the browser 'Cookie:' header value (only if anonymous access fails)
  --delay 1.5           seconds between HTTP requests (be polite; do not lower below 1)
"""
import argparse
import json
import os
import random
import sys
import time

import gzip
import zlib

import requests

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:  # pragma: no cover
    AESGCM = None

OSHW = 'https://oshwhub.com'
PRO = 'https://pro.lceda.cn'
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/124.0 Safari/537.36')


class Client:
    def __init__(self, cookie=None, delay=1.5):
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': UA, 'Accept': 'application/json, text/plain, */*'})
        if cookie:
            self.s.headers['Cookie'] = cookie
        self.delay = delay
        self.last = 0.0

    def _wait(self):
        dt = time.time() - self.last
        if dt < self.delay:
            time.sleep(self.delay - dt + random.uniform(0, 0.3))
        self.last = time.time()

    def get(self, url, **kw):
        return self._req('GET', url, **kw)

    def post(self, url, **kw):
        return self._req('POST', url, **kw)

    def _req(self, method, url, headers=None, **kw):
        for attempt in range(4):
            self._wait()
            try:
                r = self.s.request(method, url, headers=headers, timeout=30, **kw)
                self.last_status, self.last_text = r.status_code, r.text[:600]
            except requests.RequestException as e:
                self.last_status, self.last_text = None, repr(e)[:300]
                print('  network error', e, file=sys.stderr)
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            try:
                j = r.json()
            except ValueError:
                return None
            return j
        return None


def decode_blob(raw, key_hex=None, iv_hex=None):
    """Documents of newer projects are served as a CDN file: AES-256-GCM(gzip(dataStr)).
    The key and iv come with the document record (the editor decrypts it the same way)."""
    data = raw
    if key_hex and iv_hex:
        if AESGCM is None:
            raise RuntimeError('pip install cryptography')
        key, iv = bytes.fromhex(key_hex), bytes.fromhex(iv_hex)
        try:
            data = AESGCM(key).decrypt(iv, raw, None)
        except Exception:
            # tag missing / not appended: plain GCM counter-mode decryption
            dec = Cipher(algorithms.AES(key), modes.CTR(iv + b'\x00\x00\x00\x02')).decryptor()
            data = dec.update(raw) + dec.finalize()
    if data[:2] == b'\x1f\x8b':
        try:
            data = gzip.decompress(data)
        except Exception:
            data = zlib.decompressobj(31).decompress(data)
    text = data.decode('utf-8')
    if not text.lstrip().startswith('['):
        raise ValueError('decoded blob is not a dataStr')
    return text


def resolve_datastr(cl, rec):
    """Return the dataStr of a document/symbol record: inline, or fetched from dataStrId."""
    ds = rec.get('dataStr')
    if ds:
        return ds
    url = rec.get('dataStrId')
    if not (url and str(url).startswith('http')):
        return None
    for attempt in range(3):
        cl._wait()
        try:
            r = cl.s.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code != 200:
            time.sleep(5 * (attempt + 1))
            continue
        try:
            return decode_blob(r.content, rec.get('key'), rec.get('iv'))
        except Exception as e:
            print('  decode failed', url, repr(e)[:120], file=sys.stderr)
            return None
    return None


def pro_headers(project_uuid):
    return {'path': project_uuid, 'Content-Type': 'application/json', 'Referer': f'{PRO}/editor',
            'Origin': PRO, 'Editor-Version': '4.1.78'}


def project_info(cl, key):
    j = cl.get(f'{OSHW}/api/project/{key}')
    if not j or not j.get('success'):
        return None
    return j['result']


def related(cl, seed_uuid, tags, page, size=10):
    url = f'{OSHW}/api/project/relatedProjects'
    j = cl.get(url, params={'uuid': seed_uuid, 'project_tags': ','.join(tags), 'page': page, 'pageSize': size})
    if not j or not j.get('success'):
        return [], 0
    r = j['result']
    return r.get('lists') or [], r.get('totalPage') or 0


FAIL_DIR = None
_N_DUMPS = [0]


def dump_failure(cl, info, docs, tried):
    """Keep the raw answers of the first failures (no cookies) so the next fix is data-driven."""
    if not FAIL_DIR or _N_DUMPS[0] >= 30:
        return
    _N_DUMPS[0] += 1
    os.makedirs(FAIL_DIR, exist_ok=True)
    rec = {'project': {k: info.get(k) for k in ('uuid', 'path', 'origin', 'license')},
           'docs': docs, 'tried': tried, 'proImages': (docs_extra.get('raw') or '')[:8000]}
    with open(os.path.join(FAIL_DIR, info['uuid'] + '.json'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


def list_documents(cl, puuid, uuids, alt_paths):
    """POST documents/lists with a few variants; returns (json, attempts)."""
    tried = []
    variants = [(puuid, {'uuids': uuids, 'docType': 1}), (puuid, {'uuids': uuids})]
    variants += [(ap, {'uuids': uuids, 'docType': 1}) for ap in alt_paths if ap and ap != puuid]
    if len(uuids) > 1:
        variants.append((puuid, {'uuids': uuids[:1], 'docType': 1}))
    for path, body in variants:
        j = cl.post(f'{PRO}/api/v2/documents/lists', json=body, headers=pro_headers(path))
        ok = bool(j and j.get('success') and j.get('result'))
        tried.append({'path': path, 'body_keys': sorted(body), 'n': len(body['uuids']), 'http': getattr(cl, 'last_status', None),
                      'code': (j or {}).get('code'), 'msg': (j or {}).get('message') or (j or {}).get('msg'),
                      'text': None if ok else getattr(cl, 'last_text', '')[:400]})
        if (j or {}).get('code') == 422:      # author did not share the source: respect it, stop here
            break
        if ok:
            if len(body['uuids']) < len(uuids):      # single-uuid variant worked: fetch the rest one by one
                res = list(j['result'])
                for u_ in uuids[1:]:
                    j2 = cl.post(f'{PRO}/api/v2/documents/lists', json={'uuids': [u_], 'docType': 1},
                                 headers=pro_headers(path))
                    if j2 and j2.get('success'):
                        res += j2.get('result') or []
                j = dict(j, result=res)
            return j, tried
    return None, tried


docs_extra = {}


def sheet_docs(cl, puuid):
    j = cl.get(f'{OSHW}/api/project/{puuid}/proImages')
    docs = []
    docs_extra.clear()
    docs_extra['raw'] = json.dumps(j, ensure_ascii=False)[:8000] if j else None
    alt = []
    try:
        for b in ((j or {}).get('result') or {}).get('boards') or []:
            pu = (((b.get('schematic') or {}).get('boms') or {}).get('project_uuid'))
            if pu:
                alt.append(pu)
    except AttributeError:
        pass
    docs_extra['alt_paths'] = alt
    if not j or not j.get('success'):
        return docs
    for b in (j['result'] or {}).get('boards') or []:
        sch = b.get('schematic') or {}
        for d in sch.get('documents') or []:
            # newer projects omit docType on schematic sheets
            if d.get('docType') in (1, None) and d.get('uuid'):
                docs.append({'uuid': d['uuid'], 'title': d.get('display_title') or ''})
    return docs


def fetch_bundle(cl, info):
    puuid = info['uuid']
    docs = sheet_docs(cl, puuid)
    if not docs:
        return None, 'no schematic'
    j, tried = list_documents(cl, puuid, [d['uuid'] for d in docs], docs_extra.get('alt_paths', []))
    if not j:
        dump_failure(cl, info, docs, tried)
        last = tried[-1] if tried else {}
        if last.get('code') == 422:
            return None, 'no permission (source not shared)'
        return None, f"documents/lists failed http={last.get('http')} code={last.get('code')} msg={str(last.get('msg'))[:60]}"
    sheets = []
    titles = {d['uuid']: d['title'] for d in docs}
    for d in j['result'] or []:
        if d.get('docType') not in (1, None):
            continue
        ds = resolve_datastr(cl, d)
        if ds:
            sheets.append({'uuid': d['uuid'], 'title': titles.get(d['uuid'], d.get('display_title') or d.get('title', '')),
                           'dataStr': ds})
    if not sheets:
        return None, 'empty sheets'
    dev_ids, sym_ids = set(), set()
    for sh in sheets:
        for line in sh['dataStr'].splitlines():
            if line.startswith('["ATTR"'):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r[3] == 'Device' and r[4]:
                    dev_ids.add(r[4])
                elif r[3] == 'Symbol' and r[4]:
                    sym_ids.add(str(r[4]))
    devices = {}
    ids = sorted(dev_ids)
    for k in range(0, len(ids), 50):
        j = cl.post(f'{PRO}/api/v5/devices/searchByIds', json={'uuids': ids[k:k + 50]}, headers=pro_headers(puuid))
        for d in (j or {}).get('result') or []:
            devices[d['uuid']] = {'title': d.get('title'), 'attributes': d.get('attributes') or {}}
            s = (d.get('attributes') or {}).get('Symbol')
            if s:
                sym_ids.add(s)
    symbols = {}
    ids = sorted(sym_ids)
    for k in range(0, len(ids), 50):
        j = cl.post(f'{PRO}/api/v5/components/searchByIds', json={'uuids': ids[k:k + 50], 'path': puuid},
                    headers=pro_headers(puuid))
        for s in (j or {}).get('result') or []:
            ds = resolve_datastr(cl, s)
            if ds:
                symbols[s['uuid']] = {'title': s.get('title'), 'docType': s.get('docType'), 'dataStr': ds}
    missing = [i for i in sym_ids if i not in symbols and not i.endswith('_0')]
    meta = {k: info.get(k) for k in ('uuid', 'path', 'name', 'license', 'origin', 'updated_at', 'introduction')}
    meta['author'] = (info.get('owner') or {}).get('nickname') or (info.get('creator') or {}).get('nickname')
    meta['url'] = f"{OSHW}/{info.get('path')}"
    return {'project': meta, 'sheets': sheets, 'devices': devices, 'symbols': symbols, 'missing_symbols': missing}, 'ok'


def save_json(path, obj):
    """Atomic write (tmp + rename) with retries: /mnt/c under WSL sometimes throws EIO."""
    data = json.dumps(obj, ensure_ascii=False)
    for attempt in range(4):
        tmp = path + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(data)
            os.replace(tmp, path)
            return True
        except OSError as e:
            print('  write error', e, '- retrying', file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', action='append', default=[])
    ap.add_argument('--list')
    ap.add_argument('--max', type=int, default=2000)
    ap.add_argument('--cookie')
    ap.add_argument('--delay', type=float, default=1.5)
    ap.add_argument('--expand', action='store_true', help='use every discovered project as a further seed (snowball)')
    ap.add_argument('--max-pages', type=int, default=100000, help='related-project pages per seed')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'bundles'), exist_ok=True)
    global FAIL_DIR
    FAIL_DIR = os.path.join(a.out, 'fail_dumps')
    idx_path = os.path.join(a.out, 'index.jsonl')
    done = set()
    if os.path.exists(idx_path):
        for line in open(idx_path, encoding='utf-8'):
            try:
                r = json.loads(line)
                # failed projects are retried on the next run; author-restricted ones are not
                if r.get('status') == 'ok' or str(r.get('status', '')).startswith('no permission'):
                    done.add(r['uuid'])
            except (ValueError, KeyError):
                pass
    cookie = open(a.cookie, encoding='utf-8').read().strip() if a.cookie else None
    cl = Client(cookie, max(1.0, a.delay))

    seed_q = list(a.seed)
    seeded = set()

    def candidates():
        if a.list:
            for line in open(a.list, encoding='utf-8'):
                line = line.strip().rstrip('/')
                if line:
                    yield line.split('oshwhub.com/')[-1]
        # breadth-first over "related projects"; with --expand every visited project becomes a seed too
        while seed_q:
            seed = seed_q.pop(0)
            if seed in seeded:
                continue
            seeded.add(seed)
            info = project_info(cl, seed)
            if not info:
                print('seed not found', seed)
                continue
            seeded.add(info['uuid'])
            tags = info.get('project_tags') or []
            page, total = 1, 1
            while page <= min(total, a.max_pages):
                lst, total = related(cl, info['uuid'], tags, page)
                for p in lst:
                    if p.get('origin') == 'pro':
                        yield p['uuid']
                        if a.expand and p['uuid'] not in seeded:
                            seed_q.append(p['uuid'])
                page += 1
    n_ok = 0
    with open(idx_path, 'a', encoding='utf-8') as idx:
        for key in candidates():
            if n_ok >= a.max:
                break
            if key in done:
                continue
            info = project_info(cl, key)
            if not info or info.get('origin') != 'pro' or not info.get('public', True):
                continue
            if info['uuid'] in done:
                continue
            done.add(info['uuid'])
            bundle, status = fetch_bundle(cl, info)
            rec = {'uuid': info['uuid'], 'path': info.get('path'), 'license': info.get('license'), 'status': status}
            if bundle and not save_json(os.path.join(a.out, 'bundles', info['uuid'] + '.json'), bundle):
                rec['status'] = status = 'write failed (disk?)'
                bundle = None
            if bundle:
                rec['sheets'] = len(bundle['sheets'])
                rec['missing_symbols'] = len(bundle['missing_symbols'])
                n_ok += 1
            for attempt in range(4):
                try:
                    idx.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    idx.flush()
                    break
                except OSError as e:
                    print('  index write error', e, file=sys.stderr, flush=True)
                    time.sleep(10 * (attempt + 1))
            print(f"[{n_ok}] {info.get('path')} {status} sheets={rec.get('sheets')} license={info.get('license')}", flush=True)


if __name__ == '__main__':
    main()

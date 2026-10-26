#!/usr/bin/env python3
"""Evaluation-platform entry (called by run.sh):  python3 agent.py -input <dir> -output <dir>

Reads every image in <input> (flat *.png per the official sample; sub-folders also work),
writes <output>/final_results/{case_id}_Pred.json and decision_log.json (0 tokens).
Every case gets a JSON even if parsing crashes or times out (empty but valid),
so a single bad image can never cost the other cases.
"""
import os
import sys
import json
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CANDIDATES = ['/input', '/data/input', '/workspace/input', '/app/input', '/test', '/data', '/input_data']
OUT_PATTERN = '{case}_Pred.json'
CASE_TIMEOUT = float(os.environ.get('PCB_CASE_TIMEOUT', '240'))   # seconds per image
EMPTY = {'components': {}, 'pins': {}, 'nets': {}}
RETRY_DEADLINE = float(os.environ.get('PCB_RETRY_DEADLINE', '900'))   # no retries after this (platform limit 1200 s)


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-input', '--input', dest='input', default=os.environ.get('PCB_INPUT'))
    ap.add_argument('-output', '--output', dest='output', default=os.environ.get('PCB_OUTPUT', '/output'))
    a, _ = ap.parse_known_args()
    if not a.input:
        from pcbparse.pipeline import list_images
        a.input = next((c for c in CANDIDATES if os.path.isdir(c) and list_images(c)), None)
    if not a.input:
        raise SystemExit('no input directory; use -input <dir>')
    return a


def case_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def mem_available_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 8.0


def est_case_gb(path):
    """Peak memory of one case grows with the pixel count (~0.4 GB + 400 B/pixel measured)."""
    try:
        import struct
        with open(path, 'rb') as f:
            head = f.read(24)
        if head[:8] == b'\x89PNG\r\n\x1a\n':
            w, h = struct.unpack('>II', head[16:24])
            return 0.4 + w * h * 400 / 1e9
    except Exception:
        pass
    return 1.5


def n_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 2


def worker(path, q):
    try:
        from pcbparse.pipeline import parse_image
        q.put(parse_image(path))
    except Exception as e:  # noqa
        q.put({'__error__': repr(e)})


def run_one(path, timeout=None):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(path, q))
    p.start()
    try:
        res = q.get(timeout=timeout or CASE_TIMEOUT)
    except Exception:
        res = {'__error__': 'timeout'}
    p.join(5)
    if p.is_alive():
        p.kill()
    return res


def main():
    from pcbparse.pipeline import list_images
    from concurrent.futures import ThreadPoolExecutor
    a = parse_args()
    inp, root = a.input, a.output
    out = os.path.join(root, 'final_results')
    os.makedirs(out, exist_ok=True)
    cases = list_images(inp)
    workers = int(os.environ.get('PCB_WORKERS', '0')) or max(1, min(len(cases), n_cpus() - 1, 16))
    budget = max(2.0, 0.8 * mem_available_gb())
    print(f'input={inp} output={out} cases={len(cases)} workers={workers} mem_budget={budget:.1f}GB', flush=True)
    t0 = time.time()
    import threading
    cond = threading.Condition()
    state = {'used': 0.0}
    failed = []

    def job(path, retry=False):
        need = min(est_case_gb(path), budget)
        with cond:                      # memory-aware admission: big pages do not run all at once
            while state['used'] > 0 and state['used'] + need > budget:
                cond.wait()
            state['used'] += need
        try:
            t = time.time()
            res = run_one(path, CASE_TIMEOUT * (2 if retry else 1))
        finally:
            with cond:
                state['used'] -= need
                cond.notify_all()
        cid = case_id(path)
        err = res.pop('__error__', None) if isinstance(res, dict) else 'bad result'
        if err and not retry:
            failed.append(path)
        if err:
            res = dict(EMPTY)
        if not err or retry or not os.path.exists(os.path.join(out, OUT_PATTERN.format(case=cid))):
            with open(os.path.join(out, OUT_PATTERN.format(case=cid)), 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False)
        print(f'{cid}: {len(res["components"])} comps {len(res["nets"])} nets {time.time() - t:.1f}s'
              + (f' ERROR {err}' if err else '') + (' (retry)' if retry else ''), flush=True)

    # largest pages first: they bound the total run time
    cases = sorted(cases, key=lambda p: -est_case_gb(p))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, cases))
    # one more try, one at a time, for cases that crashed / ran out of memory / timed out
    for path in failed:
        if time.time() - t0 > RETRY_DEADLINE:
            break
        job(path, retry=True)
    log = {'method': 'opencv+ppocrv5-onnx (rule-based CV, no LLM)',
           'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'processed_cases': len(cases),
           'token_usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'api_calls': 0}}
    for d in (out, root):
        with open(os.path.join(d, 'decision_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
    print(f'done {len(cases)} cases in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()

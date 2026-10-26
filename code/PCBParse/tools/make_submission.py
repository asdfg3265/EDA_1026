#!/usr/bin/env python3
"""Build submission.zip for the evaluation platform (files at the zip root, no extra folder).

    python tools/make_submission.py [out.zip]
"""
import os, sys, zipfile, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'submission.zip')
files = {'Dockerfile': 'submission/Dockerfile', 'run.sh': 'submission/run.sh', 'agent.py': 'agent.py',
         'models/ppocrv5/rec.onnx': 'models/ppocrv5/rec.onnx',
         'models/ppocrv5/ppocrv5_dict.txt': 'models/ppocrv5/ppocrv5_dict.txt'}
for p in sorted(glob.glob(os.path.join(ROOT, 'pcbparse', '*.py'))):
    files['pcbparse/' + os.path.basename(p)] = os.path.relpath(p, ROOT)
for p in sorted(glob.glob(os.path.join(ROOT, 'data', '*.json'))):
    files['data/' + os.path.basename(p)] = os.path.relpath(p, ROOT)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for arc, src in files.items():
        zi = zipfile.ZipInfo.from_file(os.path.join(ROOT, src), arc)
        if arc == 'run.sh':
            zi.external_attr = 0o100755 << 16
        with open(os.path.join(ROOT, src), 'rb') as f:
            z.writestr(zi, f.read(), zipfile.ZIP_DEFLATED)
print('written', out, f'{os.path.getsize(out) / 1e6:.1f} MB', len(files), 'files')

#!/usr/bin/env python3
"""Probe the evaluation / training machine and write env_report.json.

Usage (on the competition server and on your own GPU server):
    python tools/env_probe.py            # writes env_report.json in the current dir
Only reads information; installs nothing, sends nothing over the network
except an optional reachability check (HEAD request) to a few hosts.
"""
import json, os, platform, shutil, subprocess, sys, time


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa
        return f'ERR {e}'


R = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'platform': platform.platform(), 'python': sys.version,
     'executable': sys.executable, 'cpu_count': os.cpu_count(), 'cwd': os.getcwd()}
R['meminfo'] = sh("grep -E 'MemTotal|MemAvailable' /proc/meminfo")
R['cpu_model'] = sh("grep -m1 'model name' /proc/cpuinfo")
R['disk'] = sh('df -h . | tail -1')
R['nvidia_smi'] = sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader')
R['cuda_nvcc'] = sh('nvcc --version | tail -1')
R['docker'] = shutil.which('docker') or ''
R['in_container'] = os.path.exists('/.dockerenv')
mods = {}
for m in ['torch', 'torchvision', 'onnxruntime', 'cv2', 'numpy', 'scipy', 'PIL', 'paddle', 'paddleocr',
          'ultralytics', 'openai', 'dashscope', 'docx', 'networkx', 'shapely', 'skimage']:
    try:
        mod = __import__(m)
        mods[m] = getattr(mod, '__version__', 'ok')
    except Exception as e:  # noqa
        mods[m] = None
R['modules'] = mods
try:
    import torch
    R['torch_cuda'] = {'available': torch.cuda.is_available(), 'version': torch.version.cuda,
                       'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
except Exception:
    pass
try:
    import onnxruntime as ort
    R['ort_providers'] = ort.get_available_providers()
except Exception:
    pass
# names only, never values
R['llm_env_vars'] = sorted(k for k in os.environ if any(s in k.upper() for s in
                           ('QWEN', 'DASHSCOPE', 'OPENAI', 'API_KEY', 'LLM', 'MODEL', 'BASE_URL')))
net = {}
for host in ['https://dashscope.aliyuncs.com', 'https://pypi.org', 'https://github.com',
             'https://easyeda.com', 'https://oshwhub.com', 'https://pro.easyeda.com']:
    net[host] = sh(f'curl -s -o /dev/null -m 6 -I -w "%{{http_code}}" {host}', timeout=10)
R['network'] = net
R['pip_freeze_head'] = sh(f'{sys.executable} -m pip freeze 2>/dev/null | head -200')
out = os.path.abspath('env_report.json')
json.dump(R, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ('platform', 'python', 'cpu_count', 'nvidia_smi', 'modules', 'llm_env_vars', 'network')},
                 ensure_ascii=False, indent=1))
print('written', out)

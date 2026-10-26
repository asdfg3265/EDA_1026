#!/usr/bin/env python3
"""Usage: python run.py <input_dir> <output_dir>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcbparse.pipeline import run_dir

if __name__ == '__main__':
    run_dir(sys.argv[1], sys.argv[2])

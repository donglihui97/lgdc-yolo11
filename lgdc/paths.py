"""Locations of the data and of generated files.

LGDC_ROOT  working directory for runs, engines and results (default: the repository root).
LGDC_DATA  directory that contains neu_det/ and gc10_det/ as written by tools/prepare_data.py
           (default: $LGDC_ROOT/data/processed).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('LGDC_ROOT', str(REPO)))
DATA_ROOT = Path(os.environ.get('LGDC_DATA', str(ROOT / 'data' / 'processed')))
DATA = {'neu': DATA_ROOT / 'neu_det', 'gc10': DATA_ROOT / 'gc10_det'}
NC = {'neu': 6, 'gc10': 10}
RUNS = ROOT / 'runs'
ENGINES = ROOT / 'engines'
RESULTS = ROOT / 'results'


def data_yaml(ds):
    return str(DATA[ds] / 'data.yaml')


def resolve(p):
    """Paths in timing configs are relative to LGDC_ROOT unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

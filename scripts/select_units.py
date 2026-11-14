"""LGDC step 2: rank the units by measured benefit per accuracy cost and select them greedily.

    benefit  dt_u = (t_0 - t_u) / t_0                    at the deployment batch size (Eq. 3)
    cost     da_u = max(a_0 - a_u, eps), eps = 0.1 pp      after the 10-epoch fine-tuning (Eq. 4)
    order    by dt_u / da_u, largest first; keep the prefix whose summed benefit is closest to rho (Eq. 5)

Either read your own measurements
    --timing results/units_<tag>_b1/summary.json --sens runs/sens --ds neu --imgsz 256
or the measurements of the paper
    --table measurements/units_neu.json
"""
import argparse
import csv
import json
from pathlib import Path

import _common  # noqa: F401
from lgdc import paths
from lgdc.unit_names import PAPER_NAME


def last_map(run_dir):
    with open(Path(run_dir) / 'results.csv', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    key = next(k for k in rows[0] if 'mAP50-95' in k)
    return 100 * float(rows[-1][key])


def from_runs(timing, sens_dir, ds, imgsz):
    summary = json.loads(Path(timing).read_text())
    t0 = summary['none']['median_ms']
    a0 = last_map(Path(sens_dir) / f'{ds}_sens_none_i{imgsz}_s0')
    rows = []
    for u, s in summary.items():
        if u in ('none',) or u.startswith('_'):
            continue
        rows.append({'unit': u, 'dt_pct': 100 * (t0 - s['median_ms']) / t0,
                     'a_u': last_map(Path(sens_dir) / f'{ds}_sens_{u.replace(".", "_")}_i{imgsz}_s0')})
    return a0, rows


def select(a0, rows, rho, eps):
    for r in rows:
        r['da'] = max(a0 - r['a_u'], eps)
        r['ratio'] = r['dt_pct'] / r['da']
    rows.sort(key=lambda r: -r['ratio'])
    best_k, best_gap, total = 0, abs(rho), 0.0
    for k, r in enumerate(rows, 1):
        total += r['dt_pct']
        r['cum'] = total
        if abs(total - rho) < best_gap:
            best_k, best_gap = k, abs(total - rho)
    return rows, best_k


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', help='JSON with a0 and per-unit dt_b1_pct and a_u (see measurements/)')
    ap.add_argument('--batch', type=int, default=1, choices=[1, 16], help='batch size of the benefit (with --table)')
    ap.add_argument('--timing', help='summary.json of the per-unit timing session')
    ap.add_argument('--sens', default=str(paths.RUNS / 'sens'))
    ap.add_argument('--ds', choices=['neu', 'gc10'])
    ap.add_argument('--imgsz', type=int)
    ap.add_argument('--rho', type=float, default=25.0, help='target latency reduction in percent')
    ap.add_argument('--eps', type=float, default=0.1, help='cost floor in mAP percentage points')
    a = ap.parse_args()
    if a.table:
        t = json.loads(Path(a.table).read_text())
        a0 = t['a0']
        rows = [{'unit': r['unit'], 'dt_pct': r[f'dt_b{a.batch}_pct'], 'a_u': r['a_u']} for r in t['units']]
    else:
        a0, rows = from_runs(a.timing, a.sens, a.ds, a.imgsz)
    rows, k = select(a0, rows, a.rho, a.eps)
    print(f'a0 = {a0:.2f}   target rho = {a.rho:.1f}%')
    print(f'{"":2s}{"unit":10s}{"paper":>6s}{"dt %":>8s}{"da pp":>8s}{"ratio":>8s}{"sum dt %":>10s}')
    for i, r in enumerate(rows):
        mark = '*' if i < k else ' '
        print(f'{mark} {r["unit"]:10s}{PAPER_NAME.get(r["unit"], ""):>6s}{r["dt_pct"]:8.2f}{r["da"]:8.3f}'
              f'{r["ratio"]:8.1f}{r["cum"]:10.2f}')
    sel = [r['unit'] for r in rows[:k]]
    print('selected:', ','.join(sel), ' (' + ','.join(PAPER_NAME.get(u, u) for u in sel) + ')')

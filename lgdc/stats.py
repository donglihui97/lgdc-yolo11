"""Moving-block bootstrap and GPU clock telemetry for the timing sessions."""
import random
import statistics
import subprocess
import time


def block_ci(values, seed=0, reps=2000, block=5):
    """95% moving-block bootstrap interval of the median (fixed seed)."""
    rng = random.Random(seed)
    n = len(values)
    block = min(block, n)
    meds = []
    for _ in range(reps):
        sample = []
        while len(sample) < n:
            i = rng.randrange(0, n - block + 1)
            sample.extend(values[i:i + block])
        meds.append(statistics.median(sample[:n]))
    meds.sort()
    return [meds[int(.025 * reps)], meds[int(.975 * reps) - 1]]


def telemetry(stop, rows):
    """Sample the SM clock, temperature and throttle reasons once per second until stop is set."""
    while not stop.is_set():
        out = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
        rows.append((time.time(), out))
        stop.wait(1.0)


def clock_states(rows):
    return sorted(set(r[1].split(', ')[0] + ' MHz / throttle ' + r[1].split(', ')[-1] for r in rows if r[1]))

"""Run in a fresh process with an empty NUMBA_CACHE_DIR for each dimension."""
import json
import os
from pathlib import Path
import sys
import time
from numerical_chan_numba import hypervolume

d = int(sys.argv[1])
points = [tuple(1 + (i + k) % 5 for i in range(d)) for k in range(5)]
start = time.perf_counter()
value = hypervolume(points)
first = time.perf_counter() - start
start = time.perf_counter()
second_value = hypervolume(points)
second = time.perf_counter() - start
assert value == second_value
row = {'dimension': d, 'points': points, 'value': value,
       'first_use_seconds': first, 'second_use_seconds': second,
       'scope': 'Module imports excluded; first call includes actual JIT compilation '
                'and computation with a fresh empty cache. Not pure compiler time.'}
Path(sys.argv[2]).write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))

import re
from collections import defaultdict

profiler_file = "profiler/fit-profiler1.txt"

func_times = defaultdict(float)

# Regex to match cProfile lines
pattern = re.compile(
    r"\s*(\S+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.+)"
)

with open(profiler_file, "r") as f:
    for line in f:
        match = pattern.match(line)
        if match:
            ncalls, cumtime, func = match.groups()
            func_times[func.strip()] += float(cumtime)

# Sort functions by cumulative time descending
sorted_funcs = sorted(func_times.items(), key=lambda x: x[1], reverse=True)

print("Top 20 slowest functions (by cumulative time):")
for func, t in sorted_funcs[:20]:
    print(f"{func:60} {t:.6f} sec")

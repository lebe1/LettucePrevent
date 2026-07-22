import glob
import os

stats_files = sorted(glob.glob("*_stats.txt"))
output_file = "aggregated_stats.txt"

with open(output_file, "w") as out:
    for filepath in stats_files:
        out.write(f"=== {filepath} ===\n\n")
        with open(filepath, "r") as f:
            out.write(f.read())
        out.write("\n\n")

print(f"Aggregated {len(stats_files)} files into {output_file}")

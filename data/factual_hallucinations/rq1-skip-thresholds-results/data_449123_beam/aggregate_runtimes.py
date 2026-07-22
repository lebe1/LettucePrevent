import json
import glob
import os

files = sorted(glob.glob("**/*_generations.json", recursive=True))

output_file = "runtime_stats.txt"

with open(output_file, "w") as out:
    for filepath in files:
        with open(filepath, "r") as f:
            data = json.load(f)

        meta = data[-1]["_meta"]
        generations = data[:-1]

        durations = [entry["duration_seconds"] for entry in generations]
        n = len(durations)
        sorted_durations = sorted(durations)
        mid = n // 2
        median = (sorted_durations[mid] if n % 2 != 0
                  else (sorted_durations[mid - 1] + sorted_durations[mid]) / 2)

        out.write(f"=== {os.path.basename(filepath)} ===\n\n")
        out.write(f"Total duration (meta) : {meta['duration_seconds']:.2f}s\n")
        out.write(f"Num generations       : {n}\n")
        out.write(f"Mean                  : {sum(durations) / n:.2f}s\n")
        out.write(f"Median                : {median:.2f}s\n")
        out.write(f"Min                   : {min(durations):.2f}s\n")
        out.write(f"Max                   : {max(durations):.2f}s\n")
        out.write("\n\n")

print(f"Processed {len(files)} files into {output_file}")

import json
from collections import Counter

# Load dataset
input_file = "../data/ragtruth_data.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter and count prompts with task_type == "Summary"
summary_prompts = [entry["prompt"] for entry in data if entry.get("task_type") == "Summary"]
prompt_counts = Counter(summary_prompts)

# Convert to desired format
output_data = [{"prompt": prompt, "counts": count} for prompt, count in prompt_counts.items()]

# Save output to file
output_file = "../data/ragtruth_unique_summary_prompts.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(output_data)} unique prompts to {output_file}")

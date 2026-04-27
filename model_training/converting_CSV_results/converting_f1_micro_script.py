import pandas as pd

df = pd.read_csv("eval_recall_micro.csv")  # adjust filename as needed

# Extract unique run names from column headers
# Column pattern: "<run_name> - eval/recall_micro"
f1_cols = [c for c in df.columns if c.endswith("- eval/recall_micro") and "__" not in c]
run_names = [c.replace(" - eval/recall_micro", "") for c in f1_cols]

records = []
for run, col in zip(run_names, f1_cols):
    series = df[col].dropna()
    if series.empty:
        continue
    final_f1 = series.iloc[-1]
    max_f1 = series.max()
    # Find the global_step at which the maximum occurred
    max_idx = series.idxmax()
    step_at_max = df.loc[max_idx, "train/global_step"]
    records.append({
        "Run": run,
        "Final eval/recall_micro": round(final_f1, 6),
        "Max. eval/recall_micro": round(max_f1, 6),
        "Step at Max.": int(step_at_max)
    })

summary = pd.DataFrame(records)
# Sort by run number (ascending)
summary["_num"] = summary["Run"].str.extract(r"(\d+)$").astype(int)
summary = summary.sort_values("_num").drop(columns="_num").reset_index(drop=True)

print(summary.to_string(index=False))
summary.to_csv("sweep_summary.csv", index=False)
import pandas as pd

df = pd.read_csv("train_loss.csv")

# Extract unique run names from column headers
# Column pattern: "<run_name> - eval/f1_micro"
loss_cols = [c for c in df.columns if c.endswith("- eval/f1_micro") and "__" not in c]
run_names = [c.replace(" - eval/f1_micro", "") for c in loss_cols]

records = []
for run, col in zip(run_names, loss_cols):
    series = df[col].dropna()
    if series.empty:
        continue
    final_loss = series.iloc[-1]
    min_loss = series.min()
    # Find the global_step at which the minimum occurred
    min_idx = series.idxmin()
    step_at_min = df.loc[min_idx, "train/global_step"]
    records.append({
        "Run": run,
        "Final eval/f1_micro": round(final_loss, 6),
        "Min. eval/f1_micro": round(min_loss, 6),
        "Step at Min.": int(step_at_min)
    })

summary = pd.DataFrame(records)
# Sort by run number (ascending)
summary["_num"] = summary["Run"].str.extract(r"(\d+)$").astype(int)
summary = summary.sort_values("_num").drop(columns="_num").reset_index(drop=True)

print(summary.to_string(index=False))
summary.to_csv("sweep_summary.csv", index=False)
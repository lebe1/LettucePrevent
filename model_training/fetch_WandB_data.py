import wandb
import pandas as pd
from pathlib import Path

api = wandb.Api()
runs = api.runs("lebeccard/your-project-name")  # entity/project

out_dir = Path("results/wandb_export")
out_dir.mkdir(parents=True, exist_ok=True)

summary_rows = []
for run in runs:
    # Full metric history (all logged steps, all metrics)
    history = run.history()  # pandas DataFrame
    history.to_csv(out_dir / f"{run.name}_history.csv", index=False)

    # Hyperparameters + final summary metrics in one overview table
    row = {"run_name": run.name, "run_id": run.id, "state": run.state}
    row.update({f"config/{k}": v for k, v in run.config.items()})
    row.update({f"summary/{k}": v for k, v in run.summary.items()
                if not k.startswith("_")})
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(out_dir / "sweep_overview.csv", index=False)
print(f"Exported {len(summary_rows)} runs to {out_dir}")
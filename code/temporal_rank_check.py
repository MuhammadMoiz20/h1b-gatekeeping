"""Rank metrics for the FY2022 temporal test reported in the paper.

Re-executes the setup and temporal cells of 04_models.ipynb (cells 1, 3, 5, 13)
so the fitted models are identical, then reports how well the pre-2022
predictions rank FY2022 employer-years. Writes output/tables/temporal_rank_metrics.csv.
Run from anywhere inside the repository: python code/temporal_rank_check.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

ROOT = next(p for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents)
            if (p / "code" / "src").is_dir())
nb = json.load(open(ROOT / "code" / "04_models.ipynb"))
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
ns = {}
for i in (1, 3, 5, 13):
    exec(compile(cells[i], f"04_models cell {i}", "exec"), ns)
    if i == 1:  # keep cell 5's split diagnostic write out of the committed tables
        ns["TABLES"] = Path(tempfile.mkdtemp())

y, y_bin, w = ns["y"], ns["y_bin"], ns["weights"]
te = ns["te_t"]
rows = []
for model, pred in (("LightGBM", np.asarray(ns["pred_gbm_t"])),
                    ("WLS", np.asarray(ns["pred_wls_t"]))):
    yt, wt, yb = y[te], w[te], y_bin[te]
    rep = np.repeat(np.arange(len(yt)), wt.astype(int))
    dec = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    lo = np.average(yt[dec == 0], weights=wt[dec == 0])
    hi = np.average(yt[dec == 9], weights=wt[dec == 9])
    mse = np.average((yt - pred) ** 2, weights=wt)
    bias2 = (np.average(pred, weights=wt) - np.average(yt, weights=wt)) ** 2
    rows.append({
        "model": model, "test_year": 2022, "rows": len(te),
        "weighted_r2": r2_score(yt, pred, sample_weight=wt),
        "spearman": spearmanr(yt, pred).correlation,
        "spearman_petition_weighted": spearmanr(yt[rep], pred[rep]).correlation,
        "auc_any_denial": roc_auc_score(yb, pred),
        "auc_any_denial_weighted": roc_auc_score(yb, pred, sample_weight=wt),
        "actual_rate_lowest_pred_decile": lo, "actual_rate_highest_pred_decile": hi,
        "pred_mean": np.average(pred, weights=wt), "actual_mean": np.average(yt, weights=wt),
        "bias2_share_of_mse": bias2 / mse,
    })
out = pd.DataFrame(rows)
out.to_csv(ROOT / "output" / "tables" / "temporal_rank_metrics.csv", index=False)
print(out.round(4).to_string(index=False))

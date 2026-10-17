"""Out-of-fold evaluation of the three baseline models (training set only).

Saves predictions, threshold tables and plots to reports/. The test set is not used.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # save figures to files, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from src.config import REPORTS_DIR
from src.models.evaluation import calibration_table, pooled_scores, threshold_table
from src.models.data import load_train_xy
from src.models.out_of_fold import get_oof_probabilities
from src.models.run_baselines import get_baseline_models

logger = logging.getLogger(__name__)
THRESHOLDS = np.round(np.arange(0.2, 0.85, 0.1), 1)


def save_figures(y: pd.Series, oof: pd.DataFrame, names: list[str]) -> None:
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in names:
        precision, recall, _ = precision_recall_curve(y, oof[name])
        ax.plot(recall, precision, label=f"{name} (PR-AUC {average_precision_score(y, oof[name]):.2f})")
    ax.axhline(y.mean(), color="grey", linestyle="--", label=f"floor ({y.mean():.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (out-of-fold, training set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in names:
        fpr, tpr, _ = roc_curve(y, oof[name])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="floor")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (out-of-fold, training set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in names:
        table = calibration_table(y, oof[name])
        ax.plot(table["mean_predicted"], table["actual_rate"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Actual attrition rate", title="Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = load_train_xy()
    leaver_share = y.mean()
    all_models = get_baseline_models(scale_pos_weight=(1 - leaver_share) / leaver_share)
    models = {name: m for name, m in all_models.items() if not name.startswith("dummy")}

    oof = pd.DataFrame({"EmployeeNumber": X["EmployeeNumber"].to_numpy(), "attrition": y.to_numpy()})
    for name, model in models.items():
        logger.info("Out-of-fold predictions for %s ...", name)
        oof[name] = get_oof_probabilities(model, X, y)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)
    names = list(models)

    print(f"\nTraining rows: {len(y)}, leavers: {int(y.sum())} ({leaver_share:.3f})")

    print("\n=== POOLED OUT-OF-FOLD SCORES ===")
    print(pd.DataFrame({n: pooled_scores(y, oof[n]) for n in names}).T.round(3).to_string())

    tables = []
    for name in names:
        table = threshold_table(y, oof[name], THRESHOLDS)
        tables.append(table.assign(model=name))
        print(f"\n=== THRESHOLD TABLE: {name} ===")
        print(table.round(3).to_string(index=False))
    pd.concat(tables).to_csv(REPORTS_DIR / "threshold_tables.csv", index=False)

    for name in names:
        print(f"\n=== CALIBRATION: {name} (5 equal-sized groups) ===")
        print(calibration_table(y, oof[name]).round(3).to_string(index=False))

    save_figures(y, oof, names)
    logger.info("Saved predictions, tables and figures to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()

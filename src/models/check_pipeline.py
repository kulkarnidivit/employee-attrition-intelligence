"""Run the preprocessing pipeline on the real TRAINING data (all four switch combinations)."""

from sklearn.dummy import DummyClassifier

from src.features.preprocessing import build_pipeline
from src.models.data import load_train_xy


def main() -> None:
    X, y = load_train_xy()
    print(f"Train: X={X.shape}, leavers={int(y.sum())} ({y.mean():.3f})\n")

    for feature_set in ("raw", "engineered"):
        for include_sensitive in (False, True):
            pipe = build_pipeline(DummyClassifier(), feature_set, include_sensitive, scale_numeric=True)
            out = pipe[:-1].fit_transform(X, y)
            print(
                f"{feature_set:10s} sensitive={str(include_sensitive):5s} -> "
                f"{out.shape[1]} columns, NaN={int(out.isna().sum().sum())}"
            )

    default = build_pipeline(DummyClassifier())[:-1].fit_transform(X, y)
    print("\nDefault columns (engineered, sensitive OFF):")
    print(", ".join(default.columns))


if __name__ == "__main__":
    main()

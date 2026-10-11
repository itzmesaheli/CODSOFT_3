"""
CODSOFT Task 3 - Customer Churn Prediction
------------------------------------------
Predicts whether a customer will leave (churn) using Logistic Regression,
Random Forest and Gradient Boosting.

Dataset: put ONE churn CSV in ./data/ (any name ending in .csv). Works with:
    - Churn_Modelling.csv  (bank customers, target column "Exited")
    - Telco customer churn (target column "Churn")
The script finds the target column and the ID columns automatically.

Usage:
    pip install pandas scikit-learn matplotlib joblib
    python churn_predictor.py
    python churn_predictor.py --score data/new_customers.csv     # score new customers
"""

import argparse
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data")
OUT_DIR = Path("output")
MODEL_PATH = OUT_DIR / "churn_model.joblib"

TARGET_CANDIDATES = ["exited", "churn", "churned", "attrition", "left"]
ID_HINTS = ["rownumber", "customerid", "customer_id", "customer id", "surname", "name", "id"]


# ---------------------------------------------------------------- data loading
def find_dataset() -> Path:
    files = sorted(DATA_DIR.glob("*.csv")) + sorted(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit("No CSV/XLSX found. Put the churn dataset inside the 'data' folder.")
    return files[0]


def find_target(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.strip().lower() in TARGET_CANDIDATES:
            return col
    raise SystemExit(
        f"Could not find the churn column. Columns are: {list(df.columns)}\n"
        f"Rename the target column to 'Exited' or 'Churn'."
    )


def encode_target(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        series = series.str.strip().str.lower().map(mapping)
    return series.astype(int)


def prepare_features(df: pd.DataFrame, target: str | None) -> pd.DataFrame:
    """Drop ID-like columns and the target; fix numbers stored as text."""
    df = df.copy()
    drop = [c for c in df.columns if c.strip().lower() in ID_HINTS]
    if target:
        drop.append(target)
    X = df.drop(columns=drop, errors="ignore")

    for col in X.select_dtypes(exclude="number").columns:
        converted = pd.to_numeric(X[col].astype(str).str.strip().replace("", np.nan), errors="coerce")
        if converted.notna().mean() > 0.9:          # e.g. Telco "TotalCharges"
            X[col] = converted
        elif X[col].nunique() > 50:                  # free-text / high-cardinality column
            X = X.drop(columns=col)
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )


def make_pipeline(X: pd.DataFrame, clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])


# ------------------------------------------------------------------- training
def train_and_evaluate() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: load
    path = find_dataset()
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    target = find_target(df)
    y = encode_target(df[target])
    X = prepare_features(df, target)
    print(f"[1] Loaded {path.name}: {len(df):,} customers, {X.shape[1]} features")
    print(f"    Target '{target}': {y.mean():.1%} churned ({int(y.sum()):,} of {len(y):,})")
    print(f"    Features used: {list(X.columns)}\n")

    # Step 2: split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[2] Train: {len(X_tr):,} | Test: {len(X_te):,}\n")

    # Step 3: compare models
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results = {}
    print("[3] Model comparison (churn = positive class)")
    print(f"    {'Model':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'AUC':>8}")
    for name, clf in candidates.items():
        start = time.time()
        pipe = make_pipeline(X_tr, clf).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        scores = (
            accuracy_score(y_te, pred), precision_score(y_te, pred),
            recall_score(y_te, pred), f1_score(y_te, pred), roc_auc_score(y_te, proba),
        )
        results[name] = (scores[4], pipe, pred)
        print(f"    {name:<22}" + "".join(f"{s:>8.4f}" for s in scores) + f"  ({time.time() - start:.0f}s)")

    best_name = max(results, key=lambda k: results[k][0])
    _, best_pipe, best_pred = results[best_name]
    print(f"\n[4] Best model (by ROC-AUC): {best_name}\n")
    print(classification_report(y_te, best_pred, target_names=["stayed", "churned"], digits=4))

    # Step 5: plots
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, best_pred, display_labels=["stayed", "churned"], ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, pipe, _) in results.items():
        RocCurveDisplay.from_estimator(pipe, X_te, y_te, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Step 6: what drives churn?
    clf = best_pipe.named_steps["clf"]
    names = best_pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=names)
    else:
        imp = pd.Series(np.abs(clf.coef_[0]), index=names)
    top = imp.sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax, color="#38bdf8")
    ax.set_title(f"Top features driving churn - {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("[5] Top churn drivers:", ", ".join(top.index[::-1][:5]))

    # Step 7: retrain best model on ALL data and save
    final = make_pipeline(X, candidates[best_name]).fit(X, y)
    joblib.dump({"model": final, "columns": list(X.columns)}, MODEL_PATH)
    print(f"[6] Final model trained on all data and saved to {MODEL_PATH}")


# ----------------------------------------------------------------- prediction
def score_file(csv_path: str) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found. Run the script without --score first.")
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_path)
    X = prepare_features(df, None)[bundle["columns"]]
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["predicted_churn"] = (proba >= 0.5).astype(int)
    dest = OUT_DIR / "scored_customers.csv"
    out.to_csv(dest, index=False)
    print(f"Scored {len(out):,} customers -> {dest}")
    print(f"Predicted to churn: {int(out['predicted_churn'].sum()):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODSOFT Task 3 - Customer Churn Prediction")
    parser.add_argument("--score", metavar="CSV", help="Score customers from a CSV file")
    args = parser.parse_args()

    if args.score:
        score_file(args.score)
    else:
        train_and_evaluate()

"""
Trains the champion credit-default-risk model (same synthetic dataset and
preprocessing pipeline used earlier in this project) and saves the fitted
pipeline to model/champion_model.pkl for the FastAPI service to load.
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    MODEL = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=7,
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    MODEL = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=7,
    )

RANDOM_STATE = 7
NUMERIC_FEATURES = ["age", "income", "loan_amount", "credit_score", "debt_to_income"]
LOW_CARD_CAT = ["employment_type"]
HIGH_CARD_CAT = ["state"]
ALL_FEATURES = NUMERIC_FEATURES + LOW_CARD_CAT + HIGH_CARD_CAT


def make_synthetic_data(n=3000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 75, n).astype(float)
    income = np.round(rng.lognormal(mean=10.5, sigma=0.5, size=n), 2)
    loan_amount = np.round(rng.uniform(1000, 50000, n), 2)
    credit_score = rng.integers(300, 850, n).astype(float)
    debt_to_income = np.round(rng.uniform(0, 0.6, n), 3)
    employment_type = rng.choice(
        ["Salaried", "Self-employed", "Unemployed", "Retired"], n, p=[0.55, 0.25, 0.1, 0.1]
    ).astype(object)
    state = rng.choice([f"ST{i:02d}" for i in range(25)], n)

    for arr, frac in [(age, 0.03), (income, 0.05), (credit_score, 0.04), (debt_to_income, 0.02)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        arr[idx] = np.nan
    emp_idx = rng.choice(n, int(n * 0.03), replace=False)
    employment_type[emp_idx] = None

    logit = (
        -1.0 - 0.01 * (credit_score - 650) + 3.0 * debt_to_income + 0.00002 * loan_amount
        - 0.01 * (income / 1000) + 0.8 * (employment_type == "Unemployed")
        + rng.normal(0, 0.7, n)
    )
    logit = np.nan_to_num(logit, nan=0.0)
    prob_default = 1 / (1 + np.exp(-logit))
    default = (rng.random(n) < prob_default).astype(int)

    return pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount, "credit_score": credit_score,
        "debt_to_income": debt_to_income, "employment_type": employment_type, "state": state,
        "default": default,
    })


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_enc", TargetEncoder(random_state=RANDOM_STATE)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("low_card_cat", low_card_pipeline, LOW_CARD_CAT),
        ("high_card_cat", high_card_pipeline, HIGH_CARD_CAT),
    ])
    return Pipeline([("preprocess", preprocessor), ("clf", MODEL)])


def main():
    df = make_synthetic_data()
    X = df[ALL_FEATURES]
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    print("Test ROC-AUC: ", round(roc_auc_score(y_test, proba), 4))
    print("Test Precision:", round(precision_score(y_test, pred), 4))
    print("Test Recall:   ", round(recall_score(y_test, pred), 4))
    print("Test F1:       ", round(f1_score(y_test, pred), 4))

    joblib.dump(pipeline, "model/champion_model.pkl")
    print("\nSaved model/champion_model.pkl")


if __name__ == "__main__":
    main()

import optuna
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocess import load_raw_data, clean_data, get_feature_target

RANDOM_STATE = 42


def objective(trial, X_train, y_train):
    C = trial.suggest_float("C", 1e-4, 1e2, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, l1_ratio=l1_ratio, class_weight=class_weight,
            solver="saga", max_iter=3000,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()


def run_tuning(n_trials=50):
    df = load_raw_data("data/raw/big_startup_secsees_dataset.csv")
    df = clean_data(df)
    X, y = get_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    print("Best CV ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)

    best_params = study.best_params
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=best_params["C"], l1_ratio=best_params["l1_ratio"],
            class_weight=best_params["class_weight"], solver="saga", max_iter=3000,
        )),
    ])
    final_pipeline.fit(X_train, y_train)

    preds = final_pipeline.predict(X_test)
    probs = final_pipeline.predict_proba(X_test)[:, 1]
    print("\nFinal holdout test performance:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    joblib.dump(final_pipeline, "models/tuned_logreg.pkl")
    return final_pipeline, study


if __name__ == "__main__":
    run_tuning(n_trials=50)
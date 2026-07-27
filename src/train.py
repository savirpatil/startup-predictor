import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocess import load_raw_data, clean_data, get_feature_target


def train_baseline_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    return pipeline


if __name__ == "__main__":
    df = load_raw_data("/Users/savirpatil/Projects/startup-predictor/data/raw/big_startup_secsees_dataset.csv")
    df = clean_data(df)
    X, y = get_feature_target(df)
    model = train_baseline_model(X, y)
    joblib.dump(model, "models/baseline_logreg.pkl")
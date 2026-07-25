import numpy as np
import pandas as pd

DROP_COLUMNS = [
    "Unnamed: 0", "Unnamed: 6", "id", "object_id", "name",
    "closed_at", "founded_at", "first_funding_at", "last_funding_at",
    "state_code", "state_code.1", "category_code",
    "latitude", "longitude", "zip_code", "city",
    "labels",
]

MILESTONE_COLS = ["age_first_milestone_year", "age_last_milestone_year"]


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target
    df["target"] = (df["status"] == "acquired").astype(int)

    # Missing-milestone flag before imputing
    df["no_milestone_reached"] = df[MILESTONE_COLS[0]].isnull().astype(int)
    df[MILESTONE_COLS] = df[MILESTONE_COLS].fillna(0)

    # Skewed funding amount -> log scale
    df["funding_total_usd_log"] = np.log1p(df["funding_total_usd"])

    # Drop redundant / leaking / unusable columns
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    df = df.drop(columns=["status", "funding_total_usd"])

    return df


def get_feature_target(df: pd.DataFrame):
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


if __name__ == "__main__":
    df = load_raw_data("/Users/savirpatil/Projects/startup-predictor/data/raw/startup data.csv")
    df = clean_data(df)
    X, y = get_feature_target(df)
    print(X.shape, y.value_counts().to_dict())
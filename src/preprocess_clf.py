import numpy as np
import pandas as pd
import pycountry
import pycountry_convert as pc

TOP_CATEGORIES = [
    "Biotechnology", "Software", "Curated Web", "Advertising", "E-Commerce",
    "Mobile", "Enterprise Software", "Games", "Analytics", "Health Care",
]
CONTINENT_MAP = {
    "NA": "NorthAmerica", "SA": "SouthAmerica", "EU": "Europe",
    "AS": "Asia", "AF": "Africa", "OC": "Oceania",
}


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _country_to_continent(code3):
    if pd.isnull(code3):
        return "Unknown"
    try:
        country = pycountry.countries.get(alpha_3=code3)
        if country is None:
            return "Unknown"
        cont_code = pc.country_alpha2_to_continent_code(country.alpha_2)
        return CONTINENT_MAP.get(cont_code, "Unknown")
    except Exception:
        return "Unknown"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["status"].isin(["closed", "acquired", "ipo"])].copy()
    df["target"] = df["status"].isin(["acquired", "ipo"]).astype(int)

    # Funding amount: "-" means unreported
    df["funding_total_usd"] = df["funding_total_usd"].replace("-", np.nan).astype(float)
    df["funding_missing"] = df["funding_total_usd"].isnull().astype(int)
    median_funding = df["funding_total_usd"].median()
    df["funding_total_usd"] = df["funding_total_usd"].fillna(median_funding)
    df["funding_total_usd_log"] = np.log1p(df["funding_total_usd"])

    for c in ["founded_at", "first_funding_at", "last_funding_at"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["age_first_funding_year"] = (df["first_funding_at"] - df["founded_at"]).dt.days / 365.25
    df["age_last_funding_year"] = (df["last_funding_at"] - df["founded_at"]).dt.days / 365.25

    df["age_missing"] = df["age_first_funding_year"].isnull().astype(int)
    for c in ["age_first_funding_year", "age_last_funding_year"]:
        df[c] = df[c].fillna(df[c].median())

    # Time between first and last round -- distinct signal from raw age,
    # replaces age_first_funding_year to fix multicollinearity (VIF 85+)
    df["funding_gap"] = (df["age_last_funding_year"] - df["age_first_funding_year"]).clip(lower=0)
    df["funding_velocity"] = df["funding_rounds"] / df["age_last_funding_year"].clip(lower=0.25)

    # Continent
    df["continent"] = df["country_code"].apply(_country_to_continent)
    for cont in ["NorthAmerica", "Europe", "Asia", "SouthAmerica", "Oceania", "Africa"]:
        df[f"is_{cont.lower()}"] = (df["continent"] == cont).astype(int)

    # Category, top N + Other
    df["category_list"] = df["category_list"].fillna("Unknown")
    df["primary_category"] = df["category_list"].str.split("|").str[0]
    for cat in TOP_CATEGORIES:
        col = "is_" + cat.lower().replace(" ", "").replace("-", "")
        df[col] = (df["primary_category"] == cat).astype(int)
    df["is_othercategory"] = (~df["primary_category"].isin(TOP_CATEGORIES)).astype(int)

    feature_cols = (
        ["funding_rounds", "funding_total_usd_log", "funding_missing",
         "age_last_funding_year", "funding_gap", "age_missing",
         "funding_velocity"]
        + [f"is_{c.lower()}" for c in ["NorthAmerica", "Europe", "Asia", "SouthAmerica", "Oceania", "Africa"]]
        + ["is_" + cat.lower().replace(" ", "").replace("-", "") for cat in TOP_CATEGORIES]
        + ["is_othercategory", "target"]
    )
    return df[feature_cols]


def get_feature_target(df: pd.DataFrame):
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


if __name__ == "__main__":
    df = load_raw_data("data/raw/big_startup_secsees_dataset.csv")
    df = clean_data(df)
    X, y = get_feature_target(df)
    print(X.shape, y.value_counts().to_dict())
from pathlib import Path
import numpy as np
import pandas as pd

TOP_CATEGORIES = [
    "Biotechnology",
    "Software",
    "Curated Web",
    "Advertising",
    "E-Commerce",
    "Mobile",
    "Enterprise Software",
    "Games",
    "Analytics",
    "Health Care",
]

CONTINENT_MAP = {
    "north america": "NorthAmerica",
    "europe": "Europe",
    "asia": "Asia",
    "south america": "SouthAmerica",
    "oceania": "Oceania",
    "africa": "Africa",
}

INDUSTRY_MAPPING = {
    "Internet software & services": "Software",
    "Fintech": "Software",
    "Artificial intelligence": "Analytics",
    "Artificial Intelligence": "Analytics",
    "E-commerce & direct-to-consumer": "E-Commerce",
    "Consumer & retail": "E-Commerce",
    "Health": "Health Care",
    "Data management & analytics": "Analytics",
    "Mobile & telecommunications": "Mobile",
    "Cybersecurity": "Enterprise Software",
    "Edtech": "Curated Web",
    "Supply chain, logistics, & delivery": "Other",
    "Hardware": "Other",
    "Auto & transportation": "Other",
    "Travel": "Other",
    "Other": "Other",
}


def parse_currency(val) -> float:
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip().replace("$", "").replace(",", "")
    if not val_str or val_str == "-":
        return np.nan
    multiplier = 1.0
    if val_str.endswith(("B", "b")):
        multiplier, val_str = 1e9, val_str[:-1]
    elif val_str.endswith(("M", "m")):
        multiplier, val_str = 1e6, val_str[:-1]
    elif val_str.endswith(("K", "k")):
        multiplier, val_str = 1e3, val_str[:-1]
    try:
        return float(val_str) * multiplier
    except ValueError:
        return np.nan


def load_raw_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-transform target valuation with 99th percentile capping
    df["valuation_usd"] = df["Valuation"].apply(parse_currency)
    val_cap = df["valuation_usd"].quantile(0.99)
    df["valuation_usd_winsorized"] = df["valuation_usd"].clip(upper=val_cap)
    df["valuation_usd_log"] = np.log1p(
        df["valuation_usd_winsorized"].clip(lower=0)
    )

    # Industry mapping & hot encoding
    df["industry_clean"] = (
        df["Industry"].fillna("Other").astype(str).str.strip()
    )
    df["mapped_category"] = (
        df["industry_clean"].map(INDUSTRY_MAPPING).fillna("Other")
    )
    for cat in TOP_CATEGORIES:
        col_name = "is_" + cat.lower().replace(" ", "").replace("-", "")
        df[col_name] = (df["mapped_category"] == cat).astype(int)
    df["is_othercategory"] = (df["mapped_category"] == "Other").astype(int)

    # Continent encoding
    def standardize_continent(cont):
        if pd.isnull(cont):
            return "Unknown"
        return CONTINENT_MAP.get(str(cont).strip().lower(), "Unknown")

    df["continent_clean"] = df["Continent"].apply(standardize_continent)
    continents = [
        "NorthAmerica",
        "Europe",
        "Asia",
        "SouthAmerica",
        "Oceania",
        "Africa",
    ]
    for cont in continents:
        df[f"is_{cont.lower()}"] = (df["continent_clean"] == cont).astype(int)

    # Years to unicorn status calculation
    df["founded_year_clean"] = pd.to_numeric(
        df["Year Founded"].astype(str).str.extract(r"(\d{4})")[0],
        errors="coerce",
    )
    df["age_missing"] = df["founded_year_clean"].isnull().astype(int)
    df["founded_year_clean"] = df["founded_year_clean"].fillna(
        df["founded_year_clean"].median()
    )

    df["date_joined_clean"] = pd.to_datetime(
        df["Date Joined"], errors="coerce"
    )
    joined_year = df["date_joined_clean"].dt.year.fillna(
        df["date_joined_clean"].dt.year.median()
    )
    df["years_to_unicorn"] = (joined_year - df["founded_year_clean"]).clip(
        lower=0
    )

    # Funding & Intensity metrics
    df["funding_usd"] = df["Funding"].apply(parse_currency)
    df["funding_missing"] = df["funding_usd"].isnull().astype(int)
    df["funding_usd"] = df["funding_usd"].fillna(df["funding_usd"].median())
    df["funding_usd_log"] = np.log1p(df["funding_usd"].clip(lower=0))

    raw_capital_intensity = df["funding_usd"] / (df["years_to_unicorn"] + 1)
    df["capital_intensity_log"] = np.log1p(
        raw_capital_intensity.clip(lower=0)
    )

    # Industry benchmarks
    df["is_na_tech"] = (
        (df["continent_clean"] == "NorthAmerica")
        & (df["mapped_category"].isin(["Software", "Analytics"]))
    ).astype(int)

    industry_means = df.groupby("mapped_category")["funding_usd_log"].transform("mean")
    df["funding_vs_industry_avg"] = df["funding_usd_log"] - industry_means
    df["is_funding_above_industry_avg"] = (
        df["funding_vs_industry_avg"] > 0
    ).astype(int)

    return df


def get_feature_target(df: pd.DataFrame):
    continents = [
        "NorthAmerica",
        "Europe",
        "Asia",
        "SouthAmerica",
        "Oceania",
        "Africa",
    ]
    
    feature_cols = (
        [
            "funding_usd_log",
            "years_to_unicorn",
            "capital_intensity_log",
            "funding_vs_industry_avg",
            "is_funding_above_industry_avg",
            "is_na_tech",
            "funding_missing",
            "age_missing",
        ]
        + [f"is_{c.lower()}" for c in continents]
        + [
            "is_" + cat.lower().replace(" ", "").replace("-", "")
            for cat in TOP_CATEGORIES
        ]
        + ["is_othercategory"]
    )

    X = df[feature_cols]
    y = df["valuation_usd_log"]
    return X, y


if __name__ == "__main__":
    df = load_raw_data("data/raw/Unicorn_Companies.csv")
    df = clean_data(df)
    X, y = get_feature_target(df)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
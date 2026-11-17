"""
ingestor.py

ingests and cleans the ASC Income Dataset
"""
import argparse
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource

def preprocess():
    """
    Preprocess by dropping columns with missing values and removing columns not used in aggregation
    """

def clean_rows(data, features, min_age=0):
    """
    """
    # ensure that all values are numeric
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # drop rows where personal income missing
    data = data.dropna(subset=['PINCP']).copy()

    # Optional population restriction
    if min_age > 0:
        data = data[data["AGEP"] >= min_age].copy()
    return data

def create_weighted_columns(data, features):
    """
    Create weighted columns
    """
    for column in set(features) - {"PWGTP", "PINCP"}:
        data[f"{column}_weighted_sum"] = (
            data[column].fillna(0) * data["PWGTP"]
        ).astype("int64")
        data[f"{column}_weight"] = data["PWGTP"].where(data[column].notna(), 0).astype("int64")
    return data

def download_and_clean_ascincome(states, year, horizon, features, min_age, weighted_columns=True, download=False, info=True):
    """
    Download and clean the ASC Income Dataset
    """
        # Download raw ACS person-level data.
    source = ACSDataSource(
        survey_year=year,
        horizon=horizon,
        survey="person",
    )
    raw = source.get_data(states=states, download=download)

    if features == "all":
        features = ["AGEP","COW","SCHL","MAR","POBP","RELP","WKHP", "OCCP","SEX","RAC1P","PINCP","PWGTP"]

    missing_columns = sorted(set(features) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Downloaded data is missing expected columns: {missing_columns}")

    data = raw[features].copy()
    rows_before = len(data)
    data = clean_rows(data, features, min_age)

    # create target (50k)
    data["income_over_50k"] = (data["PINCP"] > 50_000).astype("int8")

    if weighted_columns:
        data = create_weighted_columns(data, features)

    for column in set(features) - {"PWGTP", "PINCP"}:
        data[f"{column}_weighted_sum"] = (
            data[column].fillna(0) * data["PWGTP"]
        ).astype("int64")
        data[f"{column}_weight"] = data["PWGTP"].where(data[column].notna(), 0).astype("int64")

    # Add column for missing data
    for column in features:
        data[f"{column}_missing"] = data[column].isna().astype("int8")
        data[column] = data[column].fillna(0)

    output_columns = features + ["income_over_50k"] + [f"{c}_missing" for c in features]
    if weighted_columns:
        output_columns = output_columns + [f"{c}_weighted_sum" for c in features] + [
            f"{c}_weight" for c in features]
    
    cleaned = data[output_columns].reset_index(drop=True)

    if info:
        print(f"States: {', '.join(states)}")
        print(f"Survey: {year} ACS {horizon}")
        print(f"Rows downloaded: {rows_before:,}")
        print(f"Rows after target/age filtering: {len(cleaned):,}")
        print(f"Rows removed: {rows_before - len(cleaned):,}")
        print("\nIncome-label counts:")
        print(cleaned["income_over_50k"].value_counts().sort_index())

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Download and clean folktables ACSIncome data."
    )
    parser.add_argument(
            "--states",
            default="MA",
            help="Comma-separated state abbreviations. Default: MA",
        )
    parser.add_argument("--year", default="2024", help="ACS survey year. Default: 2024")
    parser.add_argument(
        "--horizon",
        default="1-Year",
        choices=["1-Year", "5-Year"],
        help="ACS survey period. Default: 1-Year",
    )
    parser.add_argument(
        "--output",
        default="parquet",
        help="Output as CSV or Parquet. Default: ascincome_clean.parquet",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=18,
        help="Keep records at or above this age. Use 0 to disable. Default: 18",
    )
    parser.add_argument(
            "--features",
            default="AGEP,COW,SCHL,MAR,POBP,RELP,WKHP,SEX,RAC1P,PINCP,PWGTP",
            help="Comma-separated feature names. Default: all features, excluding OCCP",
        )
    args = parser.parse_args()

    states = [state.strip().upper() for state in args.states.split(",") if state.strip()]
    if not states:
        raise ValueError("Please provide at least one state.")

    features = {feature.strip() for feature in args.features.split(",") if feature.strip()}

    cleaned = download_and_clean_ascincome(states=states,
                                           year=args.year,
                                           horizon=args.horizon, 
                                           features=features, 
                                           min_age=args.min_age)
    if args.output.lower().endswith(".csv"):
        cleaned.to_csv(args.output, index=False)
    else:
        if not args.output.lower().endswith(".parquet"):
            print("Output file does not end with .csv or .parquet. Defaulting to parquet.")
            output_path = Path("ascincome_clean.parquet")
        else:
            output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(output_path, index=False)

if __name__ == "__main__":
    main()
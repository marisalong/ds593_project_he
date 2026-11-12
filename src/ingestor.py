"""
ingestor.py

ingests and cleans the ASC Income Dataset
"""
import argparse
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource


def download_and_clean_ascincome(states, year, horizon, features, min_age, info=True):
    """
    Download and clean the ASC Income Dataset
    """
        # Download raw ACS person-level data.
    source = ACSDataSource(
        survey_year=year,
        horizon=horizon,
        survey="person",
    )
    raw = source.get_data(states=states, download=True)

    if features == "all":
        features = ["AGEP","COW","SCHL","MAR","POBP","RELP","WKHP", "OCCP","SEX","RAC1P","PINCP","PWGTP"]

    missing_columns = sorted(set(features) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Downloaded data is missing expected columns: {missing_columns}")

    # Convert relevant columns to numeric. Non-numeric or missing values become NaN.
    data = raw[features].copy()
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    rows_before = len(data)

    # A target cannot be created when personal income is missing.
    data = data.dropna(subset=['PINCP']).copy()

    # Optional population restriction. This is explicit so you can change it.
    if min_age > 0:
        data = data[data["AGEP"] >= min_age].copy()

    # Create the binary ACSIncome label.
    data["income_over_50k"] = (data["PINCP"] > 50_000).astype("int8")

    # Create weight columns
    for column in set(features) - {"PWGTP", "PINCP"}:
        data[f"{column}_weighted_sum"] = (
            data[column].fillna(0) * data["PWGTP"]
        ).astype("int64")
        data[f"{column}_weight"] = data["PWGTP"].where(data[column].notna(), 0).astype("int64")

    # Preserve feature missingness explicitly. Fill missing feature values with 0
    # only after adding a separate missingness indicator.
    for column in features:
        data[f"{column}_missing"] = data[column].isna().astype("int8")
        data[column] = data[column].fillna(0)

    output_columns = features + [f"{c}_missing" for c in features] + [
        "income_over_50k"
    ]
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
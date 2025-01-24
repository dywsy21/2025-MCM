import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Read CSVs (#file: summerOly_athletes.csv, #file: summerOly_hosts.csv, etc.)
    df_athletes = pd.read_csv("data/summerOly_athletes.csv")
    df_hosts = pd.read_csv("data/summerOly_hosts.csv")
    df_medals = pd.read_csv("data/summerOly_medal_counts.csv")
    df_programs = pd.read_csv("data/summerOly_programs.csv")

    # Merge/join as needed
    df_merged = df_medals.merge(df_hosts, how="left", on="Year")
    df_merged = df_merged.merge(df_athletes, how="left", on=["NOC", "Year"])

    # Create feature columns and target columns (gold medals, total medals)
    df_merged["host_advantage"] = (df_merged["Host_NOC"] == df_merged["NOC"]).astype(int)
    y_gold = df_merged["Gold"]
    y_total = df_merged["Total"]
    X = df_merged[["Year", "host_advantage"]]
    y = pd.concat([y_gold, y_total], axis=1)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test


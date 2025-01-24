import pandas as pd
import numpy as np

def process_data():
    # Read data
    df_medals = pd.read_csv("data/provided_data/summerOly_medal_counts.csv")
    df_hosts = pd.read_csv("data/provided_data/summerOly_hosts.csv")
    df_athletes = pd.read_csv("data/provided_data/summerOly_athletes.csv")
    
    # Get unique NOC-Sport combinations and merge with medals data
    df_athletes_sports = df_athletes[["NOC", "Sport"]].drop_duplicates()
    
    # Modify df_sport_noc to count team medals only once
    df_sport_noc = df_athletes[["NOC", "Sport", "Year", "Medal", "Event"]].dropna(subset=["Medal"])
    # Count medals by type for each NOC-Sport-Year-Event combination (counting each event only once)
    medal_counts = df_sport_noc.groupby(["NOC", "Sport", "Year", "Event", "Medal"]).size().reset_index()
    # Now count the number of events (medals) for each NOC-Sport-Year-Medal combination
    medal_counts = medal_counts.groupby(["NOC", "Sport", "Year", "Medal"]).size().reset_index(name="Count")
    
    # Pivot to get separate columns for each medal type
    medal_counts_pivot = medal_counts.pivot_table(
        index=["NOC", "Sport", "Year"],
        columns="Medal",
        values="Count",
        fill_value=0
    ).reset_index()
    
    # Rename columns to match original format
    medal_counts_pivot = medal_counts_pivot.rename(columns={
        "Gold": "Gold",
        "Silver": "Silver",
        "Bronze": "Bronze"
    })
    
    # Create result dataframe
    result_df = pd.DataFrame()
    years = sorted(medal_counts_pivot["Year"].unique())
    
    for year in years:
        year_data = medal_counts_pivot[medal_counts_pivot["Year"] == year]
        
        # Add medal columns for this year
        for medal_type in ["Gold", "Silver", "Bronze"]:
            temp_df = pd.merge(
                df_athletes_sports,
                year_data[["NOC", "Sport", medal_type]],
                on=["NOC", "Sport"],
                how="left"
            )
            result_df[f"{medal_type}_{year}"] = temp_df[medal_type].fillna(0)
        
        # Add host country column
        host = df_hosts[df_hosts["Year"] == year]["Host"].iloc[0] if year in df_hosts["Year"].values else ""
        result_df[f"Host_country_{year}"] = host

    # Add NOC and Sport columns back
    result_df = pd.concat([df_athletes_sports, result_df], axis=1)
    
    # Save output
    result_df.to_csv("data/generated_training_data/sport_oriented/out.csv", index=False)

if __name__ == "__main__":
    process_data()

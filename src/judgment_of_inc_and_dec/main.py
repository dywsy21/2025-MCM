# 2024 medal counts for each country: 2024_count.csv
# 2028 projected medal counts for each country: 2028_pre_not_rounded.csv

IMPROVEMENT_THRESHOLD = 15

def judge_changes_2024_2028(csv_2024, csv_2028, country_noc_map, threshold=IMPROVEMENT_THRESHOLD):
    import pandas as pd

    # Load country to NOC mapping
    country_noc = pd.read_csv(country_noc_map)
    # Some countries have multiple NOC codes separated by semicolons - take first one
    country_noc['NOC'] = country_noc['NOC'].str.split(';').str[0]
    country_noc_dict = dict(zip(country_noc['Country'], country_noc['NOC']))

    # Load and process 2024 data
    df_2024 = pd.read_csv(csv_2024)
    df_2024['NOC'] = df_2024['Country'].map(country_noc_dict)
    df_2024 = df_2024.dropna(subset=['NOC'])  # Drop rows where country couldn't be mapped to NOC
    df_2024['NOC'] = df_2024['NOC'].astype(str).str.strip()

    # Load and process 2028 data  
    df_2028 = pd.read_csv(csv_2028)
    df_2028['NOC'] = df_2028['NOC'].astype(str).str.strip()
    df_2028 = df_2028[['NOC','B','S','G','Tot']]
    df_2028 = df_2028.rename(columns={'B':'Bronze','S':'Silver','G':'Gold','Tot':'Total'})

    # Merge on NOC
    merged = pd.merge(df_2024, df_2028, on='NOC', how='inner', suffixes=('_2024','_2028'))
    
    print(f"Successfully matched {len(merged)} countries")
    print(merged[['NOC', 'Country', 'Gold_2024', 'Silver_2024', 'Bronze_2024', 
                 'Gold_2028', 'Silver_2028', 'Bronze_2028']])

    # Create results dataframe
    results = []

    for _, row in merged.iterrows():
        noc = row['NOC']
        gold_24, silver_24, bronze_24 = row['Gold_2024'], row['Silver_2024'], row['Bronze_2024']
        gold_28, silver_28, bronze_28 = row['Gold_2028'], row['Silver_2028'], row['Bronze_2028']
        total_24, total_28 = row['Total_2024'], row['Total_2028']
        
        print(f"Checking {noc}...: {gold_24}, {silver_24}, {bronze_24} -> {gold_28}, {silver_28}, {bronze_28}")
        print(f"Total: {total_24} -> {total_28}\n")

        # Check percentage difference
        improvement_cond1 = (total_24 > 0) and ((total_28 - total_24) / total_24 > threshold/100)
        decline_cond1 = (total_24 > 0) and ((total_24 - total_28) / total_24 > threshold/100)

        # Check no decrease & at least one increase
        improvement_cond2 = (gold_28 >= gold_24 and silver_28 >= silver_24 and bronze_28 >= bronze_24) and \
                            (gold_28 > gold_24 or silver_28 > silver_24 or bronze_28 > bronze_24)

        # Check no increase & at least one decrease
        decline_cond2 = (gold_28 <= gold_24 and silver_28 <= silver_24 and bronze_28 <= bronze_24) and \
                        (gold_28 < gold_24 or silver_28 < silver_24 or bronze_28 < bronze_24)

        # Determine change and reason
        if improvement_cond1 or improvement_cond2:
            change = 'Inc'
            reason = '1' if improvement_cond1 else '2'
            if improvement_cond1 and improvement_cond2:
                reason = '1, 2'
        elif decline_cond1 or decline_cond2:
            change = 'Dec'
            reason = '1' if decline_cond1 else '2'
            if decline_cond1 and decline_cond2:
                reason = '1, 2'
        else:
            change = 'None'
            reason = ''

        results.append({
            'NOC': noc,
            'Change': change,
            'Reason': reason
        })

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['Change'] != 'None']  # Remove unchanged countries
    
    # Sort results - 'Inc' first, then 'Dec'
    results_df['Sort'] = results_df['Change'].map({'Inc': 0, 'Dec': 1})
    results_df = results_df.sort_values(['Sort', 'NOC']).drop('Sort', axis=1)
    
    results_df.to_csv('results_changes.csv', index=False)
    
    return results_df

if __name__ == '__main__':
    results = judge_changes_2024_2028(
        "2024_count.csv",
        "2028_pre_not_rounded.csv",
        "data/generated_training_data/misc/Country_NOC.csv"
    )
    print("\nResults saved to results_changes.csv")
    print(results)

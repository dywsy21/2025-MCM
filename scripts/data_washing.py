# 1. store city -> NOC dict from data/generated_training_data/City_NOC.csv
print(1)
import pandas as pd
from sympy import idiff
city_noc_df = pd.read_csv("data/generated_training_data/City_NOC.csv")
city_noc_dict = dict(zip(city_noc_df["City"], city_noc_df["NOC"]))

# 2. store event -> team people count from data/generated_training_data/Events.csv
print(2)
events_df = pd.read_csv("data/generated_training_data/Events_no_comma.csv")
event_team_people_count = dict(zip(events_df["Event"], events_df["Team_size"]))

# 3. The ultimate format is id | Gold_1896 | Silver_1896 | Bronze_1896 | Host_NOC_1896 | ... | Gold_2024 | Silver_2024 | Bronze_2024 | Host_NOC_2024 | NOC | First_Year_In_Match, create these columns first and fill with 0
print(3)
years = range(1896, 2025, 4)
columns = ["id"] + [f"{medal}_{year}" for year in years for medal in ["Gold", "Silver", "Bronze", "Host_NOC"]] + ["NOC", "First_Year_In_Match"]
data_df = pd.DataFrame(columns=columns)
# fill data_df with 0
# data_df["id"] = range(1, 1 + len(city_noc_dict) * len(events_df))

# 4. Read data from data/generated_training_data/summerOly_athletes_ID_nocomma.csv, extract ID column and fill it in the id column of data_df
print(4)
athletes_df = pd.read_csv("data/generated_training_data/summerOly_athletes_IDascend_nocomma.csv")
data_df["id"] = athletes_df["ID"]

# 5. Read data from data/generated_training_data/summerOly_athletes_ID_nocomma.csv, extract Medal column. The medal column is in the format of "Gold", "Silver", "Bronze", "No medal". Fill the corresponding columns in data_df with 1
print(5)
for index, row in athletes_df.iterrows():
    if index % 1000 == 0:
        print(index)
    year = row["Year"]
    medal = row["Medal"]
    name, event = row["ID"].split("_")
    # row["ID"] = f"{name}_{event}"
    if medal in ["Gold", "Silver", "Bronze"]:
        data_df.at[index, f"{medal}_{year}"] = 1.0 / event_team_people_count[event]

# 6. Read data from data/generated_training_data/summerOly_athletes_ID_nocomma.csv, extract NOC column. Fill the NOC column in data_df with the corresponding NOC, check if id is the same. use loc
print(6)
data_df["NOC"] = athletes_df["NOC"]

# 7. Read data from data/generated_training_data/ID_FirstYear.csv (two columns: id, First_Year_In_Match), extract First_Year_In_Match column. Fill the First_Year_In_Match column in data_df with the correct id.
print(7)
first_year_df = pd.read_csv("data/generated_training_data/ID_FirstYear.csv")
# generate a dict to map id to first year
id_first_year_dict = {k.lower(): v for k, v in zip(first_year_df["ID"], first_year_df["First_Year_In_Match"])}
# fill the First_Year_In_Match column in data_df
# data_df["First_Year_In_Match"] = data_df["id"].map(id_first_year_dict)
for i, row in data_df.iterrows():
    data_df.at[i, "First_Year_In_Match"] = id_first_year_dict[row["id"].lower()]

# 8. Save the data_df to data/generated_training_data/ult.csv
print(8)
data_df.to_csv("data/generated_training_data/ult.csv", index=False)

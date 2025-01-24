import pandas as pd

# Load Event -> Team_size dictionary
events_df = pd.read_csv('data/generated_training_data/misc/Events_no_comma.csv')
event_team_size = dict(zip(events_df['Event'], events_df['Team_size']))

# Load athletes data
athletes_df = pd.read_csv('data/generated_training_data/misc/summerOly_athletes_ID_nocomma.csv')

# Ensure every event is present in the dictionary
missing_events = set(athletes_df['Event']) - set(event_team_size.keys())
if missing_events:
    raise ValueError(f"Missing team size information for events: {missing_events}")

# Add Team_size column
athletes_df['Team_size'] = athletes_df['Event'].map(event_team_size)

# Save the updated dataframe to a new CSV file
athletes_df.to_csv('data/generated_training_data/misc/summerOly_athletes_ID_nocomma_with_team_size.csv', index=False)

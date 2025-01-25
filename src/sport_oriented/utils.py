import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the Olympics data"""
    df = pd.read_csv(filepath)
    # Convert Year to int type and handle any non-numeric values
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    # Drop any rows with NaN values
    df = df.dropna()
    # Remove rows where Host is '#N/A'
    df = df[df['Host'] != '#N/A']
    # Aggregate medals by country and year
    medal_counts = df.groupby(['NOC', 'Year', 'Sport'])[['Bronze', 'Silver', 'Gold']].sum().reset_index()
    return medal_counts

def create_features(df, target_year, n_matches):
    """Create features for each country based on historical performance"""
    features = []
    labels = []
    
    # Get unique years and sort them as numpy array
    years = np.sort(df['Year'].unique().astype(int))
    # Get training years
    mask = years < target_year
    training_years = years[mask][-n_matches:]
    
    for country in df['NOC'].unique():
        country_data = df[df['NOC'] == country]
        
        # Create features for target year - 4
        if len(training_years) >= n_matches:
            past_data = {year: country_data[country_data['Year'] == year] for year in training_years}
            
            if all(len(past_data[year]) > 0 for year in training_years):
                feature_row = {
                    'NOC': country,
                    'Year': target_year - 4
                }
                
                # Add historical medal counts
                for i, year in enumerate(training_years):
                    year_data = past_data[year]
                    feature_row.update({
                        f'bronze_{i}': year_data['Bronze'].sum(),
                        f'silver_{i}': year_data['Silver'].sum(),
                        f'gold_{i}': year_data['Gold'].sum()
                    })
                    
                # Add sport-specific features
                for sport in df['Sport'].unique():
                    sport_medals = [past_data[year][past_data[year]['Sport'] == sport][['Bronze', 'Silver', 'Gold']].sum() 
                                  for year in training_years]
                    if any(medal.sum() > 0 for medal in sport_medals):
                        feature_row.update({
                            f'{sport}_bronze': np.mean([m['Bronze'] for m in sport_medals]),
                            f'{sport}_silver': np.mean([m['Silver'] for m in sport_medals]),
                            f'{sport}_gold': np.mean([m['Gold'] for m in sport_medals])
                        })
                
                features.append(feature_row)
                
                # Add labels (actual medals for target_year - 4)
                actual = country_data[country_data['Year'] == target_year - 4][['Bronze', 'Silver', 'Gold']].sum()
                labels.append([actual['Bronze'], actual['Silver'], actual['Gold']])
    
    # Add additional NaN handling in feature creation
    features = pd.DataFrame(features)
    labels = np.array(labels)
    
    # Drop any rows with NaN values
    nan_mask = ~np.isnan(labels).any(axis=1)
    features = features[nan_mask].reset_index(drop=True)
    labels = labels[nan_mask]
    
    # Fill any remaining NaN values in features with 0
    features = features.fillna(0)
    
    return features, labels

def prepare_prediction_features(df, target_year, n_matches):
    """Prepare features for prediction"""
    features = []
    # Get unique years and sort them as numpy array
    years = np.sort(df['Year'].unique().astype(int))
    # Get training years
    mask = years < target_year
    training_years = years[mask][-n_matches:]
    
    for country in df['NOC'].unique():
        country_data = df[df['NOC'] == country]
        past_data = {year: country_data[country_data['Year'] == year] for year in training_years}
        
        if all(len(past_data[year]) > 0 for year in training_years):
            feature_row = {
                'NOC': country,
                'Year': target_year
            }
            
            # Add historical features using same logic as create_features
            for i, year in enumerate(training_years):
                year_data = past_data[year]
                feature_row.update({
                    f'bronze_{i}': year_data['Bronze'].sum(),
                    f'silver_{i}': year_data['Silver'].sum(),
                    f'gold_{i}': year_data['Gold'].sum()
                })
                
            for sport in df['Sport'].unique():
                sport_medals = [past_data[year][past_data[year]['Sport'] == sport][['Bronze', 'Silver', 'Gold']].sum() 
                              for year in training_years]
                if any(medal.sum() > 0 for medal in sport_medals):
                    feature_row.update({
                        f'{sport}_bronze': np.mean([m['Bronze'] for m in sport_medals]),
                        f'{sport}_silver': np.mean([m['Silver'] for m in sport_medals]),
                        f'{sport}_gold': np.mean([m['Gold'] for m in sport_medals])
                    })
                    
            features.append(feature_row)
    
    # Create DataFrame and handle NaN values
    features = pd.DataFrame(features)
    
    # Fill NaN values with 0
    features = features.fillna(0)
    
    return pd.DataFrame(features)

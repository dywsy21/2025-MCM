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

def get_all_features(df, n_matches):
    """Get all possible feature names"""
    feature_names = ['NOC', 'Year']
    
    # Add historical medal count features
    for i in range(n_matches):
        feature_names.extend([f'bronze_{i}', f'silver_{i}', f'gold_{i}'])
    
    # Add sport-specific features for all sports
    for sport in df['Sport'].unique():
        feature_names.extend([
            f'{sport}_bronze',
            f'{sport}_silver',
            f'{sport}_gold'
        ])
    
    return feature_names

def ensure_consistent_features(features_df, all_features):
    """Ensure dataframe has all expected features, filling missing ones with 0"""
    for feature in all_features:
        if feature not in features_df.columns and feature not in ['NOC', 'Year']:
            features_df[feature] = 0
    return features_df[all_features]

def create_features(df, target_year, n_matches):
    """Create features for each country based on historical performance"""
    # Get all possible feature names first
    all_features = get_all_features(df, n_matches)
    
    features = []
    labels = []
    
    # Get unique years and sort them as numpy array
    years = np.sort(df['Year'].unique().astype(int))
    
    # Ensure we only use n_matches most recent years
    if len(years) > n_matches + 1:  # +1 for evaluation year
        years = years[-(n_matches+1):]
    
    # Get training years (all except last year which is evaluation year)
    training_years = years[:-1]
    evaluation_year = years[-1]
    
    if len(training_years) != n_matches:
        print(f"Warning: Only found {len(training_years)} years of training data, expected {n_matches}")
    
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
    features = ensure_consistent_features(features, all_features)
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
    # Get all possible feature names first
    all_features = get_all_features(df, n_matches)
    
    features = []
    
    # Get unique years and sort them as numpy array
    years = np.sort(df['Year'].unique().astype(int))
    
    # Ensure we only use n_matches most recent years
    if len(years) > n_matches:
        years = years[-n_matches:]
        
    training_years = years
    
    if len(training_years) != n_matches:
        print(f"Warning: Only found {len(training_years)} years of training data, expected {n_matches}")
    
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
    features = ensure_consistent_features(features, all_features)
    
    # Fill NaN values with 0
    features = features.fillna(0)
    
    return pd.DataFrame(features)

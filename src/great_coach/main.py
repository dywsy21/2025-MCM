# Brief: This file contains the pattern identification code of the great coach effect

# The Great Coach Effect is a phenomenon where the presence of a highly skilled coach significantly improves the performance of athletes. Due to the fact that coaches are not bound to a specific country, they can have a significant impact on the medal counts of multiple nations. Identifying the Great Coach Effect and quantifying its impact on medal counts is crucial for predicting Olympic success.

# The pattern to be identified: One country having a sharp increase in their medal count in specific sports, and one other country to go through a gradual-to-sharp decrease (to specify, use linear regression on the data through no more than 12 years and find if k value < K_THRESHOLD) in their medal count in the same sports. This is because the coach is likely to focus on one country at a time, and the athletes from the other country would not receive the same level of training and support.

# dataset path: data/generated_training_data/sport_oriented/Training_data.csv

import csv
from config import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def plot_coach_effect(pairs, title="Great Coach Effect Analysis"):
    G = nx.DiGraph()
    
    edge_labels = {}
    for pair in pairs:
        source_country = pair[0]
        target_country = pair[1]
        sport = pair[2]
        years = f"{pair[5]}-{pair[6]}"
        
        G.add_node(source_country)
        G.add_node(target_country)
        
        edge_key = (source_country, target_country)
        if edge_key in edge_labels:
            edge_labels[edge_key] += f"\n{sport}\n({years})"
        else:
            edge_labels[edge_key] = f"{sport}\n({years})"
            G.add_edge(source_country, target_country)

    plt.figure(figsize=(20, 20))
    
    pos = nx.circular_layout(G, scale=2)
    
    # Draw nodes with increased size and alpha
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=3000, alpha=0.8)
    
    # Draw curved edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20,
                          connectionstyle='arc3, rad=0.2')  # Add curve to edges
    
    # Draw node labels with larger font
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge labels with adjusted position for curved edges
    edge_labels_pos = {}
    for (node1, node2), label in edge_labels.items():
        # Calculate the midpoint with offset for the curved edge
        p1 = pos[node1]
        p2 = pos[node2]
        edge_labels_pos[(node1, node2)] = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                font_size=8, 
                                bbox=dict(facecolor='white', 
                                        edgecolor='none',
                                        alpha=0.7))
    
    plt.title(title, pad=20, size=16, fontweight='bold')
    plt.axis('off')
    
    # Add legend - moved to bottom right with adjusted appearance
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', alpha=0.8, 
              label='Country'),
        Patch(facecolor='white', edgecolor='gray', 
              label='Potential Coach Movement / Advantage Transfer')
    ]
    plt.legend(handles=legend_elements, 
              loc='lower right',  # Changed from 'center' to 'lower right'
              fontsize=12,
              bbox_to_anchor=(0.98, 0.02),  # Fine-tune position
              framealpha=0.9,  # Make legend background more visible
              edgecolor='gray')  # Add border to legend box
    
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", 
                bbox_inches='tight', dpi=300)
    plt.show()

def analyze_coach_effect_impact(df, pairs, granularity='sport'):
    """Analyze the impact of coach effect across different tiers"""
    # Calculate total medals for each country
    country_totals = df.groupby('NOC')[['Gold', 'Silver', 'Bronze']].sum()
    country_totals['TotalMedals'] = country_totals['Gold']*3 + country_totals['Silver']*2 + country_totals['Bronze']
    
    # Initialize data structures for analysis
    tier_impacts = {i: {'increases': [], 'decreases': [], 
                       'relative_increases': [], 'relative_decreases': []} 
                   for i in range(1, 6)}
    
    # Calculate impacts for each pair
    for pair in pairs:
        inc_country, dec_country = pair[0], pair[1]
        sport_or_event = pair[2]
        overlap_start, overlap_end = pair[5], pair[6]
        
        # Calculate absolute changes
        inc_data = df[(df['NOC'] == inc_country) & 
                     (df[granularity] == sport_or_event) & 
                     (df['Year'] >= overlap_start) & 
                     (df['Year'] <= overlap_end)]
        dec_data = df[(df['NOC'] == dec_country) & 
                     (df[granularity] == sport_or_event) & 
                     (df['Year'] >= overlap_start) & 
                     (df['Year'] <= overlap_end)]
        
        inc_change = inc_data['TotalMedals'].mean()
        dec_change = -dec_data['TotalMedals'].mean()  # Negative to represent decrease
        
        # Calculate relative changes
        if inc_country in country_totals.index:
            inc_relative = inc_change / country_totals.loc[inc_country, 'TotalMedals'] * 100
        else:
            inc_relative = 0
            
        if dec_country in country_totals.index:
            dec_relative = dec_change / country_totals.loc[dec_country, 'TotalMedals'] * 100
        else:
            dec_relative = 0
        
        # Add to appropriate tier
        inc_tier = df[df['NOC'] == inc_country]['Tier'].iloc[0]
        dec_tier = df[df['NOC'] == dec_country]['Tier'].iloc[0]
        
        tier_impacts[inc_tier]['increases'].append(inc_change)
        tier_impacts[dec_tier]['decreases'].append(dec_change)
        tier_impacts[inc_tier]['relative_increases'].append(inc_relative)
        tier_impacts[dec_tier]['relative_decreases'].append(dec_relative)
    
    # Calculate averages
    avg_impacts = {i: {
        'avg_increase': np.mean(data['increases']) if data['increases'] else 0,
        'avg_decrease': np.mean(data['decreases']) if data['decreases'] else 0,
        'avg_relative_increase': np.mean(data['relative_increases']) if data['relative_increases'] else 0,
        'avg_relative_decrease': np.mean(data['relative_decreases']) if data['relative_decreases'] else 0
    } for i, data in tier_impacts.items()}
    
    # Plot results
    plot_tier_impacts(avg_impacts, granularity)
    
    return avg_impacts

def plot_tier_impacts(avg_impacts, granularity):
    """Plot the average impacts across tiers and save to CSV"""
    tiers = list(avg_impacts.keys())
    increases = [avg_impacts[t]['avg_increase'] for t in tiers]
    decreases = [avg_impacts[t]['avg_decrease'] for t in tiers]
    rel_increases = [avg_impacts[t]['avg_relative_increase'] for t in tiers]
    rel_decreases = [avg_impacts[t]['avg_relative_decrease'] for t in tiers]
    
    # Save to CSV
    impact_df = pd.DataFrame({
        'Tier': tiers,
        'Average_Increase': increases,
        'Average_Decrease': decreases,
        'Relative_Increase_Percent': rel_increases,
        'Relative_Decrease_Percent': rel_decreases
    })
    impact_df.to_csv(f'tier_impacts_{granularity}.csv', index=False)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(tiers))
    width = 0.35
    
    # Absolute changes plot
    ax1.bar(x - width/2, increases, width, label='Average Increase', color='green', alpha=0.6)
    ax1.bar(x + width/2, decreases, width, label='Average Decrease', color='red', alpha=0.6)
    ax1.set_ylabel('Absolute Medal Count Change')
    ax1.set_title(f'Absolute Impact by Tier ({granularity} level)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Tier {t}' for t in tiers])
    ax1.legend()
    
    # Relative changes plot
    ax2.bar(x - width/2, rel_increases, width, label='Relative Increase (%)', color='green', alpha=0.6)
    ax2.bar(x + width/2, rel_decreases, width, label='Relative Decrease (%)', color='red', alpha=0.6)
    ax2.set_ylabel('Relative Change (%)')
    ax2.set_title(f'Relative Impact by Tier ({granularity} level)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Tier {t}' for t in tiers])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'coach_effect_impact_{granularity}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tier_impacts_from_csv(csv_path, granularity):
    """Plot the average impacts across tiers from CSV"""
    impact_df = pd.read_csv(csv_path)
    
    tiers = impact_df['Tier']
    increases = impact_df['Average_Increase']
    decreases = impact_df['Average_Decrease']
    rel_increases = impact_df['Relative_Increase_Percent']
    rel_decreases = impact_df['Relative_Decrease_Percent']
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(tiers))
    width = 0.35
    
    # Absolute changes plot
    ax1.bar(x - width/2, increases, width, label='Average Increase', color='green', alpha=0.6)
    ax1.bar(x + width/2, decreases, width, label='Average Decrease', color='red', alpha=0.6)
    ax1.set_ylabel('Absolute Medal Count Change')
    ax1.set_title(f'Absolute Impact by Tier ({granularity} level)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Tier {t}' for t in tiers])
    ax1.legend()
    
    # Relative changes plot
    ax2.bar(x - width/2, rel_increases, width, label='Relative Increase (%)', color='green', alpha=0.6)
    ax2.bar(x + width/2, rel_decreases, width, label='Relative Decrease (%)', color='red', alpha=0.6)
    ax2.set_ylabel('Relative Change (%)')
    ax2.set_title(f'Relative Impact by Tier ({granularity} level)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Tier {t}' for t in tiers])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'coach_effect_impact_{granularity}_from_csv.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_great_coach_effect_by_sport(csv_path):
    df = pd.read_csv(csv_path)
    # Weighted medal calculation: Gold=3, Silver=2, Bronze=1
    df['TotalMedals'] = df['Gold'] * 3 + df['Silver'] * 2 + df['Bronze']
    grouped = df.groupby(['NOC','Sport','Year'])['TotalMedals'].sum().reset_index()

    possible_increases = []
    possible_decreases = []

    nocs = grouped['NOC'].unique()
    sports = grouped['Sport'].unique()

    for noc in tqdm(nocs, desc="Processing NOCs"):
        for sport in sports:
            subset = grouped[(grouped['NOC'] == noc) & (grouped['Sport'] == sport)]
            if len(subset) > 1:
                sub_sorted = subset.sort_values('Year').tail(12)
                X = sub_sorted[['Year']]
                y = sub_sorted['TotalMedals']
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                min_year, max_year = sub_sorted['Year'].min(), sub_sorted['Year'].max()
                if slope > POS_K_LOWER:
                    possible_increases.append((noc, sport, slope, min_year, max_year))
                elif slope < NEG_K_UPPER:
                    possible_decreases.append((noc, sport, slope, min_year, max_year))

    # Pair up
    pairs = []
    for inc in possible_increases:
        for dec in possible_decreases:
            if inc[1] == dec[1]:  # same sport
                overlap_start = max(inc[3], dec[3])
                overlap_end = min(inc[4], dec[4])
                if overlap_start <= overlap_end - OVERLAP_GAP_LOWER and overlap_end - overlap_start <= OVERLAP_GAP_UPPER:
                    pairs.append((inc[0], dec[0], inc[1], inc[2], dec[2], overlap_start, overlap_end))

    with open("analysis_result_by_sport.txt", "w") as f:
        f.write("Great Coach Effect Analysis Results\n\n")
        f.write("Possible Increases:\n")
        for inc in possible_increases:
            f.write(f"- NOC: {inc[0]}, Sport: {inc[1]}, Slope: {inc[2]:.3f}\n")
        f.write("\nPossible Decreases:\n")
        for dec in possible_decreases:
            f.write(f"- NOC: {dec[0]}, Sport: {dec[1]}, Slope: {dec[2]:.3f}\n")
        f.write("\nPaired Increases/Decreases:\n")
        for pair in pairs:
            f.write(f"- Sport: {pair[2]}, Overlap Years [{pair[5]}-{pair[6]}], "
                    f"Increase: {pair[0]}(slope={pair[3]:.3f}) / Decrease: {pair[1]}(slope={pair[4]:.3f})\n")

    plot_coach_effect(pairs, "Great Coach Effect Analysis (Sport Level)")

    # Add impact analysis
    avg_impacts = analyze_coach_effect_impact(df, pairs, 'Sport')
    
    # Write additional impact analysis to file
    with open("analysis_result_by_sport.txt", "a") as f:
        f.write("\nImpact Analysis by Tier:\n")
        for tier in range(1, 6):
            f.write(f"\nTier {tier}:\n")
            f.write(f"  Average Increase: {avg_impacts[tier]['avg_increase']:.2f} medals\n")
            f.write(f"  Average Decrease: {avg_impacts[tier]['avg_decrease']:.2f} medals\n")
            f.write(f"  Relative Increase: {avg_impacts[tier]['avg_relative_increase']:.2f}%\n")
            f.write(f"  Relative Decrease: {avg_impacts[tier]['avg_relative_decrease']:.2f}%\n")
    
    return possible_increases, possible_decreases, pairs, avg_impacts

def analyze_great_coach_effect_by_event(csv_path):
    df = pd.read_csv(csv_path)
    # Weighted medal calculation: Gold=3, Silver=2, Bronze=1
    df['TotalMedals'] = df['Gold'] * 3 + df['Silver'] * 2 + df['Bronze']
    grouped = df.groupby(['NOC','Event','Year'])['TotalMedals'].sum().reset_index()

    possible_increases = []
    possible_decreases = []

    nocs = grouped['NOC'].unique()
    events = grouped['Event'].unique()

    for noc in tqdm(nocs, desc="Processing NOCs"):
        for event in events:
            subset = grouped[(grouped['NOC'] == noc) & (grouped['Event'] == event)]
            if len(subset) > 1:
                sub_sorted = subset.sort_values('Year').tail(12)
                X = sub_sorted[['Year']]
                y = sub_sorted['TotalMedals']
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                min_year, max_year = sub_sorted['Year'].min(), sub_sorted['Year'].max()
                if slope > POS_K_LOWER:
                    possible_increases.append((noc, event, slope, min_year, max_year))
                elif slope < NEG_K_UPPER:
                    possible_decreases.append((noc, event, slope, min_year, max_year))

    # Pair up
    pairs = []
    for inc in possible_increases:
        for dec in possible_decreases:
            if inc[1] == dec[1]:  # same event
                overlap_start = max(inc[3], dec[3])
                overlap_end = min(inc[4], dec[4])
                if overlap_start <= overlap_end and overlap_end - overlap_start <= OVERLAP_GAP_UPPER:
                    pairs.append((inc[0], dec[0], inc[1], inc[2], dec[2], overlap_start, overlap_end))

    with open("analysis_result_by_event.txt", "w") as f:
        f.write("Great Coach Effect Analysis Results (Event Level)\n\n")
        f.write("Possible Increases:\n")
        for inc in possible_increases:
            f.write(f"- NOC: {inc[0]}, Event: {inc[1]}, Slope: {inc[2]:.3f}\n")
        f.write("\nPossible Decreases:\n")
        for dec in possible_decreases:
            f.write(f"- NOC: {dec[0]}, Event: {dec[1]}, Slope: {dec[2]:.3f}\n")
        f.write("\nPaired Increases/Decreases:\n")
        for pair in pairs:
            f.write(f"- Event: {pair[2]}, Overlap Years [{pair[5]}-{pair[6]}], "
                    f"Increase: {pair[0]}(slope={pair[3]:.3f}) / Decrease: {pair[1]}(slope={pair[4]:.3f})\n")

    plot_coach_effect(pairs, "Great Coach Effect Analysis (Event Level)")

    # Add impact analysis
    avg_impacts = analyze_coach_effect_impact(df, pairs, 'Event')
    
    # Write additional impact analysis to file
    with open("analysis_result_by_event.txt", "a") as f:
        f.write("\nImpact Analysis by Tier:\n")
        for tier in range(1, 6):
            f.write(f"\nTier {tier}:\n")
            f.write(f"  Average Increase: {avg_impacts[tier]['avg_increase']:.2f} medals\n")
            f.write(f"  Average Decrease: {avg_impacts[tier]['avg_decrease']:.2f} medals\n")
            f.write(f"  Relative Increase: {avg_impacts[tier]['avg_relative_increase']:.2f}%\n")
            f.write(f"  Relative Decrease: {avg_impacts[tier]['avg_relative_decrease']:.2f}%\n")
    
    return possible_increases, possible_decreases, pairs, avg_impacts

def calculate_and_plot_tier2_gains():
    df = pd.read_csv("data/generated_training_data/sport_oriented/Training_data_tier.csv")
    df['TotalMedals'] = df['Gold']*3 + df['Silver']*2 + df['Bronze']

    # Filter tier 2 countries
    tier2 = df[df['Tier'] == 2].copy()

    # Group by NOC and Year, sum total medals
    grouped = tier2.groupby(['NOC', 'Year'])['TotalMedals'].sum().reset_index()

    # Find earliest and latest year totals for each country
    earliest_totals = grouped.groupby('NOC').first().reset_index()
    latest_totals = grouped.groupby('NOC').last().reset_index()

    # Calculate absolute & relative gains
    results = []
    for noc in earliest_totals['NOC'].unique():
        earliest_val = earliest_totals[earliest_totals['NOC'] == noc]['TotalMedals'].values[0]
        latest_val = latest_totals[latest_totals['NOC'] == noc]['TotalMedals'].values[0]
        absolute_gain = latest_val - earliest_val
        relative_gain = (absolute_gain / earliest_val * 100) if earliest_val else 0
        results.append((noc, absolute_gain, relative_gain))

    # Sort and pick top 10 by absolute gain
    sorted_abs = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    # Sort and pick top 10 by relative gain
    sorted_rel = sorted(results, key=lambda x: x[2], reverse=True)[:10]

    # Plot top 10 absolute gains
    noc_abs = [r[0] for r in sorted_abs]
    abs_vals = [r[1] / 2 for r in sorted_abs]
    plt.figure(figsize=(10, 5))
    plt.bar(noc_abs, abs_vals, color='skyblue')
    plt.title('Top 10 Tier 2 Countries by Absolute Medal Gain')
    plt.ylabel('Absolute Gain')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tier2_top10_absolute_gain.png', dpi=300)
    plt.show()

    # Plot top 10 relative gains
    noc_rel = [r[0] for r in sorted_rel]
    rel_vals = [r[2] for r in sorted_rel]
    plt.figure(figsize=(10, 5))
    plt.bar(noc_rel, rel_vals, color='orange')
    plt.title('Top 10 Tier 2 Countries by Relative Medal Gain (%)')
    plt.ylabel('Relative Gain (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tier2_top10_relative_gain.png', dpi=300)
    plt.show()

def main(min_granularity):
    csv_path = "data/generated_training_data/sport_oriented/Training_data_tier.csv"
    if min_granularity == 'sport':
        analyze_great_coach_effect_by_sport(csv_path)
    elif min_granularity == 'event':
        analyze_great_coach_effect_by_event(csv_path)

if __name__ == "__main__":
    # main(min_granularity='sport')
    # plot_tier_impacts_from_csv('tier_impacts_Sport.csv', 'Sport')
    
    calculate_and_plot_tier2_gains()
    

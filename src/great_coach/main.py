# Brief: This file contains the pattern identification code of the great coach effect

# The Great Coach Effect is a phenomenon where the presence of a highly skilled coach significantly improves the performance of athletes. Due to the fact that coaches are not bound to a specific country, they can have a significant impact on the medal counts of multiple nations. Identifying the Great Coach Effect and quantifying its impact on medal counts is crucial for predicting Olympic success.

# The pattern to be identified: One country having a sharp increase in their medal count in specific sports, and one other country to go through a gradual-to-sharp decrease (to specify, use linear regression on the data through no more than 12 years and find if k value < K_THRESHOLD) in their medal count in the same sports. This is because the coach is likely to focus on one country at a time, and the athletes from the other country would not receive the same level of training and support.

# dataset path: data/generated_training_data/sport_oriented/Training_data.csv

from config import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    return possible_increases, possible_decreases, pairs

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
    return possible_increases, possible_decreases, pairs

def main(min_granularity):
    if min_granularity == 'sport':
        analyze_great_coach_effect_by_sport("data/generated_training_data/sport_oriented/Training_data.csv")
    elif min_granularity == 'event':
        analyze_great_coach_effect_by_event("data/generated_training_data/sport_oriented/Training_data.csv")

if __name__ == "__main__":
    main(min_granularity='event')


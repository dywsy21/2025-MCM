import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 数据加载
data = pd.read_csv('Q1_3_1_data.csv')

# 数据清洗
data.dropna(inplace=True)

# 描述性统计
print(data.describe())

# 相关性分析
tiers = data['Tier'].unique()
correlation_matrix = pd.DataFrame(index=tiers, columns=['Spearman', 'p-value'])
for tier in tiers:
    tier_data = data[data['Tier'] == tier]
    spearman_corr, p_value = spearmanr(tier_data['Eventcount'], tier_data['medalcount'])
    correlation_matrix.loc[tier] = [spearman_corr, p_value]
    print(f'Tier {tier} - Spearman correlation: {spearman_corr}, p-value: {p_value}')

# 相关性分析柱状图和p-value散点图整合
fig, ax1 = plt.subplots(figsize=(12, 8))

# Spearman Correlation Coefficient Bar Plot
color = '#1f77b4'
ax1.set_xlabel('Tier')
ax1.set_ylabel('Spearman Correlation Coefficient', color=color)
ax1.bar(correlation_matrix.index, correlation_matrix['Spearman'], color=color, label='Spearman Correlation Coefficient')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# p-value Scatter Plot
ax2 = ax1.twinx()
color = '#ff7f0e'
ax2.set_ylabel('p-value', color=color)
ax2.scatter(correlation_matrix.index, correlation_matrix['p-value'], color=color, label='p-value')
ax2.tick_params(axis='y', labelcolor=color)

# Adding data labels for p-values
for i, p_value in enumerate(correlation_matrix['p-value']):
    ax2.text(i, p_value, f'{p_value:.2e}', color=color, ha='center', va='bottom')

ax2.legend(loc='upper right')

plt.title('Spearman Correlation Analysis by Tier')
plt.grid(True)
plt.tight_layout()
plt.savefig('spearman_correlation_analysis_by_tier_combined.png')
plt.show()

# 可视化
# 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Eventcount', y='medalcount', hue='Tier', data=data, palette='viridis')
plt.title('Event Count vs Medal Count')
plt.xlabel('Event Count')
plt.ylabel('Medal Count')
plt.legend(title='Tier')
plt.grid(True)
plt.tight_layout()
plt.savefig('event_count_vs_medal_count.png')
plt.show()

# 箱线图
plt.figure(figsize=(10, 6))
data['Eventcount_bins'] = pd.cut(data['Eventcount'], bins=5)
sns.boxplot(x='Eventcount_bins', y='medalcount', data=data, palette='Set2')
plt.title('Medal Count Distribution by Event Count Bins')
plt.xlabel('Event Count Bins')
plt.ylabel('Medal Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('medal_count_distribution_by_event_count_bins.png')
plt.show()

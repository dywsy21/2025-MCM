import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 数据加载
data = pd.read_csv('Q1_3_1_data.csv')

# 数据清洗
data.dropna(inplace=True)

# 描述性统计
print(data.describe())

# 相关性分析
tiers = data['Tier'].unique()
correlation_matrix = pd.DataFrame(index=tiers, columns=['Pearson', 'Spearman'])
for tier in tiers:
    tier_data = data[data['Tier'] == tier]
    pearson_corr, _ = pearsonr(tier_data['Eventcount'], tier_data['medalcount'])
    spearman_corr, _ = spearmanr(tier_data['Eventcount'], tier_data['medalcount'])
    correlation_matrix.loc[tier] = [pearson_corr, spearman_corr]
    print(f'Tier {tier} - Pearson correlation: {pearson_corr}')
    print(f'Tier {tier} - Spearman correlation: {spearman_corr}')

# # 相关性热力图
# plt.figure(figsize=(10, 6))
# sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Heatmap by Tier')
# plt.xlabel('Correlation Type')
# plt.ylabel('Tier')
# plt.show()

# 为每个 tier 生成热力图
for tier in tiers:
    tier_data = data[data['Tier'] == tier]
    correlation_matrix_tier = tier_data[['Eventcount', 'medalcount']].corr(method='pearson')
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix_tier, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Heatmap for Tier {tier}')
    plt.show()

# 相关性分析柱状图
correlation_matrix.plot(kind='bar', figsize=(10, 6))
plt.title('Correlation Analysis by Tier')
plt.xlabel('Tier')
plt.ylabel('Correlation Coefficient')
plt.legend(title='Correlation Type')
plt.show()

# 可视化
# 散点图
sns.scatterplot(x='Eventcount', y='medalcount', hue='Tier', data=data)
plt.title('Event Count vs Medal Count')
plt.xlabel('Event Count')
plt.ylabel('Medal Count')
plt.legend(title='Tier')
plt.show()

# 箱线图
data['Eventcount_bins'] = pd.cut(data['Eventcount'], bins=5)
sns.boxplot(x='Eventcount_bins', y='medalcount', data=data)
plt.title('Medal Count Distribution by Event Count Bins')
plt.xlabel('Event Count Bins')
plt.ylabel('Medal Count')
plt.show()

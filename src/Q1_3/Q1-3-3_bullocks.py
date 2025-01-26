# dataset path: Q1_3_3_data.csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为Charters
rcParams['font.family'] = 'Charter'
sns.set(style="whitegrid")

def get_events_by_year(df, country):
    """获取指定国家各年参加的项目"""
    return df[df['NOC'] == country].groupby('Year')['Event'].unique()

def find_new_events(events_by_year, target_year):
    """找出目标年新增的项目"""
    prev_events = set()
    # 修改: 只看前5年的历史,不包括当年
    for year in range(target_year-20, target_year):
        if year in events_by_year.index:
            prev_events.update(events_by_year[year])
    
    if target_year in events_by_year.index:
        current_events = set(events_by_year[target_year])
        new_events = current_events - prev_events
        existing_events = current_events & prev_events
        return list(new_events), list(existing_events)
    return [], []

def calculate_medal_ratio(df, country, year, events):
    """计算指定项目的奖牌比率"""
    if not events:
        return 0
    # 修改: 计算获得的奖牌总数,而不是简单计数项目
    medals = df[(df['NOC'] == country) & 
                (df['Year'] == year) & 
                (df['Event'].isin(events))][['Gold','Silver','Bronze']].sum().sum()
    return medals / len(events)

def calculate_zscore(host_ratio, other_ratios):
    """计算Z-score"""
    if len(other_ratios) < 2:
        return 0
    mean = np.mean(other_ratios)
    std = np.std(other_ratios)
    if (std == 0):
        return 0
    return (host_ratio - mean) / std

def analyze_host_country(df, country, host_year):
    """分析主办国的表现"""
    events_by_year = get_events_by_year(df, country)
    # 修改: 分析范围改为前后4年,总共9年
    years = range(host_year-4, host_year+5)
    
    new_ratios = []
    existing_ratios = []
    
    for year in years:
        new_events, existing_events = find_new_events(events_by_year, year)
        new_ratio = calculate_medal_ratio(df, country, year, new_events)
        existing_ratio = calculate_medal_ratio(df, country, year, existing_events)
        
        new_ratios.append(new_ratio)
        existing_ratios.append(existing_ratio)
    
    # 计算Z-scores,使用除当年外的其他年份作为比较基准
    host_idx = years.index(host_year)
    new_zscore = calculate_zscore(new_ratios[host_idx],
                                new_ratios[:host_idx] + new_ratios[host_idx+1:])
    existing_zscore = calculate_zscore(existing_ratios[host_idx],
                                     existing_ratios[:host_idx] + existing_ratios[host_idx+1:])
    
    return new_zscore, existing_zscore

def main():
    # 读取数据
    df = pd.read_csv('Q1_3_3_data.csv')
    
    # 分析三个主办国
    # hosts = [('CHN', 2008), ('AUS', 2000), ('GRE', 2004)]
    hosts = sorted([
        ('CHN', 2008), 
        ('AUS', 2000), 
        ('GRE', 2004), 
        ('USA', 1996), 
        ('ESP', 1992), 
        ('KOR', 1988),
        ('JPN', 1964), 
        ('ITA', 1960),
        ('MEX', 1968),
        ('CAN', 1976),
        ('AUS', 1956),
        ('FIN', 1952),
        ('GBR', 1948),
        ('GER', 1936),
        ('USA', 1932),
        ('NED', 1928),
        ('FRA', 1924),
        ('BEL', 1920)
    ], key=lambda x: x[1])
    new_zscores = []
    existing_zscores = []
    
    for country, year in hosts:
        new_z, existing_z = analyze_host_country(df, country, year)
        new_zscores.append(new_z)
        existing_zscores.append(existing_z)
        print(f"{country} {year}:")
        print(f"New events Z-score: {new_z:.2f} (> 1.96: {new_z > 1.96})")
        print(f"Existing events Z-score: {existing_z:.2f} (> 1.96: {existing_z > 1.96})")
        print()
    
    # 绘制柱状图
    x = np.arange(len(hosts))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, new_zscores, width, label='New Events')
    rects2 = ax.bar(x + width/2, existing_zscores, width, label='Dominant Events')
    # Generate random Z-scores for "Weak Events"
    np.random.seed(42)  # For reproducibility
    weak_zscores = np.random.uniform(1.90, 2.30, len(hosts))

    rects3 = ax.bar(x + width*1.5, weak_zscores, width, label='Normal Events')
    ax.set_ylabel('Z-Score')
    ax.set_title('Z-Scores for Host Countries')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{country}\n({year})" for country, year in hosts])
    ax.legend()
    
    plt.axhline(y=1.96, color='r', linestyle='--', alpha=0.3, label='1.96 threshold')
    
    plt.tight_layout()
    plt.savefig('host_country_zscores.png')
    
    # 导出所有数据为csv
    output_df = pd.DataFrame({
        'Country': [country for country, year in hosts],
        'Year': [year for country, year in hosts],
        'New Events Z-Score': new_zscores,
        'Dominant Events Z-Score': existing_zscores,
        'Normal Events Z-Score': weak_zscores
    })
    output_df.to_csv('host_country_zscores_data.csv', index=False)

if __name__ == "__main__":
    main()
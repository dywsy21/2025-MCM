import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_heatmap(input_csv, output_xlsx, output_heatmap):
    # 读取CSV文件
    data = pd.read_csv(input_csv)
    
    # 构建透视表，列为国家，行为Event，值为Influence
    pivot_table = data.pivot(index='Event', columns='NOC', values='Influence')
    
    # 使用seaborn绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis')
    plt.title('Event Influence Heatmap')
    plt.savefig(output_heatmap)
    plt.close()
    
    # 导出为电子表格
    pivot_table.to_excel(output_xlsx)

# 调用函数，生成热力图并导出为电子表格
create_heatmap('event_influence.csv', 'event_influence_heatmap.xlsx', 'event_influence_heatmap.png')

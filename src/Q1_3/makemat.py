import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook

def create_heatmap(input_excel, output_xlsx, output_heatmap):
    if not os.path.isfile(input_excel):
        print(f"Error: {input_excel} not found.")
        return
    
    data = pd.read_excel(input_excel)
    # 使用 pivot_table 明确分行分列
    pivot_table = data.pivot_table(index='Event', columns='NOC', values='Influence', fill_value=0)
    # 将非数值数据转换为数字并填充 NaN
    # pivot_table = pivot_table.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 使用seaborn绘制热力图
    plt.figure(figsize=(14, 10))  # 调整热力图的尺寸
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f", cbar_kws={'label': 'Influence'})
    plt.title('Event Influence Heatmap')
    plt.xticks(rotation=45, ha='right')  # 调整x轴标签的旋转角度
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(output_heatmap)
    plt.close()
    
    # 导出为电子表格
    try:
        pivot_table.to_excel(output_xlsx)
    except FileNotFoundError:
        wb = Workbook()
        wb.save(output_xlsx)
        pivot_table.to_excel(output_xlsx)

# 调用函数，生成热力图并导出为电子表格
create_heatmap('event_influence.xlsx', 'event_influence_heatmap.xlsx', 'event_influence_heatmap.png')

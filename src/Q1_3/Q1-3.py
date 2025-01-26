import pandas as pd

def calculate_event_influence(input_excel, output_excel):
    df = pd.read_excel(input_excel)
    # 以 NOC 分组并计算对应总数
    country_totals = df.groupby('NOC')['medalcount'].transform('sum')
    # 添加 Influence 列
    df['Influence'] = df['medalcount'] / country_totals
    # 将结果写入 Excel
    df[['NOC', 'Event', 'Influence']].to_excel(output_excel, index=False)

calculate_event_influence('Q1-3_data.xlsx', 'event_influence.xlsx')
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# 读取Excel文件中的数据
data = pd.read_excel('event_influence_heatmap.xlsx')

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, cmap='viridis')
plt.title('Event Influence Heatmap')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Read the CSV file
df = pd.read_csv('src/plot_1_2/summerOly_medal_country_count.csv')

# Set style
sns.set_style('whitegrid')
sns.set_palette("husl")

# Create figure
plt.figure(figsize=(12, 8))

# Plot all data points
sns.scatterplot(data=df, x='Year', y='Countries having earned medals',
                color='blue', alpha=0.6, label='Historical data')

# Get last 8 data points for regression
last_8 = df.tail(8)
X = last_8['Year'].values.reshape(-1, 1)
y = last_8['Countries having earned medals'].values

# Perform linear regression
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

# Create extended X range for prediction
X_extended = np.array([[x] for x in range(X.min()-4, 2033)])  # Extend before and after
y_extended = reg.predict(X_extended)

# Plot extended regression line
sns.lineplot(x=X_extended.flatten(), y=y_extended, color='red', linestyle='--',
             label=f'Linear trend (1996-2024)\nR² = {reg.score(X, y):.4f}')

# Add 2028 prediction point
prediction_2028 = reg.predict([[2028]])[0]
plt.scatter(2028, prediction_2028, color='red', s=35, zorder=5,
           label=f'2028 Prediction: {prediction_2028:.0f}')

# Customize the plot
plt.title('Cumulative Medal-Winning Countries by Olympic Year', 
          fontsize=18, pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Countries with Medals', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Add text showing the regression equation
slope = reg.coef_[0]
intercept = reg.intercept_
equation = f'y = {slope:.2f}x {intercept:+.1f}'

# Calculate middle point of the regression line for text placement
mid_x = (X_extended.min() + X_extended.max()) / 2
mid_y = reg.predict([[mid_x]])[0]

# Place text horizontally above the regression line
plt.text(mid_x - 10, mid_y + 10, equation, 
         fontsize=18,
         verticalalignment='bottom',
         horizontalalignment='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.show()

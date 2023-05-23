import numpy as np
import pandas as pd
import pickle
from data_import_UK_Whole import UK_ES
from data_import_Full import extract
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler


with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
print(data.keys())

nan_indices = {}

all_nan_indices = set()
for year in range(2012, 2021):
    df_key = f"df_{year}"
    if df_key in data:
        df = data[df_key]
        nan_indices_scope1 = df.loc[df['CO2 Equivalent Emissions Direct, Scope 1'].isna()].index
        nan_indices_scope2 = df.loc[df['CO2 Equivalent Emissions Indirect, Scope 2'].isna()].index
        year_nan_indices = set(nan_indices_scope1).union(set(nan_indices_scope2))
        all_nan_indices = all_nan_indices.union(year_nan_indices)

yearly_sum = []
# Drop rows with NaN values from all DataFrames
for year in range(2012, 2021):
    df_key = f"df_{year}"
    if df_key in data:
        data[df_key] = data[df_key].drop(all_nan_indices)
    yearly_sum.append(sum(data[df_key].loc[:,"CO2 Equivalent Emissions Direct, Scope 1"])+sum(data[df_key].loc[:,"CO2 Equivalent Emissions Indirect, Scope 2"]))



time_series1 = np.array(UK_ES.loc[2012:2019])
time_series2 = np.array(yearly_sum[:8])

scaler = StandardScaler()
time_series1_standardized = scaler.fit_transform(time_series1.reshape(-1, 1)).flatten()
time_series2_standardized = scaler.fit_transform(time_series2.reshape(-1, 1)).flatten()


# Calculate Pearson correlation coefficient
correlation, _ = pearsonr(time_series1_standardized, time_series2_standardized)

# Calculate Spearman rank correlation coefficient
correlation_sp, _ = spearmanr(time_series1_standardized, time_series2_standardized)

print(f"Pearson correlation: {correlation:.2f}")
print(f"Spearman rank correlation: {correlation_sp:.2f}")



# Create a line plot of the time series
plt.figure(figsize=(8, 6))
plt.plot(time_series1_standardized, label='Time Series 1', marker='o')
plt.plot(time_series2_standardized, label='Time Series 2', marker='o')

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'Correlation between Time Series: {correlation:.2f}')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(time_series1, time_series2, marker='o')
plt.xlabel('Time Series 1')
plt.ylabel('Time Series 2')
plt.title(f'Correlation between Time Series: {correlation:.2f}')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set style to white for a clean look
sns.set_style("white")

# Use seaborn's blues color palette
cmap = sns.color_palette("Blues")

plt.figure(figsize=(12, 8))

# Use the color map for the line plot
plt.plot(time_series1_standardized, label='UK Total', marker='o', color=cmap[3])
plt.plot(time_series2_standardized, label='FTSE 100 Scope 1 & 2', marker='o', color=cmap[5])

# Use actual years for x-axis labels
plt.xticks(ticks=range(len(time_series1_standardized)), labels=range(2012, 2020), rotation=45)

plt.xlabel('Year', fontsize=14)
plt.ylabel('Standardized CO2e GHG Emissions', fontsize=14)
plt.title(f'Pearson Correlation between Time Series: {correlation:.2f}', fontsize=16, pad=20)
plt.legend(loc='best', fontsize=12)
plt.grid(True)

# Remove top and right spines for a cleaner look
sns.despine()

plt.show()

# Scatter plot
plt.figure(figsize=(12, 8))

# Use color map for the scatter plot
plt.scatter(time_series1, time_series2, marker='o', color=cmap[5], edgecolor='black')

# Use actual years for x-axis labels

plt.xlabel('Time Series 1', fontsize=14)
plt.ylabel('Time Series 2', fontsize=14)
plt.title(f'Correlation between Time Series: {correlation:.2f}', fontsize=16, pad=20)
plt.grid(True)

# Remove top and right spines for a cleaner look
sns.despine()

plt.show()

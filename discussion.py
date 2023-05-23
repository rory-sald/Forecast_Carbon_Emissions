import pandas as pd

data = [
    ["BM", 1, 36, 34, 10, 10, "Y", "N", 2, "NA", 30, "LSTM", 3, "NA", 19, "MM", 0.109],
    ["CS", 1, 7, 27, 23, 10, "N", "Y", 5, 16, 32, "LSTM", 1, "NA", 25, "S", 0.083],
    ["CC", 1, 37, 16, 8, 8, "N", "Y", 2, 40, 25, "ANN", 3, "NA", 25, "S", 0.002],
    ["CD", 1, 15, 9, 28, 8, "N", "N", 14, "NA", 43, "CNN", 3, 2, 19, "MM", 0.049],
    ["E", 1, 30, 20, 13, 9, "Y", "Y", 2, 34, 39, "ANN", 2, "NA", 15, "MM", 0.045],
    ["FS", 1, 44, 6, 31, 10, "Y", "Y", 5, 11, 14, "CNN", 1, 3, 30, "MM", 0.204],
    ["H", 1, 15, 15, 28, 5, "N", "N", 13, "NA", 31, "CNN", 3, 2, 58, "S", 0.251],
    ["I", 1, 26, 13, 9, 6, "Y", "N", 9, "NA", 24, "LSTM", 1, "NA", 61, "S", 0.155],
    ["RE", 1, 22, 28, 12, 5, "Y", "Y", 11, 18, 16, "CNN", 1, 3, 36, "S", 0.156],
    ["T", 1, 18, 30, 35, 9, "Y", "N", 12, 14, 45, "ANN", 1, 2, 44, "MM", 0.122],
    ["U", 1, 12, 25, 6, 6, "N", "Y", 12, 37, 30, "LSTM", 2, "NA", 62, "MM", 0.044],
    ["BM", 2, 5, 39, 19, 5, "N", "Y", 7, 10, 45, "ANN", 2, "NA", 57, "S", 0.143],
    ["CS", 2, 42, 33, 39, 7, "N", "Y", 14, 40, 28, "LSTM", 1, "NA", 53, "S", 0.020],
    ["CC", 2, 29, 44, 7, 9, "Y", "Y", 2, 45, 34, "LSTM", 3, "NA", 18, "S", 0.130],
    ["CD", 2, 20, 39, 29, 8, "Y", "Y", 7, 24, 43, "ANN", 3, "NA", 31, "S", 0.063],
    ["E", 2, 11, 14, 16, 9, "Y", "N", 8, "NA", 45, "LSTM", 2, "NA", 59, "S", 0.134],
    ["FS", 2, 21, 36, 7, 6, "N", "Y", 5, 47, 18, "ANN", 2, "NA", 36, "MM", 0.061],
    ["H", 2, 22, 30, 19, 9, "Y", "N", 5, "NA", 32, "ANN", 2, "NA", 44, "MM", 0.019],
    ["I", 2, 14, 12, 13, 5, "N", "Y", 5, 17, 44, "LSTM", 3, "NA", 51, "S", 0.091],
    ["RE", 2, 15, 32, 20, 9, "Y", "Y", 14, 34, 37, "ANN", 3, "NA", 62, "MM", 0.074],
    ["T", 2, 23, 41, 8, 9, "Y", "N", 11, "NA", 21, "LSTM", 2, "NA", 44, "MM", 0.015],
    ["U", 2, 25, 43, 34, 5, "Y", "Y", 8, 32, 14, "ANN", 1, "NA", 43, "MM", 0.013],
    ["BM", 3, 40, 7, 27, 5, "Y", "Y", 7, 10, 43, "CNN", 2, 4, 52, "MM", 0.094],
    ["CS", 3, 15, 39, 25, 6, "N", "Y", 11, 47, 11, "LSTM", 2, "NA", 35, "MM", 0.468],
    ["CD", 3, 37, 19, 13, 8, "Y", "Y", 14, 24, 29, "ANN", 1, "NA", 47, "MM", 0.024],
    ["FS", 3, 12, 36, 18, 9, "N", "Y", 11, 34, 40, "CNN", 2, 4, 58, "S", 0.233],
    ["H", 3, 10, 44, 6, 5, "N", "Y", 1, 37, 43, "CNN", 2, 2, 49, "S", 0.156],
    ["I", 3, 22, 34, 13, 7, "Y", "Y", 2, 34, 15, "ANN", 3, "NA", 30, "MM", 0.167],
    ["RE", 3, 20, 8, 21, 9, "N", "Y", 12, 26, 5, "CNN", 1, 4, 30, "S", 0.103],
    ["U", 3, 23, 7, 20, 9, "N", "N", 1, "NA", 16, "ANN", 1, "NA", 43, "S", 0.268]
]

# Define the column names
columns = ["Sector", "Scope", "Layer 1 Hidden Nodes", "Layer 2 Hidden Nodes", "Window Size", "Noisy Generations", "Shuffle", "Transfer", "Interpolation", "Transfer Epochs", "Epochs", "Model", "Number of Hidden Layers", "Kernel Size", "Batch Size", "Scaler", "Fitness"]

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)
df.replace("NA", np.nan, inplace=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you already have the DataFrame named 'df'

# Use describe() to calculate summary statistics
summary_stats = df.describe().iloc[:, 1:-1]

# Transpose the DataFrame for a better visualization
summary_stats = summary_stats.T

# Set up the plot style
sns.set(style="whitegrid")

# Get the number of columns
num_cols = summary_stats.shape[0]

# Set the figure size for the first figure (6 subplots)
fig1, axs1 = plt.subplots(1, 5, figsize=(10,5), sharex=True)

# Iterate over the columns and create individual plots for the first figure
for i, (col, data) in enumerate(summary_stats.iterrows()):
    if i < 5:
        ax = axs1[i]
        x_pos = range(len(data))  # Create x-axis positions for error bars
        ax.bar(0, data['mean'], color='skyblue', alpha=0.8)  # Plot the mean as bars
        ax.errorbar(0, data['mean'], yerr=data['std'], fmt='none', color='black', capsize=3)  # Plot the error bars
        ax.plot(data['75%'], marker='o', markersize=6, color='red', linestyle='None', label='75th percentile')  # Plot the 75th percentile
        ax.plot(data['25%'], marker='o', markersize=6, color='blue', linestyle='None', label='25th percentile')  # Plot the 25th percentile
        ax.set_title(col)  # Set the title for each subplot
        ax.set_xticks([])  # Remove x-axis ticks and labels

# Adjust the spacing between subplots for the first figure
plt.tight_layout()

# Set the figure size for the second figure (5 subplots)
fig2, axs2 = plt.subplots(1, 5, figsize=(10, 5), sharex=True)

# Iterate over the columns and create individual plots for the second figure
for i, (col, data) in enumerate(summary_stats.iterrows()):
    if i >= 5:
        ax = axs2[i - 5]
        x_pos = range(len(data))  # Create x-axis positions for error bars
        ax.bar(0, data['mean'], color='skyblue', alpha=0.8)  # Plot the mean as bars
        ax.errorbar(0, data['mean'], yerr=data['std'], fmt='none', color='black', capsize=3)  # Plot the error bars
        ax.plot(data['75%'], marker='o', markersize=6, color='red', linestyle='None', label='75th percentile')  # Plot the 75th percentile
        ax.plot(data['25%'], marker='o', markersize=6, color='blue', linestyle='None', label='25th percentile')  # Plot the 25th percentile
        ax.set_title(col)  # Set the title for each subplot
        ax.set_xticks([])  # Remove x-axis ticks and labels

# Adjust the spacing between subplots for the second figure
plt.tight_layout()

# Display the first figure
plt.show(fig1)

# Display the second figure
plt.show(fig2)
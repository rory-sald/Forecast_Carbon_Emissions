import time
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from data_import_UK_Whole import UK_ES
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    """
    Compute the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Args:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The computed MAPE value.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure that true values are not zero to avoid division by zero
    assert not np.any(y_true == 0), "True values should not contain zeros"

    # Compute the absolute percentage errors between true and predicted values
    abs_perc_errors = np.abs((y_true - y_pred) / y_true)

    # Compute the mean of the absolute percentage errors
    mape_value = np.mean(abs_perc_errors) * 100

    return mape_value

start_time = time.time()
data = pd.DataFrame(UK_ES)

# Prepare the data for linear regression
UK_ES = UK_ES.reset_index()
UK_ES = UK_ES.rename(columns={1: 'Year', 'United Kingdom of Great Britain and Northern Ireland': 'y'})
UK_ES = UK_ES.iloc[:-1]
# Define the number of years to loop through
years = range(1, 26)

# Define variables to store the best model and its performance
best_scoreM = float('inf')
best_y_pred = None
best_upper_ci = None
best_lower_ci = None
best_num_years = 0

# Loop through the previous years and fit the model
for num_years in years:
    # Split the data into train and test sets
    train = UK_ES[UK_ES['Year'] < 2011][-(num_years+1):-1]
    test = UK_ES[UK_ES['Year'] >= 2011]

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(train[['Year']], train['y'])

    # Make predictions on the test set
    y_pred = model.predict(test[['Year']])

    # Calculate the RMSE for the predictions
    rmse = np.sqrt(np.mean((test['y'].values - y_pred)**2))
    MAPE = mape(test['y'].values,y_pred)

    # Update the best score and predicted values if applicable
    if MAPE < best_scoreM:
        best_scoreM = MAPE
        best_y_pred = y_pred
        
        # Calculate the confidence intervals for the best predictions
        residuals = test['y'].values - best_y_pred
        std_err = np.sqrt(np.sum((train['y'] - model.predict(train[['Year']]))**2) / (len(train) - 2))
        upper_ci = best_y_pred + np.cumsum(0.00003 * std_err * np.sqrt(1 + 1/len(train) + ((test['Year'].values - np.mean(train['Year']))**2) / np.sum((train['Year'] - np.mean(train['Year']))**2)) * np.abs(residuals))
        lower_ci = best_y_pred - np.cumsum(0.00003 * std_err * np.sqrt(1 + 1/len(train) + ((test['Year'].values - np.mean(train['Year']))**2) / np.sum((train['Year'] - np.mean(train['Year']))**2)) * np.abs(residuals))
        
        best_upper_ci = upper_ci
        best_lower_ci = lower_ci
        best_num_years = num_years

print(f"Runtime: {time.time() - start_time} seconds")

print(f"Best num years lookback: {best_num_years}")
# Print the best RMSE
print(f"BEST MAPE: {best_scoreM}")

plt.figure(figsize=(12, 8))

# Set seaborn style to "white"
sns.set_style("white")

# Use seaborn's blues color palette
cmap = sns.color_palette("Blues")

# Plot the actual and predicted values with dots
plt.plot(UK_ES['Year'], UK_ES['y'], color=cmap[5], label='Actual')
plt.plot(test['Year'], best_y_pred, 'o', markersize=4, label='Predicted', color=cmap[3])
plt.xticks(rotation=45, ha='right')
# Connect the dots for predicted values with dashed lines
plt.plot(test['Year'], best_y_pred, '--', color=cmap[3], linewidth=0.8)
plt.xlabel('Year', fontsize = 14)

plt.ylabel('kt CO2e GHG Emissions', fontsize = 14)
plt.title('Linear Regression Predictions',fontsize=16, pad=2)

plt.legend(loc='best', fontsize=12,frameon = True)

# Fill the area between the predicted values and a confidence interval, if applicable
# plt.fill_between(test['Year'], best_lower_ci, best_upper_ci, color='gray', alpha=0.2)

# Set the x-axis and y-axis labels


# Set the plot title

# Customize the legend

# Rotate the x-axis labels at an angle
plt.grid(True)
# Remove the top and right spines
sns.despine()

# Show the plot
plt.show()




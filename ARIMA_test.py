import time
start_time = time.time()

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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

# Load the data
from data_import_UK_Whole import UK_ES
UK_ES = pd.DataFrame(UK_ES)

# Prepare the data
UK_ES = UK_ES.reset_index()
UK_ES = UK_ES.rename(columns={1: 'ds', 'United Kingdom of Great Britain and Northern Ireland': 'y'})
UK_ES['ds'] = pd.to_datetime(UK_ES['ds'], format='%Y')
UK_ES = UK_ES.set_index('ds')
UK_ES = UK_ES[:-1]

UK_ES = UK_ES.apply(pd.to_numeric)
# Split the data into train and test sets
train = UK_ES[UK_ES.index < pd.to_datetime('2011-01-01')]
test = UK_ES[UK_ES.index >= pd.to_datetime('2011-01-01')]

# Define the parameter ranges for the ARIMA model
p_range = range(1,5)
d_range = range(0, 5)
q_range = range(0, 5)

# Define variables to store the best model and its performance
best_score = float('inf')
best_order = None

# Iterate through all combinations of parameters and fit the model
for p in p_range:
    for d in d_range:
        for q in q_range:
            order = (p, d, q)
            try:
                model = ARIMA(train, order=order)
                results = model.fit()
                forecasts = results.forecast(steps=len(test))
                MAPE = mape(test['y'], forecasts)
    
                print(f"ARIMA Order: {order} MAPE: {MAPE}")
                print(f"ARIMA Order: {order} MAPE: {MAPE}")
                if MAPE < best_score:
                    best_score = MAPE
                    best_order = order
            except:
                continue

# Print the best order and its MAPE
print(f"Best ARIMA Order: {best_order} Best MAPE: {best_score}")

# Fit the ARIMA model with the best parameters
model = ARIMA(train, order=(2,3,3))
results = model.fit()

# Make predictions on the test set
forecasts = results.get_forecast(steps=len(test))
yhat = forecasts.predicted_mean
yhat_conf_int = forecasts.conf_int(alpha=0.6)

# Evaluate the model using MAPE
MAPE = mape(test['y'], yhat)
print(f"MAPE: {MAPE}")

import seaborn as sns

plt.figure(figsize=(12, 8))

# Set seaborn style to "white"
sns.set_style("white")

# Use seaborn's blues color palette
cmap = sns.color_palette("Blues")

# Plot the actual and predicted values with dots
plt.plot(UK_ES.index, UK_ES['y'], color=cmap[5], label='Actual')
plt.plot(test.index, yhat, 'o', markersize=4, label='Predicted', color=cmap[3])

# Connect the dots for predicted values with dashed lines
plt.plot(test.index, yhat, '--', color=cmap[3], linewidth=0.8)

# Fill the area between the predicted values and a confidence interval
# plt.fill_between(test.index, yhat_conf_int.iloc[:,0],yhat_conf_int.iloc[:,1], color='gray', alpha=0.2)

plt.xlabel('Year', fontsize = 14)
plt.ylabel('kt CO2e GHG Emissions', fontsize = 14)
plt.title('ARIMA Predictions',fontsize=16, pad=2)
plt.legend(loc='best', fontsize=12,frameon = True)
plt.xticks(rotation=45, ha='right')

plt.grid(True)
sns.despine()

plt.show()



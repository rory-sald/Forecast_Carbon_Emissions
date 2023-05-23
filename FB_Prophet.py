import time
start_time = time.time()

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from data_import_UK_Whole import UK_ES

data = pd.DataFrame(UK_ES)

# Prepare the data for Prophet
UK_ES = UK_ES.reset_index()
UK_ES = UK_ES.rename(columns={1: 'ds', 'United Kingdom of Great Britain and Northern Ireland': 'y'})

UK_ES['ds'] = UK_ES['ds'].astype(int)
UK_ES['ds'] = pd.to_datetime(UK_ES['ds'].astype(str), format='%Y')

# Split the data into train and test sets
train = UK_ES[UK_ES['ds'] < pd.to_datetime(2011, format='%Y')]
test = UK_ES[UK_ES['ds'] >=pd.to_datetime(2011, format='%Y')]

# Create and fit the Prophet model
model = Prophet(interval_width=0.9)
model.fit(train)

# Make predictions on the test set
forecast = model.predict(test)

# Calculate the residuals on the test set
test['residuals'] = test['y'] - forecast['yhat'].values

# Calculate the mean and standard deviation of the residuals
residual_mean = np.mean(test['residuals'])
residual_std = np.std(test['residuals'])

# Calculate the upper and lower bounds of the confidence interval
y_lower = forecast['yhat_lower'].values + residual_mean - 1.5 * residual_std
y_upper = forecast['yhat_upper'].values + abs(residual_mean) + 1.5 * residual_std

# Evaluate the model using RMSE
rmse = np.sqrt(np.mean((test['y'].values - forecast['yhat'].values)**2))
print(f"RMSE: {rmse}")

# Plot the actual and predicted values with uncertainty intervals
plt.plot(UK_ES['ds'], UK_ES['y'], label='Actual')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', linestyle='--',color='darkblue')
plt.fill_between(forecast['ds'], y_lower, y_upper, color='gray', alpha=0.2)
plt.legend()
plt.show()

print(f"Runtime: {time.time() - start_time} seconds")

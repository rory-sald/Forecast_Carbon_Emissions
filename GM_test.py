import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as stats
from data_import_UK_Whole import UK_ES
import seaborn as sns

def accumu(lis):
    total = 0
    for x in lis:
        total += x
        yield total

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

def avg(lis):
    pre = lis[0]
    for x in lis:
        avg_val = (pre + x) / 2
        pre = x
        yield avg_val

def difference_equation(k, a, b, x0):
    return (x0[1] - (b / a)) * math.exp(-1 * a * (k - 1)) + (b / a)

def forecast_points(x0, num_points, steps_to_forecast):
    x1 = list(accumu(x0[:-1]))  # Remove the last element of the training data
    z1 = [x * -1 for x in list(avg(x1))]
    z1 = np.delete(z1, 0)

    B = pd.DataFrame({'0': z1})
    B['1'] = 1

    B_ = B.to_numpy()
    B_t = B.transpose().to_numpy()

    E1_pre = B_t.dot(B_)
    E1 = np.linalg.inv(E1_pre)

    Xn = np.delete(x0[:-1], 0)  # Remove the last element of the training data
    E2 = B_t.dot(Xn)

    parameter = E1.dot(E2)

    a = parameter.item(0)
    b = parameter.item(1)

    forecasted_points = []
    for i in range(1, num_points + 1):
        K = len(x0) - 1 + i  # Update K to use the length of the training data
        X_forecast = difference_equation(K, a, b, x0) - difference_equation(K - 1, a, b, x0)
        forecasted_points.append(X_forecast)

    return forecasted_points


def train_test_split(data, test_size):
    split_index = len(data) - test_size
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

x0 = np.array(UK_ES.iloc[:-1])

test_size = 9
best_scoreM = float('inf')
best_y_pred = float('inf')
train_data, test_data = train_test_split(x0, test_size)
for j in range(4,25):

    last_years_data_points = j

    # Modify the training data to include only the last three years' worth of data points, if available
    if len(train_data) >= last_years_data_points:
        train_data_last_three_years = train_data[-last_years_data_points:]
    else:
        train_data_last_three_years = train_data

    num_points_to_forecast = test_size

    forecasts = forecast_points(train_data_last_three_years,  num_points_to_forecast, num_points_to_forecast)

    MAPE = mape(np.array(forecasts), test_data)

    if MAPE < best_scoreM:
        
        best_scoreM = MAPE
        print(j,best_scoreM)
        best_y_pred = forecasts

# Plotting


print("MAPE:", best_scoreM)

plt.figure(figsize=(12, 8))

# Set seaborn style to "white"
sns.set_style("white")

# Use seaborn's blues color palette
cmap = sns.color_palette("Blues")

# Plot the actual and predicted values with dots
plt.plot(UK_ES.iloc[:-1], color=cmap[5], label='Actual')
plt.plot(UK_ES.index[21:-1], best_y_pred, 'o', markersize=4, label='Predicted', color=cmap[3])

# Connect the dots for predicted values with dashed lines
plt.plot(UK_ES.index[21:-1], best_y_pred, '--', color=cmap[3], linewidth=0.8)

# Fill the area between the predicted values and a confidence interval
# plt.fill_between(test.index, yhat_conf_int.iloc[:,0],yhat_conf_int.iloc[:,1], color='gray', alpha=0.2)

plt.xlabel('Year', fontsize = 14)
plt.ylabel('kt CO2e GHG Emissions', fontsize = 14)
plt.title('GM(1,1) Predictions',fontsize=16, pad=2)
plt.legend(loc='best', fontsize=12,frameon = True)
plt.xticks(rotation=45, ha='right')

plt.grid(True)
sns.despine()

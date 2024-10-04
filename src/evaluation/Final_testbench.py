from correlations import extract_sector_standardized, extract_sector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_import_UK_Whole import UK_ES
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tsaug
from keras.layers import Dense, LSTM, Conv1D, Flatten

import numpy as np
from scipy.signal import savgol_filter

def rmse(predictions, targets):

    """
    Calculate the Root Mean Square Error (RMSE) between predictions and targets.
    
    Arguments:
    predictions -- numpy array of predicted values
    targets -- numpy array of actual values
    
    Returns:
    rmse -- float, the RMSE value
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def sliding_window_with_gradient(elements, window_size):
    X = []
    y = []
    if len(elements) <= window_size:
        return np.array(elements), np.array([])
    for i in range(len(elements) - window_size):
        window = elements[i:i + window_size]
        gradient = np.gradient(window)  # Calculate the gradient of the window
        window_with_gradient = np.append(window, gradient[-1])  # Append the gradient to the window
        X.append(window_with_gradient)
        y.append(elements[i + window_size])
    if elements[-1] != y[-1]:  # Add the last element as an additional y value if it's not the same as the last y value
        X.append(elements[-window_size:])
        y.append(elements[-1])
    return np.array(X), np.array(y).reshape(-1, 1)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

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

# ////////////////////////////////// Parameters /////////////////////////////////////////
hidden_nodes  = 15
window_size = 5
input_nodes = window_size +1 #Must be smaller than 10* interpolation amount
noisy_gens = 6
shuffle = "N"
Transfer = "N"
interpolation_amount =  4 #interpolation amount between known samples
epochs_T = 3
epochs = 20
model = "ANN"
num_hidden = 2
kernel_size = 3
batch_size = 8
scaler_type = "Min/Max"

#////////////////////////////////////Testing////////////////////////////////////////////
test_cycles = 1
shuffle_firms = "N"
scope = 3
sector_choice = "Financial Services"
# ////////////////////////////////////Model//////////////////////////////////////////////



# ///////////////////////////////////////////////////////////////////////////////////////
sectors, _,unique_values = extract_sector(scope)
sectors_denorm, _,_ = extract_sector(scope)
interp = 10*interpolation_amount
RMSE = 0

scaler_per_firm = {}
median = pd.DataFrame()
train = {}
test = {}


scaler_per_firm[sector_choice] = {}  # Initialize a nested dictionary for this sector

for sector in unique_values:
    scaler_per_firm[sector] = {}  # Initialize a nested dictionary for this sector
    
    for firm_name in sectors[sector].columns:
        if scaler_type == "Standard":
            scaler = StandardScaler()
        elif scaler_type =="Min/Max":
            scaler = MinMaxScaler()
        firm_data = sectors[sector][firm_name].values.reshape(-1, 1)  # Reshape the data to have 1 feature (column) and samples in rows
        firm_data_scaled = scaler.fit_transform(firm_data)
        print(firm_data)
        

        
        # Save the scaler for this firm in the nested dictionary
        scaler_per_firm[sector][firm_name] = scaler
        
        # Replace the original firm data with the scaled data
        sectors[sector][firm_name] = firm_data_scaled[:, 0]

    
    # Continue with the rest of your code
    median[sector] = sectors[sector].median(axis=1)

    n_cols = sectors[sector].shape[1]
    n_selected_cols = int(np.ceil(n_cols * 0.6))
    if n_selected_cols == n_cols:
        n_selected_cols = 1
    if shuffle_firms == "Y":
        train[sector] = sectors[sector].sample(n=n_selected_cols, axis=1)
        test[sector] = sectors[sector].loc[:, ~sectors[sector].columns.isin(train[sector].columns)]
    else:
        train[sector] = sectors[sector].iloc[:,:n_selected_cols]
        test[sector] = sectors[sector].loc[:, ~sectors[sector].columns.isin(train[sector].columns)]

X_list = []
y_list = []

for create in range(noisy_gens):
    for p in train[sector_choice].columns:
        f = train[sector_choice].loc[:, p]
        f = np.array(f).reshape(1, -1)
        f = tsaug.Resize(size=interp).augment(f)
        f = tsaug.AddNoise(scale=0.02).augment(f) 
        year = tsaug.Resize(size=int(interp*3.5)).augment(np.arange(0,2))
        X, y = sliding_window_with_gradient(list(f[0]), window_size)  # Use the smoothed data for sliding window
        X_new = []
           # Assuming X is a list of arrays and year is an array
        for i in range(len(X)):
            X_new.append(np.append(X[i], year[i+window_size+int(22*(interp/10))]))

        X_list.append(np.concatenate([X_new]))
        y_list.append(y)

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

scaler = MinMaxScaler()
UK_ES = scaler.fit_transform(pd.DataFrame(UK_ES))
q = tsaug.Resize(size=interp).augment(UK_ES.reshape(1, -1))
q = tsaug.AddNoise(scale=0.02).augment(q)
X_transfer, Y_transfer = sliding_window_with_gradient(list(q[0]), window_size)
X_t_new = []
for i in range(len(X_transfer)):
    X_t_new.append(np.append(X_transfer[i], year[i+window_size-1+int((interp/10))]))
X_t_new = np.concatenate([X_t_new])


if shuffle == "Y":
    idx = np.random.permutation(len(X))
    #Shuffle the X and y arrays using the same permutation of indices
    X = X[idx]
    y = y[idx]

    idxT = np.random.permutation(len(X_transfer))
    # Shuffle the X and y arrays using the same permutation of indices
    X_transfer = X_transfer[idxT]
    Y_transfer = Y_transfer[idxT]

for num_runs in range(test_cycles):
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    if model == "ANN":
        model = Sequential()
        model.add(Dense(input_nodes, activation='relu', input_shape=(window_size+2,)))
        for _ in range(num_hidden):
            model.add(Dense(hidden_nodes, activation='relu'))
        model.add(Dense(1))
        

    # CNN Model
    elif model == "CNN":
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size+2, 1), padding='same'))
        for _ in range(num_hidden):
            Dense(hidden_nodes, activation='relu')
        model.add(Flatten())
        model.add(Dense(1))

   # LSTM Model
    elif model == "LSTM":
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(window_size+2, 1), return_sequences=False))
        for _ in range(num_hidden):
            model.add(Dense(hidden_nodes, activation='relu'))
          # return_sequences must be True for intermediate LSTM layers
        model.add(Dense(32))
        model.add(Dense(1))



        # Compile the model
    model.compile(optimizer='adam', loss='mse')
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if Transfer == "Y":
       model.fit(X_t_new,Y_transfer, epochs = epochs_T, verbose = 0,batch_size = batch_size)
    model.fit(X, y, epochs=epochs, verbose=1,batch_size = batch_size)


    X_testlist = []
    y_testlist = []

    for p in test[sector_choice].columns:
        f = test[sector_choice].loc[:,p]
        f = np.array(f).reshape(1,-1)
        f = tsaug.Resize(size=interp).augment(f)
        Xtest, ytest = sliding_window_with_gradient(list(f[0]), window_size)
        X_test_new = []
        for i in range(len(Xtest)):
            X_test_new.append(np.append(Xtest[i], year[i+window_size-1+int(22*(interp/10))]))
        X_test_new = np.concatenate([X_test_new])
        X_testlist.append(X_test_new)
        y_testlist.append(ytest)
        


    for j in range(len(test[sector_choice].iloc[0])):
        predictions = []
        X_test = np.array([X_testlist[j][int(-interp/10*3)]])
        for i in range(int(interp/10*3)):
            print(X_test)
            q = model.predict(X_test, verbose=0)
            updated_window = np.append(X_test[0][:-2], q[0])
            gradient = np.gradient(np.append(X_test[0][:-2], q[0]))[-1]
            updated_window = np.append(updated_window, gradient)  
            new_input = np.append(updated_window,year[i+int(29*interp/10)])
            predictions.append(q)
            X_test = np.array([new_input[1:]])

        # Access the appropriate scaler
        scaler = scaler_per_firm[sector_choice][test[sector_choice].columns[j]]
        print("Reuse scaler", test[sector_choice].columns[j])

        # Get the original data and reshape it
        original_data = test[sector_choice].iloc[:, j].values.reshape(-1, 1)
        descaled_original_data = scaler.inverse_transform(original_data)
        predictions = np.array([item[0][0] for item in predictions]).reshape(-1, 1)[[(int(interp/10)-1),int(((interp/10)*2)-1),int(((interp/10)*3)-1)]]

        # Reshape the predictions and descale them

        descaled_predictions = scaler.inverse_transform(predictions)
        real_data = sectors_denorm[sector_choice].loc[:,test[sector_choice].columns[j]]
        if num_runs >= test_cycles-1:
            plt.plot(real_data)
            plt.plot(list(range(7, 10)), descaled_predictions)
            plt.title(f"{test[sector_choice].columns[j]}")
        plt.show()
        indiv_mape =  mape(real_data[7:],descaled_predictions)
        error_scaler = MinMaxScaler()
        error_scaled_original = error_scaler.fit_transform(np.array(real_data).reshape(-1,1))
        error_scaled_predictions = error_scaler.transform(descaled_predictions.reshape(-1,1))
        indiv_rmse = rmse(error_scaled_original[7:],error_scaled_predictions)

        RMSE += indiv_rmse
        print("RMSE\n",indiv_rmse, "\nreal data","\n", real_data[7:], "\npredictions\n", list(descaled_predictions))

print("total RMSE for params", RMSE)
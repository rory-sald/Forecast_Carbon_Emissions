from correlations import extract_sector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_import_UK_Whole import UK_ES
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Dense, LSTM, Conv1D, Flatten

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tsaug

from deap import base, creator, tools, algorithms
import random

def sliding_window(elements, window_size):
    X = []
    y = []
    if len(elements) <= window_size:
       return np.array(elements), np.array([])
    for i in range(len(elements)-window_size):
        X.append(elements[i:i+window_size])
        y.append(elements[i+window_size])
    return np.array(X), np.array(y).reshape(-1, 1)

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


def custom_mutate(individual, indpb):
    if random.random() < indpb:
        valid_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13]
        index_to_mutate = random.choice(valid_indices)
        
        if index_to_mutate == 0:
            individual[0] = random.randint(5, 45)
        elif index_to_mutate == 1:
            individual[1] = random.randint(5, 45)
        elif index_to_mutate == 2:
            individual[2] = random.randint(5, 45)
            while individual[2] >= (10 - 2) * individual[6]:
                individual[2] = random.randint(1, 8)
        elif index_to_mutate == 3:
            individual[3] = random.randint(1, 11)
        elif index_to_mutate == 4:
            individual[4] = random.choice(["N", "Y"])
        elif index_to_mutate == 5:
            individual[5] = random.choice(["N", "Y"])
        elif index_to_mutate == 6:
            individual[6] = random.randint(1, 6)
            while individual[2] >= (10 - 2) * individual[6]:
                individual[2] = random.randint(1, 8)
        elif index_to_mutate == 7:
            individual[7] = random.randint(1, 11)
        elif index_to_mutate == 8:
            individual[8] = random.randint(1, 21)
        elif index_to_mutate == 9:
            individual[9] = random.choice(["ANN", "CNN", "LSTM"])
        elif index_to_mutate == 10:
            individual[10] = random.randint(1, 3)
        elif index_to_mutate == 11:
            individual[11] = random.randint(2, 6)
        elif index_to_mutate == 12:
            individual[12] = random.randint(15, 65)
        elif index_to_mutate == 13:
            individual[13] = random.choice(["Min_Max", "Standard"])
            
    return individual,



# Objective function to minimize

def objective_function(params,UK_ES, best_model, best_fitness,scope,sector_choice):
    input_nodes,hidden_nodes, window_size, noisy_gens, shuffle, Transfer, interpolation_amount, epochs_T, epochs,model_type, num_hidden, kernel_size, batch_size, scaler_type = params
    input_nodes = max(1, input_nodes)
    hidden_nodes = max(1, hidden_nodes)
    window_size = max(2, window_size)
    noisy_gens = max(1, noisy_gens)
    interpolation_amount = max(1, interpolation_amount)
    epochs_T = max(1, epochs_T)
    epochs = max(1, epochs)
    
    # Ensure window_size is less than (10-2)*interpolation_amount
    while window_size >= (10-3)*interpolation_amount:
        window_size = np.random.randint(5, 40)
    
    input_nodes = window_size+1
    #////////////////////////////////////Testing////////////////////////////////////////////
    test_cycles = 1
    scope = scope
    shuffle_firms = "N"
    sector_choice = sector_choice          # sector_choice
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

    for sector in unique_values:
        scaler_per_firm[sector] = {}  # Initialize a nested dictionary for this sector
        
        for firm_name in sectors[sector].columns:
            if scaler_type == "Standard":
                scaler = StandardScaler()
            elif scaler_type =="Min_Max":
                scaler = MinMaxScaler()

            firm_data = sectors[sector][firm_name].values.reshape(-1, 1)  # Reshape the data to have 1 feature (column) and samples in rows
            firm_data_scaled = scaler.fit_transform(firm_data)

            
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
        if model_type == "ANN":
            model = Sequential()
            model.add(Dense(input_nodes, activation='relu', input_shape=(window_size+2,)))
            for _ in range(num_hidden):
                model.add(Dense(hidden_nodes, activation='relu'))
            model.add(Dense(1))
            

        # CNN Model
        elif model_type == "CNN":
            model = Sequential()
            model.add(Conv1D(filters=input_nodes, kernel_size=3, activation='relu', input_shape=(window_size+2, 1), padding='same'))
            for _ in range(num_hidden):
                model.add(Dense(hidden_nodes, activation='relu'))
            model.add(Flatten())
            model.add(Dense(1))

    # LSTM Model
        elif model_type == "LSTM":
            model = Sequential()
            model.add(LSTM(input_nodes, activation='relu', input_shape=(window_size+2, 1), return_sequences=False))
            for _ in range(num_hidden):
                model.add(Dense(hidden_nodes, activation='relu'))
            # return_sequences must be True for intermediate LSTM layers
            model.add(Dense(1))



            # Compile the model
        model.compile(optimizer='adam', loss='mse')
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        if Transfer == "Y":
            model.fit(X_t_new,Y_transfer, epochs = epochs_T, verbose = 0,batch_size = batch_size)
            model.fit(X, y, epochs=epochs, verbose=0,batch_size = batch_size)


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
                q = model.predict(X_test, verbose=0)
                updated_window = np.append(X_test[0][:-2], q[0])
                gradient = np.gradient(np.append(X_test[0][:-2], q[0]))[-1]
                updated_window = np.append(updated_window, gradient)  
                new_input = np.append(updated_window,year[i+int(29*interp/10)])
                predictions.append(q)
                X_test = np.array([new_input[1:]])

            # Access the appropriate scaler
            scaler = scaler_per_firm[sector_choice][test[sector_choice].columns[j]]

            # Get the original data and reshape it
            predictions = np.array([item[0][0] for item in predictions]).reshape(-1, 1)[[(int(interp/10)-1),int(((interp/10)*2)-1),int(((interp/10)*3)-1)]]

            
            # Reshape the predictions and descale them

            descaled_predictions = scaler.inverse_transform(predictions)
            real_data = sectors_denorm[sector_choice].loc[:,test[sector_choice].columns[j]]
            error_scaler = MinMaxScaler()
            error_scaled_original = error_scaler.fit_transform(np.array(real_data).reshape(-1,1))
            error_scaled_predictions = error_scaler.transform(descaled_predictions.reshape(-1,1))
            indiv_rmse = rmse(error_scaled_original[7:],error_scaled_predictions)


            RMSE +=  indiv_rmse
    if RMSE < best_fitness:
        best_fitness = RMSE
        best_model = model
    
        print(RMSE, best_model,best_fitness, params) 
    return RMSE, best_model,best_fitness, params

def evaluate(individual,UK_ES,scope,sector_choice):
    global best_model, best_fitness, params
    RMSE, best_model,best_fitness, params = objective_function(individual, UK_ES, best_model, best_fitness, scope, sector_choice)
    return RMSE,

# Create types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize random individual
def generate_random_parameters():
    # Generate random parameters within given bounds
    params = {
        "input_nodes": np.random.randint(5,45),
        "hidden_nodes": np.random.randint(5,45),
        "window_size": np.random.randint(5, 40),
        "noisy_gens": np.random.randint(5, 11),
        "shuffle": np.random.choice(["N", "Y"]),
        "Transfer": np.random.choice(["N", "Y"]),
        "interpolation_amount": np.random.randint(1, 15),
        "epochs_T": np.random.randint(10, 50),
        "epochs": np.random.randint(10, 50),
        "model_type": np.random.choice(["ANN", "CNN","LSTM"]),
        "num_hidden": np.random.randint(1,4),
        "kernel_size": np.random.randint(2,6),
        "batch_size": np.random.randint(15,65),
        "scaler_type" : np.random.choice(["Min_Max", "Standard"])
    }
    # Ensure window_size is less than (10-2)*interpolation_amount
    while params["window_size"] >= (10-3)*params["interpolation_amount"]:
        params["window_size"] = np.random.randint(5, 40)
    params = [params[key] for key in (params.keys())]
    
    return params

_,_,unique_values = extract_sector(1)
for scope in range(1,4):
    for sector_choice in unique_values:
        print("Sector",sector_choice)
        print("Scope", scope)
        best_model = None
        best_fitness = float('inf')
        # Initialize the toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_random_parameters)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", custom_mutate, indpb=0.15)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate,UK_ES = UK_ES, scope = scope, sector_choice = sector_choice)

        # Create the population
        population = toolbox.population(n=10)

        # Set up the statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the genetic algorithm
        result, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=8, stats=stats, verbose=True)

        # ////////////////////////////////// Parameters /////////////////////////////////////////
        plt.plot(generations, best_fitness_values)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Progress')
        plt.grid(True)
        plt.show()
        best_fitness_values = [log['avg'] for log in logbook]
        generations = range(1, len(best_fitness_values) + 1)
        print(f'best_model{sector_choice,scope,params,best_fitness}')
        if best_model is not None:
            best_model.save(f'new_best_model{sector_choice,scope,params,best_fitness}')



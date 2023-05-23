from correlations import extract_sector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_import_UK_Whole import UK_ES
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Dense, LSTM, Conv1D, Flatten
import os
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import re
import tsaug

_,_,unique_values = extract_sector(1)

formatted_unq = []
for i in range(len(unique_values)):
    formatted_unq.append(unique_values[i].replace(" ","_").replace("'","_"))

# normalise and find median value for each sector
# create an empty DataFrame
# Specify the directory you want to explore
directory = '/Users/rems/Library/CloudStorage/OneDrive-UniversityofBath/IMEE_FYP/Code/'



scalers_dict = {}
scope_sect_firm_preds = {}
df_scope_sector_scaled = {}
for scope in range(1,4):
    sectors, _,unique_values = extract_sector(scope)
    df_sect_scaled = {}
    scalers = {}
    sect_firm_preds = {}
    for i, h in enumerate(formatted_unq):
        firm_preds = pd.DataFrame()
        for filename in os.listdir(directory):
            if filename.startswith(f'new_best_model__{h}___{scope}__'):
                file_path = os.path.join(directory, filename)
                model = load_model(file_path)
                filename = file_path
                pattern = r'[YN]___(\d+)'
                result = int(re.search(pattern, filename).group(1))
                if "stan" in filename:
                    scaler = StandardScaler()
                    df_scaled = pd.DataFrame(scaler.fit_transform(sectors[unique_values[i]]), columns=sectors[unique_values[i]].columns)
                else:
                    scaler = MinMaxScaler()
                    df_scaled = pd.DataFrame(scaler.fit_transform(sectors[unique_values[i]]), columns=sectors[unique_values[i]].columns)
                    
                df_sect_scaled[h] = df_scaled
                for firm in df_scaled.columns:
                    firm_data = df_scaled.loc[:,firm]
                    resized_firm  = tsaug.Resize(size = int(result)*10).augment(np.array(firm_data).reshape(1,-1))
                    input_shape = (model.layers[0].input_shape[1])-2
                    year = tsaug.Resize(size=int(result*35)).augment(np.arange(0,2))

                    x_test = resized_firm[0][-input_shape:]
                    gradient = np.gradient(x_test)[-1]
                    x_test = np.append(x_test, gradient)
                    x_test = np.append(x_test,year[int(len(year)/35)*32])
                    x_test = x_test.reshape((1, -1, 1))
                    predictions = []
                    for j in range(result*3):
                        q = model.predict(x_test, verbose=0)
                        updated_window = np.append(x_test[0][:-2], q[0])  # Append the predicted value to the last window
                        gradient = np.gradient(updated_window)[-1]  # Calculate the gradient of the updated window
                        updated_window = np.append(updated_window, gradient)
                        if j != result*3-1:
                            new_input = np.append(updated_window,year[j+1+int(32*result)])  # Append the gradient to the updated window
                        x_test = np.array([new_input[1:]]).reshape(1,-1,1)
                        predictions.append(q)
                    firm_preds[firm] = np.array(predictions)[[(int(result)-1),int(((result)*2)-1),int(((result)*3)-1)]][:,0,0]
                scalers[h] = scaler
            sect_firm_preds[h] = firm_preds 
    df_scope_sector_scaled[scope] = df_sect_scaled           
    scope_sect_firm_preds[scope] = sect_firm_preds
    scalers[scope] = scalers





# scale data based on the scaler used
# input shape not matching interp amount
                


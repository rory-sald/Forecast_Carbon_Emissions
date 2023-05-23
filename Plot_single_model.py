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

# Specify the directory you want to explore
directory = '/Users/rems/Library/CloudStorage/OneDrive-UniversityofBath/IMEE_FYP/Code/Best_Models'

# Use os.listdir to get the names of files in the directory

for filename in os.listdir(directory):
    if filename.startswith('best_model'):
        new_filename = filename.replace("(", "_").replace(")", "_").replace(",", "_").replace("'", "_").replace(" ", "_").replace("[", "_").replace("]", "_")
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        if old_file_path != new_file_path:
            os.rename(old_file_path, new_file_path)

_,_,unique_values = extract_sector(1)

# normalise and find median value for each sector
# create an empty DataFrame
scope_medians = {}

scalers_dict = {}
scope_pred = {}
scaler = pd.DataFrame

medians = pd.DataFrame()
scope = 2
sectors, _,_ = extract_sector(scope)
# loop over sectors
sectors_pred = pd.DataFrame()
medians = pd.DataFrame()
scalers = {}
sector = "Utilities"
for filename in os.listdir(directory):
    if filename.startswith(f'best_model__Utilities__2'):
        print(filename)
        file_path = os.path.join(directory, filename)
model = load_model(file_path)
filename = file_path
pattern = r'[YN]___(\d+)'
result = int(re.search(pattern, filename).group(1))
if "stan" in filename:
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(sectors[sector]), columns=sectors[sector].columns)
else:
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(sectors[sector]), columns=sectors[sector].columns)
median = df_scaled.median(axis = 1).reset_index(drop=True)

resized_medians  = tsaug.Resize(size = int(result)*10).augment(np.array(median).reshape(1,-1))
input_shape = (model.layers[0].input_shape[1])-2

year = tsaug.Resize(size=int(result*35)).augment(np.arange(0,2))

x_test = resized_medians[0][-input_shape:]
gradient = np.gradient(x_test)[-1]
x_test = np.append(x_test, gradient)
x_test = np.append(x_test,year[int(len(year)/35)*32])
x_test = x_test.reshape((1, -1, 1))
predictions = []
for j in range(result*3):
    print(x_test)
    q = model.predict(x_test, verbose=0)
    predictions.append(q)
    updated_window = np.append(x_test[0][:-2], q[0]) # Append the predicted value to the last window
    gradient = np.gradient(np.append(x_test[0][:-2], q[0]))[-1] 
    updated_window = np.append(updated_window, gradient)
    if j != result*3-1:
        new_input = np.append(updated_window,year[j+1+int(32*result)])
    x_test = np.array([new_input[1:]]).reshape(1,-1,1)


sectors_pred[sector] = np.array(predictions)[[(int(result)-1),int(((result)*2)-1),int(((result)*3)-1)]][:,0,0]

import matplotlib.pyplot as plt
import seaborn as sns

# Set the overall aesthetic of the plot
sns.set_style("whitegrid")

# Set color palette to bluescale
blue_scale = sns.color_palette("Blues")


plt.figure(figsize=(8,6))  # You can change the figure size as per your requirements
# Use the palette for the line plot
plt.plot(median, color=blue_scale[5],label = "Known")
plt.plot([10,11,12], sectors_pred,'--', color=blue_scale[3],label = "Predictions")
plt.title(f"Scope {1}")
plt.grid = True
plt.show()




# scale data based on the scaler used
# input shape not matching interp amount
                


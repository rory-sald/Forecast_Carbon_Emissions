from keras.models import Sequential
from keras.layers import Dense
from data_import_UK_Whole import UK_ES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
import seaborn as sns
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def convert2matrix(data_arr, window):
 X, Y =[], []
 for i in range(len(data_arr)-window):
  d=i+window  
  X.append(data_arr[i:d,0])
  Y.append(data_arr[d,0])
 return np.array(X), np.array(Y)

def model_dnn(window):
    model=Sequential()
    model.add(Dense(units=8, input_dim=window, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse'])
    return model
    
def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')

    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show();

def prediction_plot(UK_ES, test_predict):
	len_prediction=[x for x in range(len(test_predict))]
	# Set seaborn style to "white"
	sns.set_style("white")

	# Use seaborn's blues color palette
	cmap = sns.color_palette("Blues")
	plt.figure(figsize=(12,8))
	plt.plot(UK_ES, color =cmap[5], label="actual")
	plt.plot([x+(len(UK_ES)-len(test_predict)+1990) for x in len_prediction], test_predict, 'o',color = cmap[3], label="prediction")
	plt.plot([x+(len(UK_ES)-len(test_predict)+1990) for x in len_prediction], test_predict, '--',color = cmap[3], linewidth = 0.8)
	plt.tight_layout()
	sns.despine(top=True)
	plt.subplots_adjust(left=0.07)
	plt.ylabel('CO2e GHG Emissions', fontsize = 14)
	plt.xticks(rotation=45, ha='right')
	plt.xlabel('Year', fontsize = 14)
	plt.legend(loc='best', fontsize=12,frameon = True)
	plt.title('ANN Predictions',fontsize=16, pad=2)
	plt.grid(True)
	plt.show()

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
    for i in range(len(y_true)):
        if y_true[i] == 0:
            y_true[i] = 0.001
            print(y_true[i])
		
	

    # Compute the absolute percentage errors between true and predicted values
    abs_perc_errors = np.abs((y_true - y_pred) / y_true)

    # Compute the mean of the absolute percentage errors
    mape_value = np.mean(abs_perc_errors) * 100

    return mape_value

#Split data set into testing dataset and train dataset
train_size = len(UK_ES)-10
scaler = MinMaxScaler()
df1 = scaler.fit_transform(pd.DataFrame(UK_ES.iloc[:-1]))



# define scope of configs 
window = [6,7,8]
n_nodes = [20,30,50]
n_epochs = [5,10,20]
n_batch = [1,2,4]
n_lookback = [9,10,11]

try:
	best_MAPE = np.load("ANN_best_MAPE")
except:
	best_MAPE = 10
MAPEA = []
for p in n_lookback:
	for i in window:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					
					train, test = df1[p:train_size,:],df1[train_size:len(df1),:]
					#convert dataset into right shape in order to input into the DNN
					trainX, trainY = convert2matrix(train, i)
					testX, testY = convert2matrix(test, i)

					model=Sequential()
					model.add(Dense(i, input_dim=i, activation='relu'))
					model.add(Dense(j, activation='relu'))
					model.add(Dense(1))
					model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse'])


					history=model.fit(trainX,trainY, epochs=k, batch_size=l, verbose=0, validation_data=(testX,testY))
					# train_score = model.evaluate(trainX, trainY, verbose=0)
					# Save the model to a file
					
					# test_score = model.evaluate(testX, testY, verbose=0)
					future_predictions = []
					x_input = np.reshape(np.concatenate([trainX[-1], [trainY[-1]]])[1:],(1,i))
					#x_input = trainX[-1]

					for m in range(9):
						pred = model.predict(x_input, verbose = 0)
						future_predictions.append(float(pred))
						x_input = np.concatenate([x_input[:,1:], pred.reshape(1,1)], axis=1)
									# model_loss(history)
					predictions =  future_predictions
					
					# prediction_plot(UK_ES, scaler.inverse_transform(np.array(predictions).reshape(-1,1)))
					MAPE = mape(scaler.inverse_transform(np.array(test).reshape(-1,1)),scaler.inverse_transform(np.array(predictions).reshape(-1,1)))
					MAPEA.append([MAPE,[i,j,k,l,p]])
					print("config", i,j,k,l,p)
					print("MAPE",MAPE)
					
					if MAPE <= best_MAPE:
						best_MAPE = MAPE
						best_model = model
						best_params = (i,j,k,l,p,"ANN")

best_model.save(f'my_model{best_params}')
np.save("ANN_best_MAPE", best_MAPE)
print(f'my_model{best_params}',"best")
RMSE = sorted(MAPEA)
print(MAPEA)
				

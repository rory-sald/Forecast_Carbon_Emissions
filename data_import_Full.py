import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
import pickle

data = {}
blank_df = pd.DataFrame({"Constituent Name":[np.nan],"CO2 Equivalent Emissions Total":[np.nan],"CO2 Equivalent Emissions Direct, Scope 1" : [np.nan], "CO2 Equivalent Emissions Indirect, Scope 2": [np.nan],"CO2 Equivalent Emissions Indirect, Scope 3":[np.nan]}, index = [np.nan])

df = pd.read_excel(r'"Input Directory"/FTSE 100 DATA.xlsx',sheet_name= "NEW") # change input directory to your directory
df.columns = df.iloc[1]
df = df.set_index("Constituent Name")
df = df.iloc[2:,]
df = df.dropna(how = "all")

# read data for each year, remove duplicated columns for each year ie where quarterly results are all the same 
for i in range(0, 52, 5):
    data[f"df_{int(i/5+2012)}"] = df.iloc[:,[1,2,59-i,60-i,61-i]]
    print(int(i/5+2012))


def extract(constituent_name, year, scope):
    selection = pd.DataFrame()
    

    if year == all:
        for j in range(2012,2023,1):
                selection = pd.concat([selection,(data[f"df_{j}"].loc[data[f"df_{j}"].index == constituent_name])])
                if len(selection) <= j - 2012:
                    #selection = pd.concat(selection(blank_df))
                    print("skip")      
                    print(constituent_name)
        selection.insert(0,"Year",[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022])
            
    else:       
        selection = data[f"df_{year}"].loc[data[f"df_{year}"].index == constituent_name]
        selection.insert(0,"Year",year)
    if scope == 1:
        selection = selection.loc[:,["CO2 Equivalent Emissions Direct, Scope 1"]]
    
    if scope == 2:
        selection = selection.loc[:,["CO2 Equivalent Emissions Indirect, Scope 2"]]

    if scope == 3:
        selection = selection.loc[:,["CO2 Equivalent Emissions Indirect, Scope 3"]]
    
    return selection

def test_acf(constituent_name, scope):
    test_model = np.asarray(extract(constituent_name, all, scope).iloc[:,0])
    print(scope)
    print(test_model)
    years = [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
    print(years)
    test_df = pd.DataFrame({f"Scope {scope} Emissions" : test_model, "year": years} )
    test_df.set_index("year",inplace = True)
    x = range(2012,2023,1)

    print(test_df)

    # Plot the ACF and PACF to help determine the AR and MA orders
    test_df.plot()
    plot_acf(test_df)
    autocorrelation_plot(test_df)

with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)


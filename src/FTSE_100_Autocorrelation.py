import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product
import matplotlib.pyplot as plt
import warnings
import pickle

from data_import_Full import extract

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

def AutoCorrelation(constituent_name):


    total_ES_df = pd.DataFrame()
    all_ES_S1_df = pd.DataFrame()
    all_ES_S2_df = pd.DataFrame()
    all_ES_S3_df = pd.DataFrame()

    def get_common_constituents(df):
        common_constituents = set(df.iloc[:, 0].values)  # Start with the constituents in the first column
        
        for col in df.columns[1:]:  # Iterate over the rest of the columns
            current_year_constituents = set(df[col].values)  # Get the constituents for the current column
            common_constituents = common_constituents.intersection(current_year_constituents)
        
        return list(common_constituents)


    # Loop through all of the years of data
    for i in range(2012, 2022,1):
            # Clean the data by removing rows with non numeric values in them, for Scope 1, 2 and 3
            clean1 = data[f"df_{i}"][pd.to_numeric(data[f"df_{i}"]['CO2 Equivalent Emissions Direct, Scope 1'], errors='coerce').notnull()] 
            clean2 = data[f"df_{i}"][pd.to_numeric(data[f"df_{i}"]['CO2 Equivalent Emissions Indirect, Scope 2'], errors='coerce').notnull()] 
            clean3 = data[f"df_{i}"][pd.to_numeric(data[f"df_{i}"]['CO2 Equivalent Emissions Indirect, Scope 3'], errors='coerce').notnull()] 
    
            for j in range(len(clean3)): 
                all_ES_S1_df[f"{i}"] = pd.Series(clean1.index)
                all_ES_S2_df[f"{i}"] = pd.Series(clean2.index)
                all_ES_S3_df[f"{i}"] = pd.Series(clean3.index)


    common_constituents_S1 = get_common_constituents(all_ES_S1_df)
    common_constituents_S2 = get_common_constituents(all_ES_S2_df)
    common_constituents_S3 = get_common_constituents(all_ES_S3_df)


    S1_sum = pd.DataFrame()
    s1 = 0
    S2_sum = pd.DataFrame()
    s2 = 0
    S3_sum = pd.DataFrame()
    s3 = 0
    for j in range(2012,2022,1):
        s1 = 0
        s2 = 0
        s3 = 0
        for i in common_constituents_S1:
            s1 += np.array(data[f"df_{j}"].loc[[i],["CO2 Equivalent Emissions Direct, Scope 1"]])
        S1_sum[f"{j}"] = [s1]
        for m in common_constituents_S2:
            s2 += np.array(data[f"df_{j}"].loc[[m],["CO2 Equivalent Emissions Indirect, Scope 2"]])
        S2_sum[f"{j}"] = [s2]       
        for p in common_constituents_S3:
            s3 += np.array(data[f"df_{j}"].loc[[p],["CO2 Equivalent Emissions Indirect, Scope 3"]])
        S3_sum[f"{j}"] = [s3]    
        
    S1_sum = S1_sum.rename(index ={0:"CO2 Equivalent Emissions Direct, Scope 1"})
    S2_sum = S2_sum.rename(index ={0:"CO2 Equivalent Emissions Indirect, Scope 2"})
    S3_sum = S3_sum.rename(index ={0:"CO2 Equivalent Emissions Indirect, Scope 3"})
    total_ES_df = pd.concat([S1_sum, S2_sum, S3_sum]).T



    firm_df = extract(constituent_name, all,0)[0:10].iloc[:,[0,3,4,5]]
    firm_df.set_index('Year', inplace=True)


    # total_es_df and firm_df should have columns named 'Year' and 'Scope 1', 'Scope 2', and 'Scope 3'

    # Convert the columns to float data type
    total_es_df = total_ES_df.astype(float)
    total_es_df.index.name = 'Year'
    firm_df = firm_df.astype(float)

    # Calculate the autocorrelation for each scope
    autocorrelations = {}
    for scope in ['CO2 Equivalent Emissions Direct, Scope 1', 'CO2 Equivalent Emissions Indirect, Scope 2', 'CO2 Equivalent Emissions Indirect, Scope 3']:
        # Normalize the data by subtracting the mean and dividing by the standard deviation
        # Compute the Pearson correlation coefficient using np.corrcoef()
        correlation = np.corrcoef(total_es_df[scope], firm_df[scope])[0, 1]
        autocorrelations[scope] = correlation

    # Print the results
    print("Autocorrelations:")
    print(autocorrelations)

    plt.subplot(211)
    plt.plot(firm_df["CO2 Equivalent Emissions Direct, Scope 1"])
    plt.subplot(212)
    plt.plot(total_es_df["CO2 Equivalent Emissions Direct, Scope 1"])
    plt.show()
    plt.subplot(211)
    plt.plot(firm_df["CO2 Equivalent Emissions Indirect, Scope 2"])
    plt.subplot(212)
    plt.plot(total_es_df["CO2 Equivalent Emissions Indirect, Scope 2"])
    plt.show()
    plt.subplot(211)
    plt.plot(firm_df["CO2 Equivalent Emissions Indirect, Scope 3"])
    plt.subplot(212)
    plt.plot(total_es_df["CO2 Equivalent Emissions Indirect, Scope 3"])
    plt.show()

AutoCorrelation("TESCO ORD")
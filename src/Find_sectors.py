import yfinance as yf
import requests
import pandas as pd
import pickle
import numpy as np
from yahooquery import Ticker


with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
print(data.keys())

Constituents = data["df_2012"]
z = []

for i in range(len(Constituents)):
        print("Constituent Name", Constituents.index[i])
        try:
            j = Constituents["New RICs"].iloc[i]
            sbux = yf.Ticker(f"{j}")
            z.append(sbux.info['symbol'])
        except:
            print("TRY OLD RIC")
            try:
                j = Constituents["Old RICs"].iloc[i]
                sbux = yf.Ticker(f"{j}")
                print(sbux,"2")
                z.append(sbux.info['symbol'])
            except:
                print("error")
                

            

symbols = set(z)
tickers = Ticker(symbols, asynchronous=True)
datasi = tickers.get_modules("summaryProfile quoteType")

dfsi = pd.DataFrame.from_dict(datasi).T
dataframes = [pd.json_normalize(dfsi[module].apply(lambda x: {} if not isinstance(x, dict) else x)) for module in ['summaryProfile', 'quoteType']]

dfsi = pd.concat(dataframes, axis=1)

symbols = dfsi['symbol'][114:,]

tickers = Ticker(symbols, asynchronous=True)
datasi1 = tickers.get_modules("summaryProfile quoteType")

dfsi1 = pd.DataFrame.from_dict(datasi1).T
dataframes = [pd.json_normalize(dfsi1[module].apply(lambda x: {} if not isinstance(x, dict) else x)) for module in ['summaryProfile', 'quoteType']]

dfsi1 = pd.concat(dataframes, axis=1)

tickers_sectors = pd.concat([dfsi[['symbol','sector']].iloc[0:114,],dfsi1[['symbol','sector']]])

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
Constituents = data["df_2012"]


# Merge on 'New RICs'
merged_new_rics = pd.merge(Constituents, tickers_sectors, left_on='New RICs', right_on='symbol', how='left')

# Merge on 'Old RICs'
merged_old_rics = pd.merge(Constituents, tickers_sectors, left_on='Old RICs', right_on='symbol', how='left')

# Combine the sector columns from both merges
combined_sector = merged_new_rics['sector'].fillna(merged_old_rics['sector'])

# Update the 'sector' column in the Constituents dataframe while keeping its index the same
Constituents.loc[:, 'sector'] = combined_sector.values

Empty=Constituents[Constituents.loc[:,'sector'].isna()]

Constituents.loc["ADMIRAL GROUP ORD"]['sector'] = "Financial Services"
Constituents.loc["ARM HOLDINGS ORD"]['sector'] = "Technology"
Constituents.loc["AUTO TRAD/D"]['sector'] = "Consumer Cyclical"
Constituents.loc["AVAST PLC ORD"]['sector'] = "Technology"
Constituents.loc["AVEVA GROUP ORD"]['sector'] = "Technology"
Constituents.loc["BAE SYSTEMS ORD"]['sector'] = "Industrials"
Constituents.loc["BERKELEY GRP UTS"]['sector'] = "Consuner Cyclical"
Constituents.loc["BILLITON ORD"]['sector'] = "Basic Materials"
Constituents.loc["BT GROUP ORD"]['sector'] = "Communication Services"
Constituents.loc["DIRECT LINE INSURANCE ORD SHS"]['sector'] = "Financial Services"
Constituents.loc["DIXONS CARPHONE PLC"]['sector'] = "Consumer Cyclical"
Constituents.loc["ELECTROCOMPONENTS ORD"]['sector'] = "Industrials"
Constituents.loc["EVRAZ PLC ORD"]['sector'] = "Basic Materials"
Constituents.loc["GKN ORD"]['sector'] = "Technology"
Constituents.loc["HOMESERVE ORD"]['sector'] = ""
Constituents.loc["GRP 4 SECURICOR ORD"]['sector'] = "Industrials"
Constituents.loc["INMARSAT ORD"]['sector'] = "Communication Services"
Constituents.loc["INTU PROPERTIES PLC"]['sector'] = "Real Estate"
Constituents.loc["JACKSON FINANCIAL INC - PLACEHOLDER"]['sector'] = ""
Constituents.loc["JUST EAT ORD SHS"]['sector'] = "Consumer Cyclical"
Constituents.loc["MEGGITT PLC"]['sector'] = "Industrials"
Constituents.loc["MERLIN ENTERTAINMENT GROUP ORD SHS"]['sector'] = "Consuner Cyclical"
Constituents.loc["MICRO FOCUS INTERNATIONAL CASH DUMMY"]['sector'] = "Technology"
Constituents.loc["MICRO FOCUS ORD"]['sector'] = "Technology"
Constituents.loc["NMC HEALTH PLC ORD"]['sector'] = "Healthcare"
Constituents.loc["PADDY POWER BETFAIR PLC ORD"]['sector'] = "Consumer Cyclical"
Constituents.loc["PROVIDENT FINCL ORD"]['sector'] = "Financial Services"
Constituents.loc["RANDGOLD RESRCS ORD"]['sector'] = "Basic Materials"
Constituents.loc["RECKITT BNCSR GRP ORD"]['sector'] = "Consumer Defensive"
Constituents.loc["REXAM PLC"]['sector'] = ""
Constituents.loc["ROYAL DUTCH SHELL ORD SH A"]['sector'] = "Basic Materials"
Constituents.loc["ROYAL DUTCH SHELL ORD SH B"]['sector'] = "Basic Materials"
Constituents.loc["ROYAL MAIL ORD SHS"]['sector'] = "Industrials"
Constituents.loc["RS GROUP PLC ORD"]['sector'] = "Industrials"
Constituents.loc["RSA INSURANCE GROUP ORD"]['sector'] = "Financial Services"
Constituents.loc["SABMILLER ORD"]['sector'] = "Consumer Cyclical"
Constituents.loc["SHELL PLC ORD"]['sector'] = "Basic Materials"
Constituents.loc["SHIRE ORD"]['sector'] = "Healthcare"
Constituents.loc["SKY PLC"]['sector'] = "Communication Services"
Constituents.loc["STANDARD LIFE ABERDEEN ORD"]['sector'] = "Financial Services"
Constituents.loc["TUI AG N"]['sector'] = "Consumer Cyclical"
Constituents.loc["WM MORRISON SUPERMARKETS ORD"]['sector'] = "Consumer Defensive"
Constituents.loc["WORLDPAY GROUP ORD SHS"]['sector'] = "Financial Services"


all_sectors = Constituents.loc[:,'sector']

# create a Pandas dataframe with "all_sectors" as a column
df = pd.DataFrame({'Sector': all_sectors})

# export the dataframe to an Excel file called "FTSE_sectors.xlsx"
df.to_excel('FTSE_sectors.xlsx', index=True)
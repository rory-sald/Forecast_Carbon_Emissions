
import pandas as pd

df = pd.read_excel(r'"Input directory here"/SovereignEmissionsDataset.xlsx') # change to the directory the file is stored on
df.columns = df.iloc[1]
df = df.iloc[2:,]
df = df.set_index("Party")
df = df.iloc[:,1:32]
df = df.rename(columns={'Last Inventory Year (2020)': 2020.0})


# select just UK data
UK_ES = df.loc["United Kingdom of Great Britain and Northern Ireland"]

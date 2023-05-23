#calc the UK total emissions at 100% in 1990 and 0 in 2050, normalized
#assume firm is at same level as uk total in 2012
#plot each firm with their emissions from 2012 to 2024, normalized
#if firm is above 0 at 2050 curve then they fail
# if firm is below then pass
from data_import_UK_Whole import UK_ES
import matplotlib.pyplot as plt
import seaborn as sns
from Plot_Final_model import scope_medians, scope_pred
from plot_final_year import scope_medians, scope_pred
from correlations import extract_sector, stack
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math

def scale_list(input_list,level):
    # Get the first value
    first_value = input_list[0]
    
    # Scale the rest of the list based on the first value
    scaled_list = [(i / first_value) * level for i in input_list]
    
    return scaled_list

sectors,_,_ = extract_sector(1)
df_percentage = sectors["Financial Services"].apply(lambda x: (x / x.max()) * 100, axis=0)
median = df_percentage.median(axis = 1).reset_index(drop=True)

# Scale the UK_ES list
UK_ES = list(UK_ES[:2019])
scaled_UK_ES = scale_list(UK_ES,100)

final_scaled_es1 = {}

for scope in range(1,4):
    sectors,_,unique_values = extract_sector(scope)
    scaled_sector = pd.DataFrame()
    for i, sector in enumerate(unique_values):
        df_percentage = sectors[sector].apply(lambda x: (x / x.max()) * 100, axis=0)
        median = df_percentage.median(axis = 1).reset_index(drop=True)

        values_to_scale = pd.Series(scope_medians1[scope].iloc[:,i])
        min_value = np.min(median)
        max_value = np.max(median)

        values_to_scale = values_to_scale.values.reshape(-1, 1)


        scaler = MinMaxScaler(feature_range=(min_value, max_value))


        scaled_values = scaler.fit_transform(values_to_scale)


        scaled_values = pd.Series(scaled_values.flatten())
        scaled_pred =  scaler.transform(np.array(scope_pred1[scope].iloc[:,i]).reshape(-1,1))
        combined_values = np.append(scaled_values, scaled_pred.flatten())
        scaled_sector[sector] = scale_list(combined_values[-6:],scaled_UK_ES[29])
        
    final_scaled_es[scope] = scaled_sector
  


years = np.array([2019, 2030, 2050])
percentages = np.array([56.25856567507, 50, 0])

num_of_sectors = len(unique_values)


num_of_pages = math.ceil(num_of_sectors / 6)

plt.figure(figsize=(10,6))



sector_index = 8
sector = unique_values[sector_index]



poly = np.polyfit(years, percentages, 2)


f = np.poly1d(poly)


years_new = np.linspace(2019, 2050, 100)

# Create a subplot for each sector
# Plot
plt.plot(years_new, f(years_new),label='On Track Emissions', color = 'red')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title(f'{sector} Percentage Emissions')

cmap1 = sns.color_palette("Greens")
cmap2 = sns.color_palette("Blues")
y1 = final_scaled_es[1][sector][:-3]
y2 = final_scaled_es1[2][sector][:-3]
y3 = final_scaled_es[3][sector][:-3]
y4 = final_scaled_es[1][sector][-4:]
y5 = final_scaled_es1[2][sector][-4:]
y6 = final_scaled_es[3][sector][-4:]



plt.plot(range(1990,2020), scaled_UK_ES,label = "UK Total Emissions" ,color = 'black')
plt.legend()
plt.plot(range(2019, 2022),y1,color = cmap1[5],label = "Known Scope 1 Emissions")
plt.plot(range(2019, 2022),y2,color = cmap1[3],label = "Known Scope 2 Emissions")
plt.plot(range(2019, 2022),y3,color = cmap1[1],label = "Known Scope 3 Emissions")
plt.plot(range(2021, 2025),y4,'--',color = cmap2[5],label = "Predicted Scope 1 Emissions")
plt.plot(range(2021, 2025),y5, '--',color = cmap2[3],label = "Predicted Scope 2 Emissions")
plt.plot(range(2021, 2025),y6,'--',color = cmap2[1],label = "Predicted Scope 3 Emissions")
plt.xlim(2018, 2025) 


plt.legend()
plt.show()
print("Scope 1 differance:", y4[5]- 55.277386760077206)
print("Scope 2 differance:", y5[5]- 55.277386760077206)
print("Scope 3 differance:", y6[5]- 55.277386760077206)

y1 = final_scaled_es1[1][sector][:-3]
y2 = final_scaled_es1[2][sector][:-3]
y3 = final_scaled_es1[3][sector][:-3]
y4 = final_scaled_es1[1][sector][-4:]
y5 = final_scaled_es1[2][sector][-4:]
y6 = final_scaled_es1[3][sector][-4:]
print("Scope 1 differance1:", y4[5]- 55.277386760077206)
print("Scope 2 differance1:", y5[5]- 55.277386760077206)
print("Scope 3 differance1:", y6[5]- 55.277386760077206)
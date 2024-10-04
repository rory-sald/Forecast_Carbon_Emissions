from data_import_Full import extract
import matplotlib
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_import_UK_Whole import UK_ES
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import tsaug

def all_sector_corr(scope):
    _,sector_normalized,_ = extract_sector(scope)
    # Calculate the correlations between columns
    correlations = sector_normalized.corr(method = "pearson")

    # Create a heatmap to visualize the correlations
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Use a diverging color palette
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap=cmap, center=0, cbar_kws={'pad': 0.05})

    # Increase the font size of the annotations and the title
    plt.title(f'Scope {scope} Sector Correlation Heatmap', fontsize=16, pad = 5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
def indiv_sector_corr(scope):
    sectors, _, unique_values = extract_sector(scope)
    for sector in unique_values:
        # Initialize an empty dictionary for the column mappings
        column_mapping = {}
        valid_i = 0

        # Iterate over the columns
        for column in sectors[sector].columns:
            # Get the corresponding value in RICs based on the column name
            data = extract(column, 2012, 0).loc[:, "New RICs"]

            # Map the column name to the value in RICs
            if column in data.index:
                column_mapping[column] = data.loc[column]

        # Rename the columns of 'sectors' DataFrame using the dictionary mapping
        sectors[sector].rename(columns=column_mapping, inplace=True)
    
    # Create the first plot with 6 subplots (page 1)
    fig1, axs1 = plt.subplots(3, 2, figsize=(8.27, 11.69), dpi=300, tight_layout=True)
    axs1 = axs1.ravel()  # Flatten the array for easier iteration
    
    # Create the second plot with 5 subplots (page 2)
    fig2, axs2 = plt.subplots(3, 2, figsize=(8.27, 11.69), dpi=300, tight_layout=True)
    axs2 = axs2.ravel()  # Flatten the array for easier iteration
    
    for i, sector in enumerate(unique_values):
        # Calculate the correlations between columns
        correlations = sectors[sector].corr(method="pearson")
        if len(sectors[sector].columns) <= 1:
            valid_i += -1
            continue
        else:
            i = i + valid_i
        
        # Plot the correlation heatmap on the corresponding subplot
        if i < 6:
            axs1[i].imshow(correlations, cmap='RdYlBu_r', vmin=-1, vmax=1)
            axs1[i].set_xticks(range(len(correlations.columns)))
            axs1[i].set_yticks(range(len(correlations.index)))
            axs1[i].set_xticklabels(correlations.columns, fontsize=8, rotation='vertical')
            axs1[i].set_yticklabels(correlations.index, fontsize=8)
            axs1[i].set_title(f'{sector} Correlation Heatmap', fontsize=8, pad=5)
            axs1[i].tick_params(bottom=False, left=False)
        else:
            j = i - 6
            axs2[j].imshow(correlations, cmap='RdYlBu_r', vmin=-1, vmax=1)
            axs2[j].set_xticks(range(len(correlations.columns)))
            axs2[j].set_yticks(range(len(correlations.index)))
            axs2[j].set_xticklabels(correlations.columns, fontsize=8, rotation='vertical')
            axs2[j].set_yticklabels(correlations.index, fontsize=8)
            axs2[j].set_title(f'{sector} Correlation Heatmap', fontsize=8)
            axs2[j].tick_params(bottom=False, left=False)

    # Normalize the colormap to the range of your data
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    # Create a scalar mappable object with the colormap and the normalization
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])

    # Remove the last subplot
    fig2.delaxes(axs2[-1])
    for p in range(abs(valid_i)):
        fig2.delaxes(axs2[-1-(p+1)])

    # Create a new subplot for the colorbar in the same location as the last subplot
    cax = fig2.add_subplot(3, 2, (6+valid_i))

    cb = fig2.colorbar(sm, cax=cax, orientation='vertical')
    cb.ax.yaxis.set_ticks_position('left')

    # Add a text annotation as the colorbar title
    cb_title = "Pearson Correlation Heatmap Legend"
    cax.annotate(cb_title, xy=(0.5, 1.03), xycoords='axes fraction',
                fontsize= 8, ha='center')

    # Adjust the colorbar label position
    cb.set_label("")

    # Optionally, adjust the colorbar tick labels fontsize
    cb.ax.tick_params(labelsize='smaller')


    # Save and show the figures as you were doing before
    plt.figure(fig1.number)
    plt.savefig(f'correlation_heatmaps_scope{scope}_page1.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(fig2.number)
    plt.savefig(f'correlation_heatmaps_scope{scope}page2.png', dpi=300, bbox_inches='tight')
    plt.show()


#fix this
def sector_vs_UK_corr(UK_ES, scope):
    _, sector_normalized, unique_values = extract_sector(scope)
    sector_UKwhole = pd.DataFrame(UK_ES[2012:2019])
    sector_UKwhole = sector_UKwhole.reset_index()
    
    fig, axs = plt.subplots(4, 3, figsize=(17, 22))
    axs = axs.ravel()  # Flatten the array for easier iteration

    # Calculate the correlations between columns
    correlations = sector_normalized.corr(method="pearson")

    # Create a heatmap to visualize the correlations
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Use a diverging color palette
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap=cmap, center=0, cbar = False)

    # Define the new xtick labels and their positions
    xtick_labels = ['FS', 'BM', 'T', 'I', 'CD', 'H', 'CC', 'E', 'RS', 'CS', 'U']
    xtick_positions = np.arange(0.5, len(xtick_labels) + 0.5)
    plt.xticks(xtick_positions, xtick_labels, fontsize=12)

    # Define the new ytick labels and their positions
    ytick_labels = ['FS', 'BM', 'T', 'I', 'CD', 'H', 'CC', 'E', 'RS', 'CS', 'U']
    ytick_positions = np.arange(0.5, len(ytick_labels) + 0.5)
    plt.yticks(ytick_positions, ytick_labels, fontsize=12)

    # Increase the font size of the annotations and the title
    plt.title(f'Scope {scope} Sector Correlation Heatmap', fontsize=14, pad=3)


    for i, sector in enumerate(unique_values):
        if sector_UKwhole.empty:
            sector_UKwhole = sector_normalized[sector]
        else:
            sector_UKwhole[sector] = sector_normalized[sector]

        scaler = StandardScaler()
        time_series1_standardized = scaler.fit_transform(np.asarray(sector_UKwhole.loc[:, "United Kingdom of Great Britain and Northern Ireland"]).reshape(-1, 1)).flatten()
        time_series2_standardized = scaler.fit_transform(np.asarray(sector_UKwhole.loc[:, sector]).reshape(-1, 1)).flatten()

        correlation, _ = pearsonr(time_series1_standardized, time_series2_standardized)
        sp_correlation, _ = spearmanr(time_series1_standardized, time_series2_standardized)

        cmap = sns.color_palette("Blues")
        mapping = {j: year for j, year in enumerate(range(2012, 2020), start=0)}

        axs[i].plot(time_series1_standardized, label='UK Total', marker='o', color=cmap[3])
        axs[i].plot(time_series2_standardized, label='Sector', marker='o', color=cmap[5])
        axs[i].grid(True)
        axs[i].set_xticks(list(mapping.keys()))
        axs[i].set_xticklabels(list(mapping.values()), rotation=45, ha='right')
        axs[i].set_title(f'Correlation between UK Total and {sector}: {correlation:.2f}', fontsize=14)
        axs[i].legend()

    plt.tight_layout()
    plt.show()

def stack(unique_values, sectors, e):
    sector_yearly_emissions = pd.DataFrame()
    for sector_name in unique_values:
        sector_df = sectors[sector_name]
        yearly_emissions = sector_df.sum(axis=1)
        sector_yearly_emissions[sector_name] = yearly_emissions

    # Transpose the DataFrame so that sectors become rows and years become columns
    sector_yearly_emissions = sector_yearly_emissions.T
    sector_yearly_emissions['Total Emissions'] = sector_yearly_emissions.sum(axis=1)
    sector_yearly_emissions = sector_yearly_emissions.sort_values(by='Total Emissions', ascending=False)
    sector_yearly_emissions = sector_yearly_emissions.drop(columns='Total Emissions')

    # Create a custom colormap from blue to green
    cmap = LinearSegmentedColormap.from_list('blue_to_green', ['blue', 'green'])

    # Generate the colors using the custom colormap
    num_sectors = len(unique_values)
    colors = cmap(np.linspace(0, 1, num_sectors))

    # Create the stacked area chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(range(2012, 2022), sector_yearly_emissions, labels=unique_values,colors = colors)
    ax.set_title(f'Stacked Area Chart of Scope {e} Emissions by Sector (2012-2022)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions')
    ax.legend(loc='upper left', labels=sector_yearly_emissions.index)


    plt.show()
    return sector_yearly_emissions

def extract_sector_standardized(e):
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    

    df_sectors = pd.read_excel(r'/Users/rems/Library/CloudStorage/OneDrive-UniversityofBath/IMEE_FYP/Code/FTSE_sectors.xlsx')
    new_index = df_sectors.iloc[:,0]  # Extract the first row values
    df_sectors = df_sectors.drop('Constituent Name', axis = 1)
    df_sectors.index = new_index
    nan_indices = {}

    for year in range(2012, 2022):
        data[f'df_{year}'] = data[f'df_{year}'].join(df_sectors)



    unique_values = data['df_2012']['Sector'].unique()[:-1]

    total = pd.DataFrame()

    df = data["df_2012"]
    sectors = {}
    sector_normalized = pd.DataFrame()
    for i in unique_values:
        firm_sum = pd.DataFrame()  # Initialize an empty DataFrame
        x = df[df['Sector'] == i]
        for j in x.index:
            q = extract(j, all, e)[:-1]
            q = q.reset_index(drop=True)  # Reset the index
            if e ==1:
                q = q.rename(columns={"CO2 Equivalent Emissions Direct, Scope 1": j})  # Rename the columns
                if q.isna().any().any():
                    continue
            elif e == 2:
                q = q.rename(columns={"CO2 Equivalent Emissions Indirect, Scope 2": j})  # Rename the columns
                if q.isna().any().any():
                    continue
            else:
                q = q.rename(columns={"CO2 Equivalent Emissions Indirect, Scope 3": j})  # Rename the columns
                if q.isna().any().any():
                    continue
            if firm_sum.empty:
                q = np.array(q).reshape(1,-1)
                q = tsaug.Resize(size=100).augment(q)
                q = tsaug.AddNoise(scale=0.02).augment(q)
                firm_sum = pd.DataFrame(q[0],columns = [j])
            else:
                # If not, append the column from q to firm_sum
                b = np.array(q[j]).reshape(1,-1)
                b = tsaug.Resize(size=100).augment(b)
                b = tsaug.AddNoise(scale=0.02).augment(b)
                firm_sum[j] = b[0]

        sector_normalized[i] = firm_sum.astype(float).apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis = 1)
        # sector_normalized[i] = firm_sum.astype(float)..mean(axis = 1)
        sectors[i] = firm_sum.astype(float)
    return sectors,  unique_values

def extract_sector(e):
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    

    df_sectors = pd.read_excel(r'/Users/rems/Library/CloudStorage/OneDrive-UniversityofBath/IMEE_FYP/Code/FTSE_sectors.xlsx')
    new_index = df_sectors.iloc[:,0]  # Extract the first row values
    df_sectors = df_sectors.drop('Constituent Name', axis = 1)
    df_sectors.index = new_index
    nan_indices = {}

    for year in range(2012, 2022):
        data[f'df_{year}'] = data[f'df_{year}'].join(df_sectors)



    unique_values = data['df_2012']['Sector'].unique()[:-1]

    total = pd.DataFrame()

    df = data["df_2012"]
    sectors = {}
    sector_normalized = pd.DataFrame()
    for i in unique_values:
        firm_sum = pd.DataFrame()  # Initialize an empty DataFrame
        x = df[df['Sector'] == i]
        for j in x.index:
            q = extract(j, all, e)[:-1]
            q = q.reset_index(drop=True)  # Reset the index
            if e ==1:
                q = q.rename(columns={"CO2 Equivalent Emissions Direct, Scope 1": j})  # Rename the columns
                if q.isna().any().any():
                    continue
            elif e == 2:
                q = q.rename(columns={"CO2 Equivalent Emissions Indirect, Scope 2": j})  # Rename the columns
                if q.isna().any().any():
                    continue
            else:
                q = q.rename(columns={"CO2 Equivalent Emissions Indirect, Scope 3": j})  # Rename the columns
                if q.isna().any().any():
                    continue
            if firm_sum.empty:
                
                firm_sum = q
            else:

                firm_sum[j] = q[j]

        sector_normalized[i] = firm_sum.astype(float).apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis = 1)
        # sector_normalized[i] = firm_sum.astype(float)..mean(axis = 1)
        sectors[i] = firm_sum.astype(float)
    return sectors, sector_normalized, unique_values




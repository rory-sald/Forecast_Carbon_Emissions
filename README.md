# Forecasting Carbon Emissions in FTSE 100 Companies

## Project Overview
This project aims to predict Scope 1, 2, and 3 carbon emissions for companies listed in the FTSE 100 index using a variety of statistical and machine learning models. The goal is to provide insights into the carbon emission trends across sectors and aid in identifying companies that are on track to meet the UK’s carbon neutrality targets.

## Key Objectives
- **Modeling Carbon Emissions**: Applying different forecasting techniques to predict emissions for various sectors.
- **Performance Comparison**: Assessing the accuracy and applicability of models such as ARIMA, ANN, CNN, LSTM, and Grey Models.
- **Sector-Specific Insights**: Identifying sectoral differences in carbon emissions and their implications on policy and investment.

## Data Sources
- **Carbon Emissions Data**: Data from Refinitiv for FTSE 100 companies (2012-2022) used for modeling Scope 1, 2, and 3 emissions.
- **UK Total Emissions**: Historical data from the UNFCCC to provide context and benchmarks for national emission trends.

*Note*: Due to legal restrictions, the dataset is not included in this repository.

## Models Used
1. **Linear Regression**: Simple model to benchmarj performance.
2. **ARIMA**: Time series forecasting for emissions with autoregressive components.
3. **ANN (Artificial Neural Networks)**: Predicting carbon emissions using past trends and factors.
4. **LSTM (Long Short-Term Memory)**: Sequential model capable of capturing long-term dependencies in emission data.
5. **CNN (Convolutional Neural Networks)**: Applied for time series data with specific patterns.
6. **Grey Models**: Used for smaller datasets with unknown characteristics.

## Methodology
The models were trained on a sliding window dataset created from historical carbon emissions data. Data preprocessing techniques, such as linear interpolation and scaling, were employed to handle missing data and prepare the input features. Each model was tuned and evaluated using metrics like RMSE and R-squared.

## Results
- **Best Performing Models**: ANN models showed the highest performance across all three emission scopes (1, 2, and 3). 
- **Sector Insights**: Different sectors exhibited varied trends in emission reductions, with the Energy and Industrials sectors being among the highest emitters, while Financial Services had comparatively low direct emissions.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rory-sald/Forecast_Carbon_Emissions.git
   ```

2. **Install the necessary dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the models**:
   - Use the scripts in the `src/` directory for:
     - **Data preprocessing**: Files in `src/data_processing/`.
     - **Model training**: Files in `src/models/`.
     - **Evaluation**: Files in `src/evaluation/`.

## Data Availability
Scope 3 emissions are estimated for some companies due to a lack of mandatory reporting.

## Generalization
The models are trained on a specific dataset (FTSE 100), so caution should be taken when applying them to other datasets or companies.

## Future Work
### Model Enhancements
Further exploration of model optimization techniques such as transfer learning and more granular hyperparameter tuning.

### Additional Data
Expanding the dataset with more recent emissions data and incorporating other economic variables.

## License
This project is for educational purposes, and the data provided by Refinitiv is not included due to licensing restrictions.

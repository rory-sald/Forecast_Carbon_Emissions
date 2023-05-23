from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
import plotly
from collections import defaultdict
import warnings
from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results


warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Create a date range for the "ts" column
date_range = pd.date_range(start='2020-01-01', end='2022-12-01', freq='MS')

# Generate random sales data
np.random.seed(42)
sales_data = np.arange(36)

# Create the DataFrame
df = pd.DataFrame(data={'ts': date_range, 'count': sales_data})

print(df.head())


config = ForecastConfig(
     metadata_param=MetadataParam(time_col="ts", value_col="count"),  # Column names in `df`
     model_template=ModelTemplateEnum.AUTO.name,  # AUTO model configuration
     forecast_horizon=24,   # Forecasts 24 steps ahead
     coverage=0.95,         # 95% prediction intervals
 )

# Creates forecasts
forecaster = Forecaster()
result = forecaster.run_forecast_config(df=df, config=config)

# Accesses results
result.forecast     # Forecast with metrics, diagnostics
result.backtest     # Backtest with metrics, diagnostics
result.grid_search  # Time series CV result
result.model        # Trained model
result.timeseries   # Processed time series with plotting functions

forecast = result.forecast
fig = forecast.plot()
plotly.io.show(fig)
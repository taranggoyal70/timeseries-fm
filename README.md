# Chronos-2 Multivariate Forecasting Experiments

Converted from Jupyter Notebook to Python modules for submission.

## Research Questions

1. Do multivariate (MV) methods produce better predictions than univariate (UV) ones when foundation models are used?
2. Is MV forecasting accuracy better for stocks versus interest rates?
3. Is MV forecasting better when both stocks and interest rates are forecast together?
4. Can we build a large-scale "world" forecasting model?

## Project Structure

```
chronos2_experiments/
├── data_loader.py              # Downloads Mag-7 stocks and FRED interest rates
├── metrics_calculator.py       # Calculates RMSE and MAPE metrics
├── chronos_experiment_runner.py # Runs UV vs MV forecasting experiments
├── main.py                     # Main script to run experiments
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Experiment

```bash
python main.py
```

### Use Individual Modules

```python
from data_loader import DataLoader
from chronos_experiment_runner import ChronosExperimentRunner
from datetime import datetime

# Load data
loader = DataLoader()
stocks_df = loader.download_stocks(start_date="2000-01-01")

# Run experiment
runner = ChronosExperimentRunner()
stocks_df['item_id'] = 'stocks'

result = runner.run_single_experiment(
    df=stocks_df,
    target_date=datetime(2025, 3, 31),
    n=252,
    m=21,
    series_name='NVDA',
    dataset_type='stocks'
)

print(result)
```

## Modules

### data_loader.py
- `DataLoader` class
- Downloads Magnificent-7 stocks (K=7)
- Downloads FRED interest rates (K=10)
- Creates combined dataset (K=17)

### metrics_calculator.py
- `MetricsCalculator` class
- Calculates RMSE: `[1/m * sum((x-y)^2)]^(1/2)`
- Calculates MAPE: `1/m * sum(|x-y|/|x|) * 100`
- Compares UV vs MV metrics

### chronos_experiment_runner.py
- `ChronosExperimentRunner` class
- Loads Chronos-2 model from HuggingFace
- Runs univariate (UV) forecasting
- Runs multivariate (MV) forecasting
- Compares UV vs MV performance

### main.py
- Orchestrates complete experiment workflow
- Downloads data
- Initializes Chronos-2
- Runs test experiment (n=252, m=21, t=03/31/2025)
- Displays results

## Data

- **Stocks**: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
- **Interest Rates**: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
- **Date Range**: 2000-01-01 to present
- **Frequency**: Business days (B)

## Experiment Parameters

- **n**: Context length (252 = 1 year of trading days)
- **m**: Prediction length (21 = 1 month of trading days)
- **α**: n/252 (years of context)
- **Target Date**: 2025-03-31

## Output

Results include:
- UV RMSE and MAPE
- MV RMSE and MAPE
- Improvement percentages
- Boolean indicators for MV superiority
- Actual values and predictions

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Internet connection (for data download)

## Notes

- First run downloads Chronos-2 model (~478MB)
- Data is downloaded fresh on each run
- Uses business day frequency (excludes weekends/holidays)
- Forward-fills missing values

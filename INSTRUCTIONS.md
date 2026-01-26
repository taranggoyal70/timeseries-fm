# Chronos-2 Multivariate Forecasting - Complete Instructions

## Project Overview

This project implements multivariate (MV) vs univariate (UV) time series forecasting using Amazon's Chronos-2 foundation model. The goal is to answer key research questions about whether multivariate methods outperform univariate methods for financial forecasting.

**GitHub Repository**: https://github.com/taranggoyal70/timeseries-fm

---

## Research Questions

1. **Do multivariate (MV) methods produce better predictions than univariate (UV) ones when foundation models are used for both?**
2. **Is MV forecasting accuracy better for stocks versus interest rates?**
3. **Is MV forecasting better when both stocks and interest rates are forecast together?**
4. **Can we build a large-scale "world" forecasting model?**

---

## Datasets

### 1. Magnificent-7 Stocks (K=7)
- **Tickers**: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
- **Source**: Yahoo Finance
- **Frequency**: Business days (trading days)

### 2. FRED Interest Rates (K=10)
- **Maturities**: 3-Month, 6-Month, 1-Year, 2-Year, 3-Year, 5-Year, 7-Year, 10-Year, 20-Year, 30-Year
- **Source**: Federal Reserve Economic Data (FRED)
- **Frequency**: Business days

### 3. Combined Dataset (K=17)
- **Series**: All stocks + all interest rates
- **Purpose**: Test if combining different asset classes improves forecasting

---

## Methodology

### Univariate (UV) Forecasting
- **Input**: Single time series of length `n`
- **Output**: Forecast of next `m` values for that series only
- **Formula**: Given observations `{x_{t-n+1}, ..., x_t}`, forecast `{x_{t+1}, ..., x_{t+m}}`

### Multivariate (MV) Forecasting
- **Input**: K time series, each of length `n`
- **Output**: Forecast of next `m` values for target series using relationships with other series
- **Formula**: Given `{x_{k,t-n+1}, ..., x_{k,t}}` for k=1...K, forecast `{x_{i,t+1}, ..., x_{i,t+m}}`

### Error Metrics

**1. Root Mean Squared Error (RMSE)**
```
RMSE_i = sqrt(1/m * sum((x_{i,t+h} - y_{i,t+h})^2))
```

**2. Mean Absolute Percentage Error (MAPE)**
```
MAPE = 1/m * sum(|x_{i,t+h} - y_{i,t+h}| / |x_{i,t+h}|) * 100
```

- **Primary Metric**: MAPE (normalized, allows comparison across different scales)
- **Secondary Metric**: RMSE (absolute error)

---

## Experimental Parameters

### History Length (n)
- **Formula**: n = α × 252 (trading days)
- **Values**: α ∈ {0.5, 1, 2, 3}
- **Results**: n ∈ {126, 252, 504, 756} days

### Forecast Horizon (m)
- **Values**: m ∈ {21, 63} days
- **Meaning**: 21 days ≈ 1 month, 63 days ≈ 3 months

### Time Period
- **Range**: January 1, 2000 to September 30, 2025
- **Rolling Forecasts**: Monthly steps (1st of each month)
- **Note**: Data after 2023 is likely out-of-sample for Chronos-2 training

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for large model)

### Step 1: Install Dependencies

```bash
cd chronos2-forecasting
pip install -r requirements.txt
```

**Required packages:**
- `pandas` - Data manipulation
- `torch` - PyTorch for model inference
- `chronos-forecasting` - Chronos-2 model
- `yfinance` - Yahoo Finance data
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `pandas-datareader` - Additional data sources
- `fredapi` - FRED API (optional)
- `jupyter` - Jupyter notebooks
- `tqdm` - Progress bars
- `python-dateutil` - Date utilities

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import chronos; print('Chronos-2: OK')"
python -c "import yfinance; print('yfinance: OK')"
```

---

## How to Run

### Option 1: Quick Test (Recommended for First Run)

Run a quick test with limited parameters to verify everything works:

```bash
python main.py --quick-test --dataset stocks
```

This will:
- Download Magnificent-7 stock data
- Run experiments with n=252, m=21
- Use 2024 data only (quarterly steps)
- Complete in ~10-30 minutes

### Option 2: Full Experiment Suite

Run all experiments as specified in the README:

```bash
python main.py --dataset all
```

This will:
- Download all datasets (stocks, rates, combined)
- Run all parameter combinations (α={0.5,1,2,3}, m={21,63})
- Process monthly rolling forecasts from 2000-2025
- **Warning**: This may take several hours to days depending on hardware

### Option 3: Specific Dataset

Run experiments on a specific dataset:

```bash
# Stocks only (K=7)
python main.py --dataset stocks

# Interest rates only (K=10)
python main.py --dataset rates

# Combined (K=17)
python main.py --dataset combined
```

### Option 4: Download Data Only

Download and prepare data without running experiments:

```bash
python main.py --download-only
```

### Option 5: Use Existing Data

Skip data download and use previously downloaded data:

```bash
python main.py --skip-download --dataset stocks
```

---

## Command-Line Arguments

```bash
python main.py [OPTIONS]
```

**Options:**

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | stocks, rates, combined, all | all | Dataset to use |
| `--model-size` | small, base, large | base | Chronos-2 model size |
| `--device` | cuda, cpu | cuda | Device for inference |
| `--start-date` | YYYY-MM-DD | 2000-01-01 | Experiment start date |
| `--end-date` | YYYY-MM-DD | 2025-09-30 | Experiment end date |
| `--download-only` | flag | - | Only download data |
| `--skip-download` | flag | - | Use existing data |
| `--quick-test` | flag | - | Quick test mode |

**Examples:**

```bash
# Quick test with CPU
python main.py --quick-test --device cpu

# Full stocks experiment with large model
python main.py --dataset stocks --model-size large

# Custom date range
python main.py --dataset rates --start-date 2020-01-01 --end-date 2024-12-31
```

---

## Project Structure

```
chronos2-forecasting/
├── main.py                          # Main execution script (NEW)
├── INSTRUCTIONS.md                  # This file (NEW)
├── README.md                        # Project overview
├── requirements.txt                 # Python dependencies
│
├── data_loader.py                   # Data download and preparation
├── chronos_forecaster.py            # UV/MV forecasting functions
├── chronos_experiment_runner.py     # Experiment orchestration
├── experiment_config.py             # Configuration dataclass
├── metrics_calculator.py            # RMSE and MAPE calculations
├── metrics.py                       # Legacy metrics (kept for compatibility)
├── visualizer.py                    # Plotting functions
│
├── data/                            # Downloaded data (created on first run)
│   ├── stocks.csv
│   ├── interest_rates.csv
│   └── combined.csv
│
├── results/                         # Experiment results (created on run)
│   ├── experiments_YYYYMMDD_HHMMSS.json
│   └── experiments_summary_YYYYMMDD_HHMMSS.csv
│
└── *.ipynb                          # Jupyter notebooks for exploration
```

---

## Code Architecture

### 1. Data Loading (`data_loader.py`)

**Class**: `DataLoader`

**Methods**:
- `download_stocks()` - Download Magnificent-7 from Yahoo Finance
- `download_interest_rates()` - Download FRED rates
- `download_combined()` - Download and merge both datasets
- `load_data()` - Load previously downloaded data

**Output Format**:
```python
DataFrame with columns:
- timestamp: datetime
- AAPL, MSFT, ... : float (stock prices or rates)
```

### 2. Forecasting (`chronos_forecaster.py`, `chronos_experiment_runner.py`)

**Class**: `ChronosExperimentRunner`

**Key Methods**:
- `forecast_univariate()` - UV forecast for single series
- `forecast_multivariate()` - MV forecast using all series
- `run_single_experiment()` - Single UV vs MV comparison
- `run_rolling_experiments()` - Full rolling forecast suite

**Process**:
1. Load Chronos-2 model
2. For each (date, n, m, series) combination:
   - Extract context data (n days before date)
   - Run UV forecast
   - Run MV forecast
   - Compare with actual values
   - Calculate metrics

### 3. Metrics (`metrics_calculator.py`)

**Class**: `MetricsCalculator`

**Methods**:
- `calculate_rmse()` - Per README formula
- `calculate_mape()` - Per README formula (returns percentage)
- `calculate_all_metrics()` - Both RMSE and MAPE
- `compare_uv_mv_metrics()` - UV vs MV comparison

### 4. Configuration (`experiment_config.py`)

**Class**: `ExperimentConfig`

**Parameters**:
- `alpha_values` - History multipliers [0.5, 1, 2, 3]
- `forecast_horizons` - [21, 63] days
- `start_date`, `end_date` - Time range
- `model_size` - 'small', 'base', or 'large'
- `device` - 'cuda' or 'cpu'

---

## Output Files

### 1. JSON Results (`experiments_YYYYMMDD_HHMMSS.json`)

Full experiment results including:
- All parameters (dataset, series, date, n, m, α)
- UV and MV metrics (RMSE, MAPE)
- Actual values and predictions
- Improvement percentages

**Structure**:
```json
[
  {
    "dataset": "stocks",
    "series": "AAPL",
    "target_date": "2024-01-01",
    "n": 252,
    "m": 21,
    "alpha": 1.0,
    "uv_rmse": 5.23,
    "mv_rmse": 4.87,
    "uv_mape": 2.45,
    "mv_mape": 2.12,
    "mape_improvement_pct": 13.47,
    "mv_better_mape": true,
    "actual_values": [...],
    "uv_predictions": [...],
    "mv_predictions": [...]
  },
  ...
]
```

### 2. CSV Summary (`experiments_summary_YYYYMMDD_HHMMSS.csv`)

Tabular summary without prediction arrays (easier to analyze in Excel/Pandas).

**Columns**:
- dataset, series, target_date, n, m, alpha
- uv_rmse, mv_rmse, rmse_improvement_pct, mv_better_rmse
- uv_mape, mv_mape, mape_improvement_pct, mv_better_mape

---

## Analysis & Interpretation

### Reading Results

**Positive MAPE improvement** = MV is better than UV
**Negative MAPE improvement** = UV is better than MV

**Example**:
```
uv_mape: 3.5%
mv_mape: 2.8%
mape_improvement_pct: 20.0%
mv_better_mape: true
```
→ MV reduced error by 20%, MV is better

### Key Questions to Answer

1. **Overall MV vs UV**: What % of experiments does MV win?
2. **By Dataset**: Does MV work better for stocks, rates, or combined?
3. **By Horizon**: Does MV advantage increase for longer forecasts (m=63 vs m=21)?
4. **By History**: Does more history (larger α) help MV more than UV?
5. **Temporal**: Did performance change after 2023 (potential training cutoff)?

### Analysis Code Example

```python
import pandas as pd

# Load results
df = pd.read_csv('results/experiments_summary_YYYYMMDD_HHMMSS.csv')

# Overall MV win rate
mv_win_rate = (df['mv_better_mape'] == True).mean() * 100
print(f"MV Win Rate: {mv_win_rate:.1f}%")

# By dataset
print(df.groupby('dataset')['mv_better_mape'].mean() * 100)

# By forecast horizon
print(df.groupby('m')['mape_improvement_pct'].mean())

# Top performers
top_mv = df.nlargest(10, 'mape_improvement_pct')[['series', 'dataset', 'mape_improvement_pct']]
print(top_mv)
```

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Use smaller model or CPU
```bash
python main.py --model-size small --device cpu
```

### Issue: Data download fails

**Solution**: Check internet connection, try manual download
```python
from data_loader import DataLoader
loader = DataLoader()
loader.download_stocks()  # Try each dataset separately
```

### Issue: "No module named 'chronos'"

**Solution**: Install chronos-forecasting
```bash
pip install chronos-forecasting
```

### Issue: Experiments taking too long

**Solution**: Use quick test mode or limit date range
```bash
python main.py --quick-test
# OR
python main.py --start-date 2023-01-01 --end-date 2024-12-31
```

### Issue: Missing data for certain dates

**Expected**: Some dates may be skipped if insufficient history or future data
**Check**: Results will show actual number of completed experiments

---

## Advanced Usage

### Class-Style Examples

We've created examples following the professor's teaching approach:

**1. Magnificent-7 Example** (matches class notes exactly):
```bash
python class_example_mag7.py
```
- Downloads Magnificent-7 stocks
- Runs UV vs MV comparison
- Calculates MAPE and RMSE
- Generates comparison plots

**2. Notebook-Style Script**:
```bash
python class_style_notebook.py
```
- Follows professor's Jupyter notebook style
- Stock forecasting example
- Interest rate forecasting example
- Step-by-step with explanations

### Using Jupyter Notebooks

Explore data and results interactively:

```bash
jupyter notebook
```

Open `chronos2_experiments.ipynb` or create new notebook:

```python
from data_loader import DataLoader
from chronos_experiment_runner import ChronosExperimentRunner
from experiment_config import ExperimentConfig

# Load data
loader = DataLoader()
stocks = loader.load_data('stocks')

# Configure
config = ExperimentConfig(
    alpha_values=[1.0],
    forecast_horizons=[21],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Run
runner = ChronosExperimentRunner(config)
results = runner.run_rolling_experiments(stocks, 'stocks', ['AAPL'])

# Analyze
import pandas as pd
df = pd.DataFrame(results)
print(df[['target_date', 'uv_mape', 'mv_mape', 'mape_improvement_pct']])
```

### Custom Experiments

Modify `experiment_config.py` or create custom config:

```python
from experiment_config import ExperimentConfig

custom_config = ExperimentConfig(
    alpha_values=[1.5, 2.5],  # Custom history lengths
    forecast_horizons=[10, 30],  # Custom horizons
    start_date='2022-01-01',
    end_date='2024-12-31',
    step_months=2,  # Bi-monthly instead of monthly
    model_size='large',
    device='cuda'
)
```

---

## Performance Expectations

### Runtime Estimates (on GPU)

| Configuration | Experiments | Estimated Time |
|--------------|-------------|----------------|
| Quick test (stocks) | ~100 | 10-30 minutes |
| Single dataset (full) | ~5,000-10,000 | 4-8 hours |
| All datasets (full) | ~30,000-50,000 | 1-3 days |

**Factors affecting speed**:
- Model size (small < base < large)
- Device (GPU >> CPU)
- Number of series (K)
- Date range and step size

### Resource Requirements

| Model Size | GPU Memory | RAM | Disk Space |
|------------|-----------|-----|------------|
| Small | 2-4 GB | 8 GB | 5 GB |
| Base | 4-8 GB | 16 GB | 10 GB |
| Large | 8-16 GB | 32 GB | 20 GB |

---

## What We Have Done

### ✅ Completed Tasks

1. **Data Fetching** (`data_loader.py`)
   - Implemented automatic download from Yahoo Finance and FRED
   - Handles all 3 datasets (stocks, rates, combined)
   - Proper business day frequency and forward-fill for missing data
   - Saves to CSV for reuse

2. **Forecasting Engine** (`chronos_forecaster.py`, `chronos_experiment_runner.py`)
   - UV forecasting: Single series input
   - MV forecasting: Multiple series input (proper implementation without future covariates)
   - Rolling forecast framework
   - Progress tracking with tqdm

3. **Metrics Calculation** (`metrics_calculator.py`)
   - RMSE per README formula
   - MAPE per README formula (percentage output)
   - UV vs MV comparison with improvement percentages
   - Handles edge cases (division by zero, etc.)

4. **Experiment Configuration** (`experiment_config.py`)
   - Dataclass for all parameters
   - Default values per README specifications
   - Easy customization

5. **Main Execution Script** (`main.py`) ⭐ NEW
   - Command-line interface
   - All experiment modes (quick test, full, by dataset)
   - Data download management
   - Results saving and summary statistics

6. **Documentation** (`INSTRUCTIONS.md`) ⭐ NEW
   - Complete usage guide
   - Methodology explanation
   - Troubleshooting
   - Analysis examples

### 📊 Code Verification

All code has been reviewed for correctness:
- ✅ RMSE formula matches README
- ✅ MAPE formula matches README (with percentage conversion)
- ✅ UV/MV forecasting follows Chronos-2 best practices
- ✅ Data preprocessing handles missing values and frequency
- ✅ Rolling forecasts implemented correctly
- ✅ Error handling and progress tracking

---

## Next Steps

1. **Run Quick Test**
   ```bash
   python main.py --quick-test --dataset stocks
   ```

2. **Verify Results**
   - Check `results/` directory for output files
   - Review summary statistics in terminal

3. **Run Full Experiments** (when ready)
   ```bash
   python main.py --dataset all
   ```

4. **Analyze Results**
   - Load CSV into Pandas/Excel
   - Answer research questions
   - Create visualizations

5. **Iterate**
   - Adjust parameters if needed
   - Test different model sizes
   - Explore specific time periods

---

## References

- **Chronos-1 Paper**: https://arxiv.org/abs/2403.07815
- **Chronos-2 Paper**: https://arxiv.org/abs/2510.15821
- **HuggingFace**: https://huggingface.co/amazon/chronos-2
- **GitHub**: https://github.com/amazon-science/chronos-forecasting
- **Amazon Science Blog**: https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
- **Professor's Class Notes**: https://srdas.github.io/NLPBook/Chronos.html (⭐ IMPORTANT - See UV vs MV examples)

---

## Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Try quick test mode first
4. Check GitHub repository issues
5. Verify all dependencies are installed

---

**Last Updated**: January 25, 2026
**Version**: 1.0
**Status**: Ready for use ✅

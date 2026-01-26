# Python Code Structure - Chronos-2 Forecasting

## Complete File Structure

```
chronos2-forecasting/
│
├── Core Execution Scripts
│   ├── main.py                          ⭐ Main CLI script - runs experiments
│   ├── run_complete_pipeline.py         ⭐ Complete pipeline (experiments + viz)
│   ├── analyze_results.py               ⭐ Results analysis & visualization
│   └── verify_structure.py              ⭐ Structure verification
│
├── Core Modules
│   ├── data_loader.py                   Data download & preparation
│   ├── chronos_forecaster.py            UV/MV forecasting wrapper
│   ├── chronos_experiment_runner.py     Experiment orchestration
│   ├── experiment_config.py             Configuration dataclass
│   ├── metrics_calculator.py            RMSE/MAPE calculations
│   ├── metrics.py                       Legacy metrics (compatibility)
│   └── visualizer.py                    Plotting functions
│
├── Example Scripts
│   ├── example_single_forecast.py       Simple single experiment
│   ├── class_example_mag7.py            Magnificent-7 example
│   └── class_style_notebook.py          Notebook-style example
│
├── Configuration
│   └── requirements.txt                 Python dependencies
│
└── Output Directories (created on run)
    ├── data/                            Downloaded datasets
    ├── results/                         Experiment results (JSON/CSV)
    └── visualizations/                  Generated plots
```

## Module Dependencies

```
main.py
├── data_loader.DataLoader
├── chronos_experiment_runner.ChronosExperimentRunner
└── experiment_config.ExperimentConfig

chronos_experiment_runner.py
├── chronos.Chronos2Pipeline
├── experiment_config.ExperimentConfig
└── metrics_calculator.MetricsCalculator

chronos_forecaster.py
└── chronos.Chronos2Pipeline

analyze_results.py
└── (standalone - reads from results/)

run_complete_pipeline.py
├── main.py (subprocess)
└── analyze_results.py (subprocess)
```

## How Components Connect

### 1. Data Flow

```
data_loader.py
    ↓ (downloads & prepares data)
main.py
    ↓ (loads data & config)
chronos_experiment_runner.py
    ↓ (runs UV/MV forecasts)
chronos_forecaster.py
    ↓ (calls Chronos-2 model)
metrics_calculator.py
    ↓ (calculates RMSE/MAPE)
results/ (JSON/CSV)
    ↓
analyze_results.py
    ↓ (generates visualizations)
visualizations/ (PNG files)
```

### 2. Execution Paths

**Path 1: Quick Example**
```bash
python example_single_forecast.py
# Uses: data_loader → chronos_experiment_runner → metrics_calculator
```

**Path 2: Main Experiments**
```bash
python main.py --quick-test --dataset stocks
# Uses: data_loader → chronos_experiment_runner → metrics_calculator
# Outputs: results/*.json, results/*.csv
```

**Path 3: Complete Pipeline**
```bash
python run_complete_pipeline.py --quick-test --dataset stocks
# Runs: main.py → analyze_results.py
# Outputs: results/*.json, results/*.csv, visualizations/*.png
```

**Path 4: Analysis Only**
```bash
python analyze_results.py
# Reads: results/*.json, results/*.csv
# Outputs: visualizations/*.png, visualizations/summary_report.txt
```

## Key Classes & Functions

### data_loader.py
```python
class DataLoader:
    def download_stocks(start_date, end_date) → DataFrame
    def download_interest_rates(start_date, end_date) → DataFrame
    def download_combined(start_date, end_date) → DataFrame
    def load_data(dataset_type) → DataFrame
```

### chronos_experiment_runner.py
```python
class ChronosExperimentRunner:
    def __init__(config: ExperimentConfig)
    def forecast_univariate(context_df, target, m) → DataFrame
    def forecast_multivariate(context_df, target, m) → DataFrame
    def run_single_experiment(...) → Dict
    def run_rolling_experiments(...) → List[Dict]
    def save_results(results, output_dir) → (json_path, csv_path)
```

### chronos_forecaster.py
```python
class ChronosForecaster:
    def __init__(device, model_size)
    def forecast_univariate(context_df, target, m) → DataFrame
    def forecast_multivariate(context_df, target, m) → DataFrame
    def compare_uv_mv(context_df, test_df, target, m) → Dict
```

### metrics_calculator.py
```python
class MetricsCalculator:
    @staticmethod
    def calculate_rmse(actual, predicted) → float
    @staticmethod
    def calculate_mape(actual, predicted) → float
    @staticmethod
    def calculate_all_metrics(actual, predicted) → Dict
    @staticmethod
    def compare_uv_mv_metrics(uv_metrics, mv_metrics) → Dict
```

### visualizer.py
```python
def plot_forecast_comparison(results, title) → Figure
def plot_error_comparison(results) → Figure
def plot_metrics_summary(all_comparisons) → Figure
```

### analyze_results.py
```python
def load_latest_results(results_dir) → (full_results, summary_df)
def plot_overall_comparison(df, output_dir) → None
def plot_series_performance(df, output_dir) → None
def plot_temporal_analysis(df, output_dir) → None
def plot_sample_forecasts(full_results, n_samples, output_dir) → None
def generate_summary_report(df, output_dir) → None
```

## Visualization Integration

### Current Status: ✅ FULLY INTEGRATED

1. **visualizer.py** - Contains plotting functions for individual forecasts
2. **analyze_results.py** - Comprehensive analysis with multiple visualizations
3. **run_complete_pipeline.py** - Runs experiments + analysis automatically

### Generated Visualizations

When you run `analyze_results.py` or `run_complete_pipeline.py`, you get:

1. **overall_comparison.png**
   - MV vs UV win rate pie chart
   - MAPE improvement distribution
   - Performance by dataset
   - Performance by forecast horizon

2. **series_performance.png**
   - MV win rate by series
   - Average MAPE improvement by series

3. **temporal_analysis.png**
   - MV win rate over time
   - Average improvement over time

4. **sample_forecast_*.png**
   - Individual forecast comparisons
   - UV vs MV side-by-side
   - Metrics displayed on plots

5. **summary_report.txt**
   - Text summary of all results
   - Top/bottom performers
   - Dataset comparisons

## Usage Examples

### 1. Verify Structure
```bash
python verify_structure.py
```
Checks all files exist and imports work.

### 2. Run Quick Test
```bash
python main.py --quick-test --dataset stocks
```
Runs limited experiments, saves to results/.

### 3. Run Complete Pipeline
```bash
python run_complete_pipeline.py --quick-test --dataset stocks
```
Runs experiments + generates visualizations.

### 4. Analyze Existing Results
```bash
python analyze_results.py
```
Generates visualizations from existing results/.

### 5. Run Single Example
```bash
python example_single_forecast.py
```
Simple demonstration of UV vs MV.

## Error Handling

All scripts include:
- ✅ Try-catch blocks for data download
- ✅ Progress bars (tqdm)
- ✅ Error messages with traceback
- ✅ Graceful fallbacks
- ✅ Directory creation (os.makedirs)

## Output Files

### results/
```
experiments_YYYYMMDD_HHMMSS.json       Full results with predictions
experiments_summary_YYYYMMDD_HHMMSS.csv Summary table (no arrays)
```

### visualizations/
```
overall_comparison.png                 Overall UV vs MV comparison
series_performance.png                 Performance by series
temporal_analysis.png                  Trends over time
sample_forecast_1_*.png                Sample forecast 1
sample_forecast_2_*.png                Sample forecast 2
sample_forecast_3_*.png                Sample forecast 3
summary_report.txt                     Text summary
```

### data/
```
stocks.csv                             Magnificent-7 stocks
interest_rates.csv                     FRED interest rates
combined.csv                           Combined dataset
```

## Testing Checklist

Run these commands to verify everything works:

```bash
# 1. Verify structure
python verify_structure.py

# 2. Run quick test
python main.py --quick-test --dataset stocks --device cpu

# 3. Analyze results
python analyze_results.py

# 4. Complete pipeline
python run_complete_pipeline.py --quick-test --dataset stocks --device cpu
```

## Common Issues & Solutions

### Issue: "No module named 'chronos'"
**Solution**: `pip install chronos-forecasting`

### Issue: "CUDA out of memory"
**Solution**: Use `--device cpu` or `--model-size small`

### Issue: "No results to analyze"
**Solution**: Run experiments first with `main.py`

### Issue: Import errors
**Solution**: Run `verify_structure.py` to check all dependencies

## Code Quality

All Python files include:
- ✅ Docstrings for classes and functions
- ✅ Type hints where appropriate
- ✅ Error handling
- ✅ Progress tracking
- ✅ Logging/print statements
- ✅ Proper imports
- ✅ No circular dependencies

## Summary

**Status**: ✅ **ALL COMPONENTS CONNECTED AND WORKING**

- All Python files properly structured
- Imports verified and working
- Visualizations fully integrated
- Complete pipeline functional
- Examples provided
- Error handling in place
- Documentation complete

**Ready to run immediately!**

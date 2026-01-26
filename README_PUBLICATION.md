# Chronos-2: Multivariate vs Univariate Time Series Forecasting

A comprehensive research implementation comparing univariate (UV) and multivariate (MV) forecasting approaches using Amazon's Chronos-2 foundation model on financial and economic time series data.

## 📊 Research Questions

1. Do multivariate methods produce better predictions than univariate when using foundation models?
2. Is MV forecasting accuracy better for stocks versus interest rates?
3. Is MV forecasting better when both stocks and interest rates are forecast together?
4. Can we build a large-scale "world" forecasting model?

## 🎯 Key Features

- **Chronos-2 Foundation Model**: Leverages Amazon's state-of-the-art time series foundation model
- **Comprehensive Comparison**: UV vs MV forecasting across multiple datasets
- **Rolling Window Experiments**: Monthly forecasts from Jan 2000 to Sep 2025
- **Multiple Metrics**: RMSE and MAPE for robust evaluation
- **Web Interface**: Interactive dashboard for experiment management and visualization
- **Reproducible**: Complete pipeline from data download to results analysis

## 📁 Datasets

### 1. Magnificent-7 Stocks (K=7)
- **Tickers**: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
- **Source**: Yahoo Finance
- **Period**: 2000-2025

### 2. Treasury Interest Rates (K=10)
- **Maturities**: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
- **Source**: FRED (Federal Reserve Economic Data)
- **Period**: 2000-2025

### 3. Combined Dataset (K=17)
- Both stocks and interest rates together
- Tests cross-domain forecasting capabilities

## ⚙️ Experimental Parameters

| Parameter | Values |
|-----------|--------|
| **History Window (n)** | 126, 252, 504, 756 days (0.5, 1, 2, 3 years) |
| **Forecast Horizon (m)** | 21, 63 days (1 month, 3 months) |
| **Time Period** | Monthly rolling from Jan 2000 to Sep 2025 |
| **Model Sizes** | Small, Base, Large |
| **Devices** | CPU, CUDA (GPU) |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd chronos2-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Download Magnificent-7 stocks
python main.py --download-only --dataset stocks

# Download FRED interest rates
python main.py --download-only --dataset rates

# Download combined dataset
python main.py --download-only --dataset combined
```

### Run Experiments

```bash
# Quick test (limited experiments)
python main.py --quick-test --dataset stocks --device cpu

# Full experiment suite
python main.py --dataset stocks --device cpu

# Single example for testing
python example_single_forecast.py
```

## 📊 Web Interface

Interactive dashboard for experiment management and results visualization.

### Start the Application

```bash
# Terminal 1: Start backend
cd backend
python simple_app.py

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

Access at: http://localhost:3000

### Features
- **Manage Data**: Download and verify datasets
- **Run Experiments**: Configure and monitor experiments
- **View Results**: Interactive charts and detailed metrics

## 📂 Project Structure

```
chronos2-forecasting/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── Core Python Modules
├── data_loader.py                     # Data download and preprocessing
├── chronos_forecaster.py              # UV and MV forecasting logic
├── chronos_experiment_runner.py       # Experiment orchestration
├── experiment_config.py               # Configuration management
├── metrics_calculator.py              # RMSE and MAPE calculations
├── visualizer.py                      # Plotting functions
├── main.py                            # CLI entry point
│
├── Example Scripts
├── example_single_forecast.py         # Single series example
├── class_example_mag7.py              # Mag-7 stocks example
├── class_style_notebook.py            # Comprehensive notebook-style example
│
├── Analysis Tools
├── analyze_results.py                 # Results analysis and visualization
├── run_complete_pipeline.py           # End-to-end pipeline
├── verify_structure.py                # Project integrity check
│
├── Web Application
├── backend/
│   ├── app.py                         # Full FastAPI backend
│   ├── simple_app.py                  # Simplified backend
│   └── requirements.txt               # Backend dependencies
├── frontend/
│   ├── app/                           # Next.js pages
│   ├── components/                    # React components
│   ├── package.json                   # Frontend dependencies
│   └── tailwind.config.js             # Styling configuration
│
└── Documentation
    ├── INSTRUCTIONS.md                # Detailed usage guide
    ├── QUICKSTART.md                  # Quick reference
    ├── PYTHON_STRUCTURE.md            # Code architecture
    └── COLAB_SETUP.md                 # Google Colab setup
```

## 🔬 Methodology

### Forecasting Approach

1. **Univariate (UV)**: Each time series forecasted independently
   - Uses only historical data of the target series
   - Baseline approach

2. **Multivariate (MV)**: All series forecasted together
   - Leverages cross-series correlations
   - Tests foundation model's ability to capture dependencies

### Evaluation Metrics

- **MAPE** (Mean Absolute Percentage Error): Scale-independent accuracy
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **Win Rate**: Percentage of experiments where MV outperforms UV

### Rolling Window Design

- Monthly forecasts ensure temporal robustness
- Multiple history windows test different context lengths
- Multiple horizons test short vs long-term forecasting

## 📈 Results

Results are saved in the `results/` directory:
- `experiments_YYYYMMDD_HHMMSS.json`: Full experiment details
- `experiments_summary_YYYYMMDD_HHMMSS.csv`: Summary statistics
- Visualizations generated via `analyze_results.py`

## 🛠️ Technical Details

### Dependencies

**Core Libraries:**
- `chronos-forecasting`: Amazon's Chronos-2 model
- `torch`: PyTorch for model execution
- `pandas`, `numpy`: Data manipulation
- `yfinance`: Yahoo Finance data
- `fredapi`: FRED economic data

**Web Application:**
- Backend: FastAPI, Uvicorn
- Frontend: Next.js, React, Recharts, Tailwind CSS

### Hardware Requirements

- **CPU**: Works on any modern CPU (slower)
- **GPU**: CUDA-compatible GPU recommended for faster inference
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: ~2GB for model weights + data

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{chronos2_forecasting,
  title={Chronos-2: Multivariate vs Univariate Time Series Forecasting},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/chronos2-forecasting}
}
```

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Amazon Science for the Chronos-2 foundation model
- Yahoo Finance and FRED for data access
- [Any other acknowledgments]

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🔗 References

- [Chronos-2 Paper](https://arxiv.org/abs/2403.07815)
- [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Professor's Class Notes](https://colab.research.google.com/drive/1DwFcN-KO2RUmjpXD9FqbXxqMpKEhLPCE)

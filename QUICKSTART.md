# Chronos-2 Forecasting - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (5 minutes)

```bash
cd chronos2-forecasting
pip install -r requirements.txt
```

### Step 2: Run Example (10-30 minutes)

```bash
# Simple single forecast example
python example_single_forecast.py

# OR quick test with limited parameters
python main.py --quick-test --dataset stocks
```

### Step 3: Run Full Experiments (hours to days)

```bash
# All datasets and parameters
python main.py --dataset all

# Or specific dataset
python main.py --dataset stocks
```

---

## 📊 What This Does

Compares **Univariate (UV)** vs **Multivariate (MV)** forecasting:

- **UV**: Predict AAPL using only AAPL history
- **MV**: Predict AAPL using AAPL + MSFT + GOOGL + AMZN + META + TSLA + NVDA

**Question**: Does using multiple related series improve forecasts?

---

## 🎯 Quick Commands

```bash
# Class-style example (matches professor's notes)
python class_example_mag7.py

# Notebook-style example
python class_style_notebook.py

# Simple single forecast
python example_single_forecast.py

# Test with stocks only
python main.py --quick-test --dataset stocks

# Test with CPU (no GPU)
python main.py --quick-test --device cpu

# Download data only
python main.py --download-only

# Use existing data
python main.py --skip-download --dataset stocks

# Custom date range
python main.py --dataset rates --start-date 2023-01-01 --end-date 2024-12-31
```

---

## 📁 Output Files

Results saved to `results/` directory:

- `experiments_YYYYMMDD_HHMMSS.json` - Full results with predictions
- `experiments_summary_YYYYMMDD_HHMMSS.csv` - Summary table (open in Excel)

---

## 🔍 Understanding Results

**Key Metric**: MAPE (Mean Absolute Percentage Error)

- **Lower is better**
- **Positive improvement %** = MV is better than UV
- **Negative improvement %** = UV is better than MV

**Example Output**:
```
Target: AAPL
UV MAPE: 3.5%
MV MAPE: 2.8%
Improvement: 20.0%
Winner: Multivariate ✓
```

This means MV reduced forecast error by 20%.

---

## ⚙️ Datasets

1. **Stocks** (K=7): AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
2. **Rates** (K=10): Treasury rates from 3-month to 30-year
3. **Combined** (K=17): All stocks + all rates together

---

## 🛠️ Troubleshooting

**Out of memory?**
```bash
python main.py --model-size small --device cpu
```

**Data download fails?**
- Check internet connection
- Try again (sometimes FRED is slow)

**Taking too long?**
```bash
python main.py --quick-test  # Much faster
```

---

## 📚 Full Documentation

See `INSTRUCTIONS.md` for complete details on:
- Methodology and formulas
- All command-line options
- Code architecture
- Analysis examples
- Advanced usage

---

## 🎓 Research Questions

1. Do MV methods beat UV methods with foundation models?
2. Better for stocks vs interest rates?
3. Does combining stocks + rates help?
4. Can we build a "world" forecasting model?

Run experiments to find out! 🚀

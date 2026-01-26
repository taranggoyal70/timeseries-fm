# Google Colab Setup Guide

## 🚀 Quick Start

### Option 1: Upload Notebook to Colab

1. **Download the notebook:**
   - File: `Chronos2_Colab_Experiments.ipynb`
   - Location: `/Users/tarang/CascadeProjects/windsurf-project/chronos2-forecasting/`

2. **Upload to Google Colab:**
   - Go to https://colab.research.google.com/
   - Click **File → Upload notebook**
   - Select `Chronos2_Colab_Experiments.ipynb`

3. **Enable GPU (Recommended):**
   - Click **Runtime → Change runtime type**
   - Select **T4 GPU** or **A100 GPU**
   - Click **Save**

4. **Run the notebook:**
   - Execute cells sequentially from top to bottom
   - First cell installs all dependencies (~2 minutes)
   - Model loading takes ~1-2 minutes

### Option 2: Direct Link (After Uploading to GitHub)

If you upload the notebook to GitHub, you can create a direct Colab link:

```
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Chronos2_Colab_Experiments.ipynb
```

## 📋 What's Included in the Colab Notebook

### ✅ All Code Inline
- **No external files needed** - everything is self-contained
- Data loader class
- Metrics calculator class
- Experiment runner class
- Complete implementation

### ✅ Automatic Setup
- Installs all required packages
- Downloads data from Yahoo Finance & FRED
- Loads Chronos-2 model
- Ready to run experiments

### ✅ Complete Workflow
1. Install dependencies
2. Download data (Stocks + Interest Rates)
3. Run single test experiment (n=252, m=21, t=03/31/2025)
4. Run multiple experiments with parameter sweep
5. Analyze and visualize results
6. Save results (downloadable)

## 🎯 Key Differences from Local Setup

| Feature | Local Jupyter | Google Colab |
|---------|--------------|--------------|
| **Setup** | Manual pip install | Automatic in notebook |
| **Files** | Separate .py modules | All code inline |
| **GPU** | Requires local GPU | Free T4 GPU available |
| **Data Storage** | Local disk | Colab storage (temporary) |
| **Results** | Saved locally | Download via files.download() |
| **Runtime** | Unlimited | 12 hours max (reconnect) |

## ⚙️ Configuration Options

### Model Size
```python
# In the experiment runner initialization cell:
runner = ChronosExperimentRunner(model_size="base")  # or "small", "large"
```

**Recommendations:**
- **small**: Fastest, lower accuracy (~2GB memory)
- **base**: Recommended balance (~4GB memory) ✅
- **large**: Best accuracy, slowest (~8GB memory)

### Experiment Parameters

**Test Configuration (Quick):**
```python
test_stocks = ['AAPL', 'NVDA']  # Just 2 stocks
alpha = 1.0  # n = 252 (1 year)
m = 21  # 1 month forecast
forecast_dates = pd.date_range('2024-01-01', '2024-12-31', freq='QS')  # Quarterly
```

**Full Configuration (Per README):**
```python
all_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
alpha_values = [0.5, 1.0, 2.0, 3.0]
m_values = [21, 63]
forecast_dates = pd.date_range('2000-01-01', '2025-09-30', freq='MS')  # Monthly
```

## 💾 Saving Results

### Automatic Download
The notebook includes code to automatically download results:

```python
from google.colab import files
files.download('experiment_results.json')
files.download('experiment_summary.csv')
```

### Save to Google Drive (Optional)

Add this cell to save to your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
import shutil
shutil.copy('experiment_results.json', '/content/drive/MyDrive/chronos_results.json')
shutil.copy('experiment_summary.csv', '/content/drive/MyDrive/chronos_summary.csv')
print("✓ Saved to Google Drive")
```

## ⏱️ Runtime Estimates

### With T4 GPU:
- **Single test experiment**: ~30 seconds
- **10 experiments** (2 stocks, quarterly 2024): ~5 minutes
- **100 experiments** (subset): ~30-45 minutes
- **Full experiments** (all stocks, all params, 2000-2025): **Hours to days**

### With CPU Only:
- Approximately **5-10x slower** than GPU

## 🐛 Common Issues & Solutions

### Issue 1: "CUDA out of memory"
**Solution:** Use smaller model or reduce batch size
```python
runner = ChronosExperimentRunner(model_size="small")  # Instead of "base"
```

### Issue 2: "Runtime disconnected"
**Solution:** Colab has 12-hour limit. For long experiments:
- Save intermediate results frequently
- Use checkpointing
- Consider breaking into smaller batches

### Issue 3: "Data download failed"
**Solution:** Some FRED series may be unavailable
- The notebook includes error handling
- Will continue with available series
- Check output for which series succeeded

### Issue 4: "Module not found"
**Solution:** Re-run the first installation cell
```python
!pip install -q chronos-forecasting torch pandas numpy yfinance matplotlib seaborn tqdm python-dateutil
```

## 📊 Expected Output

### Test Experiment Output:
```
TEST RESULTS
============================================================

Univariate (UV):
  RMSE: 12.3456
  MAPE: 5.67%

Multivariate (MV):
  RMSE: 11.2345
  MAPE: 4.89%

Improvement:
  RMSE: 9.00%
  MAPE: 13.75%
  MV Better: True
```

### Visualization:
- Line plot showing Actual vs UV vs MV forecasts
- Bar charts for MV win rates
- Bar charts for MAPE improvements

## 🎓 Tips for Success

1. **Start Small:**
   - Run single test experiment first
   - Verify it works before scaling up
   - Use test configuration (2 stocks, quarterly)

2. **Monitor Resources:**
   - Check GPU memory: `!nvidia-smi`
   - Watch for disconnections
   - Save results frequently

3. **Incremental Testing:**
   - Test with 1 stock first
   - Then 2 stocks
   - Then scale to full experiments

4. **Use GPU:**
   - Always enable GPU runtime
   - T4 is sufficient for most experiments
   - A100 is faster but limited availability

5. **Save Early, Save Often:**
   - Download results after each major batch
   - Don't wait until the end
   - Use Google Drive for backup

## 📚 README Compliance

The Colab notebook implements **all** README specifications:

✅ Data series: Stocks (K=7), Rates (K=10), Combined (K=17)  
✅ Methodology: UV and MV forecasting (bug-free)  
✅ Metrics: RMSE and MAPE per formulas  
✅ Parameters: α={0.5,1,2,3}, m={21,63}  
✅ Time period: 2000-2025, monthly rolling  
✅ Single test: n=252, m=21, t=03/31/2025  
✅ Storage: JSON (full) + CSV (summary)  

## 🔗 Resources

- **Chronos-2 Paper:** https://arxiv.org/abs/2510.15821
- **HuggingFace:** https://huggingface.co/amazon/chronos-2
- **GitHub:** https://github.com/amazon-science/chronos-forecasting
- **Class Examples:** https://srdas.github.io/NLPBook/Chronos.html#chronos-2

## ✨ Next Steps

1. **Upload notebook to Colab**
2. **Enable GPU runtime**
3. **Run test experiment** (verify setup)
4. **Run subset experiments** (2 stocks, quarterly)
5. **Analyze results**
6. **Scale to full experiments** (if needed)

---

**Ready to run in Google Colab!** 🚀

All code is self-contained, no external files needed.

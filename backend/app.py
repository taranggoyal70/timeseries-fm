"""
FastAPI Backend for Chronos-2 Forecasting Web Application
Provides REST API endpoints for running experiments and viewing results
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import asyncio
from enum import Enum

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from data_loader import DataLoader
    from chronos_experiment_runner import ChronosExperimentRunner
    from experiment_config import ExperimentConfig
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Parent dir: {parent_dir}")
    raise

app = FastAPI(title="Chronos-2 Forecasting API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
experiment_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_task": "",
    "error": None
}

# Models
class DatasetType(str, Enum):
    stocks = "stocks"
    rates = "rates"
    combined = "combined"

class ExperimentRequest(BaseModel):
    dataset: DatasetType
    model_size: str = "base"
    device: str = "cuda"
    alpha_values: List[float] = [1.0]
    forecast_horizons: List[int] = [21]
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    step_months: int = 3
    quick_test: bool = True

class ExperimentStatus(BaseModel):
    running: bool
    progress: int
    total: int
    current_task: str
    error: Optional[str]

# Helper functions
def get_latest_results():
    """Get the most recent experiment results."""
    results_dir = Path("../results")
    if not results_dir.exists():
        return None
    
    json_files = sorted(results_dir.glob("experiments_*.json"))
    csv_files = sorted(results_dir.glob("experiments_summary_*.csv"))
    
    if not json_files or not csv_files:
        return None
    
    with open(json_files[-1], 'r') as f:
        full_results = json.load(f)
    
    summary_df = pd.read_csv(csv_files[-1])
    
    return {
        "full_results": full_results,
        "summary": summary_df.to_dict(orient="records"),
        "timestamp": json_files[-1].stem.split("_", 1)[1]
    }

def get_available_datasets():
    """Check which datasets are available."""
    data_dir = Path("../data")
    datasets = {}
    
    if data_dir.exists():
        datasets["stocks"] = (data_dir / "stocks.csv").exists()
        datasets["rates"] = (data_dir / "interest_rates.csv").exists()
        datasets["combined"] = (data_dir / "combined.csv").exists()
    else:
        datasets = {"stocks": False, "rates": False, "combined": False}
    
    return datasets

async def run_experiment_background(request: ExperimentRequest):
    """Run experiment in background."""
    global experiment_status
    
    try:
        experiment_status["running"] = True
        experiment_status["error"] = None
        experiment_status["current_task"] = "Initializing..."
        
        # Load data
        experiment_status["current_task"] = "Loading data..."
        loader = DataLoader()
        
        if request.dataset == "stocks":
            df = loader.load_data("stocks")
            series_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
        elif request.dataset == "rates":
            df = loader.load_data("interest_rates")
            series_names = ['DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 
                          'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        else:  # combined
            df = loader.load_data("combined")
            series_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                          'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 
                          'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
        
        # Configure experiment
        experiment_status["current_task"] = "Configuring experiment..."
        config = ExperimentConfig(
            alpha_values=request.alpha_values,
            forecast_horizons=request.forecast_horizons,
            start_date=request.start_date,
            end_date=request.end_date,
            step_months=request.step_months,
            model_size=request.model_size,
            device=request.device
        )
        
        # Run experiments
        experiment_status["current_task"] = "Running forecasts..."
        runner = ChronosExperimentRunner(config)
        results = runner.run_rolling_experiments(df, request.dataset.value, series_names)
        
        # Save results
        experiment_status["current_task"] = "Saving results..."
        runner.save_results(results)
        
        experiment_status["current_task"] = "Complete!"
        experiment_status["progress"] = 100
        
    except Exception as e:
        experiment_status["error"] = str(e)
        experiment_status["current_task"] = "Error occurred"
    finally:
        experiment_status["running"] = False

# API Endpoints

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Chronos-2 Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "datasets": "/datasets",
            "download": "/download/{dataset}",
            "experiment": "/experiment",
            "status": "/status",
            "results": "/results",
            "summary": "/summary"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/datasets")
async def get_datasets():
    """Get available datasets."""
    datasets = get_available_datasets()
    return {
        "datasets": datasets,
        "message": "Available datasets" if any(datasets.values()) else "No datasets downloaded yet"
    }

@app.post("/download/{dataset}")
async def download_dataset(dataset: DatasetType, background_tasks: BackgroundTasks):
    """Download a specific dataset."""
    try:
        loader = DataLoader()
        
        if dataset == DatasetType.stocks:
            df = loader.download_stocks()
            return {"message": f"Downloaded {len(df)} rows of stock data", "dataset": "stocks"}
        elif dataset == DatasetType.rates:
            df = loader.download_interest_rates()
            return {"message": f"Downloaded {len(df)} rows of interest rate data", "dataset": "rates"}
        else:  # combined
            df = loader.download_combined()
            return {"message": f"Downloaded {len(df)} rows of combined data", "dataset": "combined"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiment")
async def run_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """Run forecasting experiment."""
    global experiment_status
    
    if experiment_status["running"]:
        raise HTTPException(status_code=400, detail="Experiment already running")
    
    # Reset status
    experiment_status = {
        "running": True,
        "progress": 0,
        "total": 100,
        "current_task": "Starting...",
        "error": None
    }
    
    # Run in background
    background_tasks.add_task(run_experiment_background, request)
    
    return {"message": "Experiment started", "status": experiment_status}

@app.get("/status", response_model=ExperimentStatus)
async def get_status():
    """Get current experiment status."""
    return ExperimentStatus(**experiment_status)

@app.get("/results")
async def get_results():
    """Get latest experiment results."""
    results = get_latest_results()
    
    if results is None:
        raise HTTPException(status_code=404, detail="No results found. Run an experiment first.")
    
    return results

@app.get("/summary")
async def get_summary():
    """Get summary statistics from latest results."""
    results = get_latest_results()
    
    if results is None:
        raise HTTPException(status_code=404, detail="No results found")
    
    df = pd.DataFrame(results["summary"])
    
    # Calculate summary statistics
    mv_wins = (df['mv_better_mape'] == True).sum()
    total = len(df)
    
    summary = {
        "total_experiments": total,
        "mv_wins": int(mv_wins),
        "uv_wins": int(total - mv_wins),
        "mv_win_rate": float((mv_wins / total * 100) if total > 0 else 0),
        "avg_mape_improvement": float(df['mape_improvement_pct'].mean()),
        "median_mape_improvement": float(df['mape_improvement_pct'].median()),
        "datasets": df['dataset'].unique().tolist(),
        "series": df['series'].unique().tolist(),
        "timestamp": results["timestamp"]
    }
    
    # By dataset
    by_dataset = {}
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        mv_wins_subset = (df_subset['mv_better_mape'] == True).sum()
        total_subset = len(df_subset)
        by_dataset[dataset] = {
            "mv_wins": int(mv_wins_subset),
            "total": int(total_subset),
            "win_rate": float((mv_wins_subset / total_subset * 100) if total_subset > 0 else 0),
            "avg_improvement": float(df_subset['mape_improvement_pct'].mean())
        }
    
    summary["by_dataset"] = by_dataset
    
    return summary

@app.get("/results/series/{series_name}")
async def get_series_results(series_name: str):
    """Get results for a specific series."""
    results = get_latest_results()
    
    if results is None:
        raise HTTPException(status_code=404, detail="No results found")
    
    # Filter results for this series
    series_results = [r for r in results["full_results"] if r["series"] == series_name]
    
    if not series_results:
        raise HTTPException(status_code=404, detail=f"No results found for series {series_name}")
    
    return {
        "series": series_name,
        "count": len(series_results),
        "results": series_results
    }

@app.delete("/results")
async def clear_results():
    """Clear all results."""
    results_dir = Path("../results")
    if results_dir.exists():
        for file in results_dir.glob("*"):
            file.unlink()
        return {"message": "All results cleared"}
    return {"message": "No results to clear"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

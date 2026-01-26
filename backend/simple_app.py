"""
Simplified FastAPI Backend for Chronos-2 Forecasting
Works without complex imports - basic functionality
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Chronos-2 Forecasting API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory state
experiment_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_task": "",
    "error": None
}

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Chronos-2 Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "datasets": "/datasets",
            "status": "/status",
            "results": "/results",
            "summary": "/summary"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/datasets")
async def get_datasets():
    """Get available datasets."""
    parent_dir = Path(__file__).parent.parent
    data_dir = parent_dir / "data"
    
    datasets = {
        "stocks": (data_dir / "stocks.csv").exists() if data_dir.exists() else False,
        "rates": (data_dir / "interest_rates.csv").exists() if data_dir.exists() else False,
        "combined": (data_dir / "combined.csv").exists() if data_dir.exists() else False
    }
    
    return {
        "datasets": datasets,
        "message": "Available datasets" if any(datasets.values()) else "No datasets downloaded yet"
    }

@app.get("/status")
async def get_status():
    """Get current experiment status."""
    return experiment_status

@app.get("/results")
async def get_results():
    """Get latest experiment results."""
    parent_dir = Path(__file__).parent.parent
    results_dir = parent_dir / "results"
    
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No results found. Run an experiment first.")
    
    # Find latest JSON file
    json_files = sorted(results_dir.glob("experiments_*.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="No results found")
    
    import json
    with open(json_files[-1], 'r') as f:
        full_results = json.load(f)
    
    return {
        "count": len(full_results),
        "results": full_results[:10],  # Return first 10 for preview
        "message": f"Loaded {len(full_results)} experiments"
    }

@app.get("/summary")
async def get_summary():
    """Get summary statistics from latest results."""
    parent_dir = Path(__file__).parent.parent
    results_dir = parent_dir / "results"
    
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No results found")
    
    csv_files = sorted(results_dir.glob("experiments_summary_*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="No results found")
    
    import pandas as pd
    df = pd.read_csv(csv_files[-1])
    
    mv_wins = (df['mv_better_mape'] == True).sum()
    total = len(df)
    
    summary = {
        "total_experiments": int(total),
        "mv_wins": int(mv_wins),
        "uv_wins": int(total - mv_wins),
        "mv_win_rate": float((mv_wins / total * 100) if total > 0 else 0),
        "avg_mape_improvement": float(df['mape_improvement_pct'].mean()),
        "median_mape_improvement": float(df['mape_improvement_pct'].median()),
        "datasets": df['dataset'].unique().tolist(),
        "series": df['series'].unique().tolist(),
        "timestamp": csv_files[-1].stem.split("_", 2)[2]
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

@app.post("/download/{dataset}")
async def download_dataset(dataset: str):
    """Download a specific dataset."""
    parent_dir = Path(__file__).parent.parent
    main_py = parent_dir / "main.py"
    
    # Check if main.py exists
    if not main_py.exists():
        return {
            "message": f"To download {dataset} dataset, run this command in your terminal:\n\ncd {parent_dir}\npython main.py --download-only --dataset {dataset}",
            "status": "info",
            "command": f"python main.py --download-only --dataset {dataset}"
        }
    
    # For now, return instruction since subprocess can be slow
    return {
        "message": f"Download {dataset} dataset by running:\n\npython main.py --download-only --dataset {dataset}\n\nThis will download from Yahoo Finance (stocks) or FRED (rates).",
        "status": "info",
        "dataset": dataset,
        "instructions": [
            f"Open a terminal",
            f"cd {parent_dir}",
            f"python main.py --download-only --dataset {dataset}",
            "Wait 1-2 minutes for download to complete",
            "Refresh this page to see updated status"
        ]
    }

@app.post("/experiment")
async def run_experiment():
    """Run forecasting experiment."""
    parent_dir = Path(__file__).parent.parent
    
    return {
        "message": "To run experiments, use the terminal:\n\n1. Quick Test (10-30 min):\n   python main.py --quick-test --dataset stocks --device cpu\n\n2. Single Example (2-5 min):\n   python example_single_forecast.py\n\nResults will appear in the 'View Results' tab when complete.",
        "status": "info",
        "instructions": [
            "Open a terminal",
            f"cd {parent_dir}",
            "Run: python main.py --quick-test --dataset stocks --device cpu",
            "Wait for completion (10-30 minutes)",
            "Refresh 'View Results' tab to see charts"
        ],
        "quick_commands": {
            "quick_test": "python main.py --quick-test --dataset stocks --device cpu",
            "single_example": "python example_single_forecast.py",
            "full_experiment": "python main.py --dataset stocks --device cpu"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Chronos-2 API Server...")
    print("API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

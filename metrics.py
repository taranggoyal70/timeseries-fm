import numpy as np
import pandas as pd

def calculate_rmse(actual, predicted):
    actual_values = actual.values if isinstance(actual, pd.Series) else actual
    pred_values = predicted.values if isinstance(predicted, pd.Series) else predicted
    
    mse = np.mean((actual_values - pred_values) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(actual, predicted):
    actual_values = actual.values if isinstance(actual, pd.Series) else actual
    pred_values = predicted.values if isinstance(predicted, pd.Series) else predicted
    
    mape = np.mean(np.abs((actual_values - pred_values) / actual_values))
    return mape

def calculate_metrics(actual, pred_df):
    predictions = pred_df.set_index("timestamp")["predictions"]
    
    common_index = actual.index.intersection(predictions.index)
    
    if len(common_index) == 0:
        return {"rmse": np.nan, "mape": np.nan, "n_points": 0}
    
    actual_aligned = actual.loc[common_index]
    pred_aligned = predictions.loc[common_index]
    
    rmse = calculate_rmse(actual_aligned, pred_aligned)
    mape = calculate_mape(actual_aligned, pred_aligned)
    
    return {
        "rmse": rmse,
        "mape": mape,
        "n_points": len(common_index)
    }

def compare_uv_mv_metrics(results):
    actual = results["actual"]
    uv_pred = results["uv_predictions"]
    mv_pred = results["mv_predictions"]
    
    uv_metrics = calculate_metrics(actual, uv_pred)
    mv_metrics = calculate_metrics(actual, mv_pred)
    
    comparison = {
        "target": results["target"],
        "uv_rmse": uv_metrics["rmse"],
        "uv_mape": uv_metrics["mape"],
        "mv_rmse": mv_metrics["rmse"],
        "mv_mape": mv_metrics["mape"],
        "n_points": uv_metrics["n_points"],
        "mape_improvement": (uv_metrics["mape"] - mv_metrics["mape"]) / uv_metrics["mape"] * 100 if uv_metrics["mape"] > 0 else 0,
        "mv_better": mv_metrics["mape"] < uv_metrics["mape"]
    }
    
    return comparison

def print_metrics_comparison(comparison):
    print(f"\n{'='*60}")
    print(f"Target: {comparison['target']}")
    print(f"{'='*60}")
    print(f"Univariate (UV):")
    print(f"  RMSE: {comparison['uv_rmse']:.4f}")
    print(f"  MAPE: {comparison['uv_mape']:.4f} ({comparison['uv_mape']*100:.2f}%)")
    print(f"\nMultivariate (MV):")
    print(f"  RMSE: {comparison['mv_rmse']:.4f}")
    print(f"  MAPE: {comparison['mv_mape']:.4f} ({comparison['mv_mape']*100:.2f}%)")
    print(f"\nImprovement:")
    print(f"  MAPE Improvement: {comparison['mape_improvement']:.2f}%")
    print(f"  MV Better: {'YES' if comparison['mv_better'] else 'NO'}")
    print(f"  Points Evaluated: {comparison['n_points']}")
    print(f"{'='*60}\n")

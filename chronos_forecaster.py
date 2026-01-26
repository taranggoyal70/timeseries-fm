import torch
import pandas as pd
from chronos import Chronos2Pipeline

class ChronosForecaster:
    
    def __init__(self, device="cuda", model_size="base"):
        """
        Initialize Chronos-2 forecaster.
        
        Args:
            device: 'cuda' or 'cpu'
            model_size: 'small', 'base', or 'large' (base recommended for accuracy/speed balance)
        """
        print(f"Loading Chronos-2 {model_size} model...")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            f"amazon/chronos-2-{model_size}",
            device_map=device if torch.cuda.is_available() else "cpu"
        )
        print(f"Model loaded on {device if torch.cuda.is_available() else 'CPU'}")
    
    def forecast_univariate(self, context_df, target_column, prediction_length, 
                           quantile_levels=[0.1, 0.5, 0.9]):
        context_uv = context_df[["item_id", "timestamp", target_column]].copy()
        
        future_df = pd.DataFrame({
            "item_id": context_uv["item_id"].iloc[-1],
            "timestamp": pd.date_range(
                start=context_uv["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="B"
            )
        })
        
        pred_df = self.pipeline.predict_df(
            context_uv,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target=target_column
        )
        
        return pred_df
    
    def forecast_multivariate(self, context_df, target_column, prediction_length,
                             quantile_levels=[0.1, 0.5, 0.9]):
        """
        Multivariate forecast WITHOUT future covariates (proper approach).
        Chronos-2 learns relationships from historical data only.
        """
        future_df_cols = [col for col in context_df.columns if col not in ["item_id", "timestamp", target_column]]
        
        if not future_df_cols:
            print("Warning: No covariates found for multivariate forecast. Falling back to univariate.")
            return self.forecast_univariate(context_df, target_column, prediction_length, quantile_levels)
        
        # Create future_df with ONLY timestamp and item_id
        # Do NOT include future covariate values - we don't know them!
        future_df = pd.DataFrame({
            "item_id": context_df["item_id"].iloc[-1],
            "timestamp": pd.date_range(
                start=context_df["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="B"
            )
        })
        
        # Chronos-2 will use the historical relationships between series
        # to forecast the target, without needing future covariate values
        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target=target_column
        )
        
        return pred_df
    
    def compare_uv_mv(self, context_df, test_df, target_column, prediction_length,
                     quantile_levels=[0.1, 0.5, 0.9]):
        print(f"\nForecasting {target_column}...")
        print(f"  Context length: {len(context_df)}, Prediction length: {prediction_length}")
        
        print("  Running univariate forecast...")
        uv_pred = self.forecast_univariate(context_df, target_column, prediction_length, quantile_levels)
        
        print("  Running multivariate forecast...")
        mv_pred = self.forecast_multivariate(context_df, target_column, prediction_length, quantile_levels)
        
        actual = test_df.set_index("timestamp")[target_column]
        
        results = {
            "target": target_column,
            "uv_predictions": uv_pred,
            "mv_predictions": mv_pred,
            "actual": actual,
            "context_df": context_df,
            "test_df": test_df
        }
        
        return results

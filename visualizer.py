import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast_comparison(results, title=None):
    target = results["target"]
    context_df = results["context_df"]
    test_df = results["test_df"]
    uv_pred = results["uv_predictions"]
    mv_pred = results["mv_predictions"]
    actual = results["actual"]
    
    ts_context = context_df.set_index("timestamp")[target]
    ts_uv = uv_pred.set_index("timestamp")
    ts_mv = mv_pred.set_index("timestamp")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ts_context.plot(ax=ax1, label="Historical Data", color="royalblue", linewidth=1.5)
    actual.plot(ax=ax1, label="Actual (Ground Truth)", color="green", linewidth=1.5)
    ts_uv["predictions"].plot(ax=ax1, label="UV Forecast", color="tomato", linewidth=1.5, linestyle="--")
    ax1.fill_between(
        ts_uv.index,
        ts_uv["0.1"],
        ts_uv["0.9"],
        alpha=0.2,
        color="tomato",
        label="UV 80% Interval"
    )
    ax1.set_title(f"Univariate Forecast - {target}" if not title else f"{title} - UV")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ts_context.plot(ax=ax2, label="Historical Data", color="royalblue", linewidth=1.5)
    actual.plot(ax=ax2, label="Actual (Ground Truth)", color="green", linewidth=1.5)
    ts_mv["predictions"].plot(ax=ax2, label="MV Forecast", color="purple", linewidth=1.5, linestyle="--")
    ax2.fill_between(
        ts_mv.index,
        ts_mv["0.1"],
        ts_mv["0.9"],
        alpha=0.2,
        color="purple",
        label="MV 80% Interval"
    )
    ax2.set_title(f"Multivariate Forecast - {target}" if not title else f"{title} - MV")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_error_comparison(results):
    target = results["target"]
    uv_pred = results["uv_predictions"]
    mv_pred = results["mv_predictions"]
    actual = results["actual"]
    
    ts_uv = uv_pred.set_index("timestamp")["predictions"]
    ts_mv = mv_pred.set_index("timestamp")["predictions"]
    
    common_index = actual.index.intersection(ts_uv.index).intersection(ts_mv.index)
    
    uv_error = actual.loc[common_index] - ts_uv.loc[common_index]
    mv_error = actual.loc[common_index] - ts_mv.loc[common_index]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    uv_error.plot(ax=ax, label="UV Error", color="tomato", linewidth=1.5, alpha=0.7)
    mv_error.plot(ax=ax, label="MV Error", color="purple", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_title(f"Forecast Errors - {target}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (Actual - Predicted)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_metrics_summary(all_comparisons):
    df = pd.DataFrame(all_comparisons)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = range(len(df))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], df["uv_mape"], width, label="UV", color="tomato", alpha=0.7)
    axes[0].bar([i + width/2 for i in x], df["mv_mape"], width, label="MV", color="purple", alpha=0.7)
    axes[0].set_xlabel("Target Series")
    axes[0].set_ylabel("MAPE")
    axes[0].set_title("MAPE Comparison: UV vs MV")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["target"], rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    
    improvement = df["mape_improvement"]
    colors = ["green" if x > 0 else "red" for x in improvement]
    axes[1].bar(x, improvement, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Target Series")
    axes[1].set_ylabel("MAPE Improvement (%)")
    axes[1].set_title("MV Improvement over UV (positive = MV better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["target"], rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig

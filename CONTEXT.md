# Multivariate Forecasting with Foundation Models

This context defines the experimental language for comparing univariate and multivariate Chronos-2 forecasts of financial and economic time series.

## Language

**Experiment**:
One reproducible forecast evaluation defined by a panel, forecast origin, lookback window, horizon, and forecasting mode.
_Avoid_: Notebook run without recorded parameters

**Panel**:
The set of target series evaluated together: Magnificent-7 equities, Treasury rates, or the combined world panel.
_Avoid_: Dataset file, visualization

**Forecast Origin**:
The last observed date available to the model before prediction begins.
_Avoid_: Data download date, training cutoff

**Lookback Window**:
The number of observations before a Forecast Origin supplied as model context.
_Avoid_: Horizon

**Horizon**:
The number of future trading-day observations predicted and scored.
_Avoid_: Lookback Window

**Univariate Mode**:
Forecasting each target from its own historical Series.
_Avoid_: Independent forecasts executed in one batch and called multivariate

**Multivariate Mode**:
Forecasting with all Series in a Panel jointly represented so cross-series information is available to the model.
_Avoid_: World Model, unless the combined panel is used

**World Model**:
The Multivariate Mode experiment over the combined equity-and-rate Panel. It is a panel configuration, not a separately trained foundation model.
_Avoid_: New pretrained model

**Realized Value**:
An observed post-origin value used as ground truth for evaluation.
_Avoid_: Forecast, imputed future

**MAPE**:
The primary normalized forecast-error metric used for comparison across Series.
_Avoid_: Accuracy percentage, financial return

**RMSE**:
An unnormalized forecast-error metric reported within a Series or comparable scale.
_Avoid_: Cross-asset comparison without scaling

**Training-Cutoff Analysis**:
The comparison of experiments before and after the assumed Chronos-2 training-data boundary to surface possible leakage effects.
_Avoid_: Proof of the model's exact training corpus

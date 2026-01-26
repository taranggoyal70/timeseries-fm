# Multivariate Forecasting with Foundation Models

Using Chronos-2 for Economic and Financial Forecasts

## Research Questions

1. Do multivariate (MV) methods produce better predictions than univariate (UV) ones when foundation models (FMs, transformers) are used for both?

2. Is MV forecasting accuracy better for stocks versus interest rates?

3. Is MB forecasting better when both stocks and interest rates are forecast together?

4. Can we build a large-scale “world” forecasting model?

## Chronos-2

We will use Chronos-2 for all forecasts -- it is a transformer that is pretrained on thousands of time series. For inference, we enter a single or multiple series and ask it to return a forecast of one series going forward. Resources are as follows:

- Chronos-1 Paper: https://arxiv.org/abs/2403.07815
- Chronos-2 Paper: https://arxiv.org/abs/2510.15821
- HuggingFace: https://huggingface.co/amazon/chronos-2
- GitHub: https://github.com/amazon-science/chronos-forecasting
- Amazon Science: https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
- Class note examples: https://srdas.github.io/NLPBook/Chronos.html#chronos-2. (Please go over these examples to see how we compare UV models with MV models.)

## Methodology

The way Chronos-2 works is as follows:

1. UV forecasting. Enter a series $x$ of length $n$ and ask the FM to forecast the next $m$ values. If a chosen date is $t$, then the input series will be on observations $\{x_{t-n+1},x_{t-n+2},...,x_{t-1},x_t\}$ and the forecast will be over observations $\{x_{t+1},x_{t+2},...,x_{t+m}\}$.

2. MV forecasting. Enter $K$ series $x_{kt},k=1...K$, each of length $n$ and ask the FM to forecast the next $m$ values of series $x_{it}$. If a chosen date is $t$, then the input series will be on observations $\{x_{k,t-n+1},x_{k,t-n+2},...,x_{k,t-1},x_{k,t}\}, k=1...K$ and the forecast will be over observations $\{x_{i,t+1},x_{i,t+2},...,x_{i,t+m}\}$ for $i=1...K$. Let the forecasted values be $\{y_{i,t+1},y_{i,t+2},...,y_{i,t+m}\}$ for $i=1...K$.

3. For accuracy we can compare the actual values $\{x_{i,t+1}, ... ,x_{i,t+m}\}$ with the forecast values $\{y_{i,t+1}, ... ,y_{i,t+m}\}$

4. For example, if we are using the "magnificent-7" stocks as shown in the class notes, then we will use the same input data on $n$ days to forecast each of the 7 series one by one, first using the UV method, and second using the MV method. The code is provided in the class notes: https://srdas.github.io/NLPBook/Chronos.html#multivariate-stock-price-forecast

5. A single experiment: To begin, we set $n=252$ and $m=21$ days. We also need a date $t$, Let's choose 03/31/2025 as an example. Once we have the code working for a single case, then we can input different dates and also different values for $n$ and $m$.

## Error Statistics

[1] Root mean squared error (RMSE):

$$
RMSE_i = \left[\frac{1}{m} \sum_{h=1}^m \left(x_{i,t+h}-y_{i,t+h}\right)^2 \right]^{1/2}
$$

[2] Mean absolute percentage error (MAPE):

$$
MAPE = \frac{1}{m} \sum_{h=1}^m \left| \frac{x_{i,t+h}-y_{i,t+h}}{x_{i,t+h}} \right|
$$

- We compute and report these for each experiment for each of the $K$ series in each experiment. Because MAPE is normalized and RMSE is not, the main metric we care about is MAPE.

- We compute these error metrics for UV and MV for comparison and display them on the front end as well.


## Multiple Experiments

1. Vary $n$ to be multiples of 252, where the multiple is $\alpha = \{0.5, 1, 2, 3\}$.

2. Let $m = \{21, 63\}$, i.e., one month and three months forecast periods.

3. Time period: $t$ = 01/01/2000 to 09/30/2025, in steps of one month at a time, i.e., rolling forecasts. The total number of months will vary depending on $n$ and $m$. We will recognize that there may be data leakage because Chronos-2 may have been trained on some part of the time series. While we report results for the whole time period from 2000 onwards, we will compare results from periods before training completion to periods after. The training data cut off date from the paper is unclear given several datasets are used (Table 6), but it seems safe to assume a cut off date in 2023. That leaves 24 months of data (2024 and 2025) at least for testing. It will be interesting to see if forecast  errors increase substantially after 2023.

4. Prepare a single script to run the entire set of experiments for one set of time series of size $K$ and store the error metrics and $x$ and $y$ data.


## Data series

For this data, create a script to download it so that it can be re-run and updated as time passes. 

1. Stock data: Magnificent-7, ($K=7$)

2. Interest rates: Use the FRED Constant Maturity interest rates for maturities of [3 months](https://fred.stlouisfed.org/series/DGS3MO), [6 months](DGS6MO), [1 year](https://fred.stlouisfed.org/series/DGS1), [2 year](https://fred.stlouisfed.org/series/DGS2), [3 year](https://fred.stlouisfed.org/series/DGS3), [5 year](https://fred.stlouisfed.org/series/DGS5), [7 year](https://fred.stlouisfed.org/series/DGS7), [10 year](https://fred.stlouisfed.org/series/DGS10), [20 year](https://fred.stlouisfed.org/series/DGS20) and [30 year](https://fred.stlouisfed.org/series/DGS30). ($K=10$)

3. Use both together. ($K=17$)


"""
ARIMA / SARIMA Time Series Forecasting
========================================
AutoRegressive Integrated Moving Average — a classical statistical model for
univariate time series forecasting. Combines three components:

    AR(p):  autoregressive — predict from p lagged values
            y_t = c + φ₁y_{t-1} + ... + φ_p y_{t-p} + ε_t

    I(d):   integrated — differencing d times to achieve stationarity
            Δy_t = y_t - y_{t-1}

    MA(q):  moving average — predict from q lagged error terms
            y_t = μ + ε_t + θ₁ε_{t-1} + ... + θ_q ε_{t-q}

SARIMA(p,d,q)(P,D,Q,s) adds seasonal components:
    (P,D,Q,s) — seasonal AR, I, MA with period s (e.g., 12 for monthly data)

Model Identification (Box-Jenkins method):
    1. Check stationarity: ADF test + ACF/PACF plots
    2. Difference if needed (d)
    3. Read ACF → q (MA order)
    4. Read PACF → p (AR order)
    5. Fit and check AIC/BIC + residuals

Use Cases:
    - Short-term economic forecasting
    - Demand planning and inventory management
    - Energy consumption forecasting
    - Marketing spend and channel performance time series
    - Baseline model for comparing against ML forecasters

Advantages:
    - Interpretable parameters
    - Well-established theory + diagnostic tools
    - Works with small datasets (no deep learning data requirements)
    - Residual diagnostics detect model misspecification

Limitations:
    - Univariate only (no exogenous variables in base ARIMA; ARIMAX adds them)
    - Linear model — cannot capture nonlinear patterns
    - Manual order selection can be tedious (auto_arima helps)
    - Poor for long-horizon forecasting relative to modern ML methods

Fine-Tuning Recommendations:
    - Use pmdarima's auto_arima() for automatic order selection.
    - d: use ADF/KPSS test; difference until p-value < 0.05 on ADF.
    - p, q: read PACF (p) and ACF (q) plots after differencing.
    - Seasonal: identify period s from domain knowledge or periodogram.
    - Information criterion: AIC preferred for forecasting; BIC for parsimony.
    - Residual checks: Ljung-Box test for autocorrelation in residuals.
      If residuals are autocorrelated → increase p or q.
    - Forecast intervals: 80%/95% confidence intervals are standard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_time_series(
    n_obs:    int = 300,
    seasonal: bool = True,
    seed:     int = 42,
) -> pd.Series:
    """
    Generate synthetic ARMA(1,1) time series with optional seasonal component.

    Replace with your own data:
        import pandas as pd
        df = pd.read_csv("sales.csv", index_col="date", parse_dates=True)
        series = df["sales"]  # pd.Series with DatetimeIndex

    Returns:
        pd.Series with monthly DatetimeIndex
    """
    rng = np.random.RandomState(seed)
    n   = n_obs

    # ARMA(1,1) component
    phi, theta = 0.7, 0.3
    eps = rng.normal(0, 1, n)
    y   = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]

    # Trend
    y += np.linspace(0, 20, n)

    # Seasonal (12-month)
    if seasonal:
        y += 5 * np.sin(2 * np.pi * np.arange(n) / 12)

    dates = pd.date_range(start="2016-01-01", periods=n, freq="MS")
    return pd.Series(y, index=dates, name="value")


# ─────────────────────────────────────────────
# STATIONARITY & DIAGNOSTICS
# ─────────────────────────────────────────────

def adf_test(series: pd.Series, name: str = "Series") -> bool:
    """
    Augmented Dickey-Fuller test for stationarity.

    H0: series has a unit root (non-stationary)
    Reject H0 (p < 0.05) → series is stationary.

    Returns True if stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"  ADF Test ({name}):")
    print(f"    ADF Statistic: {result[0]:.4f}")
    print(f"    p-value:       {result[1]:.4f}  "
          f"({'STATIONARY' if result[1] < 0.05 else 'NON-STATIONARY'})")
    print(f"    Critical 5%:   {result[4]['5%']:.4f}")
    return result[1] < 0.05


def plot_acf_pacf(series: pd.Series, lags: int = 40) -> None:
    """
    ACF and PACF plots for order identification.

    ACF:  lags where autocorrelation is significant → MA order (q)
    PACF: lags where partial autocorrelation is significant → AR order (p)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(series.dropna(),  lags=lags, ax=ax1, title="ACF  (→ MA order q)")
    plot_pacf(series.dropna(), lags=lags, ax=ax2, title="PACF (→ AR order p)")
    plt.tight_layout()
    plt.savefig("arima_acf_pacf.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: set order based on ADF test + ACF/PACF analysis
ARIMA_CONFIG = {
    "order": (1, 1, 1),  # Placeholder: (p, d, q) — use auto_arima or ACF/PACF
}

SARIMA_CONFIG = {
    "order":          (1, 1, 1),     # Placeholder: (p, d, q)
    "seasonal_order": (1, 1, 1, 12), # Placeholder: (P, D, Q, s) — s=12 for monthly
}


# ─────────────────────────────────────────────
# AUTO ORDER SELECTION
# ─────────────────────────────────────────────

def auto_order_selection(series: pd.Series,
                          p_range: range = range(0, 4),
                          q_range: range = range(0, 4),
                          d: int = 1) -> tuple:
    """
    Grid search over (p, q) with fixed d, select by AIC.

    For production: use pmdarima.auto_arima() which handles this automatically
    with stepwise search and seasonal support:
        from pmdarima import auto_arima
        model = auto_arima(series, seasonal=True, m=12, stepwise=True)
    """
    best_aic = np.inf
    best_pq  = (1, 1)
    results  = []

    for p in p_range:
        for q in q_range:
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                results.append((p, q, m.aic, m.bic))
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_pq  = (p, q)
            except Exception:
                pass

    df = pd.DataFrame(results, columns=["p", "q", "AIC", "BIC"]).sort_values("AIC")
    print("\n  Top 5 models by AIC:")
    print(df.head().to_string(index=False))
    print(f"\n  Best order: ARIMA({best_pq[0]}, {d}, {best_pq[1]})  AIC={best_aic:.2f}")
    return best_pq[0], d, best_pq[1]


# ─────────────────────────────────────────────
# TRAINING & FORECASTING
# ─────────────────────────────────────────────

def train_arima(series: pd.Series, order: tuple) -> ARIMA:
    """Fit ARIMA model."""
    model = ARIMA(series, order=order)
    return model.fit()


def train_sarima(series: pd.Series, order: tuple, seasonal_order: tuple) -> SARIMAX:
    """Fit SARIMA model."""
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)


def forecast(fitted_model, steps: int = 24,
             alpha: float = 0.05) -> pd.DataFrame:
    """
    Generate h-step ahead forecast with confidence intervals.

    Args:
        fitted_model: fitted ARIMA/SARIMA result
        steps:        forecast horizon
        alpha:        significance level (0.05 = 95% CI)

    Returns:
        DataFrame with columns: forecast, lower_ci, upper_ci
    """
    pred = fitted_model.get_forecast(steps=steps)
    ci   = pred.conf_int(alpha=alpha)

    df = pd.DataFrame({
        "forecast":  pred.predicted_mean,
        "lower_ci":  ci.iloc[:, 0],
        "upper_ci":  ci.iloc[:, 1],
    })
    return df


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(fitted_model, test: pd.Series) -> dict:
    """
    Evaluate on test set: RMSE, MAE, MAPE.

    Uses dynamic one-step-ahead predictions from the model.
    """
    pred = fitted_model.predict(start=test.index[0], end=test.index[-1])
    rmse = np.sqrt(mean_squared_error(test, pred))
    mae  = mean_absolute_error(test, pred)
    mape = np.mean(np.abs((test - pred) / (test + 1e-8))) * 100

    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def residual_diagnostics(fitted_model) -> None:
    """
    Residual diagnostics: plot residuals, ACF, histogram.
    Run Ljung-Box test for remaining autocorrelation.
    """
    resid = fitted_model.resid.dropna()

    # Ljung-Box test: H0 = no autocorrelation in residuals
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    p_val = lb["lb_pvalue"].values[0]
    print(f"  Ljung-Box p-value (lag 10): {p_val:.4f}  "
          f"({'PASS — no autocorrelation' if p_val > 0.05 else 'FAIL — residual autocorrelation'})")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3))
    axes[0].plot(resid.values, linewidth=0.7)
    axes[0].axhline(0, color="red", linewidth=0.8)
    axes[0].set_title("Residuals")

    axes[1].hist(resid.values, bins=30, color="steelblue", alpha=0.7)
    axes[1].set_title("Residual Histogram")

    plot_acf(resid, lags=30, ax=axes[2], title="Residual ACF")
    plt.suptitle("ARIMA — Residual Diagnostics")
    plt.tight_layout()
    plt.savefig("arima_residuals.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_forecast(series: pd.Series, forecast_df: pd.DataFrame,
                  title: str = "ARIMA Forecast") -> None:
    """Plot observed series with forecast and confidence interval."""
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, label="Observed", color="steelblue", linewidth=1.5)
    plt.plot(forecast_df.index, forecast_df["forecast"],
             label="Forecast", color="tomato", linewidth=2.0, linestyle="--")
    plt.fill_between(forecast_df.index,
                     forecast_df["lower_ci"],
                     forecast_df["upper_ci"],
                     alpha=0.25, color="tomato", label="95% CI")
    plt.axvline(series.index[-1], color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("arima_forecast.png", dpi=150)
    plt.show()


def plot_train_test_forecast(series: pd.Series, train: pd.Series,
                              test: pd.Series, fitted_model) -> None:
    """Plot fitted values on train + forecast on test."""
    fitted = fitted_model.fittedvalues
    pred   = fitted_model.predict(start=test.index[0], end=test.index[-1])

    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train.values, label="Train", color="steelblue",  linewidth=1.5)
    plt.plot(test.index,  test.values,  label="Test",  color="black",       linewidth=1.5, alpha=0.7)
    plt.plot(fitted.index, fitted.values, label="Fitted", color="orange",   linewidth=1.0, alpha=0.8)
    plt.plot(pred.index,   pred.values,   label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
    plt.axvline(train.index[-1], color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("ARIMA — Train/Test Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("arima_train_test.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== ARIMA / SARIMA Time Series Forecasting ===")

    # Placeholder: replace with your time series data
    series = generate_time_series(n_obs=300, seasonal=True)
    print(f"Series: {len(series)} observations  [{series.index[0].date()} → {series.index[-1].date()}]")

    # Train/test split (last 24 months = test)
    train = series[:-24]
    test  = series[-24:]

    # Stationarity check
    print("\nStationarity check:")
    stationary = adf_test(series, "Original")
    if not stationary:
        adf_test(series.diff().dropna(), "First difference")

    # ACF/PACF for order identification
    plot_acf_pacf(series.diff().dropna())

    # Auto order selection
    print("\nAuto order selection:")
    p, d, q = auto_order_selection(train)

    # Fit ARIMA
    print(f"\nFitting ARIMA({p},{d},{q})...")
    arima_result = train_arima(train, order=(p, d, q))
    print(arima_result.summary().tables[0])

    print("\nEvaluation (test set):")
    evaluate(arima_result, test)

    # Residual diagnostics
    print("\nResidual diagnostics:")
    residual_diagnostics(arima_result)

    # Fit SARIMA
    print("\nFitting SARIMA(1,1,1)(1,1,1,12)...")
    sarima_result = train_sarima(train,
                                  order=SARIMA_CONFIG["order"],
                                  seasonal_order=SARIMA_CONFIG["seasonal_order"])
    print("\nSARIMA evaluation:")
    evaluate(sarima_result, test)

    # Forecast
    fc = forecast(sarima_result, steps=24)
    plot_forecast(series, fc, title="SARIMA — 24-Month Forecast")
    plot_train_test_forecast(series, train, test, arima_result)


if __name__ == "__main__":
    main()

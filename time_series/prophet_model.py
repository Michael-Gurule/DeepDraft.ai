"""
Meta Prophet Time Series Forecasting
=======================================
An additive regression model designed for business time series with strong
seasonal patterns and trend changes. Decompose the series into:

    y(t) = trend(t) + seasonality(t) + holidays(t) + noise(t)

Components:
    Trend:       Piecewise linear or logistic growth, with automatic
                 changepoint detection.
    Seasonality: Fourier series approximation of periodic patterns.
                 Yearly (order=10), weekly (order=3), daily (order=4).
    Holidays:    User-defined holiday effects (regression on indicator dummies).
    Regressor:   Additional external regressors (add_regressor).

Why Prophet vs ARIMA:
    - Handles multiple seasonalities natively (ARIMA cannot)
    - Robust to missing data and outliers
    - Automatic changepoint detection
    - Requires minimal hyperparameter tuning
    - Works with daily/weekly/yearly data; ARIMA excels at hourly/stationary data

Use Cases:
    - Marketing channel performance forecasting
    - Sales demand forecasting with holiday effects
    - Web traffic forecasting
    - Energy/utility consumption with seasonal patterns
    - Any series with strong seasonal patterns + occasional trend shifts

Fine-Tuning Recommendations:
    - changepoint_prior_scale (default 0.05): controls trend flexibility.
      Higher (0.1–0.5) → more flexible trend (risk of overfitting).
      Lower (0.001–0.01) → smoother trend (risk of underfitting).
      Most impactful parameter to tune.
    - seasonality_prior_scale (default 10): amplitude of seasonality components.
      Reduce if seasonality is too noisy.
    - holidays_prior_scale (default 10): holiday effect amplitude.
    - n_changepoints (default 25): number of potential trend changepoints.
      Changepoints are placed uniformly in the first 80% of training data.
    - seasonality_mode: 'additive' (default) for constant-amplitude seasonality.
      'multiplicative' when seasonal amplitude grows with trend level.
    - fourier_order: increasing order (e.g., 20 for yearly) captures finer
      seasonal patterns but risks overfitting.
    - Cross-validation: use prophet's built-in cross_validation() for proper
      time series CV with rolling origin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_business_series(
    n_days: int = 730,
    seed:   int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic business time series (daily) with:
        - Linear trend
        - Weekly seasonality
        - Yearly seasonality
        - Trend changepoints
        - Injected holiday effects

    Replace with your own data (Prophet requires 'ds' and 'y' columns):
        df = pd.read_csv("daily_revenue.csv")
        df = df.rename(columns={"date": "ds", "revenue": "y"})
        df["ds"] = pd.to_datetime(df["ds"])

    Returns:
        DataFrame with columns: ds (datetime), y (float)
    """
    rng   = np.random.RandomState(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    # Trend with changepoints
    trend = np.linspace(100, 180, n_days)
    trend[400:] += 30  # trend shift

    # Weekly seasonality
    weekly = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)

    # Yearly seasonality
    yearly = 20 * np.sin(2 * np.pi * dates.dayofyear / 365.25)

    # Holiday spikes (Christmas, Black Friday)
    holiday_effect = np.zeros(n_days)
    for i, d in enumerate(dates):
        if d.month == 12 and d.day in range(22, 27):
            holiday_effect[i] = 25
        if d.month == 11 and d.day in range(24, 29) and d.dayofweek == 4:
            holiday_effect[i] = 20

    # Noise
    noise = rng.normal(0, 5, n_days)

    y = trend + weekly + yearly + holiday_effect + noise
    return pd.DataFrame({"ds": dates, "y": y.astype(np.float32)})


def make_holiday_df() -> pd.DataFrame:
    """
    Define custom holidays for Prophet.

    Replace with your country/event holidays:
        from prophet.make_holidays import make_holidays_df
        holidays = make_holidays_df(year_list=[2022, 2023, 2024], country="US")

    Returns:
        DataFrame with columns: holiday (str), ds (datetime), lower_window, upper_window
    """
    holidays = pd.DataFrame({
        "holiday":      ["christmas"] * 4 + ["black_friday"] * 2,
        "ds":           pd.to_datetime([
                            "2022-12-25", "2023-12-25", "2024-12-25", "2025-12-25",
                            "2022-11-25", "2023-11-24",
                        ]),
        "lower_window": [-3, -3, -3, -3, -1, -1],
        "upper_window": [1, 1, 1, 1, 1, 1],
    })
    return holidays


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: changepoint_prior_scale is most important to tune
PROPHET_CONFIG = {
    "changepoint_prior_scale":   0.05,   # Placeholder: 0.001–0.5 (higher = more flexible)
    "seasonality_prior_scale":   10.0,   # Placeholder: 0.01–100
    "holidays_prior_scale":      10.0,   # Placeholder: 0.01–100
    "seasonality_mode":          "additive",   # Placeholder: 'additive' or 'multiplicative'
    "yearly_seasonality":        True,
    "weekly_seasonality":        True,
    "daily_seasonality":         False,   # Placeholder: True for sub-daily data
    "interval_width":            0.95,    # 95% uncertainty interval
}


def build_prophet(config: dict = None, holidays: pd.DataFrame = None) -> Prophet:
    """
    Build and configure a Prophet model.

    To add custom seasonalities:
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)

    To add regressors (external features):
        model.add_regressor("temperature")
        # Then add column "temperature" to training df
    """
    if config is None:
        config = PROPHET_CONFIG
    model = Prophet(**config)
    if holidays is not None:
        model.holidays = holidays
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: Prophet, df_train: pd.DataFrame,
             df_test: pd.DataFrame) -> dict:
    """
    Evaluate on test set: RMSE, MAE, MAPE.
    """
    future  = model.make_future_dataframe(periods=len(df_test), freq="D")
    forecast_df = model.predict(future)

    # Align predictions to test period
    pred_test = forecast_df.set_index("ds").loc[df_test["ds"]]["yhat"]
    y_true    = df_test.set_index("ds")["y"]

    rmse = np.sqrt(mean_squared_error(y_true, pred_test))
    mae  = mean_absolute_error(y_true, pred_test)
    mape = np.mean(np.abs((y_true - pred_test) / (y_true + 1e-8))) * 100

    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    return {"rmse": rmse, "mae": mae, "mape": mape}


# ─────────────────────────────────────────────
# CROSS-VALIDATION (Prophet built-in)
# ─────────────────────────────────────────────

def prophet_cross_validate(model: Prophet, df: pd.DataFrame,
                             initial: str = "365 days",
                             period:  str = "60 days",
                             horizon: str = "90 days") -> pd.DataFrame:
    """
    Prophet's built-in time series cross-validation.

    Rolling-origin forward validation:
        - initial:  training period for first cutoff
        - period:   spacing between cutoff dates
        - horizon:  forecast horizon to evaluate

    Returns cv_df with yhat, y, ds, cutoff columns.
    """
    print(f"  CV: initial={initial}, period={period}, horizon={horizon}")
    cv_df = cross_validation(model, initial=initial, period=period,
                              horizon=horizon, parallel="processes")
    metrics = performance_metrics(cv_df)
    print("\n  Cross-validation metrics (median):")
    for col in ["rmse", "mae", "mape"]:
        if col in metrics.columns:
            print(f"    {col.upper()}: {metrics[col].median():.4f}")
    return cv_df


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_changepoint_prior(df: pd.DataFrame,
                            scales: list = None) -> float:
    """
    Grid search over changepoint_prior_scale using Prophet CV RMSE.
    Most impactful parameter for trend flexibility.

    Returns best scale.
    """
    if scales is None:
        scales = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]

    rmses = []
    for scale in scales:
        try:
            m = Prophet(changepoint_prior_scale=scale,
                        yearly_seasonality=True,
                        weekly_seasonality=True)
            m.fit(df)
            cv = cross_validation(m, initial="365 days", period="60 days",
                                   horizon="60 days", parallel="processes")
            pm = performance_metrics(cv)
            rmse = pm["rmse"].mean()
            rmses.append(rmse)
            print(f"  changepoint_prior_scale={scale}  RMSE={rmse:.4f}")
        except Exception as e:
            print(f"  scale={scale} failed: {e}")
            rmses.append(np.inf)

    best_scale = scales[np.argmin(rmses)]
    plt.figure(figsize=(8, 4))
    plt.semilogx(scales, rmses, "o-", color="steelblue")
    plt.axvline(best_scale, color="red", linestyle="--",
                label=f"Best: {best_scale}")
    plt.xlabel("changepoint_prior_scale (log scale)")
    plt.ylabel("CV RMSE")
    plt.title("Prophet — Changepoint Prior Scale Tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prophet_tuning.png", dpi=150)
    plt.show()

    print(f"\n  Best changepoint_prior_scale: {best_scale}")
    return best_scale


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_forecast(model: Prophet, forecast_df: pd.DataFrame,
                  df: pd.DataFrame, test_start: str = None) -> None:
    """Plot forecast with uncertainty bands and optional test split."""
    fig = model.plot(forecast_df, figsize=(12, 5))
    ax  = fig.axes[0]
    ax.set_title("Prophet — Forecast")
    if test_start:
        ax.axvline(pd.to_datetime(test_start), color="red",
                   linestyle="--", label="Test start")
        ax.legend()
    plt.tight_layout()
    plt.savefig("prophet_forecast.png", dpi=150)
    plt.show()


def plot_components(model: Prophet, forecast_df: pd.DataFrame) -> None:
    """
    Plot model components: trend, weekly seasonality, yearly seasonality, holidays.
    Key for interpreting what the model learned.
    """
    fig = model.plot_components(forecast_df, figsize=(10, 8))
    plt.suptitle("Prophet — Component Decomposition", y=1.02)
    plt.tight_layout()
    plt.savefig("prophet_components.png", dpi=150)
    plt.show()


def plot_changepoints(model: Prophet, df: pd.DataFrame,
                       forecast_df: pd.DataFrame) -> None:
    """
    Plot trend with detected changepoints highlighted.
    Helps identify when major trend shifts occurred.
    """
    import matplotlib.dates as mdates
    from prophet.plot import add_changepoints_to_plot

    fig = model.plot(forecast_df, figsize=(12, 5))
    ax  = fig.axes[0]
    add_changepoints_to_plot(ax, model, forecast_df)
    ax.set_title("Prophet — Trend Changepoints")
    plt.tight_layout()
    plt.savefig("prophet_changepoints.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Meta Prophet Time Series Forecasting ===")

    # Placeholder: replace with your dataset
    df = generate_business_series(n_days=730)
    print(f"Series: {len(df)} daily observations  "
          f"[{df['ds'].min().date()} → {df['ds'].max().date()}]")

    # Train/test split (last 90 days = test)
    test_start = df["ds"].iloc[-90]
    df_train   = df[df["ds"] < test_start]
    df_test    = df[df["ds"] >= test_start]
    print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")

    # Build and fit model
    holidays = make_holiday_df()
    model    = build_prophet(holidays=holidays)
    model.fit(df_train)

    # Forecast
    future      = model.make_future_dataframe(periods=90, freq="D")
    forecast_df = model.predict(future)

    print("\nEvaluation (test set):")
    evaluate(model, df_train, df_test)

    # Cross-validation
    print("\nCross-validation:")
    prophet_cross_validate(model, df_train,
                            initial="365 days",
                            period="60 days",
                            horizon="60 days")

    # Hyperparameter tuning (optional — takes time)
    # tune_changepoint_prior(df_train)

    # Visualizations
    plot_forecast(model, forecast_df, df, test_start=str(test_start.date()))
    plot_components(model, forecast_df)
    plot_changepoints(model, df, forecast_df)


if __name__ == "__main__":
    main()

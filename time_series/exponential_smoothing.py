"""
Exponential Smoothing (ETS)
============================

Classical statistical forecasting family. Each method is optimal for a specific
data generating process — fast to fit, interpretable, and strong baselines.

Methods covered:
  SES  — Simple Exponential Smoothing (level only; no trend/seasonality)
  Holt — Double Exponential Smoothing (level + additive trend)
  Holt-Winters — Triple Exponential Smoothing (level + trend + seasonality)
  ETS  — Error/Trend/Seasonality framework (statsmodels; covers all variants)
  Theta — Theta decomposition method (M3 competition winner)

ETS component notation:
  Error:       A (additive) or M (multiplicative)
  Trend:       N (none), A (additive), Ad (additive damped), M (multiplicative)
  Seasonality: N (none), A (additive), M (multiplicative)

  ETS(A,A,A)  — additive error, additive trend, additive seasonality (Holt-Winters)
  ETS(A,Ad,M) — additive error, damped trend, multiplicative seasonality
  ETS(A,N,N)  — SES

Update equations (Holt-Winters additive):
  ℓ_t = α(y_t - s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})      ← level
  b_t = β*(ℓ_t - ℓ_{t-1}) + (1-β*)b_{t-1}                ← trend
  s_t = γ(y_t - ℓ_{t-1} - b_{t-1}) + (1-γ)s_{t-m}        ← seasonal

  h-step forecast: ŷ_{t+h|t} = ℓ_t + h·b_t + s_{t+h-m(k+1)}

Parameters:
  α ∈ (0,1): level smoothing    (# Placeholder: larger = faster adaptation)
  β ∈ (0,1): trend smoothing    (# Placeholder: smaller = slower trend change)
  γ ∈ (0,1): seasonal smoothing (# Placeholder:)
  φ ∈ (0,1): damping factor     (# Placeholder: 0.8-0.98 typical; 1.0 = no damping)

Rule of thumb for data patterns → method:
  Stationary, no trend, no seasonality   → SES
  Trend, no seasonality                  → Holt
  Trend + seasonality                    → Holt-Winters
  Unknown / automatic                    → ETS auto-selection (AIC)
  Series with strong trend from zero     → Holt-Winters multiplicative trend

Fine-tuning recommendations:
  - seasonal_periods (#Placeholder: 12 monthly, 4 quarterly, 7 daily, 52 weekly)
  - trend (#Placeholder: "add" | "mul" | "add_damped" | None)
  - seasonal (#Placeholder: "add" | "mul" | None)
  - damped_trend (#Placeholder: True): dampens long-horizon trend; reduces overconfidence
  - use_boxcox (#Placeholder: False or lambda): stabilize variance in multiplicative series
  - initialization_method (#Placeholder: "estimated"): let statsmodels estimate initial state

References:
  Holt (1957) — Forecasting Seasonals and Trends by Exponentially Weighted Averages
  Winters (1960) — Forecasting Sales by Exponentially Weighted Moving Averages
  Hyndman et al. (2002) — A State Space Framework for Automatic Forecasting
  Hyndman & Khandakar (2008) — Automatic Time Series Forecasting (auto.arima / ETS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing,
    Holt,
)
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "seasonal_periods": 12,          # # Placeholder: 12 monthly, 4 quarterly, 7 daily
    "forecast_horizon": 24,          # # Placeholder:
    "train_fraction":   0.8,         # # Placeholder:
    "trend":            "add",       # # Placeholder: "add" | "mul" | None
    "seasonal":         "add",       # # Placeholder: "add" | "mul" | None
    "damped_trend":     True,        # # Placeholder:
    "alpha":            None,        # # Placeholder: None = optimize
    "beta":             None,        # # Placeholder: None = optimize
    "gamma":            None,        # # Placeholder: None = optimize
    "phi":              None,        # # Placeholder: None = optimize (damping parameter)
    "use_boxcox":       False,       # # Placeholder: True if variance grows with level
    "initialization":  "estimated", # # Placeholder:
    "seed":             42,
}


# ─────────────────────────────────────────────
# Synthetic Data
# ─────────────────────────────────────────────

def generate_synthetic_data(
    n: int = 120,
    trend_slope: float = 2.0,      # # Placeholder:
    seasonal_amplitude: float = 15.0,  # # Placeholder:
    noise_std: float = 3.0,        # # Placeholder:
    seasonal_periods: int = 12,
    seed: int = 42,
) -> pd.Series:
    """
    Generate synthetic monthly time series with trend + seasonality + noise.
    Replace with your real data: pd.read_csv(...).set_index("date")["value"]
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend    = 100 + trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_periods)
    noise    = rng.normal(0, noise_std, n)
    y = trend + seasonal + noise
    idx = pd.date_range("2015-01", periods=n, freq="ME")
    return pd.Series(y, index=idx, name="value")


def load_airline_data() -> pd.Series:
    """
    Classic Box-Jenkins airline passengers dataset (monthly, 1949-1960).
    Strong multiplicative seasonality + trend.
    """
    try:
        import statsmodels.api as sm
        data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
        y = pd.Series(data["value"].values)
        y.index = pd.date_range("1949-01", periods=len(y), freq="ME")
        return y
    except Exception:
        return generate_synthetic_data()


# ─────────────────────────────────────────────
# Model Fitting
# ─────────────────────────────────────────────

def fit_ses(y_train: pd.Series, alpha: float = None) -> dict:
    """
    Simple Exponential Smoothing — level only, no trend/seasonality.
    Best for stationary series.
    """
    model = SimpleExpSmoothing(y_train, initialization_method="estimated")
    fit = model.fit(smoothing_level=alpha, optimized=(alpha is None))
    print(f"SES: α={fit.params['smoothing_level']:.4f}")
    return {"model": model, "fit": fit, "name": "SES"}


def fit_holt(y_train: pd.Series, damped: bool = False) -> dict:
    """
    Holt's Linear Trend (double exponential smoothing).
    damped=True prevents over-shooting for long horizons.
    """
    model = Holt(y_train, damped_trend=damped, initialization_method="estimated")
    fit = model.fit(optimized=True)
    label = "Holt Damped" if damped else "Holt Linear"
    print(f"{label}: α={fit.params['smoothing_level']:.4f}  "
          f"β={fit.params['smoothing_trend']:.4f}"
          + (f"  φ={fit.params['damping_trend']:.4f}" if damped else ""))
    return {"model": model, "fit": fit, "name": label}


def fit_holt_winters(
    y_train: pd.Series,
    trend: str = "add",
    seasonal: str = "add",
    seasonal_periods: int = 12,
    damped_trend: bool = False,
    use_boxcox: bool = False,
) -> dict:
    """
    Holt-Winters triple exponential smoothing.
    trend:    "add" (linear) | "mul" (exponential) | None
    seasonal: "add" (fixed amplitude) | "mul" (proportional amplitude)
    Use multiplicative seasonal for series where seasonal variation grows with level.
    """
    model = ExponentialSmoothing(
        y_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        use_boxcox=use_boxcox,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    label = f"HW({trend or 'N'},{seasonal or 'N'}{'D' if damped_trend else ''})"
    print(f"{label}: α={fit.params['smoothing_level']:.4f}  "
          f"β={fit.params.get('smoothing_trend', 0):.4f}  "
          f"γ={fit.params.get('smoothing_seasonal', 0):.4f}")
    return {"model": model, "fit": fit, "name": label}


def fit_ets_auto(y_train: pd.Series, seasonal_periods: int = 12) -> dict:
    """
    Automatic ETS model selection by AIC.
    Tries multiple error/trend/seasonal combinations and returns best by AIC.
    """
    best_aic = np.inf
    best_result = None

    configs = [
        ("add",  "add",  "add"),
        ("add",  "add",  "mul"),
        ("add",  None,   "add"),
        ("add",  None,   "mul"),
        ("mul",  "add",  "mul"),
        ("mul",  "mul",  "mul"),
        ("add",  "add",  None),
        ("add",  None,   None),
    ]
    for trend, seasonal, error in configs:
        try:
            model = ExponentialSmoothing(
                y_train,
                trend=trend, seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_result = {
                    "model": model, "fit": fit,
                    "name": f"ETS_auto({error},{trend or 'N'},{seasonal or 'N'})",
                    "aic": fit.aic,
                }
        except Exception:
            pass

    if best_result:
        print(f"Best ETS: {best_result['name']}  AIC={best_result['aic']:.2f}")
    return best_result


def fit_theta(y_train: pd.Series, seasonal_periods: int = 12, theta: float = 2.0) -> dict:
    """
    Theta method: decomposes series via two theta lines, forecasts separately.
    theta=2 is standard (won M3 competition). theta → ∞ gives SES.
    """
    try:
        model = ThetaModel(y_train, period=seasonal_periods, deseasonalize=True)
        fit = model.fit()
        # statsmodels applies theta at forecast time, not at fit time.
        print(f"Theta: θ={theta}")
        return {"model": model, "fit": fit, "name": f"Theta(θ={theta})", "theta": theta}
    except Exception as e:
        print(f"Theta model failed: {e}")
        return fit_ses(y_train)


# ─────────────────────────────────────────────
# Forecasting and Evaluation
# ─────────────────────────────────────────────

def forecast(result: dict, h: int) -> pd.Series:
    """Generate h-step ahead point forecast."""
    if "theta" in result:
        return result["fit"].forecast(h, theta=result["theta"])
    try:
        return result["fit"].forecast(h)
    except Exception:
        return result["fit"].predict(start=len(result["fit"].fittedvalues),
                                     end=len(result["fit"].fittedvalues) + h - 1)


def prediction_interval(result: dict, h: int, alpha: float = 0.05,
                        n_sims: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Returns the point forecast with (1-alpha) prediction intervals.

    Holt-Winters fits (SES, Holt, ExponentialSmoothing) have no analytic
    interval method in statsmodels, so the interval comes from simulating
    n_sims future paths with resampled additive errors and taking quantiles.
    Theta fits provide analytic intervals via prediction_intervals().
    """
    fit = result["fit"]
    fc = forecast(result, h)

    if hasattr(fit, "prediction_intervals"):          # ThetaModelResults
        pi = fit.prediction_intervals(h, theta=result["theta"], alpha=alpha)
        return pd.DataFrame({"forecast": fc, "lower": pi.iloc[:, 0], "upper": pi.iloc[:, 1]})

    sims = fit.simulate(h, anchor="end", repetitions=n_sims, error="add",
                        random_state=seed)
    sims = pd.DataFrame(np.asarray(sims).reshape(h, -1), index=fc.index)
    return pd.DataFrame({
        "forecast": fc,
        "lower":    sims.quantile(alpha / 2, axis=1),
        "upper":    sims.quantile(1 - alpha / 2, axis=1),
    })


def evaluate(result: dict, y_test: pd.Series) -> dict:
    """Compute RMSE, MAE, MAPE for a model against test data."""
    h = len(y_test)
    fc = forecast(result, h)
    fc_vals = fc.values[:h]
    y_vals  = y_test.values[:h]

    rmse = np.sqrt(mean_squared_error(y_vals, fc_vals))
    mae  = mean_absolute_error(y_vals, fc_vals)
    mape = np.mean(np.abs((y_vals - fc_vals) / (y_vals + 1e-8))) * 100

    return {"model": result["name"], "rmse": rmse, "mae": mae, "mape": mape}


def compare_models(y: pd.Series, cfg: dict) -> pd.DataFrame:
    """Train all ETS variants and compare forecast accuracy."""
    n = len(y)
    split = int(n * cfg["train_fraction"])
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    m = cfg["seasonal_periods"]
    h = len(y_test)

    print(f"\nDataset: n={n} | train={split} | test={h} | seasonality={m}\n")

    results_raw = [
        fit_ses(y_train),
        fit_holt(y_train, damped=False),
        fit_holt(y_train, damped=True),
        fit_holt_winters(y_train, trend="add", seasonal="add",
                          seasonal_periods=m, damped_trend=cfg["damped_trend"]),
        fit_holt_winters(y_train, trend="add", seasonal="mul",
                          seasonal_periods=m, damped_trend=cfg["damped_trend"]),
        fit_ets_auto(y_train, seasonal_periods=m),
        fit_theta(y_train, seasonal_periods=m),
    ]
    results_raw = [r for r in results_raw if r is not None]

    metrics = [evaluate(r, y_test) for r in results_raw]
    df = pd.DataFrame(metrics).sort_values("rmse")
    print("\n" + df.to_string(index=False))
    return df, results_raw, y_train, y_test


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_decomposition(y: pd.Series, seasonal_periods: int = 12):
    """Additive STL-style visual decomposition of the series."""
    from statsmodels.tsa.seasonal import STL
    stl = STL(y, period=seasonal_periods, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("STL Decomposition", fontsize=12)
    for ax, component, label in zip(
        axes,
        [y, res.trend, res.seasonal, res.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        ax.plot(component, lw=1.2)
        ax.set_ylabel(label); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig("ets_decomposition.png", dpi=150)
    plt.show()
    print("Saved: ets_decomposition.png")


def plot_forecasts(results_raw: list, y_train: pd.Series, y_test: pd.Series, cfg: dict):
    """Plot all model forecasts vs actuals on the test set."""
    h = len(y_test)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_raw)))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_train.index, y_train, "k-", lw=1.5, label="Train", alpha=0.7)
    ax.plot(y_test.index,  y_test,  "k--", lw=1.5, label="Actual")

    for res, color in zip(results_raw, colors):
        try:
            fc = forecast(res, h)
            ax.plot(y_test.index, fc.values[:h], lw=1.2, color=color,
                    label=res["name"], alpha=0.85)
        except Exception:
            pass

    ax.axvline(y_train.index[-1], color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Exponential Smoothing — Model Comparison")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ets_forecasts.png", dpi=150)
    plt.show()
    print("Saved: ets_forecasts.png")


def plot_prediction_intervals(result: dict, y_train: pd.Series, y_test: pd.Series):
    """Plot forecast with 80% and 95% prediction intervals."""
    h = len(y_test)
    ci95 = prediction_interval(result, h, alpha=0.05)
    ci80 = prediction_interval(result, h, alpha=0.20)
    idx = y_test.index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_train.index, y_train, "k-", lw=1.5, label="Train")
    ax.plot(idx, y_test, "k--", lw=1.5, label="Actual")
    ax.plot(idx, ci95["forecast"], "b-", lw=2, label=result["name"])
    ax.fill_between(idx, ci95["lower"], ci95["upper"], alpha=0.15, color="blue", label="95% PI")
    ax.fill_between(idx, ci80["lower"], ci80["upper"], alpha=0.25, color="blue", label="80% PI")
    ax.axvline(y_train.index[-1], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"Forecast with Prediction Intervals — {result['name']}")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ets_prediction_intervals.png", dpi=150)
    plt.show()
    print("Saved: ets_prediction_intervals.png")


def plot_comparison_bar(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Exponential Smoothing Model Comparison", fontsize=12)
    for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
        ax.barh(df["model"][::-1], df[metric][::-1], color="steelblue", edgecolor="white")
        ax.set_xlabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("ets_comparison.png", dpi=150)
    plt.show()
    print("Saved: ets_comparison.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}\n")

    # ── Dataset ─────────────────────────────────────────────────────
    # Airline passengers (classic benchmark with multiplicative seasonality)
    y = load_airline_data()
    print(f"Series: n={len(y)} | range=[{y.min():.1f}, {y.max():.1f}]")

    # ── Decomposition ───────────────────────────────────────────────
    print("\n=== STL Decomposition ===")
    plot_decomposition(y, seasonal_periods=CONFIG["seasonal_periods"])

    # ── Model comparison ────────────────────────────────────────────
    print("\n=== Model Comparison ===")
    df, results_raw, y_train, y_test = compare_models(y, CONFIG)
    plot_forecasts(results_raw, y_train, y_test, CONFIG)
    plot_comparison_bar(df)

    # ── Best model prediction intervals ─────────────────────────────
    print("\n=== Prediction Intervals (Best Model) ===")
    best_result = results_raw[0]   # already sorted by RMSE
    hw = fit_holt_winters(y_train, trend="add", seasonal="mul",
                           seasonal_periods=CONFIG["seasonal_periods"],
                           damped_trend=True)
    plot_prediction_intervals(hw, y_train, y_test)

    print(f"\nBest model: {df.iloc[0]['model']}  RMSE={df.iloc[0]['rmse']:.4f}")


if __name__ == "__main__":
    main()

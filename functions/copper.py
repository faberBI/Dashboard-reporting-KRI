import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
import optuna
warnings.filterwarnings("ignore")

def make_lag_df(series, n_lags):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df.dropna().reset_index(drop=True)

def plot_copper_forecast(df_model, result_df_annual):
    fig, ax = plt.subplots(figsize=(14, 7))

    # ✅ Year-End esplicito (invece di 'Y')
    df_yearly = df_model.resample("YE").mean()

    # Prezzo storico
    ax.plot(df_yearly.index, df_yearly["Copper"], label="Historical Price in €", color="blue")

    # Previsioni future
    ax.plot(
        result_df_annual.index,
        result_df_annual["Mean_Forecast"],
        label="Forecast Average",
        color="orange",
        linestyle="--"
    )

    # Banda di incertezza (usa colonne se presenti)
    if "CP_Lower_95" in result_df_annual.columns and "GARCH_Upper_95" in result_df_annual.columns:
        ax.fill_between(
            result_df_annual.index,
            result_df_annual["CP_Lower_95"],
            result_df_annual["GARCH_Upper_95"],
            color="green",
            alpha=0.2,
            label="Adjusted Forecast"
        )

    ax.set_title("Historical and Forecasted Copper Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price in Euro")
    ax.grid(True)
    ax.legend()
    return fig

def plot_var_vs_budget(result_df_annual):
    # Copia il DataFrame
    df_plot = result_df_annual.copy()
    # Assicurati che l'indice sia datetime
    df_plot.index = pd.to_datetime(df_plot.index)
    # Estrai solo gli anni per il plot
    years = df_plot.index.year

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Grafico barre VaR_vs_budget
    ax.bar(years, df_plot["VaR_vs_budget"], color='#00196c', alpha=0.7)
    
    ax.set_title("VaR vs Budget per anno")
    ax.set_xlabel("Anno")
    ax.set_ylabel("VaR vs Budget (EUR)")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    return fig

def monte_carlo_forecast_cp_from_disk(
    series,
    N_SIM=10_000,
    end_date=None,
    random_seed=42
):
    """
    Random Walk additivo sul PREZZO + Conformal/Bootstrap sul PREZZO.
    Optuna ottimizza drift_window, cal_window e alpha (coverage vs width).
    Restituisce lo stesso schema colonne usato nella tua app.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    rng = np.random.default_rng(random_seed)

    # -----------------------------
    # 0) Sanitizzazione input
    # -----------------------------
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    series = pd.to_numeric(series, errors="coerce").dropna()

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("La serie deve avere DatetimeIndex (Time come index).")

    series = series[~series.index.duplicated()].sort_index()

    if len(series) < 40:
        raise ValueError("Serie troppo corta: servono almeno ~40 osservazioni per Optuna+Conformal robusto.")

    price = series.astype(float)
    dP = price.diff().dropna().values   # ΔP_t
    P  = price.values                  # P_t

    # -----------------------------
    # Helpers RW
    # -----------------------------
    def rolling_drift(dP_hist: np.ndarray, window: int) -> float:
        w = min(window, len(dP_hist))
        if w <= 0:
            return 0.0
        return float(np.mean(dP_hist[-w:]))

    def one_step_predict_price(P_prev: float, mu: float) -> float:
        return float(P_prev + mu)

    # -----------------------------
    # 2) Optuna objective (TimeSeriesSplit, no leakage)
    #    obiettivo: width piccolo + penalità se coverage < target
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=5)
    idx_all = np.arange(1, len(P))  # t=1..T-1

    def objective(trial: optuna.Trial) -> float:
        drift_window = trial.suggest_int("drift_window", 12, 120)
        cal_window   = trial.suggest_int("cal_window",   12, 120)
        alpha_trial  = trial.suggest_float("alpha", 0.01, 0.2)

        widths = []
        penalties = []

        for tr_idx, va_idx in tscv.split(idx_all):
            tr_points = idx_all[tr_idx]
            va_points = idx_all[va_idx]

            # errori 1-step nel train fold
            err_train = []
            for t in tr_points:
                mu_t = rolling_drift(dP_hist=dP[:t], window=drift_window)
                p_hat = one_step_predict_price(P_prev=P[t-1], mu=mu_t)
                err_train.append(P[t] - p_hat)

            err_train = np.asarray(err_train, dtype=float)
            w = min(cal_window, len(err_train))
            if w < 5:
                return 1e9

            q_hat = np.quantile(np.abs(err_train[-w:]), 1 - alpha_trial)

            # validazione: coverage e width
            covered = 0
            total = 0
            width_sum = 0.0

            for t in va_points:
                mu_t = rolling_drift(dP_hist=dP[:t], window=drift_window)
                p_hat = one_step_predict_price(P_prev=P[t-1], mu=mu_t)

                lower = p_hat - q_hat
                upper = p_hat + q_hat

                y_true = P[t]
                total += 1
                covered += int(lower <= y_true <= upper)
                width_sum += (upper - lower)

            cov = covered / max(1, total)
            avg_width = width_sum / max(1, total)

            target = 1 - alpha_trial
            # penalità forte solo per undercoverage (se over-coverage va bene ma è inefficiente -> width già penalizza)
            penalty = max(0.0, target - cov) * 1000.0

            widths.append(avg_width)
            penalties.append(penalty)

        return float(np.mean(widths) + np.mean(penalties))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=300, show_progress_bar=False)

    best = study.best_params
    DRIFT_W = int(best["drift_window"])
    CAL_W   = int(best["cal_window"])
    ALPHA_OPT = float(best["alpha"])

    # -----------------------------
    # 3) Calibra residui 1-step e q_hat finale
    # -----------------------------
    err_hist = []
    for t in range(1, len(P)):
        mu_t = rolling_drift(dP_hist=dP[:t], window=DRIFT_W)
        p_hat = one_step_predict_price(P_prev=P[t-1], mu=mu_t)
        err_hist.append(P[t] - p_hat)

    err_hist = np.asarray(err_hist, dtype=float)
    w = min(CAL_W, len(err_hist))
    resid_pool = err_hist[-w:]

    # q_hat 1-step (utile per debug; multi-step useremo bootstrap)
    q_hat = float(np.quantile(np.abs(resid_pool), 1 - ALPHA_OPT))

    # -----------------------------
    # 4) Date future
    # -----------------------------
    last_date = pd.to_datetime(price.index.max()).normalize()
    if end_date is None:
        end_date = last_date + pd.DateOffset(years=5)
    else:
        end_date = pd.to_datetime(end_date)

    if end_date <= last_date:
        raise ValueError(
            f"end_date ({end_date.date()}) deve essere successiva all’ultima data disponibile ({last_date.date()})."
        )

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        end=end_date,
        freq="ME"
    )
    H = len(future_dates)

    # -----------------------------
    # 5) Drift “oggi”
    # -----------------------------
    mu0 = rolling_drift(dP_hist=dP, window=DRIFT_W)
    p0 = float(price.iloc[-1])

    # -----------------------------
    # 6) Simulazione bootstrap multi-step sul PREZZO
    #    P_{t+1} = P_t + mu0 + shock_t
    # -----------------------------
    shocks = rng.choice(resid_pool, size=(N_SIM, H), replace=True)
    increments = mu0 + shocks
    sim_paths = p0 + np.cumsum(increments, axis=1)
    sim_paths = np.maximum(sim_paths, 0.01)

    lower_q = np.percentile(sim_paths, 100 * (ALPHA_OPT / 2.0), axis=0)
    upper_q = np.percentile(sim_paths, 100 * (1 - ALPHA_OPT / 2.0), axis=0)

    # (per compatibilità, mappiamo questi su "GARCH_*" e "CP_*")
    lower_95 = lower_q
    upper_95 = upper_q
    cp_lower = lower_q
    cp_upper = upper_q

    # -----------------------------
    # 7) Output identico (schema colonne)
    # -----------------------------
    final_forecast = pd.DataFrame(
        {
            "Mean_Forecast": (cp_lower + upper_95) / 2.0,
            "GARCH_Lower_95": lower_95,
            "GARCH_Upper_95": upper_95,
            "CP_Lower_95": cp_lower,
            "CP_Upper_95": cp_upper,
        },
        index=future_dates
    )

    df_yearly = final_forecast.resample("YE").mean()

    return final_forecast, df_yearly


def full_copper_forecast(link_df, price_col='Copper', N_SIM=1000, alpha=0.05, DIST="ged", calibration_size_pct=0.05):

    # ================= Preprocessing =================
    df = pd.read_excel(link_df)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time").reset_index(drop=True)
    series = pd.to_numeric(df[price_col], errors="coerce").dropna()

    BEST_LAG = 1 

    # ================= Dataset con lag =================
    data = make_lag_df(series, BEST_LAG)
    dates = df["Time"].iloc[-len(data):]  # allineamento con data dopo lag

    n_total = len(data)
    n_train = int(n_total * 0.9)
    n_cal = int(n_total * calibration_size_pct)
    n_test = n_total - n_train - n_cal

    train = data.iloc[:n_train]
    calibration = data.iloc[n_train:n_train+n_cal]
    test = data.iloc[n_train+n_cal:]

    X_train, y_train = train.drop("y", axis=1), train["y"]
    X_cal, y_cal = calibration.drop("y", axis=1), calibration["y"]
    X_test, y_test = test.drop("y", axis=1), test["y"]

    dates_train = dates.iloc[:n_train]
    dates_cal = dates.iloc[n_train:n_train+n_cal]
    dates_test = dates.iloc[n_train+n_cal:]

    # ================= CatBoost finale =================
    BEST_PARAMS = {'iterations': 96, 'depth': 3, 'learning_rate': 0.075, 
                   'l2_leaf_reg': 1.278, 'max_ctr_complexity': 2, 'min_data_in_leaf': 24}
    cat_model = CatBoostRegressor(**BEST_PARAMS, loss_function="RMSE", verbose=False,
                                  monotone_constraints=[1]*BEST_LAG)
    cat_model.fit(X_train, y_train)

    # ================= GARCH sui residui =================
    residuals = y_train - cat_model.predict(X_train)
    garch = arch_model(residuals, vol="Garch", p=1, q=1, mean="Zero", dist=DIST)
    garch_fit = garch.fit(disp="off")
    sigma_test = np.sqrt(garch_fit.conditional_volatility[-len(X_test):])

    # ================= Monte Carlo sul test set =================
    np.random.seed(42)
    sim_paths = np.zeros((N_SIM, len(X_test)))
    for sim in range(N_SIM):
        if DIST == "t":
            z = np.random.standard_t(df=garch_fit.params["nu"], size=len(X_test))
        else:
            z = np.random.standard_normal(len(X_test))
        sim_paths[sim, :] = cat_model.predict(X_test) + sigma_test * z

    y_test_pred_mean = sim_paths.mean(axis=0)
    y_test_lower = np.percentile(sim_paths, 100*alpha/2, axis=0)
    y_test_upper = np.percentile(sim_paths, 100*(1-alpha/2), axis=0)

    # ================= Conformal Prediction =================
    sigma_cal = np.sqrt(garch_fit.conditional_volatility[-len(X_cal):])
    sim_paths_cal = np.zeros((N_SIM, len(X_cal)))
    for sim in range(N_SIM):
        if DIST == "t":
            z = np.random.standard_t(df=garch_fit.params["nu"], size=len(X_cal))
        else:
            z = np.random.standard_normal(len(X_cal))
        sim_paths_cal[sim, :] = cat_model.predict(X_cal) + sigma_cal * z

    y_cal_pred_mean = sim_paths_cal.mean(axis=0)
    conformity_scores = np.abs(y_cal.values - y_cal_pred_mean)
    q_hat = np.quantile(conformity_scores, 1-alpha)

    cp_lower_test = y_test_pred_mean - q_hat
    cp_upper_test = y_test_pred_mean + q_hat

    # ================= Plot =================
    start_date = pd.Timestamp("2015-01-01")

    mask_train = dates_train >= start_date
    mask_cal = dates_cal >= start_date
    mask_test = dates_test >= start_date

    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(dates_train[mask_train], y_train.values[mask_train], label="Train", color="black", linewidth=1.5)
    ax.plot(dates_cal[mask_cal], y_cal.values[mask_cal], label="Calibration", color="gray", linewidth=1.5)
    ax.plot(dates_test[mask_test], y_test.values[mask_test], label="Test (reale)", color="blue", linewidth=2)
    ax.plot(dates_test[mask_test], y_test_pred_mean[mask_test], label="Forecast Test (CatBoost + GARCH MC)", color="orange", linestyle="--", linewidth=2)
    ax.fill_between(dates_test[mask_test], cp_lower_test[mask_test], cp_upper_test[mask_test], color='orange', alpha=0.2, label='CP 95%')
    ax.axvline(x=dates_test.iloc[0], color="gray", linestyle=":", linewidth=1)

    ax.set_title("Forecast Test Hybrid Model con Conformal Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


def make_lag_df(series, n_lags):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df.dropna().reset_index(drop=True)

def plot_copper_forecast(df_model, result_df_annual):
    fig, ax = plt.subplots(figsize=(14,7))
    df_yearly = df_model.resample('Y').mean()
    # Prezzo storico
    ax.plot(df_yearly.index, df_yearly['Copper'], label='Historical Price in €', color='blue')

    # Previsioni future
    ax.plot(result_df_annual.index, result_df_annual['Mean_Forecast'], label='Forecast Average', color='orange', linestyle='--')

    # Banda di incertezza
    if 'CP_Lower_95' in result_df_annual.columns and 'GARCH_Upper_95' in result_df_annual.columns:
        ax.fill_between(result_df_annual.index, 
                        result_df_annual['CP_Lower_95'], 
                        result_df_annual['GARCH_Upper_95'], 
                        color='green', alpha=0.2, label='Adjusted Forecast')

    ax.set_title('Historical and Forecasted Copper Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price in Euro')
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



def monte_carlo_forecast_cp_from_disk(series, cat_model_path="utils/catboost_model.cbm", garch_model_path="utils/garch_model.pkl", params_path="utils/model_params.pkl", N_SIM=1000, alpha=0.05, end_date=None, random_seed=42):

    np.random.seed(random_seed)

    # -----------------------------
    # Caricamento modelli
    # -----------------------------
    cat_model = CatBoostRegressor()
    cat_model.load_model(cat_model_path)

    with open(garch_model_path, "rb") as f:
        garch_fit = pickle.load(f)

    with open(params_path, "rb") as f:
        params_loaded = pickle.load(f)
    
    BEST_LAG = params_loaded["BEST_LAG"]
    CALIBRATION_H = params_loaded.get("CALIBRATION_H", 24)

    # -----------------------------
    # Orizzonte in mesi
    # -----------------------------
    last_date = pd.Timestamp.now().normalize()
    if end_date is None:
        end_date = last_date + pd.DateOffset(years=5)
    else:
        end_date = pd.to_datetime(end_date)

    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                 end=end_date, freq='M')
    H = len(future_dates)

    # -----------------------------
    # Monte Carlo simulation
    # -----------------------------
    last_series = series.copy()
    sim_paths = np.zeros((N_SIM, H))

    garch_fc = garch_fit.forecast(horizon=H)
    sigma = np.sqrt(garch_fc.variance.values[-1])
    DIST = garch_fit.model.distribution.name.lower()

    for sim in range(N_SIM):
        path_series = last_series.copy()
        if DIST == "t":
            z = np.random.standard_t(df=garch_fit.params["nu"], size=H)
        else:
            z = np.random.standard_normal(H)
        for h in range(H):
            lags = path_series.iloc[-BEST_LAG:].values
            X_future = pd.DataFrame([lags], columns=[f"lag_{i+1}" for i in range(BEST_LAG)])
            mu = cat_model.predict(X_future)[0]
            eps = sigma[h] * z[h]
            y_next = mu + eps
            sim_paths[sim, h] = y_next
            path_series = pd.concat([path_series, pd.Series([y_next])], ignore_index=True)

    # -----------------------------
    # Fan chart GARCH
    # -----------------------------
    forecast_mean = sim_paths.mean(axis=0)
    lower_95 = np.percentile(sim_paths, 100*alpha/2, axis=0)
    upper_95 = np.percentile(sim_paths, 100*(1-alpha/2), axis=0)

    # -----------------------------
    # Conformal Prediction
    # -----------------------------
    data_cp = make_lag_df(series, BEST_LAG)
    calibration_data = data_cp.iloc[-CALIBRATION_H:]
    X_cal = calibration_data.drop("y", axis=1)
    y_cal = calibration_data["y"].values
    y_cal_pred = cat_model.predict(X_cal)

    # Conformity score normalizzato con GARCH
    garch_cal_fc = garch_fit.forecast(horizon=CALIBRATION_H)
    sigma_cal = np.sqrt(garch_cal_fc.variance.values[-1])
    conformity_scores = np.abs((y_cal - y_cal_pred) / sigma_cal)
    q_hat = np.quantile(conformity_scores, 1 - alpha)

    # Intervalli CP adattivi
    cp_lower = forecast_mean - q_hat * sigma
    cp_upper = forecast_mean + q_hat * sigma

    final_forecast = pd.DataFrame({
        "Mean_Forecast": (cp_lower+upper_95)/2,
        "GARCH_Lower_95": lower_95,
        "GARCH_Upper_95": upper_95,
        "CP_Lower_95": cp_lower,
        "CP_Upper_95": cp_upper
    }, index=future_dates)

    df_yearly = final_forecast.resample('Y').mean()
    
    return final_forecast, df_yearly

def full_copper_forecast(link_df, price_col='Copper', forecast_horizon_years=5, N_SIM=1000,
                         CALIBRATION_H=24, alpha=0.05, DIST="ged", optuna_trials=300):
    """
    Funzione completa: selezione lag, tuning CatBoost con Optuna, fit CatBoost,
    GARCH sui residui, Monte Carlo forecast e Conformal Prediction.
    
    Args:
        df: DataFrame con colonne 'Time' e prezzi.
        price_col: nome colonna dei prezzi.
        forecast_horizon_years: anni di forecast.
        N_SIM: numero simulazioni Monte Carlo.
        CALIBRATION_H: ultimi periodi per Conformal Prediction.
        alpha: livello di significatività per Conformal Prediction.
        DIST: distribuzione GARCH ('t' o 'ged').
        optuna_trials: numero trial Optuna.
        
    Returns:
        final_forecast: DataFrame con forecast e intervalli.
        fig: matplotlib figure con plot storico + test + forecast.
    """
    
    # =========================================================
    # Preprocessing
    # =========================================================
    df = pd.read_excel(link_df)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time").reset_index(drop=True)
    series = pd.to_numeric(df[price_col], errors="coerce")
    
    # Funzione lag features
    def make_lag_df(series, n_lags):
        df_lag = pd.DataFrame({"y": series})
        for lag in range(1, n_lags+1):
            df_lag[f"lag_{lag}"] = df_lag["y"].shift(lag)
        return df_lag.dropna()
    
    # =========================================================
    # 1) Best lag
    # =========================================================
    lag_scores = []
    for n_lags in range(1, 10):
        data = make_lag_df(series, n_lags)
        split = int(len(data)*0.9)
        train, test = data.iloc[:split], data.iloc[split:]
        X_train, y_train = train.drop("y", axis=1), train["y"]
        X_test, y_test = test.drop("y", axis=1), test["y"]
        model = CatBoostRegressor(iterations=461, depth=7, learning_rate=0.0185698166207864,
                                  l2_leaf_reg=9.9687496613796, loss_function="RMSE", verbose=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds)
        lag_scores.append((n_lags, rmse))
    
    BEST_LAG = min(lag_scores, key=lambda x:x[1])[0]
    print(f"✅ Best lag: {BEST_LAG}")
    
    # =========================================================
    # Dataset definitivo
    # =========================================================
    data = make_lag_df(series, BEST_LAG)
    split = int(len(data)*0.9)
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train.drop("y", axis=1), train["y"]
    X_test, y_test = test.drop("y", axis=1), test["y"]
    monotone_constraints = [1]*BEST_LAG
    
    # =========================================================
    # 2) Optuna CatBoost
    # =========================================================
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 10, 100),
            "depth": trial.suggest_int("depth", 2, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "loss_function": "RMSE",
            "verbose": False,
            "monotone_constraints": [1],
            "bootstrap_type": "Bayesian",
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 2),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 30)
        }
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)
        preds = model.predict(X_test)
        return np.sqrt(mean_squared_error(y_test, preds))
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=True)
    
    BEST_PARAMS = study.best_params
    print("✅ Best CatBoost params:", BEST_PARAMS)
    
    # =========================================================
    # 3) CatBoost finale
    # =========================================================
    cat_model = CatBoostRegressor(**BEST_PARAMS, loss_function="RMSE", verbose=False,
                                  monotone_constraints=monotone_constraints)
    cat_model.fit(X_train, y_train)
    
    # =========================================================
    # 4) GARCH sui residui
    # =========================================================
    residuals = y_train - cat_model.predict(X_train)
    garch = arch_model(residuals, vol="Garch", p=1, q=1, mean="Zero", dist=DIST)
    garch_fit = garch.fit(disp="off")
    
    # =========================================================
    # 5) Monte Carlo forecast
    # =========================================================
    H = forecast_horizon_years * 12
    last_series = series.copy()
    sim_paths = np.zeros((N_SIM, H))
    
    garch_fc = garch_fit.forecast(horizon=H)
    sigma = np.sqrt(garch_fc.variance.values[-1])
    DIST_LOWER = garch_fit.model.distribution.name.lower()
    
    for sim in range(N_SIM):
        path_series = last_series.copy()
        if DIST_LOWER=="t":
            z = np.random.standard_t(df=garch_fit.params["nu"], size=H)
        else:
            z = np.random.standard_normal(H)
        for h in range(H):
            lags = path_series.iloc[-BEST_LAG:].values
            X_future = pd.DataFrame([lags], columns=X_train.columns)
            mu = cat_model.predict(X_future)[0]
            eps = sigma[h]*z[h]
            y_next = mu + eps
            sim_paths[sim,h] = y_next
            path_series = pd.concat([path_series, pd.Series([y_next])], ignore_index=True)
    
    # =========================================================
    # Fan chart GARCH
    # =========================================================
    forecast_mean = sim_paths.mean(axis=0)
    lower_95 = np.percentile(sim_paths, 100*alpha/2, axis=0)
    upper_95 = np.percentile(sim_paths, 100*(1-alpha/2), axis=0)
    
    # =========================================================
    # Conformal Prediction
    # =========================================================
    calibration_data = data.iloc[-CALIBRATION_H:]
    X_cal = calibration_data.drop("y", axis=1)
    y_cal = calibration_data["y"]
    y_cal_pred = cat_model.predict(X_cal)
    conformity_scores = np.abs(y_cal - y_cal_pred)
    q_hat = np.quantile(conformity_scores, 1-alpha)
    
    cp_lower = forecast_mean - q_hat
    cp_upper = forecast_mean + q_hat
    
    # =========================================================
    # DataFrame finale
    # =========================================================
    final_forecast = pd.DataFrame({
        "Step": np.arange(1,H+1),
        "Mean_Forecast": forecast_mean,
        "GARCH_Lower_95": lower_95,
        "GARCH_Upper_95": upper_95,
        "CP_Lower_95": cp_lower,
        "CP_Upper_95": cp_upper
    })
    
    # =========================================================
    # Plot storico + test + forecast
    # =========================================================
    dates = df["Time"].iloc[-len(series):]
    y_real = series.values
    train_size = len(X_train)
    test_size = len(X_test)
    dates_train = dates.iloc[:train_size]
    dates_test = dates.iloc[train_size:train_size+test_size]
    y_train_real = y_real[:train_size]
    y_test_real = y_real[train_size:train_size+test_size]
    y_test_pred = cat_model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(dates_train, y_train_real, label="Storico Train", color="black", linewidth=1.5)
    ax.plot(dates_test, y_test_real, label="Storico Test (reale)", color="blue", linewidth=2)
    ax.plot(dates_test, y_test_pred, label="Forecast Test (CatBoost)", color="orange", linestyle="--", linewidth=2)
    ax.axvline(x=dates_test.iloc[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_title("Storico + Forecast Test con Conformal Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    
    return final_forecast, fig



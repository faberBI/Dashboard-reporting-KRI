import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate
import pickle
import matplotlib.pyplot as plt


def make_lag_df(series, n_lags):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df.dropna().reset_index(drop=True)

def plot_copper_forecast(df_model, result_df_annual):
    fig, ax = plt.subplots(figsize=(14,7))

    # Prezzo storico
    ax.plot(df_model.index, df_model['Copper'], label='Historical Price in €', color='blue')

    # Previsioni future
    ax.plot(result_df_annual.index, result_df_annual['Mean_Forecast'], label='Forecast Average', color='orange', linestyle='--')

    # Banda di incertezza
    if 'CP_Lower_95' in result_df_annual.columns and 'upper_95' in result_df_annual.columns:
        ax.fill_between(result_df_annual.index, 
                        result_df_annual['CP_Lower_95'], 
                        result_df_annual['upper_95'], 
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




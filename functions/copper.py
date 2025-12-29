import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

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


def full_copper_forecast(link_df, price_col='Copper', N_SIM=1000, alpha=0.05, DIST="ged", calibration_size=24):

    # ================= Preprocessing =================
    df = pd.read_excel(link_df)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time").reset_index(drop=True)
    series = pd.to_numeric(df[price_col], errors="coerce").dropna()

    BEST_LAG = 1 

    # ================= Dataset definitivo =================
    data = make_lag_df(series, BEST_LAG)
    split_train = int(len(data)*0.7)
    split_cal = split_train + calibration_size

    train = data.iloc[:split_train]
    calibration = data.iloc[split_train:split_cal]
    test = data.iloc[split_cal:]

    X_train, y_train = train.drop("y", axis=1), train["y"]
    X_cal, y_cal = calibration.drop("y", axis=1), calibration["y"]
    X_test, y_test = test.drop("y", axis=1), test["y"]

    monotone_constraints = [1]*BEST_LAG

    # ================= CatBoost finale =================
    BEST_PARAMS = {'iterations': 96, 'depth': 3, 'learning_rate': 0.07501216733365657, 'l2_leaf_reg': 1.2784730546433896, 'max_ctr_complexity': 2, 'min_data_in_leaf': 24}
    cat_model = CatBoostRegressor(**BEST_PARAMS, loss_function="RMSE", verbose=False,
                                  monotone_constraints=monotone_constraints)
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
        if DIST=="t":
            z = np.random.standard_t(df=garch_fit.params["nu"], size=len(X_test))
        else:
            z = np.random.standard_normal(len(X_test))
        sim_paths[sim, :] = cat_model.predict(X_test) + sigma_test * z

    y_test_pred_mean = sim_paths.mean(axis=0)
    y_test_lower = np.percentile(sim_paths, 100*alpha/2, axis=0)
    y_test_upper = np.percentile(sim_paths, 100*(1-alpha/2), axis=0)

    # ================= Conformal Prediction sul test set =================
    # Calcolata usando il set di calibrazione
    sigma_cal = np.sqrt(garch_fit.conditional_volatility[-len(X_cal):])
    sim_paths_cal = np.zeros((N_SIM, len(X_cal)))
    for sim in range(N_SIM):
        if DIST=="t":
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
    dates = df["Time"].iloc[-len(series):]
    y_real = series.values
    train_size = len(X_train)
    cal_size = len(X_cal)
    test_size = len(X_test)

    dates_train = dates.iloc[:train_size]
    dates_cal = dates.iloc[train_size:train_size+cal_size]
    dates_test = dates.iloc[train_size+cal_size:train_size+cal_size+test_size]

    y_train_real = y_real[:train_size]
    y_cal_real = y_real[train_size:train_size+cal_size]
    y_test_real = y_real[train_size+cal_size:train_size+cal_size+test_size]

    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(dates_train, y_train_real, label="Train", color="black", linewidth=1.5)
    ax.plot(dates_cal, y_cal_real, label="Calibration", color="gray", linewidth=1.5)
    ax.plot(dates_test, y_test_real, label="Test (reale)", color="blue", linewidth=2)
    ax.plot(dates_test, y_test_pred_mean, label="Forecast Test (CatBoost + GARCH MC)", color="orange", linestyle="--", linewidth=2)
    ax.fill_between(dates_test, cp_lower_test, cp_upper_test, color='orange', alpha=0.2, label='CP 95%')
    ax.axvline(x=dates_test.iloc[0], color="gray", linestyle=":", linewidth=1)
    ax.set_title("Forecast Test Hybrid Model con Conformal Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

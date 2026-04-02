import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import optuna
import yfinance as yf
import streamlit as st


def download_ecb_series(series_dict, start="2010-01-01"):
    df_final = pd.DataFrame()
    for name, key in series_dict.items():
        try:
            df = ecbdata.get_series(key, start=start)
            df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
            df = df.set_index('TIME_PERIOD')
            df = df.rename(columns={'OBS_VALUE': name})
            df_final = df_final.join(df[[name]], how='outer')
            
        except Exception as e:
            print(f"Errore scaricando {name}: {e}")
    return df_final
    
def download_yahoo_series(symbols_dict, start="2010-01-01"):
    data = yf.download(list(symbols_dict.values()), start=start)
    close = data["Close"]
    close = close.rename(columns={v: k for k, v in symbols_dict.items()})
    print("Dati Yahoo Finance scaricati")
    return close
    
def plot_predictions_streamlit(df_dropped, y_pred_train, y_pred_val, y_pred_test, train_end, val_end):
    plt.figure(figsize=(15,6))
    # Serie originale
    plt.plot(df_dropped['euribor_3m'], label="Originale", color='black')
    # Train
    train_idx = df_dropped.index[:len(y_pred_train)]
    plt.plot(train_idx, y_pred_train, label="Train Pred", color='blue')
    # Validation
    val_start = len(y_pred_train)
    val_end_idx = val_start + len(y_pred_val)
    val_idx = df_dropped.index[val_start:val_end_idx]
    plt.plot(val_idx, y_pred_val, label="Val Pred", color='orange')
    # Test
    test_idx = df_dropped.index[-len(y_pred_test):]
    plt.plot(test_idx, y_pred_test, label="Test Pred", color='green')
    plt.title("Serie originale vs Predizione completa")
    plt.legend()
    plt.grid(True)
    # Render in Streamlit
    st.pyplot(plt.gcf())
    plt.close()

    
# ============================================================
# SIMULAZIONE UNICA EURIBOR MONTE CARLO + CONFORMAL
# ============================================================
def simulate_euribor(series, df_dropped, n_sims=1000, alpha=0.05, horizon_days=3*360, plan_euribor_df=None):
    np.random.seed(234)
    
    def simulate_ou(X0, theta, mu, sigma, n_steps, dt=1.0):
        X = np.zeros(n_steps)
        X[0] = X0
        for t in range(1, n_steps):
            dW = np.random.randn() * np.sqrt(dt)
            X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
        return X

    def objective(trial):
        theta = trial.suggest_loguniform("theta", 1e-3, 1.0)
        mu = trial.suggest_uniform("mu", series.min(), series.max())
        sigma = trial.suggest_loguniform("sigma", 1e-4, 1.0)
        X_prev, X_next = series[:-1], series[1:]
        dt = 1.0
        var = sigma**2 * dt
        mean = X_prev + theta*(mu - X_prev)*dt
        log_lik = -0.5 * np.sum(((X_next - mean)**2)/var + np.log(2*np.pi*var))
        return log_lik

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=234))
    study.optimize(objective, n_trials=1000)

    theta_opt = study.best_params["theta"]
    mu_opt = study.best_params["mu"]
    sigma_opt = study.best_params["sigma"]

    n_period = horizon_days
    X0 = series[-1]
    simulations = np.zeros((n_sims, n_period))
    for i in range(n_sims):
        simulations[i, :] = simulate_ou(X0, theta_opt, mu_opt, sigma_opt, n_period)

    lower_emp = np.percentile(simulations, 100*alpha/2, axis=0)
    upper_emp = np.percentile(simulations, 100*(1-alpha/2), axis=0)
    median = np.median(simulations, axis=0)
    mean = np.mean(simulations, axis=0)

    # Conformal adjustment
    calibration_y = series[-252:]
    np.random.seed(234)
    samples_cal = np.random.choice(simulations.flatten(), size=(len(calibration_y), n_period))
    lower_cal = np.percentile(samples_cal, 2.5, axis=1)
    upper_cal = np.percentile(samples_cal, 97.5, axis=1)
    nonconformity = np.maximum(lower_cal - calibration_y, calibration_y - upper_cal)
    q_hat = np.quantile(np.append(nonconformity, np.inf), 0.95)

    lower_adj = lower_emp - q_hat
    upper_adj = upper_emp + q_hat

    while np.any(upper_adj <= lower_adj):
        mask = upper_adj <= lower_adj
        upper_adj[mask] = lower_adj[mask] + 0.05

    
    idx = pd.date_range(start=df_dropped.index[-1] + pd.Timedelta(days=1), periods=n_period, freq="D")

    if plan_euribor_df is not None:
        plan_euribor_df['Tasso'] = plan_euribor_df['Tasso'].astype(float)
    # Serie giornaliera basata sul tasso fisso annuale
    plan_rate_series = pd.Series(
        index=idx,
        data=[plan_euribor_df.loc[plan_euribor_df['Anno'] == d.year, 'Tasso'].values[0] for d in idx]
    )

    forecast_df = pd.DataFrame({
        "lower_emp": lower_emp,
        "upper_emp": upper_emp,
        "median": median,
        'mean': mean,
        "lower_adj": lower_adj,
        "upper_adj": upper_adj
    }, index=idx)
    
    forecast_quarterly = forecast_df.resample("QE").mean()
    # Media ponderata solo sulla colonna 'median'
    forecast_quarterly['median'] = (forecast_quarterly['median'] * 0.5+ plan_rate_series.resample("QE").mean() * 0.5)
    plan_q = plan_rate_series.resample("QE").mean()
    forecast_quarterly['lower_adj'] = (forecast_quarterly['lower_adj']*0.85)

    return forecast_df, forecast_quarterly
    
# ============================================================
# GRAFICO STREAMLIT
# ============================================================

def plot_full_forecast(y, df_forecast):
    plt.figure(figsize=(15,6))
    plt.plot(y, label="Originale", color='black')
    idx_forecast = df_forecast.index
    plt.plot(idx_forecast, df_forecast['median'], label='Mean Forecast', color='green', linestyle='--')
    plt.fill_between(idx_forecast, df_forecast['lower_emp'], df_forecast['upper_emp'],
                     color='red', alpha=0.2, label='Adjusted Interval (Conformal)')
    plt.title("Serie storica + Predizioni + Forecast Monte Carlo")
    plt.xlabel("Date")
    plt.ylabel("EURIBOR 3M")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()
    
def get_spread_for_date(date, spread_df):
    row = spread_df[(spread_df["From"] <= date) & (spread_df["To"] > date)]
    if not row.empty:
        return float(row["Spread"].iloc[0])
    else:
        return float(spread_df["Spread"].iloc[-1]) 

def get_plan_euribor_for_date(date, plan_df):
    year = date.year
    # Ordiniamo per sicurezza
    plan_df_sorted = plan_df.sort_values("Anno")
    # Se l'anno è presente → uso diretto
    if year in plan_df_sorted["Anno"].values:
        return plan_df_sorted.loc[plan_df_sorted["Anno"] == year, "Tasso"].values[0]
    # Se l'anno è successivo all'ultimo disponibile → ultimo valore noto
    if year > plan_df_sorted["Anno"].max():
        return plan_df_sorted.iloc[-1]["Tasso"]
    # Se l'anno è precedente al primo disponibile → primo valore noto
    if year < plan_df_sorted["Anno"].min():
        return plan_df_sorted.iloc[0]["Tasso"]
    # Fallback 
    return np.nan

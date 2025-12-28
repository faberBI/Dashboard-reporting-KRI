import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate


import numpy as np
import pandas as pd
import pickle

def get_copper_prediction(df_model, end_date, n_sims=10_000, alpha=0.05, egarch_pickle_path="egarch_fit_def.pkl"):
    """
    Simula i prezzi futuri del copper mensili usando residui EGARCH salvati,
    costruisce intervalli di previsione con conformal adjustment e aggrega annualmente.
    
    Args:
        df_model (pd.DataFrame): storico dei prezzi, con colonna 'copper_price' e indice Datetime.
        end_date (str o datetime): data di fine simulazione.
        n_sims (int): numero di simulazioni Monte Carlo.
        alpha (float): livello di significatività per gli intervalli.
        egarch_pickle_path (str): path del file pickle contenente il modello EGARCH fit.

    Returns:
        result_df (pd.DataFrame): risultati mensili con mediana, media e intervalli aggiustati.
        result_df_annual (pd.DataFrame): aggregazione annuale dei risultati.
    """
    
    # --- Carica modello EGARCH fit ---
    with open(egarch_pickle_path, "rb") as f:
        egarch_fit = pickle.load(f)
    
    egarch_model = egarch_fit.model  # oggetto arch_model associato
    
    last_date = df_model.index[-1]
    end_date = pd.to_datetime(end_date)
    
    # Indice mensile
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                 end=end_date,
                                 freq='M')
    T = len(future_dates)
    
    last_price = df_model['copper_price'].iloc[-1]
    
    # Parametri EGARCH
    params = egarch_fit.params
    sigma_hist = np.median(np.abs(egarch_fit.conditional_volatility))
    
    # Simulazioni Monte Carlo
    sim_prices = np.zeros((n_sims, T))
    
    for i in range(n_sims):
        sim = egarch_model.simulate(params, nobs=T)
        sim_resid = sim["data"]
        sim_resid = np.clip(sim_resid, -3*sigma_hist, 3*sigma_hist)
        
        # Log-returns simulati
        sim_logret = 0 + sim_resid
        sim_prices[i] = last_price * np.exp(np.cumsum(sim_logret))
    
    # DataFrame dei risultati
    result_df = pd.DataFrame({
        "median": np.median(sim_prices, axis=0),
        "average": np.mean(sim_prices, axis=0),
        "lower": np.percentile(sim_prices, 5, axis=0),
        "upper": np.percentile(sim_prices, 95, axis=0)
    }, index=future_dates)
    
    # --- Conformal adjustment con ultimi 12 mesi ---
    calibration_y = df_model['copper_price'].iloc[-12:].values
    np.random.seed(234)
    samples_cal = np.random.choice(sim_prices.flatten(), size=(len(calibration_y), T))
    lower_cal = np.percentile(samples_cal, 100*alpha/2, axis=1)
    upper_cal = np.percentile(samples_cal, 100*(1-alpha/2), axis=1)
    
    nonconformity = np.maximum(lower_cal - calibration_y[:, None],
                               calibration_y[:, None] - upper_cal)
    q_hat = np.quantile(nonconformity, 1-alpha)
    
    # Intervalli aggiustati
    lower_adj = np.maximum(result_df['lower'] - q_hat, 0)  # evita negativi
    upper_adj = result_df['upper'] + q_hat
    
    # Controlli: se mediana fuori intervallo, resetta agli originali
    lower_adj[result_df['median'] < lower_adj] = result_df['lower'][result_df['median'] < lower_adj]
    upper_adj[result_df['median'] > upper_adj] = result_df['upper'][result_df['median'] > upper_adj]
    
    result_df['lower_adj'] = lower_adj
    result_df['upper_adj'] = upper_adj
    result_df.drop(['lower', 'upper'], axis=1, inplace=True)
    
    # Aggregazione annuale
    result_df_annual = result_df.resample('Y').mean()
    
    return result_df, result_df_annual


import matplotlib.pyplot as plt

def plot_copper_forecast(df_model, result_df_annual):
    fig, ax = plt.subplots(figsize=(14,7))

    # Prezzo storico
    ax.plot(df_model.index, df_model['copper_price'], label='Historical Price', color='blue')

    # Previsioni future
    ax.plot(result_df_annual.index, result_df_annual['average'], label='Forecast Average', color='orange', linestyle='--')

    # Banda di incertezza
    if 'lower_adj' in result_df_annual.columns and 'upper_adj' in result_df_annual.columns:
        ax.fill_between(result_df_annual.index, 
                        result_df_annual['lower_adj'], 
                        result_df_annual['upper_adj'], 
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



def monte_carlo_forecast_cp(series, cat_model, garch_fit, BEST_LAG,
                             CALIBRATION_H=24, N_SIM=1000, alpha=0.05,
                             end_date=None, random_seed=42):
    import numpy as np
    import pandas as pd

    np.random.seed(random_seed)

    # Horizon in mesi
    today = pd.Timestamp.now().normalize()
    if end_date is None:
        end_date = today + pd.DateOffset(years=5)
    else:
        end_date = pd.to_datetime(end_date)

    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                 end=end_date,
                                 freq='M')
    H = len(future_dates)
    # Monte Carlo simulation
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

    # Fan chart GARCH
    forecast_mean = sim_paths.mean(axis=0)
    lower_95 = np.percentile(sim_paths, 100*alpha/2, axis=0)
    upper_95 = np.percentile(sim_paths, 100*(1-alpha/2), axis=0)

    # Conformal Prediction
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
        "Mean_Forecast": forecast_mean,
        "GARCH_Lower_95": lower_95,
        "GARCH_Upper_95": upper_95,
        "CP_Lower_95": cp_lower,
        "CP_Upper_95": cp_upper
    }, index = future_dates)

    return final_forecast


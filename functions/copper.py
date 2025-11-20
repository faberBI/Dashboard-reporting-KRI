import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate


def simulate_cb_egarch_outsample(
        copula_model, model_cb, egarch_model, egarch_fit,
        last_date, end_date, S0, n_sims=500,
        clip_low=-1, clip_high=1,
        monthly_cap=0.0025, monthly_floor=-0.002):

    last_date = pd.to_datetime(last_date)
    end_date  = pd.to_datetime(end_date)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 end=end_date, freq='B')
    T = len(future_dates)

    # ---- Copula per future X ----
    X_fut_flat = copula_model.sample(T * n_sims)
    pred_flat = model_cb.predict(X_fut_flat)
    pred_cb = pred_flat.reshape(n_sims, T)
    pred_cb = pred_cb - np.median(pred_cb, axis=0)
    # ---- EGARCH params ----
    params = egarch_fit.params


    cap_daily = np.log(1 + 0.25) / 21
    floor_daily = np.log(1 - 0.18) / 21

    # sigma storica per clipping
    sigma_hist = np.median(egarch_fit.conditional_volatility) / 100  # prudente

    sim_prices = np.zeros((n_sims, T))

    for i in range(n_sims):

        # ---- GARCH simulate ----
        sim = egarch_model.simulate(params, nobs=T)
        sim_resid = sim["data"] / 100

        # clipping controllato sui residui
        sim_resid = np.clip(sim_resid,
                            clip_low * sigma_hist,
                            clip_high * sigma_hist)
         
        
        # log returns simulati
        sim_logret = pred_cb[i] + sim_resid
        sim_logret = np.clip(sim_logret, floor_daily, cap_daily)
        sim_prices[i] = S0 * np.exp(np.cumsum(sim_logret))

        
    # ---- Raggruppamento ----
    result = pd.DataFrame({
        "median":  np.median(sim_prices, axis=0),
        "average": np.mean(sim_prices, axis=0),
        "lower":   np.percentile(sim_prices, 5, axis=0),
        "upper":   np.percentile(sim_prices, 95, axis=0),
    }, index=future_dates)

    return result, sim_prices


import matplotlib.pyplot as plt

def get_forecast_plot(df, result_df):
    history = df['PX_LAST']
    future_dates = result_df.index

    fig, ax = plt.subplots(figsize=(16,6))

    # Storico reale
    ax.plot(history.index, history.values,
            color="black", linewidth=2, label="Storico reale")

    # Mediana e media simulata
    ax.plot(future_dates, result_df['average'],
            color="purple", linestyle="--", linewidth=1.8,
            label="Media simulata")

    ax.plot(future_dates, result_df['median'],
            color="blue", linestyle="--", linewidth=1.8,
            label="Median simulata")

    # Bande 5–95%
    ax.fill_between(future_dates,
                    result_df['lower'],
                    result_df['upper'],
                    color="blue", alpha=0.20,
                    label="Intervallo 5–95%")

    # Setup grafico
    ax.set_title("Storico + Simulazioni Out-of-Sample CatBoost + EGARCH")
    ax.set_xlabel("Data")
    ax.set_ylabel("Prezzo")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig

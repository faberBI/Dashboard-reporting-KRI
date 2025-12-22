import numpy as np
import pandas as pd
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate


def get_copper_prediction(df_model, end_date, n_sims=10_000, alpha = 0.05):
  
  last_date = df_model.index[-1]
  end_date = pd.to_datetime(end_date)

  # Genera indice mensile a partire dal mese successivo a last_date
  future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                               end=end_date,
                               freq='M')
  T = len(future_dates)     
  last_price = df_model['copper_price'][-1] # prezzo corrente


  # Parametri EGARCH
  params = egarch_fit.params
  sigma_hist = np.median(np.abs(egarch_fit.conditional_volatility))  # volatilità storica

  sim_prices = np.zeros((n_sims, T))

  for i in range(n_sims):
      # Simula residui EGARCH
      sim = egarch_model.simulate(params, nobs=T)
      sim_resid = sim["data"]

      # Clipping prudente sui residui
      sim_resid = np.clip(sim_resid, -3*sigma_hist, 3*sigma_hist)

      sim_logret = 0 + sim_resid

      # Ricostruzione prezzi cumulativi
      sim_prices[i] = last_price * np.exp(np.cumsum(sim_logret))

  # Costruisci DataFrame dei risultati

  result_df = pd.DataFrame({
      "median": np.median(sim_prices, axis=0),
      "average": np.mean(sim_prices, axis=0),
      "lower": np.percentile(sim_prices, 5, axis=0),
      "upper": np.percentile(sim_prices, 95, axis=0)
  }, index=future_dates)
  
  # Ultimi 12 mesi storici dei prezzi per calibrazione
  calibration_y = df_model['copper_price'].iloc[-12:].values

  # Campionamento casuale dalle simulazioni per conformal prediction
  np.random.seed(234)
  samples_cal = np.random.choice(sim_prices.flatten(), size=(len(calibration_y), T))

  # Quantili non aggiustati
  lower_cal = np.percentile(samples_cal, 100*alpha/2, axis=1)
  upper_cal = np.percentile(samples_cal, 100*(1-alpha/2), axis=1)

  # Nonconformity
  nonconformity = np.maximum(lower_cal - calibration_y[:, None], calibration_y[:, None] - upper_cal)
  q_hat = np.quantile(nonconformity, 1-alpha)

  # Intervalli aggiustati
  lower_adj = np.maximum(result_df['lower'] - q_hat, 0)  # evita negativi
  upper_adj = result_df['upper'] + q_hat

  # Controllo: se la mediana è più piccola del lower_adj, resetta lower_adj al lower originale
  lower_adj[result_df['median'] < lower_adj] = result_df['lower'][result_df['median'] < lower_adj]
  upper_adj[result_df['median'] > upper_adj] = result_df['upper'][result_df['median'] > upper_adj]

  # Salva nel DataFrame finale
  result_df['lower_adj'] = lower_adj
  result_df['upper_adj'] = upper_adj
  result_df.drop(['lower', 'upper'], axis = 1, inplace = True)
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


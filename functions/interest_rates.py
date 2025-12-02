import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import optuna
import yfinance as yf

def download_ecb_series(series_dict, start="2010-01"):
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
    
    yahoo_symbols = {
        "sp500": "^GSPC",
        "eurusd": "EURUSD=X",
        "vix": "^VIX",
        "us10y": "^TNX",
        "oil": "CL=F",
        "gold": "GC=F",
    }

    def download_yahoo_series(symbols_dict, start="2010-01-01"):
        data = yf.download(list(symbols_dict.values()), start=start)
        close = data["Close"]
        close = close.rename(columns={v: k for k, v in symbols_dict.items()})
        print("Dati Yahoo Finance scaricati")
        return close



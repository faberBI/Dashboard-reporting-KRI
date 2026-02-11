import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import pmdarima as pm
import io
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache_resource
def fit_sarimax_model(series):
    """
    Fit SARIMAX usando parametri fissi derivati da auto_arima.
    """
    model = SARIMAX(
        series,
        order=(4,0,1),
        seasonal_order=(2,1,0,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

def forecast_monthly_prices(series, n_years=1):
    """
    Serie storica mensile -> forecast per n_years * 12 mesi
    """
    model_fit = fit_sarimax_model(series)
    forecast_periods = n_years * 12
    forecast = model_fit.forecast(steps=forecast_periods)
    return forecast


    
def get_return(path, year=2015):
    hist_pun = pd.read_excel(path)
    hist_pun["Date"] = pd.to_datetime(hist_pun["Date"])
    hist_pun["Year"] = hist_pun["Date"].dt.year
    hist_pun["Month"] = hist_pun["Date"].dt.month
    hist_pun["log_return"] = np.log(hist_pun["GMEPIT24 Index"] / hist_pun["GMEPIT24 Index"].shift(1))
    last_5y = hist_pun[hist_pun['Year'] > year]
    
    monthly_std = last_5y.groupby("Month")["log_return"].std().reset_index(name="std_log_return")
    monthly_std['std_log_return_montly'] = monthly_std['std_log_return'] * np.sqrt(21)
    
    monthly_price = last_5y.groupby("Month")["GMEPIT24 Index"].mean().reset_index(name="avg_price")
    
    return last_5y, monthly_std, monthly_price

def apply_cholesky(last_5y):
    monthly_lr = last_5y.groupby(last_5y["Date"].dt.to_period("M"))["log_return"].sum().to_frame("log_return")
    monthly_lr["Month"] = monthly_lr.index.month
    monthly_lr["Year"] = monthly_lr.index.year
    monthly_lr_matrix = monthly_lr.pivot(index="Year", columns="Month", values="log_return")
    corr_matrix = monthly_lr_matrix.corr()
    rho_hat = np.mean([corr_matrix.iloc[i, i+1] for i in range(11)])
    corr = np.fromfunction(lambda i,j: rho_hat ** np.abs(i-j), (12,12))
    L = np.linalg.cholesky(corr)
    return L

def get_garch(last_5y, rolling_window=12):
    monthly_means = last_5y.groupby(['Year', 'Month'])['GMEPIT24 Index'].mean().reset_index()
    monthly_means['Date'] = pd.to_datetime(monthly_means[['Year','Month']].assign(DAY=1))
    monthly_means = monthly_means.sort_values('Date')
    prices = monthly_means.set_index('Date')['GMEPIT24 Index']
    
    log_returns = np.log(prices).diff().dropna()
    rolling_std = log_returns.rolling(window=rolling_window).std()
    
    model = arch_model(log_returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
    fit = model.fit(disp=False)
    sigma_t = fit.conditional_volatility
    
    sigma_df = pd.DataFrame({'Date': sigma_t.index, 'sigma': sigma_t.values})
    sigma_df['Month'] = sigma_df['Date'].dt.month
    monthly_sigma = sigma_df.groupby('Month')['sigma'].mean().values
    
    return monthly_sigma, rolling_std

def simulate_prices(PUN_monthly_forecast, PUN_monthly, monthly_sigma, monthly_std, L, n_sims=100_000, seed=42):
    np.random.seed(seed)
    vol_h = monthly_sigma * np.array(PUN_monthly)
    vol_m = np.array(monthly_std['std_log_return_montly'].values) * np.array(PUN_monthly)
    vol_f = (vol_h + vol_m)/2
    
    n_total_months = len(PUN_monthly_forecast)
    n_years = n_total_months // 12
    all_years = []
    
    for i in range(n_years):
        P_mean = np.array(PUN_monthly_forecast[i*12:(i+1)*12])
        Z = np.random.normal(size=(n_sims,12))
        shocks = (Z @ L.T) * vol_f[np.newaxis, :]
        P_paths = P_mean[np.newaxis, :] + shocks
        all_years.append(P_paths)
    
    PUN_paths = np.hstack(all_years)
    return PUN_paths, shocks

def compute_VaR(df_var, VaR_95_monthly):
    df_var['Price_95perc'] = VaR_95_monthly
    df_var['Var_monthly_95_w_solar'] = (df_var['Price_95perc'] - df_var['Prezzo Budget']) * np.maximum(df_var['Fabbisogno'] - (df_var['PPA Erg'] + df_var['Forward'] + df_var['Solar']),0) *1000
    df_var['Var_monthly_95_w/o_solar'] = (df_var['Price_95perc'] - df_var['Prezzo Budget']) * np.maximum(df_var['Fabbisogno'] - (df_var['PPA Erg'] + df_var['Forward']),0) *1000
    return df_var

def plot_pun_forecast_vs_volatility(last_5y, PUN_monthly_forecast, monthly_sigma, rolling_std):
    """
    Grafico combinato:
    - PUN storico vs PUN forecastato
    - Volatilità stimata GARCH e rolling std
    """
    # Serie mensile media storica
    monthly_means = last_5y.groupby(['Year', 'Month'])['GMEPIT24 Index'].mean().reset_index()
    monthly_means['Date'] = pd.to_datetime(monthly_means[['Year','Month']].assign(DAY=1))
    monthly_means = monthly_means.sort_values('Date')
    
    # Date forecast
    n_forecast_months = len(PUN_monthly_forecast)
    last_date = monthly_means['Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=n_forecast_months, freq='MS')
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(14,6))
    
    # PUN storico
    ax1.plot(monthly_means['Date'], monthly_means['GMEPIT24 Index'], label='PUN storico', color='blue', linewidth=2)
    
    # PUN forecast
    ax1.plot(forecast_dates, PUN_monthly_forecast, label='PUN forecast', color='orange', linestyle='--', linewidth=2)
    
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Prezzo PUN (€)")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left')
    
    # Volatilità sul secondo asse
    ax2 = ax1.twinx()
    
    # Rolling std (mensile)
    rolling_std_monthly = rolling_std.resample('M').mean()
    ax2.plot(rolling_std_monthly.index, rolling_std_monthly.values, label='Rolling std log-return', color='green', linestyle=':')
    
    # Volatilità GARCH (mensile)
    months = np.arange(1,13)
    ax2.bar(months - 0.2, monthly_sigma, width=0.4, alpha=0.4, label='Volatilità condizionale GARCH', color='red')
    
    ax2.set_ylabel("Volatilità")
    ax2.legend(loc='upper right')
    
    plt.title("Prezzo PUN storico vs forecast e volatilità stimata")
    plt.tight_layout()
    st.pyplot(fig)

def plot_monthly_VaR(VaR_95_monthly, start_year=2026):
    """
    Grafico VaR 95% mensile su più anni.

    VaR_95_monthly : array o lista di valori VaR per ogni mese (lunghezza = n_years*12)
    start_year     : anno di partenza del forecast (es. 2026)
    """
    n_months = len(VaR_95_monthly)
    n_years = n_months // 12

    # Creiamo le date mese/anno
    dates = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq='MS')

    # Nomi mesi in italiano
    months_names = [
        "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
        "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
    ]

    labels = [f"{months_names[d.month-1]} {d.year}" for d in dates]

    # Creazione figura
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(labels, VaR_95_monthly, marker='o', color='crimson', linewidth=2)

    ax.set_ylabel("VaR 95% (€/MWh)")
    ax.set_title(f"VaR 95% Mensile - PUN ({n_years} year)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)



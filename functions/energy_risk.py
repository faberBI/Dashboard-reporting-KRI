import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import pmdarima as pm
import io
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
import altair as alt
import plotly.graph_objects as go

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

def adjust_first_forecast_with_partial_month(forecast, series):
    
    last_date = series["Date"].max()
    current_period = last_date.to_period("M")

    partial_data = series[
        series["Date"].dt.to_period("M") == current_period
    ]["GMEPIT24 Index"]

    partial_mean = partial_data.mean()

    weight = last_date.day / last_date.days_in_month

    forecast.iloc[0] = (
        (1 - weight) * forecast.iloc[0]
        + weight * partial_mean
    )

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
    monthly_price_year = (
        last_5y
        .set_index("Date")["GMEPIT24 Index"]
        .resample("MS")          # Month Start frequency
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"GMEPIT24 Index": "avg_price"})
    )
    return last_5y, monthly_std, monthly_price, monthly_price_year

def apply_cholesky(last_5y):
    monthly_lr = last_5y.groupby(last_5y["Date"].dt.to_period("M"))["log_return"].sum().to_frame("log_return")
    monthly_lr["Month"] = monthly_lr.index.month
    monthly_lr["Year"] = monthly_lr.index.year
    monthly_lr_matrix = monthly_lr.pivot(index="Year", columns="Month", values="log_return")
    corr_matrix = monthly_lr_matrix.corr()
    rho_hat = np.mean([corr_matrix.iloc[i, i+1] for i in range(11)])
    corr = np.fromfunction(lambda i,j: rho_hat ** np.abs(i-j), (12,12))
    L = np.linalg.cholesky(corr)
    return L, rho_hat

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
    
    return monthly_sigma, rolling_std, sigma_t

def simulate_prices(PUN_monthly_forecast, PUN_monthly, monthly_sigma, monthly_std, L, n_sims=100_000, seed=42, min_price=5.0):
    np.random.seed(seed)
    vol_h = monthly_sigma * np.array(PUN_monthly)
    vol_m = np.array(monthly_std['std_log_return_montly'].values) * np.array(PUN_monthly)
    vol_f = vol_h*0.4+ vol_m*0.6
    
    n_total_months = len(PUN_monthly_forecast)
    n_years = n_total_months // 12
    all_years = []
    
    for i in range(n_years):
        P_mean = np.array(PUN_monthly_forecast[i*12:(i+1)*12])
        Z = np.random.normal(size=(n_sims,12))
        shocks = (Z @ L.T) * vol_f[np.newaxis, :]
        P_paths = P_mean[np.newaxis, :] + shocks
        P_paths = np.clip(P_paths, min_price, None)
        all_years.append(P_paths)
    
    PUN_paths = np.hstack(all_years)
    return PUN_paths, shocks

def compute_VaR(df_var, VaR_95_monthly, cut_month, hid_month):
    
    VaR_95_2026 = VaR_95_monthly[-cut_month:] 
    df_var['scoperto_w_solar'] = np.maximum(df_var['Fabbisogno'] - (df_var['PPA Erg'] + df_var['Forward'] +	df_var['Solar']), 0)
    df_var['scoperto_w/o_solar'] = np.maximum(df_var['Fabbisogno'] - (df_var['PPA Erg'] + df_var['Forward']), 0)
    df_var['Price_95perc'] = np.concatenate([ [np.nan]*hid_month, VaR_95_2026 ])
    df_var['Var_monthly_95_w_solar'] = (df_var['Price_95perc'] - df_var['Prezzo Budget']) * np.maximum(df_var['Fabbisogno'] - (df_var['PPA Erg'] + df_var['Forward'] + df_var['Solar']),0) *1000
    df_var['Var_monthly_95_w/o_solar'] = (df_var['Price_95perc'] - df_var['Prezzo Budget']) * np.maximum(df_var['Fabbisogno'] - (df_var['PPA Erg'] + df_var['Forward']),0) *1000
    
    return df_var

def plot_volatility(rolling_std, sigma_t):
    """
    Grafico volatilità:
    - Rolling std mensile dei log-return
    - Volatilità condizionale GARCH (mensile)
    """

    # Crea figura e asse
    fig, ax = plt.subplots(figsize=(14, 6))

    # ===== PLOT LINEE =====
    rolling_std.plot(
        ax=ax, 
        label='Rolling std log-return', 
        linestyle=':', 
        linewidth=2
    )

    sigma_t.plot(
        ax=ax, 
        label='Volatilità condizionale GARCH', 
        linestyle='--', 
        linewidth=2
    )

    # ===== CUSTOMIZZAZIONE =====
    ax.set_title("Volatilità stimata")
    ax.set_ylabel("Volatilità")
    ax.set_xlabel("Data")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    # ===== STREAMLIT =====
    st.pyplot(fig, use_container_width=True)

def plot_monthly_VaR(VaR_95_monthly, cut_month, start_year=2026):
    """
    Grafico VaR 95% mensile su più anni.

    VaR_95_monthly : array o lista di valori VaR per ogni mese (lunghezza = n_years*12)
    cut_month      : numero di mesi da tagliare dall'inizio (forecast parte dopo questi mesi)
    start_year     : anno di partenza del forecast (es. 2026)
    """

    # Prendo solo i mesi che servono
    VaR_95_2026 = VaR_95_monthly[-cut_month:] 
    n_months = len(VaR_95_2026)
    n_years = n_months // 12

    # Genero le date corrispondenti agli stessi mesi
    # Inizio da start_year + cut_month mesi
    dates = pd.date_range(start=f"{start_year}-01-01", periods=len(VaR_95_monthly), freq='MS')
    dates = dates[-cut_month:]

    # Nomi mesi in italiano
    months_names = [
        "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
        "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
    ]
    labels = [f"{months_names[d.month-1]} {d.year}" for d in dates]

    # Creazione figura
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(labels, VaR_95_2026, marker='o', color='crimson', linewidth=2)

    ax.set_ylabel("VaR 95% (€/MWh)")
    ax.set_title(f"VaR 95% Mensile - PUN ({n_years} year)")
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


def plot_energy_stack_with_var(df):
    months = df['Month']

    # Volumi
    fabbisogno = df['Fabbisogno']
    ppa = df['PPA Erg']
    forward = df['Forward']
    solar = df['Solar']

    # Prezzo
    price_var = df['Var_monthly_95_w_solar'] / 1000

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ===== ASSE SINISTRO: ENERGIA =====
    ax1.stackplot(
        months,
        ppa,
        forward,
        solar,
        labels=['PPA Erg', 'Forward', 'Solar'],
        colors=['#001f6b', '#5b9bff', '#66d17a'],
        alpha=0.95
    )

    ax1.plot(
        months,
        fabbisogno,
        color='black',
        linestyle='--',
        linewidth=2.5,
        label='Fabbisogno'
    )

    ax1.set_ylabel("Energia (MWh)")
    ax1.set_xlabel("Mese")
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ===== ASSE DESTRO: PREZZO =====
    ax2 = ax1.twinx()
    ax2.plot(
        months,
        price_var,
        color='crimson',
        linewidth=3,
        label='VaR €'
    )

    ax2.set_ylabel("VaR € (k€)", color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, None)

    # ===== LEGENDA =====
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("Copertura Energetica e VaR (95%) €", fontsize=14)
    plt.tight_layout()

    return fig


def plot_var_bars(dati_fibercop):
    # Creazione colonna "Anno-Mese" per l'asse X
    dati_fibercop['Anno-Mese'] = dati_fibercop['Anno'].astype(str) + "-" + dati_fibercop['Month']
    
    # Posizione delle barre
    x = np.arange(len(dati_fibercop))
    width = 0.4  # larghezza barre
    
    # Figura
    fig, ax = plt.subplots(figsize=(14,6))
    
    # Barre Var con Solar
    bars1 = ax.bar(
        x - width/2,
        dati_fibercop['Var_monthly_95_w_solar'],
        width=width,
        label='VaR with Solar',
        color='green',
        alpha=0.7
    )
    
    # Barre Var senza Solar
    bars2 = ax.bar(
        x + width/2,
        dati_fibercop['Var_monthly_95_w/o_solar'],
        width=width,
        label='VaR w/o Solar',
        color='red',
        alpha=0.7
    )
    
    # Label e titolo
    ax.set_xticks(x)
    ax.set_xticklabels(dati_fibercop['Anno-Mese'], rotation=45, ha='right')
    ax.set_ylabel("Value@Risk (€)")
    ax.set_xlabel("Anno-Mese")
    ax.set_title("Monthly VaR w & w/o Solar")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    
    # Aggiunge i valori sopra le barre (formattati in €)
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'€{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'€{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Se Streamlit
    st.pyplot(fig, use_container_width=True)


def compute_CVaR(hedge_vector, df, PUN_paths, VaR_level=95):
    hedge_vector = np.asarray(hedge_vector).flatten()
    exposed = df["scoperto_w_solar"].values - hedge_vector
    exposed = np.maximum(exposed, 0)
    losses = np.sum(
        exposed * np.maximum(
            PUN_paths - df["Prezzo Budget"].values,
            0
        ),
        axis=1)
    VaR = np.percentile(losses, VaR_level)
    CVaR = losses[losses >= VaR].mean()
    return CVaR


def plot_monthly_coverage_stack(df, month_col="Month"):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if month_col not in df.columns:
        st.error(f"Colonna '{month_col}' non trovata! Colonne disponibili: {df.columns.tolist()}")
        return

    # =========================
    # ORDINAMENTO ASSE X
    # =========================
    mesi_italiani = ["gen", "feb", "mar", "apr", "mag", "giu",
                     "lug", "ago", "set", "ott", "nov", "dic"]

    if month_col == "Month":
        # caso mono-anno classico
        df[month_col] = pd.Categorical(
            df[month_col],
            categories=mesi_italiani,
            ordered=True
        )
    else:
        # caso multi-anno (Anno-Mese)
        df = df.sort_values(month_col)

    # =========================
    # COLONNE DI COVERAGE
    # =========================
    coverage_cols = ["Copertura", "hedge_addizionale_MWh", "scoperto_finale"]
    available_cols = [c for c in coverage_cols if c in df.columns]

    if not available_cols:
        st.error("Nessuna colonna di coverage disponibile nel DataFrame.")
        return

    selected_cols = st.multiselect(
        "Seleziona le componenti da visualizzare nel grafico:",
        options=available_cols,
        default=available_cols
    )

    if not selected_cols:
        st.warning("Seleziona almeno una colonna da visualizzare.")
        return

    # =========================
    # PREPARAZIONE DATI
    # =========================
    df_plot = df.melt(
        id_vars=month_col,
        value_vars=selected_cols,
        var_name="Tipo",
        value_name="MWh"
    )

    df_plot["MWh"] = pd.to_numeric(df_plot["MWh"], errors="coerce").fillna(0)

    df_plot_pivot = df_plot.pivot_table(
        index=month_col,
        columns="Tipo",
        values="MWh",
        fill_value=0
    )

    # =========================
    # PLOT
    # =========================
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot_pivot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel(month_col)
    ax.set_ylabel("MWh")
    ax.set_title("Monthly Coverage (PRE / POST)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


def plot_monthly_additional_hedge(df, month_col="Month"):
    """
    Grafico a barre: hedge addizionale mensile
    """
    chart = (
        alt.Chart(df)
        .mark_bar(color="#2196F3")
        .encode(
            x=alt.X(f"{month_col}:N", title="Mese"),
            y=alt.Y("Hedge_addizionale_MWh:Q", title="Hedge addizionale (MWh)"),
            tooltip=["Hedge_addizionale_MWh:Q"]
        )
    )

    st.altair_chart(chart, use_container_width=True)

def plot_cvar_reduction_over_iterations(log):
    """
    Andamento CVaR durante l'ottimizzazione greedy
    """
    # Se log è lista o DataFrame
    if log is None or len(log) == 0:
        st.warning("Nessuna iterazione di ottimizzazione disponibile.")
        return

    # Trasforma in DataFrame se non lo è già
    if not isinstance(log, pd.DataFrame):
        log_df = pd.DataFrame(log)
    else:
        log_df = log

    import altair as alt

    chart = (
        alt.Chart(log_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("iter:Q", title="Iterazione"),
            y=alt.Y("CVaR_euro:Q", title="CVaR (€)"),
            tooltip=[
                "iter:Q",
                "mese:N",
                "CVaR_euro:Q",
                "copertura_annua_pct:Q"
            ]
        )
    )

    st.altair_chart(chart, use_container_width=True)

# =========================
# FUNZIONE CVaR WRAPPER
# =========================
def CVaR(h, df_local, PUN_paths_local):
    """
    Calcola il CVaR usando un hedge vector h e il dataframe df_local.
    """
    return float(compute_CVaR(
        hedge_vector=h,
        df=df_local,
        PUN_paths=PUN_paths_local,
        VaR_level=95
    ))


def plot_hedging_dashboard(df, month_col="Anno-Mese"):
    df = df.copy()

    # =========================
    # GRAFICO 1: COSTO HEDGING
    # =========================
    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(
            x=df[month_col],
            y=df["hedge_cost"],
            mode="lines+markers",
            name="Hedge cost (€/MWh)",
            line=dict(color="black", width=2),
            marker=dict(size=7)
        )
    )

    fig_cost.update_layout(
        title="Costo hedging (€/MWh)",
        xaxis_title="Anno-Mese",
        yaxis_title="€/MWh",
        template="plotly_white",
        height=350
    )

    # =========================
    # GRAFICO 2: HEDGE ADDIZIONALE
    # =========================
    fig_hedge = go.Figure()
    fig_hedge.add_trace(
        go.Bar(
            x=df[month_col],
            y=df["hedge_addizionale_MWh"],
            name="Hedge addizionale (MWh)",
            marker_color="#ff7f0e"
        )
    )

    fig_hedge.update_layout(
        title="Hedge addizionale per mese (acquisti / de-hedging)",
        xaxis_title="Anno-Mese",
        yaxis_title="MWh",
        template="plotly_white",
        height=350
    )

    # =========================
    # GRAFICO 3: COPERTURA VS FABBISOGNO
    # =========================
    fig_cov = go.Figure()

    # Barre PRE
    fig_cov.add_bar(
        x=df[month_col],
        y=df["Copertura"],
        name="Copertura PRE",
        marker_color="#1f77b4"
    )

    fig_cov.add_bar(
        x=df[month_col],
        y=df["Scoperto_base"],
        name="Scoperto PRE",
        marker_color="#d62728"
    )

    # Barre POST
    fig_cov.add_bar(
        x=df[month_col],
        y=df["coperto_totale"],
        name="Copertura POST",
        marker_color="#2ca02c"
    )

    fig_cov.add_bar(
        x=df[month_col],
        y=df["scoperto_finale"],
        name="Scoperto POST",
        marker_color="#9467bd"
    )

    # Linea fabbisogno
    fig_cov.add_trace(
        go.Scatter(
            x=df[month_col],
            y=df["Fabbisogno"],
            name="Fabbisogno",
            mode="lines+markers",
            line=dict(color="black", width=3),
            marker=dict(size=6)
        )
    )

    fig_cov.update_layout(
        title="Copertura e scoperto PRE / POST vs Fabbisogno",
        barmode="group",
        xaxis_title="Anno-Mese",
        yaxis_title="Energia (MWh)",
        template="plotly_white",
        height=500
    )

    return fig_cost, fig_hedge, fig_cov


def read_budget_excel(file):
    df = pd.read_excel(file)

    # Normalizza nomi colonne
    df.columns = [c.strip() for c in df.columns]

    required_cols = {"Month", "Budget_Cum", "Year"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Il file deve contenere colonne: {required_cols}")

    # Eventuale cast (modifica direttamente il df)
    df["Month"] = df["Month"].astype(str)
    df["Budget_Cum"] = df["Budget_Cum"].astype(float)

    return df


def simulate_budget(
    df,
    mu=20,
    sigma_ratio=0.2,
    shape_sigma=0.15,
    n_sim=10000,
    seed=42
):
    # ============================================================
    # 1. CHECK INPUT
    # ============================================================
    required_cols = {"Year", "Month", "Budget_Cum"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Il DataFrame deve contenere: {required_cols}")

    df = df.copy()

    # Ordina correttamente per anno + mese
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month_num"], day=1))

    df = df.sort_values("date").reset_index(drop=True)

    budget_cum = df["Budget_Cum"].astype(float).values

    if not np.all(np.diff(budget_cum) >= 0):
        raise ValueError("Budget cumulato NON monotono!")

    Y_budget = budget_cum[-1]

    # ============================================================
    # 2. INCREMENTALE (MENSILE)
    # ============================================================
    monthly_budget = np.diff(np.insert(budget_cum, 0, 0.0))
    monthly_profile = monthly_budget / Y_budget

    # ============================================================
    # 3. SIMULAZIONE MONTE CARLO
    # ============================================================
    rng = np.random.default_rng(seed)

    sigma = sigma_ratio * mu
    Y_sim = np.clip(rng.normal(mu, sigma, n_sim), 0, None)

    shape = rng.lognormal(
        mean=np.log(monthly_profile + 1e-8),
        sigma=shape_sigma,
        size=(n_sim, len(df))
    )

    shape /= shape.sum(axis=1, keepdims=True)

    monthly_sim = Y_sim[:, None] * shape

    # ============================================================
    # 4. STATISTICHE MENSILI
    # ============================================================
    P50_m = np.percentile(monthly_sim, 50, axis=0)
    P90_m = np.percentile(monthly_sim, 10, axis=0)
    P95_m = np.percentile(monthly_sim, 5, axis=0)

    # ============================================================
    # 5. DATAFRAME OUTPUT
    # ============================================================
    df_out = df.copy()

    df_out["Budget"] = monthly_budget
    df_out["P50"] = P50_m
    df_out["P90"] = P90_m
    df_out["P95"] = P95_m

    # cumulati
    df_out["Cum_Budget"] = df_out["Budget"].cumsum()
    df_out["Cum_P50"] = df_out["P50"].cumsum()
    df_out["Cum_P95"] = df_out["P95"].cumsum()

    # asse x leggibile (Anno-Mese)
    x_axis = df_out["Year"].astype(str) + "-" + df_out["Month"].astype(str)

    # ============================================================
    # 6. GRAFICO CUMULATO
    # ============================================================
    fig_cum = go.Figure()

    fig_cum.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["Cum_P95"],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig_cum.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["Cum_P50"],
        fill='tonexty',
        name="Range (P50–P95)",
        mode='lines',
        line=dict(width=0),
    ))

    fig_cum.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["Cum_P50"],
        name="P50 cumulato",
        line=dict(width=3)
    ))

    fig_cum.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["Cum_Budget"],
        name="Budget cumulato",
        line=dict(width=3, dash="dash")
    ))

    fig_cum.update_layout(
        title="Produzione Cumulata (multi-anno)",
        xaxis_title="Anno-Mese",
        yaxis_title="Valore cumulato",
        template="plotly_white",
        hovermode="x unified"
    )

    # ============================================================
    # 7. GRAFICO MENSILE
    # ============================================================
    fig_monthly = go.Figure()

    fig_monthly.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["P95"],
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig_monthly.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["P50"],
        fill='tonexty',
        name="Range (P50–P95)",
        mode='lines',
        line=dict(width=0),
    ))

    fig_monthly.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["P50"],
        name="P50 mensile",
        line=dict(width=3)
    ))

    fig_monthly.add_trace(go.Scatter(
        x=x_axis,
        y=df_out["Budget"],
        name="Budget mensile",
        line=dict(width=3, dash="dash")
    ))

    fig_monthly.update_layout(
        title="Produzione Mensile (multi-anno)",
        xaxis_title="Anno-Mese",
        yaxis_title="Valore mensile",
        template="plotly_white",
        hovermode="x unified"
    )

    # ============================================================
    # 8. OUTPUT
    # ============================================================
    return {
        "df": df_out,
        "Y_budget": Y_budget,
        "plot_monthly": fig_monthly,
        "plot_cum": fig_cum
    }

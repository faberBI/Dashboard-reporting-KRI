import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from sklearn import metrics
import math
import warnings
from scipy.stats import gaussian_kde
import optuna
import io
import requests
import zipfile
import json
import subprocess
from folium.plugins import HeatMap
from PIL import Image
from arch import arch_model
from catboost import CatBoostRegressor
from copulas.multivariate import GaussianMultivariate
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import RFECV
import pickle
from datetime import datetime
from ecbdata import ecbdata
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import yfinance as yf
import pmdarima as pm
import matplotlib.pyplot as plt
import openai
import plotly.express as px
import plotly.graph_objects as go

# Library custom
from utils.data_loader import load_kri_excel, validate_kri_data
from functions.energy_risk import (
    fit_sarimax_model,
    forecast_monthly_prices,
    get_return,
    apply_cholesky,
    get_garch,
    simulate_prices,
    compute_VaR,
    plot_volatility,
    plot_monthly_VaR,
    plot_energy_stack_with_var,
    plot_var_bars,
    compute_CVaR,
    plot_monthly_coverage_stack,
    plot_monthly_additional_hedge,
    plot_cvar_reduction_over_iterations,
    CVaR,
    plot_hedging_dashboard,
    adjust_first_forecast_with_partial_month,
    read_budget_excel,
    simulate_budget)
from functions.interest_rates import (download_ecb_series, download_yahoo_series, plot_predictions_streamlit, simulate_euribor, plot_full_forecast, get_spread_for_date, get_plan_euribor_for_date)
from functions.geospatial import (get_risk_area_frane, get_risk_area_idro, get_magnitudes_for_comune)
from functions.business_interruption import (get_kri_bi, plot_kri, plot_kri_map_regioni_interattivo ,get_gpt_insights_kri)
from functions.copper import (make_lag_df, plot_copper_forecast, plot_var_vs_budget, monte_carlo_forecast_cp_from_disk, full_copper_forecast)
from functions.ebitda import (plot_top_corr_bar, get_top_correlations, simula_fattori_empiricamente, genera_template_input, load_risk_factors, parse_factors, sample_distribution, 
                            apply_uncertainty_to_params, simulate_ebitda_multi_year_blocks, simulate_ebitda_multi_year_blocks_with_ricavi, simulate_ebitda_multi_year_blocks_old, plot_k_min_max_plotly, 
                            calcola_importanza_fattori, genera_output_excel, safe_pivot, ensure_dataframe, safe_applymap)

# -----------------------
# Configurazione Streamlit
# -----------------------
openai.api_key = st.secrets["OPEN_AI_KEY"]

# Carica il logo
logo = Image.open("Image/logo_fibercop.PNG")

st.set_page_config(page_title="Risk Situation Room", page_icon=logo , layout="wide")
st.markdown("""
<div style='text-align: center;'>
""", unsafe_allow_html=True)

st.image(logo, width=300)  # logo centrato grazie al div

st.markdown("""
<h1 style='color: white; font-weight: 800; font-family: Arial, sans-serif;'>
Risk Situation Room </h1>
<p style='color: #cccccc; font-size: 18px; font-family: Arial, sans-serif;'></p>
</div>
""", unsafe_allow_html=True)

# st.set_page_config(page_title="Risk Situation Room", page_icon="📊", layout="wide")
st.title("📊 Risk Situation Room")

# -----------------------
# Selezione KRI
# -----------------------
kri_options = ["⚡ Energy Risk", "🌪️ Natural Event Risk", "🟠 Copper Price", "🛑⚡ Business Interruption","💳 Credit risk" ,"📈 Interest Rate", "💰 Liquidity Risk","📊📈 Ebitda @Risk" ]

if "kri_data" not in st.session_state:
    st.session_state.kri_data = {}

selected_kri = st.sidebar.selectbox("📑 Seleziona KRI", kri_options)

uploaded_file = st.sidebar.file_uploader(
    f"📂 Carica file Excel per {selected_kri}", type="xlsx", key=selected_kri
)

# -----------------------
# Funzione per ottenere DataFrame KRI
# -----------------------
def get_kri_dataframe(selected_kri, uploaded_file):
    df = None
    if uploaded_file:
        try:
            df = load_kri_excel(uploaded_file, selected_kri)
            if validate_kri_data(df, selected_kri):
                st.session_state.kri_data[selected_kri] = df
                st.success(f"✅ {selected_kri} aggiunto con successo!")
            else:
                st.warning(f"⚠️ File Excel non valido per {selected_kri}. Uso dati di default.")
                df = None
        except Exception as e:
            st.warning(f"⚠️ Errore nel caricamento: {e}. Uso valori di default.")
            df = None

    if df is None:
        if selected_kri == "⚡ Energy Risk":
            df = pd.DataFrame({
            "Anno": [2026]*12,
            "Month": ["gen","feb","mar","apr","mag","giu","lug","ago","set","ott","nov","dic"],
            "Fabbisogno": [115.818215, 104.882526, 117.1127113, 116.1287501, 124.1428433, 139.1228606,
                    147.6280344, 146.0052504, 128.9177564, 120.0082437, 108.5472334, 104.6362729],
            "PPA Erg": [40, 34, 38, 35, 37, 32, 33, 33, 33, 34, 34, 36],
            "Forward": [72.19178082, 65.20547945, 72.19178082, 69.8630137, 72.19178082, 69.8630137,
                72.19178082, 72.19178082, 69.8630137, 72.19178082, 69.8630137, 72.19178082],
            "Solar": [0.197, 0.197, 0.937, 0.937, 0.937, 9.347, 10.57, 11.812, 17.128, 18.37, 19.612, 28.571],
            "Prezzo Forward": [119.84, 116.34, 102.89, 90.84, 87.9, 93.51, 103.61, 103.61, 103.61, 108.17, 108.17, 108.17],
            "Prezzo Budget": [132.8, 132.7, 115.2, 98.2, 99, 108.7, 116.9, 110.4, 121.7, 124.6, 129.7, 130.4]
        })
        elif selected_kri == "🌪️ Natural Event Risk":
            df = pd.DataFrame({
                "id": [1, 2],
                "comune": ["Milano", "Capua"],
                "zona": ["B12", "C2"],
                "lat": [45.47377982648482, 41.109706286872694],
                "long": [9.179101925254832, 14.20053274338481],
                "codice_comune": ["F205", "B715"],
                "building": [200000, 250000],
                "content": [50000, 60000]
            })
        else:
            df = pd.DataFrame()
        st.warning(f"⚠️ Nessun file Excel caricato per {selected_kri}. Uso valori di default.")
        st.session_state.kri_data[selected_kri] = df

    return df

# -----------------------
# Carica o crea DataFrame
# -----------------------
df = get_kri_dataframe(selected_kri, uploaded_file)
st.subheader(f"📌 {selected_kri}")
st.dataframe(df.head())

# -----------------------
# Logica specifica KRI
# -----------------------
if selected_kri == "⚡ Energy Risk":

    st.subheader("💰 Inserisci o modifica EBITDA per anno")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile nel DataFrame!")
        st.stop()

    # ============================================================
    # 0) UTIL: normalizzazione mesi (ITA <-> num)
    # ============================================================
    month_map_ita_to_num = {
        "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
        "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
        
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,

        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,

        "Gen": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Mag": 5, "Giu": 6,
        "Lug": 7, "Ago": 8, "Set": 9, "Ott": 10, "Nov": 11, "Dic": 12

    }
    month_map_num_to_ita = {v: k for k, v in month_map_ita_to_num.items()}

    def normalize_month_num(s: pd.Series) -> pd.Series:
        """
        Converte Month in numero 1..12:
        - accetta 'gen','feb',... oppure '1','2',... oppure già numerico.
        """
        ss = s.astype(str).str.strip().str.lower()
        num_from_ita = ss.map(month_map_ita_to_num)
        num_from_num = pd.to_numeric(ss, errors="coerce")
        out = num_from_ita.fillna(num_from_num)
        return out.astype("Int64")

    def month_num_to_ita(series_num: pd.Series) -> pd.Series:
        return series_num.map(month_map_num_to_ita).astype("string")

    # ============================================================
    # 1) PREPARAZIONE DATI BASE + allineamento Month (DB italiano)
    # ============================================================
    df_base = df.copy()

    # normalizzo mese sul DB (italiano)
    df_base = df_base.assign(
        Month_num=normalize_month_num(df_base["Month"]),
        Month_ita=lambda x: month_num_to_ita(x["Month_num"])
    )

    df_base["Month"] = df_base["Month_ita"]

    # ============================================================
    # 2) INPUT EBITDA
    # ============================================================
    agg_dict = {
        "Fabbisogno": "sum",
        "PPA Erg": "sum",
        "Forward": "sum",
        "Solar": "sum",
        "Prezzo Forward": "mean",
        "Prezzo Budget": "mean"
    }

    df_grouped = df_base.groupby("Anno").agg(agg_dict).reset_index()

    if "Ebitda" not in df_grouped.columns:
        df_grouped["Ebitda"] = 1_900_000_000

    ebitda_inputs = {}
    for _, row in df_grouped.iterrows():
        anno = int(row["Anno"])
        ebitda_inputs[anno] = st.number_input(
            f"EBITDA per {anno} (€)",
            min_value=0.0,
            value=float(row["Ebitda"]),
            step=1_000_000.0,
            format="%.0f",
            key=f"ebitda_{anno}"
        )

    df_grouped["Ebitda"] = df_grouped["Anno"].map(ebitda_inputs)
    st.dataframe(df_grouped.style.format({"Ebitda": "€{:,.0f}"}))
    st.success("✅ Parametri validi, pronti per la simulazione!")

    # ============================================================
    # 3) PARAMETRI SIMULAZIONE
    # ============================================================
    n_simulations = st.number_input("Numero simulazioni", 1000, 1_000_000, 10_000, 1000)
    risk_appetite = st.number_input("Risk appetite (% EBITDA)", 0.005, 0.2, 0.01, 0.001)
    step = st.number_input("Step MWh hedging", 1000, 10000, 5000, 1000)
    alpha = st.number_input("Copertura minima fabbisogno", 0.5, 1.0, 0.85, 0.05)

    n_year = int(df_base["Anno"].nunique())
    st.metric("Numero anni simulati", n_year)

    # ============================================================
    # 4) SOLAR STOCASTICO
    # ============================================================
    st.subheader("☀️ Stress Produzione Solar")

    use_solar_stochastic = st.checkbox("Attiva Solar stocastico", value=False)
    solar_sigma_ratio = st.number_input("σ Solar", 0.01, 1.0, 0.20, 0.05)
    mu_prod = st.number_input("Produzione effettiva Solar", 0.0, 1000.0, 20.0, 5.0)
    regen_solar = st.button("🔄 Rigenera Solar")

    df_budget = None
    if use_solar_stochastic:
        df_budget = read_budget_excel("Data/Solar_Budget.xlsx")

        # ✅ ALLINEAMENTO Month del budget al DB in italiano
        # budget: crea Month_num e Month_ita e (se vuoi) Month = Month_ita
        df_budget = df_budget.copy()
        # df_budget DEVE avere colonne Year, Month, Budget_Cum (come nella tua funzione)
        df_budget = df_budget.assign(
            Month_num=normalize_month_num(df_budget["Month"]),
            Month_ita=lambda x: month_num_to_ita(x["Month_num"])
        )
        df_budget["Month"] = df_budget["Month_ita"]

        with st.expander("🔎 Check Solar Budget"):
            st.dataframe(df_budget[["Year", "Month", "Budget_Cum"]])

    # ============================================================
    # 5) SIMULAZIONE
    # ============================================================
    if st.button("💹 Esegui simulazione Energy Risk"):

        st.info("⏳ Simulazione in corso...")

        # -------------------------
        # 5.1 PUN DATA
        # -------------------------
        data_path = "Data/Pun_ts_price.xlsx"
        if not os.path.exists(data_path):
            st.error("❌ File PUN mancante: Data/Pun_ts_price.xlsx")
            st.stop()

        df_excel = pd.read_excel(data_path)
        if "Date" not in df_excel.columns or "GMEPIT24 Index" not in df_excel.columns:
            st.error("❌ Formato file errato: servono colonne 'Date' e 'GMEPIT24 Index'")
            st.stop()

        df_excel["Date"] = pd.to_datetime(df_excel["Date"])

        # -------------------------
        # 5.2 MODELLO PREZZI
        # -------------------------
        np.random.seed(42)

        last_5y, monthly_std, monthly_price, monthly_price_year = get_return(data_path)
        L, _ = apply_cholesky(last_5y)

        hist_pun = df_excel.copy()
        hist_pun["Year"] = hist_pun["Date"].dt.year
        hist_filter = hist_pun[hist_pun["Year"] > 2015]
        series_daily = hist_filter.set_index("Date")["GMEPIT24 Index"].astype(float).dropna()

        # serie mensile per forecast
        series_monthly = monthly_price_year.set_index("Date")["avg_price"]
        forecast = forecast_monthly_prices(series_monthly, n_years=n_year)

        # indice mensile coerente (MS)
        start_ms = pd.Timestamp(series_monthly.index[-1]).to_period("M").to_timestamp(how="start")
        forecast.index = pd.date_range(start=start_ms, periods=len(forecast), freq="MS")

        monthly_sigma, _, _ = get_garch(last_5y)

        PUN_paths, shocks = simulate_prices(
            forecast,
            monthly_price["avg_price"].values,
            monthly_sigma=monthly_sigma,
            monthly_std=monthly_std,
            L=L,
            n_sims=n_simulations
        )

        VaR_95_monthly = np.percentile(PUN_paths, 95, axis=0)
        st.success("✅ VaR mensile (95°) calcolato sui path PUN!")

        # -------------------------
        # 5.3 DEFINIZIONE df_risk (Solar OFF = piano, Solar ON = P95 con fallback)
        # -------------------------
        df_risk = df_base.copy()

        if use_solar_stochastic:

            if df_budget is None or df_budget.empty:
                st.error("❌ Caricare budget solar")
                st.stop()

            # cache key (evita ricalcolo inutile)
            cache_key = (float(mu_prod), float(solar_sigma_ratio), int(n_simulations), 42)

            if regen_solar or ("solar_cache_key" not in st.session_state) or (st.session_state.solar_cache_key != cache_key):

                out = simulate_budget(
                    df=df_budget,
                    mu=mu_prod,
                    sigma_ratio=solar_sigma_ratio,
                    shape_sigma=0.15,
                    n_sim=n_simulations,
                    seed=42)

                st.session_state.solar_cache_key = cache_key
                st.session_state.solar_out = out

            solar_out = st.session_state.solar_out

            st.subheader("☀️ Solar – simulazione stocastica")               
            st.plotly_chart(solar_out["plot_monthly"], use_container_width=True)              
            st.plotly_chart(solar_out["plot_cum"], use_container_width=True)

            # estraggo P95, allineo mese in ITA + num
            df_solar_p95 = solar_out["df"][["Year", "Month", "P95"]].copy()
            df_solar_p95 = df_solar_p95.assign(
                Month_num=normalize_month_num(df_solar_p95["Month"]),
                Month_ita=lambda x: month_num_to_ita(x["Month_num"])
            ).rename(columns={"Year": "Anno", "P95": "Solar_P95"})

            # merge robusto su (Anno, Month_num) e fallback al Solar piano
            df_risk = (
                df_risk
                .assign(
                    Solar_budget=lambda x: x["Solar"],
                    Month_num=normalize_month_num(df_risk["Month"]),
                    Month_ita=lambda x: month_num_to_ita(x["Month_num"])
                )
                .merge(
                    df_solar_p95[["Anno", "Month_num", "Solar_P95"]],
                    on=["Anno", "Month_num"],
                    how="left"
                )
                .assign(
                    Solar=lambda x: x["Solar_P95"].fillna(x["Solar_budget"])
                )
                .drop(columns=["Solar_P95"])
            )

            st.success("✅ Solar stocastico applicato ai valori di piano")

        else:
            st.info("ℹ️ Solar deterministico: VaR calcolato con Solar a piano")

        # -------------------------
        # 5.4 VaR Engine
        # -------------------------
        last_month = series_daily.index[-1]
        n_months = n_year * 12
        cut_month = n_months - (last_month.month - 1)
        hid_month = n_months - cut_month

        dati_fibercop = compute_VaR(df_risk, VaR_95_monthly, cut_month, hid_month)

        st.subheader("📊 Output VaR")
        st.dataframe(dati_fibercop)

        st.metric(
            "Yearly VaR (somma mensile)",
            f"€ {np.round(dati_fibercop['Var_monthly_95_w_solar'].sum(), 0):,.0f}"
        )

       # ============================================================
        # 6) OTTIMIZZAZIONE (USA SEMPRE IL PIANO)
        # ============================================================
        st.subheader("📈 Hedging Optimization Model")

        # df_opt = metriche rischio (da df_risk) + Solar_plan (da df_base)
        df_opt = (
            dati_fibercop
            .merge(
                df_base[["Anno", "Month_num", "Month", "Solar"]].rename(columns={"Solar": "Solar_plan"}),
                on=["Anno", "Month_num", "Month"],
                how="left"
            )
        )

        # IMPORTANTISSIMO: ordina per allineare con PUN_paths (mesi in ordine)
        df = (
            df_opt
            .sort_values(["Anno", "Month_num"])
            .reset_index(drop=True)
            .copy()
        )

        # ⚠️ ripristino Solar OPERATIVO (sempre piano) per ottimizzazione
        df["Solar"] = df["Solar_plan"]

        # ---- unità: lavoro in MWh
        df["Fabbisogno"] = df["Fabbisogno"] * 1000
        df["Copertura"] = (df["PPA Erg"] + df["Forward"] + df["Solar"]) * 1000

        # ---- ALLINEA unità anche per CVaR (compute_CVaR usa scoperto_w_solar)
        df["scoperto_w_solar"] = df["scoperto_w_solar"] * 1000
        if "scoperto_w/o_solar" in df.columns:
            df["scoperto_w/o_solar"] = df["scoperto_w/o_solar"] * 1000

        # per grafici/diagnostica
        df["Scoperto_base"] = df["scoperto_w_solar"]

        # Limiti annuali (risk appetite su EBITDA)
        CVaR_limit = {year: ebitda_inputs[year] * risk_appetite for year in df["Anno"].unique()}

        # Hedge cost mensile (€/MWh)
        df["hedge_cost"] = df["Prezzo Forward"] - df["Prezzo Budget"]

        # Inizializzazione hedge (MWh)
        n_months = len(df)
        hedge = np.zeros(n_months, dtype=float)

        for year in df["Anno"].unique():
            mask = df["Anno"] == year
            total_fabbisogno_year = df.loc[mask, "Fabbisogno"].sum()
            coperto_attuale_year = df.loc[mask, "Copertura"].sum()
            max_copertura_totale_year = total_fabbisogno_year * alpha
            max_needed = max(max_copertura_totale_year - coperto_attuale_year, 0)

            weights = df.loc[mask, "Scoperto_base"].values
            if weights.sum() > 0:
                weights = weights / weights.sum()
                hedge[mask.values] += max_needed * weights

        # Limite mensile (MWh)
        max_hedge_mensile = df["Fabbisogno"].values * alpha - df["Copertura"].values
        hedge = np.minimum(hedge, max_hedge_mensile)

        # Saturazione residuo anno per anno
        for year in df["Anno"].unique():
            mask = df["Anno"] == year
            residuo = (df.loc[mask, "Fabbisogno"].sum() * alpha) - (df.loc[mask, "Copertura"].sum() + hedge[mask.values].sum())

            while residuo > 0:
                spazio = max_hedge_mensile[mask.values] - hedge[mask.values]
                spazio[spazio < 0] = 0
                if spazio.sum() <= 0:
                    break
                incremento = residuo * (spazio / spazio.sum())
                hedge[mask.values] += incremento
                hedge[mask.values] = np.minimum(hedge[mask.values], max_hedge_mensile[mask.values])
                residuo = (df.loc[mask, "Fabbisogno"].sum() * alpha) - (df.loc[mask, "Copertura"].sum() + hedge[mask.values].sum())

        hedge = hedge.astype(float)

        # CVaR iniziale (globale, usato per efficienza rischio/costo)
        CVaR_current = CVaR(hedge, df, PUN_paths)

        # =========================
        # OTTIMIZZAZIONE GREEDY MULTI-ANNO
        # =========================
        iteration = 0
        log = []
        total_fabbisogno = df["Fabbisogno"].sum()

        while True:
            iteration += 1
            best_month = None
            best_efficiency = 0.0

            spazio = max_hedge_mensile - hedge
            admissible = (spazio >= step) & (df["hedge_cost"].values > 0)

            if not admissible.any():
                st.warning("⚠️ Nessun ulteriore miglioramento possibile.")
                break
            
            for m in np.where(admissible)[0]:
                hedge_test = hedge.copy()
                hedge_test[m] += step

                anno_m = df.loc[m, "Anno"]
                mask_year = (df["Anno"].values == anno_m)

                copertura_annua = (
                    (df.loc[mask_year, "Copertura"].sum() + hedge_test[mask_year].sum())
                    / df.loc[mask_year, "Fabbisogno"].sum()
                )
                if copertura_annua > alpha:
                    continue
                
                CVaR_new = CVaR(hedge_test, df, PUN_paths)
                risk_reduction = CVaR_current - CVaR_new
                cost_eur = step * df.loc[m, "hedge_cost"]
                if cost_eur <= 0:
                    continue
                
                efficiency = risk_reduction / cost_eur
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_month = m

            if best_month is None:
                st.warning("⚠️ Nessun ulteriore miglioramento possibile.")
                break
            
            # aggiorna hedge
            hedge[best_month] += step
            CVaR_current = CVaR(hedge, df, PUN_paths)

            # =========================
            # CHECK RISK APPETITE (ANNUALE) - COERENTE
            # =========================
            anno_best = df.loc[best_month, "Anno"]
            mask_year = (df["Anno"].values == anno_best)

            CVaR_year = compute_CVaR(
                hedge_vector=hedge[mask_year],
                df=df.loc[mask_year].copy(),
                PUN_paths=PUN_paths[:, mask_year]
            )

            if CVaR_year > CVaR_limit[anno_best]:
                st.warning(
                    f"⚠️ CVaR {CVaR_year:,.0f}€ oltre risk appetite {CVaR_limit[anno_best]:,.0f}€ per {anno_best}. Stop ottimizzazione."
                )
                break
            
            # log (usa CVaR_year per coerenza annuale)
            hedge_tot = hedge.sum()
            cvar_pct = (CVaR_year / ebitda_inputs[anno_best]) * 100
            copertura_tot_pct = ((df["Copertura"] + hedge).sum() / total_fabbisogno) * 100

            log.append({
                "iter": iteration,
                "mese": df.loc[best_month, "Month"],
                "anno": int(anno_best),
                "hedge_tot_MWh": float(hedge_tot),
                "CVaR_euro": float(CVaR_year),
                "CVaR_pct_EBITDA": float(cvar_pct),
                "copertura_annua_pct": float(copertura_tot_pct)
            })

            st.write(f"Iter {iteration}: CVaR anno {anno_best} = {CVaR_year:,.0f}€, Copertura totale = {copertura_tot_pct:.2f}%")

        log = pd.DataFrame(log)

        # =========================
        # OUTPUT FINALE
        # =========================
        df["hedge_addizionale_MWh"] = hedge
        df["coperto_totale"] = df["Copertura"] + hedge
        df["scoperto_finale"] = df["Fabbisogno"] - df["coperto_totale"]
        total_hedge_cost = np.sum(hedge * df["hedge_cost"].values)
        copertura_tot_pct = (df["coperto_totale"].sum() / total_fabbisogno) * 100

        st.dataframe(df)
        st.metric("CVaR finale (ultimo anno vincolante) (€)", f"€ {log['CVaR_euro'].iloc[-1]:,.0f}" if len(log) else f"€ {CVaR_current:,.0f}")
        st.metric("Costo hedge totale (€)", f"€ {total_hedge_cost:,.0f}")
        st.metric("Copertura totale (%)", f"{copertura_tot_pct:.2f}%")

        st.subheader("📊 Grafici")
        if 'Month' in df.columns:         
            month_map = {"gen": 1, "feb": 2, "mar": 3, "apr": 4,"mag": 5, "giu": 6, "lug": 7, "ago": 8,"set": 9, "ott": 10, "nov": 11, "dic": 12}
            df['Month_num'] = (df['Month'].str.lower().str.strip().map(month_map))
            df['Anno'] = df['Anno'].astype(int)
            df['Month_num'] = df['Month_num'].astype(int)
            df['date'] = pd.to_datetime(
            df['Anno'].astype(str) + "-" + df['Month_num'].astype(str).str.zfill(2),format="%Y-%m")
            df['period'] = df['date'].dt.to_period('M')
            df = df.sort_values('period')
            df['Anno-Mese'] = df['period'].astype(str)
        
        plot_monthly_coverage_stack(df, month_col="Anno-Mese")
        fig_cost, fig_hedge, fig_cov = plot_hedging_dashboard(df, month_col="Anno-Mese")
        st.plotly_chart(fig_cost, use_container_width=True)
        st.plotly_chart(fig_hedge, use_container_width=True)
        st.plotly_chart(fig_cov, use_container_width=True)

        # Monthly adjustment dataset
        dati_monthly = (dati_fibercop.groupby('Anno',as_index=False)[['Fabbisogno','PPA Erg','Forward','Solar','scoperto_w_solar','scoperto_w/o_solar','Var_monthly_95_w_solar','Var_monthly_95_w/o_solar']].sum())
        
        # Esportazione Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_base.to_excel(writer, sheet_name="dati_input", index=False)
            last_5y.to_excel(writer, sheet_name="last_5y", index=False)
            monthly_std.to_excel(writer, sheet_name="monthly_std", index=False)
            monthly_price.to_excel(writer, sheet_name="monthly_price", index=False)
            pd.DataFrame(forecast).to_excel(writer, sheet_name="PUN_forecast", index=False)
            pd.DataFrame(monthly_sigma, columns=['monthly_sigma']).to_excel(writer, sheet_name="monthly_sigma", index=False)
            pd.DataFrame(PUN_paths).to_excel(writer, sheet_name="PUN_paths", index=False)
            dati_fibercop.to_excel(writer, sheet_name="dati_var_monthly", index=False)
            dati_monthly.to_excel(writer, sheet_name = "dati_var_yearly", index = False)
            pd.DataFrame(VaR_95_monthly, columns=['VaR_95']).to_excel(writer, sheet_name="VaR_95_monthly", index=False)
            pd.DataFrame(shocks).to_excel(writer, sheet_name="shocks", index=False)
            pd.DataFrame(df).to_excel(writer, sheet_name="Hedging", index=False)
            log.to_excel(writer, sheet_name="Iteration_Algo", index=False)
        buffer.seek(0)
    
        st.download_button("💾 Scarica tutti i dati in Excel", data=buffer,
                           file_name=f"Energy_Risk_VaR_{pd.Timestamp.today().date()}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
# -----------------------
# 🌪️ Natural Event Risk
# -----------------------
elif selected_kri == "🌪️ Natural Event Risk":
    st.subheader("🌪️ Simulazione Eventi Naturali – Portafoglio Immobiliare")
    st.info("Esegui la simulazione di rischio multi-events (idro, frane, sismico, tempeste)")

    # Parametri simulazione
    n_simulazioni = st.number_input(
        "Numero di simulazioni Monte Carlo",
        min_value=1000,
        max_value=100_000,
        value=10_000,
        step=1000
    )

    # Caricamento librerie e database
    try:
        from functions.constants import classi_rischio, alpha_tilde_classi_frane, load_shapefiles_from_dropbox

        from functions.natural_events import (
            simulazione_portafoglio_con_rischi_correlati,
            calcola_vulnerabilita_intrinseca_frane,
            calcola_perdita_attesa_frane,
            vulnerabilita_profondita_pol,
            simulazione_perdita_attesa_idro,
            calculate_IEMS,
            calculate_mu_D,
            generate_damage_probability,
            calculate_value_loss,
            simulazione_perdita_attesa_sismica,
            simula_danno_tempesta
        )

        from functions.geospatial import (get_risk_area_frane, get_risk_area_idro, get_magnitudes_for_comune)

        import folium
        from streamlit_folium import st_folium
        import os
        try: 
            frane_url = st.secrets["FRANE_URL"]
            idro_url = st.secrets["IDRO_URL"]
            db_frane, db_idro = load_shapefiles_from_dropbox(frane_url, idro_url)
            
        except Exception as e:
            st.error(f"❌ Errore nel caricament dei database in formato shape : {e}")
            db_frane = pd.DataFrame()
            db_idro = pd.DataFrame()
            
        df_sismico = pd.read_excel("Data/class_comune_rischio_sismico.xlsx") if os.path.exists("Data/class_comune_rischio_sismico.xlsx") else pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Errore nel caricamento librerie o database: {e}")
        st.stop()

    # Mostra mappa immobili
    st.subheader("📍 Heatmap Immobili per Valore Building")

    if not df.empty and "lat" in df.columns and "long" in df.columns and "building" in df.columns:
        # Centra la mappa sulla media delle coordinate
        mappa = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=10)
    
        # Prepara dati per la HeatMap: [lat, long, peso]
        heat_data = [[row["lat"], row["long"], row["building"]] for idx, row in df.iterrows()]
    
        # Aggiungi la HeatMap
        HeatMap(heat_data, radius=15, max_zoom=13).add_to(mappa)
    
        st_folium(mappa, width=700, height=500)
    else:
        st.warning("📌 Nessun dato geografico disponibile per la mappa.")

    # Esecuzione simulazione
    if st.button("🚀 Avvia Simulazione Natural Event Risk"):
        with st.spinner("Esecuzione simulazione in corso..."):
            from functions.natural_events import (
                    simulazione_portafoglio_con_rischi_correlati,
                    calcola_vulnerabilita_intrinseca_frane,
                    calcola_perdita_attesa_frane,
                    vulnerabilita_profondita_pol,
                    simulazione_perdita_attesa_idro,
                    calculate_IEMS,
                    calculate_mu_D,
                    generate_damage_probability,
                    calculate_value_loss,
                    simulazione_perdita_attesa_sismica,
                    simula_danno_tempesta
                        )

            from functions.geospatial import (get_risk_area_frane, get_risk_area_idro, get_magnitudes_for_comune)
            try:
                
                results = simulazione_portafoglio_con_rischi_correlati(
                df=df,
                n_simulazioni=int(n_simulazioni),
                database_frane=db_frane,
                database_idro=db_idro,
                db_sismico=df_sismico
                    )
                st.success("✅ Simulazione completata!")

                # Mostra risultati
                st.subheader("📊 Risultati Simulazione")
                st.dataframe(results.head())

                # Grafico distribuzione perdite
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.hist(results["Perdita_aggregata_50"], bins=50, alpha=0.7)
                ax.set_title("Distribuzione Perdite Simulate")
                ax.set_xlabel("Perdita (€)")
                ax.set_ylabel("Frequenza")
                st.pyplot(fig)

                # Download Excel
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    results.to_excel(writer, index=False, sheet_name='Risultati Simulazione')
                    df.to_excel(writer, index=False, sheet_name='Immobili')
                    buffer.seek(0)

                st.download_button(
                    label="💾 Scarica risultati in Excel",
                    data=buffer,
                    file_name="Simulazione_Natural_Event_Risk.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"❌ Errore durante la simulazione: {e}")
# -----------------------
# 🟠 Copper Risk
# -----------------------
elif selected_kri == "🟠 Copper Price": 
    st.subheader("🟠 Simulazione su Copper (prezzi in euro)")
    st.info("Esegui la simulazione multivariata del copper")

    df_model = pd.read_excel('Data/copper_price.xlsx')
    if "Time" not in df_model.columns:
        raise KeyError("La colonna 'Time' non esiste nel file Excel!")

    df_model["Time"] = pd.to_datetime(df_model["Time"], errors="coerce")

    price_col = 'Copper'
    if price_col not in df_model.columns:
        raise KeyError(f"La colonna '{price_col}' non esiste nel file Excel!")

    series = pd.to_numeric(df_model[price_col], errors="coerce").dropna().reset_index(drop=True)

    # Imposta 'Time' come indice
    df_model.set_index("Time", inplace=True)
    st.dataframe(df_model[price_col])
    
    st.subheader(f"Andamento {price_col}")
    st.line_chart(df_model[[price_col]])
   
    st.info("Fonte Dati: https://www.insee.fr/en/statistiques/serie/010767327")
    
    # -----------------------------------------------
    # 📅 Selezione data finale simulazione
    # -----------------------------------------------
    end_date = st.date_input(
        "📅 Seleziona la data di fine simulazione",
        value=datetime(2028, 12, 31),
        min_value=datetime.now()
    )
    n_sims = st.slider("Number of Monte Carlo simulations", min_value=100, max_value=100_000, value=10_000, step=100)
    
    st.subheader("📦 Quantità di Copper da vendere per anno")
    start_year = df_model.index[-1].year
    end_year = end_date.year

    # Lista anni
    years = list(range(start_year, end_year + 1))
    quantities = {}
    for y in years:
        quantities[y] = st.number_input(
            f"Quantità da vendere nel {y} (in tonnellate)",
            min_value=0.0,
            step=1.0,
            value=0.0,
            format="%.2f"
        )

    budget_price = st.number_input(
        "💰 Prezzo di budget (EUR per tonnellata)",
        min_value=0.0,
        step=10.0,
        value=9000.0,
        format="%.2f"
    )

    # -----------------------------------------------
    # 🚀 RUN SIMULAZIONE
    # -----------------------------------------------
    if st.button("💹 Esegui simulazione Copper Risk"):
        st.info("Simulazione in corso...")
        result_df , result_df_annual = monte_carlo_forecast_cp_from_disk(df_model[price_col],  N_SIM=n_sims,    end_date=end_date,    random_seed=42)
        fig = plot_copper_forecast(df_model, result_df_annual)
        st.pyplot(fig)
        st.subheader("📊 Risultati Simulazione")
        result_df.index = pd.to_datetime(result_df.index)

        # -----------------------------------------------
        # 💰 Aggiunta quantità e calcolo P&L vs budget
        # -----------------------------------------------
        result_df_annual["qty"] = result_df_annual.index.year.map(quantities)
        result_df_annual["VaR_vs_budget"] = ((result_df_annual["CP_Lower_95"] - budget_price) * result_df_annual["qty"]) / 1_000_000
        result_df_annual.drop(['GARCH_Lower_95','CP_Upper_95'], axis =1, inplace = True)
        result_df_annual.columns = ['Mean_Forecast', 'Upper_95','Lower_95','qty','VaR_vs_budget']
        
        st.subheader("📘 VaR per anno (mln €)")
        result_df_annual.index = result_df_annual.index.year
        st.dataframe(result_df_annual)
        fig = plot_var_vs_budget(result_df_annual)
        st.pyplot(fig)
        # -----------------------------------------------
        # 💾 Download Excel
        # -----------------------------------------------
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=True, sheet_name='Mensile Simulazione')
            result_df_annual.to_excel(writer, index=True, sheet_name='Annuale Aggregato')
            df_model.to_excel(writer, index=True, sheet_name='Copper Price')
            buffer.seek(0)

        st.download_button(
            label="💾 Scarica risultati in Excel",
            data=buffer,
            file_name="Simulazione_Copper_price.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
elif selected_kri == "💳 Credit risk":

    st.subheader("🏦 Credit Risk – Aging & Indicatori")

    if df is None or df.empty:
        st.info("ℹ️ Nessun dato Aging disponibile. Caricare o generare i dati nella sezione precedente.")
        st.stop()

    df.columns = df.columns.str.strip()

    required_cols = [
        "Periodo",
        "TRADE RECEIVABLES (NET)",
        "Not Overdue", "1-90", "91-180",
        "181-365", "Over 365", "PROVISION"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.error(f"⚠️ Colonne mancanti nel dataset Aging: {missing_cols}")
        st.stop()

    # --------------------------
    # Parsing Periodo
    # --------------------------
    df["Periodo"] = pd.to_datetime(df["Periodo"], format="%m-%Y")

    # --------------------------
    # Raggruppamento per Periodo
    # --------------------------
    grouped = df.groupby("Periodo").sum().reset_index()

    # --------------------------
    # KPI CALCULATION
    # --------------------------
    grouped["Over90"] = (
        grouped["91-180"] +
        grouped["181-365"] +
        grouped["Over 365"]
    )

    grouped["Pct_Over_90"] = (
        grouped["Over90"] / grouped["TRADE RECEIVABLES (NET)"]
    )

    grouped["Delta_Provision"] = (
        grouped["PROVISION"].diff().fillna(0)
    )

    grouped["Aging"] = (
        0   * grouped["Not Overdue"] +
        45  * grouped["1-90"] +
        135 * grouped["91-180"] +
        270 * grouped["181-365"] +
        365 * grouped["Over 365"]
    ) / grouped["TRADE RECEIVABLES (NET)"]

    # --------------------------
    # KPI dataframe
    # --------------------------
    kpi_df = grouped[[
        "Periodo",
        "TRADE RECEIVABLES (NET)",
        "Pct_Over_90",
        "Delta_Provision",
        "Aging"
    ]].copy()

    kpi_df["Delta_Provision"] = kpi_df["Delta_Provision"].round(0)
    kpi_df["Aging"] = kpi_df["Aging"].round(0)

    st.subheader("📊 Indicatori Calcolati per Periodo")
    st.dataframe(kpi_df)

    st.subheader("📈 Grafici KPI per Periodo")

    import plotly.express as px

    # 1️⃣ Percentuale Over 90
    fig_pct = px.bar(
        kpi_df,
        x="Periodo",
        y="Pct_Over_90",
        text="Pct_Over_90",
        title="📊 Percentuale Crediti > 90 giorni per Periodo",
        color="Pct_Over_90",
        color_continuous_scale="Blues"
    )
    fig_pct.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_pct, use_container_width=True)

    # 2️⃣ Delta Provision
    fig_delta = px.bar(
        kpi_df,
        x="Periodo",
        y="Delta_Provision",
        text="Delta_Provision",
        title="💰 Delta Provision vs T-1 (€)",
        color="Delta_Provision",
        color_continuous_scale="Oranges"
    )
    fig_delta.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    st.plotly_chart(fig_delta, use_container_width=True)

    # 3️⃣ Aging medio
    fig_aging = px.bar(
        kpi_df,
        x="Periodo",
        y="Aging",
        text="Aging",
        title="⏳ Aging medio dei crediti (giorni)",
        color="Aging",
        color_continuous_scale="Greens"
    )
    fig_aging.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    st.plotly_chart(fig_aging, use_container_width=True)

    # --------------------------
    # Download Excel
    # --------------------------
    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Aging Raw")
        kpi_df.to_excel(writer, index=False, sheet_name="Indicatori KPI")
        summary = kpi_df.mean(numeric_only=True).to_frame("Value")
        summary.to_excel(writer, sheet_name="Sintesi KPI")
    buffer.seek(0)
    st.download_button(
        label="💾 Scarica file Credit Risk (Excel)",
        data=buffer,
        file_name="Credit_Risk_Aging_Indicators.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
  
elif selected_kri == "🛑⚡ Business Interruption":  
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        with st.spinner("Calcolo KRI e metriche..."):
            # Calcolo KRI e aggregazioni
            result_TFRI, WGHI_REG, WGHI_PROV, WGHI_IC, risultati_df = get_kri_bi(df)
    
        st.success("✅ Calcolo completato!")
    
        # =========================
        # Grafici principali
        # =========================
        st.markdown("""
        **Legenda KRI**  
        - 🟢 **Expected Severe Outage Rate (SOR)**: Modello MonteCarlo per stima della probabilità attesa dei disservizi con durata > soglia "alta"  
        - 🔵 **Technology Failure Risk Index (TFRI)**: Misura la vulnerabilità strutturale associata a una tipologia di causa di disservizio  
        - 🔴 **Weighted Geographical Hotspot Index (WGHI)**: Individuare concentrazioni anomale e persistenti di disservizi, ponderate per durata degli stessi, su specifiche aree geografiche (Provincia/Regione)  
        """)
        st.subheader("📊 Grafici KRI 🛑⚡")
        # plot_kri deve essere già definita nel tuo codice
        plot_kri(result_TFRI, WGHI_REG, WGHI_PROV, WGHI_IC, risultati_df, top_n=20)
        # =========================
        # Mappa interattiva regioni
        # =========================
        st.subheader("🗺️ Mappa Interattiva Weighted Geographical Hotspot Index")     
        top10 = WGHI_REG.sort_values('WGHI_reg_norm', ascending=False).head(10)
        display_df = top10[['Regioni', 'WGHI_reg_norm']].copy()
        styled_df = display_df.style.background_gradient(
        subset=['WGHI_reg_norm'],  # colonna da colorare
        cmap='RdYlGn_r',           # rosso-giallo-verde inverso
        vmin=0, vmax=1             # normalizzazione tra 0 e 1
        ).format({'WGHI_reg_norm': "{:.2f}"})  # due decimali
        st.dataframe(styled_df, use_container_width=True)
        fig = plot_kri_map_regioni_interattivo(WGHI_REG, shapefile_path='Data/Reg01012026_g_WGS84.shp', value_col='WGHI_reg_norm')
        st.plotly_chart(fig, use_container_width=True)

        insights_text = get_gpt_insights_kri(result_TFRI, WGHI_REG, WGHI_PROV, WGHI_IC, risultati_df, model="gpt-4")
        st.subheader("📊 Insight sui KRI di Business Interruption")
        st.markdown(insights_text)

        
        st.subheader("💾 Download Excel")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Sheet 1 - dati raw
            df.to_excel(writer, index=False, sheet_name='Input Data')
            # Sheet 2 - KPI principali
            result_TFRI.to_excel(writer, index=False, sheet_name='TFRI')
            # Sheet 3 - Sintesi WGHI_IC
            WGHI_IC.to_excel(writer, index=False, sheet_name='WGHI Impatto Cliente')
            # Sheet 4 - Sintesi  WGHI
            WGHI_REG.to_excel(writer, index=False, sheet_name='WGHI Regionale')
            # Sheet 5 - WGHI per regioni
            WGHI_PROV.to_excel(writer, index=False, sheet_name='WGHI Provinciale')
            # Sheet 6 - WGHI per province
            risultati_df.to_excel(writer, index=False, sheet_name='95° Percentile Probs')        
        buffer.seek(0)
        st.download_button(
            label="💾 Scarica file Excel con i KRI",
            data=buffer,
            file_name="KRI_Business_Interruption.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("📌 Carica un file Excel per visualizzare i KRI e la mappa interattiva.")
    
elif selected_kri == "📈 Interest Rate":
    import matplotlib.pyplot as plt
    
    series= {
    # --- Politica monetaria BCE ---
    "euribor_3m": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
    "deposit_rate": "FM.D.U2.EUR.4F.KR.DFR.LEV",
    "mro_rate": "FM.B.U2.EUR.4F.KR.MRR_FR.LEV",
    "marginal_lending": "FM.D.U2.EUR.4F.KR.MLFR.LEV",

    # --- Macro ---
    "inflation": "ICP.M.IT.N.000000.4.ANR",
    "core_inflation": "ICP.M.U2.N.XEF000.4.ANR",
    "unemployment": "SPF.Q.U2.UNEM.POINT.LT.Q.AVG",

    # --- Banking & liquidity ---
    "excess_liquidity": "SUP.Q.B01.W0._Z.I3017._T.SII._Z._Z._Z.PCT.C", 
    "deposit_facility_usage": "ILM.W.U2.C.L020200.U2.EUR",
    "refinancing_ops": "FM.D.U2.EUR.4F.KR.MRR_RT.LEV",
    "gdp_growth": "MNA.Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N",
    }
    
    yahoo_symbols = {
        "sp500": "^GSPC",
        "eurusd": "EURUSD=X",
        "vix": "^VIX",
        "us10y": "^TNX",
        "oil": "CL=F",
        "gold": "GC=F",
    }
    

    # ----------------------------------------
    # 3. SCARICA TUTTI I DATI
    # ----------------------------------------
    st.write("🚧Creazione database su Euribor 3mesi...🚧")
    st.write("""
    **Variabili incluse nel modello:**  
    
    📊 **Euribor / Money Market**
    - euribor 3m
    
    🏦 **Politica monetaria BCE**
    - deposit_rate
    - mro rate
    - marginal lending
    
    📈 **Macro**
    - inflation
    - core inflation
    - unemployment
    
    💰 **Banking & liquidity**
    - excess liquidity
    - deposit facility usage
    - refinancing ops
    - gdp growth
    
    💹 **Mercati finanziari (Yahoo)**
    - sp500
    - eurusd
    - vix
    - us10y
    - oil
    - gold
    """)
    
    st.subheader("📊 Trend analisi with Hybrid ML model 📊")
    results = {
    'Hybrid CB + OU': {
        'RMSE': 0.11,
        'MAE': 0.26,
        'R2': 0.83,
        'MAPE(%)': np.float64(8.13)
        }
    }

    results = pd.DataFrame(results).T 
    st.title("Hybrid ML model 📊 Results")
    st.table(results)
    
    df_ecb = download_ecb_series(series, start = '2021-01-01')
    st.dataframe(df_ecb)
    df_yahoo = download_yahoo_series(yahoo_symbols, start = '2021-01-01')
    st.dataframe(df_yahoo)
    df_all = df_ecb.join(df_yahoo, how="outer")
    df_all = df_all.sort_index().ffill()
    df_dropped = df_all.dropna()
    
       
    # ============================================================
    # STREAMLIT INTERFACCIA
    # ============================================================
    st.subheader("📊 Calcolo VaR 95% su Simulazioni Euribor 3M 📊")
    run_euribor = st.button("🚀 Simula Euribor 3M")
    n_sims = st.slider("Number of Monte Carlo simulations", min_value=100, max_value=100_000, value=10_000, step=100)
    
    if uploaded_file and run_euribor:
        tranche_df = pd.read_excel(uploaded_file, sheet_name="Tranches")
        plan_euribor_df = pd.read_excel(uploaded_file, sheet_name="Forward")
        tassi_impliciti = pd.read_excel(uploaded_file, sheet_name = 'Tassi_impliciti')
        tassi_impliciti['Data'] = pd.to_datetime(tassi_impliciti['Data'])
        
        plan_euribor_df["Anno"] = plan_euribor_df["Anno"].astype(int)
        
        st.subheader("📋 Tranche caricate dall’Excel")
        st.dataframe(tranche_df)
        series = df_dropped["euribor_3m"].values
        last_date = pd.to_datetime(df_dropped.index[-1])
    
        max_horizon_days = (pd.to_datetime(tranche_df['Maturity']).max() - last_date).days
    
        spread_df = pd.read_excel(uploaded_file, sheet_name="SpreadSchedule")
        spread_df["From"] = pd.to_datetime(spread_df["From"])
        spread_df["To"] = pd.to_datetime(spread_df["To"])
    
        # 1️⃣ Simulazione unica EURIBOR
        forecast_df, forecast_quarterly = simulate_euribor(series=series, df_dropped=df_dropped, n_sims=n_sims, horizon_days=max_horizon_days, plan_euribor_df=plan_euribor_df)
        
        st.subheader("Risultati simulazione Tassi - Euribor ")
        st.dataframe(forecast_quarterly)
        
        results_var = []
        
        # 2️⃣ Ciclo su tranche usando la simulazione unica
        for idx, row in tranche_df.iterrows():
            tranche_name = row.get("Tranche", f"T{idx+1}")
            unhedged = (row["Notional"] - row["Hedged"]) 
            # Taglio forecast fino alla maturità della tranche
            maturity_date = pd.to_datetime(row["Maturity"])
            forecast_tranche = forecast_quarterly[forecast_quarterly.index <= maturity_date]
            spread_series = forecast_tranche.index.map(lambda d: get_spread_for_date(d, spread_df))
            plan_euribor_series = forecast_tranche.index.map(lambda d: get_plan_euribor_for_date(d, plan_euribor_df))
            plan_rate = plan_euribor_series + spread_series
            var_rate = forecast_tranche["upper_adj"] + spread_series            
            var_amount = (var_rate/100) * unhedged
            plan_amount = (plan_rate/100) * unhedged
            days = forecast_tranche.index.to_series().diff().dt.days.fillna(90)
            var_cf = var_amount * (days / 360)
            plan_cf = plan_amount * (days / 360)
            kri_cashflow = np.max(var_cf-plan_cf,0)
        
            # DataFrame con indice corretto per la tranche
            df_var = pd.DataFrame({
                "Notional": row["Notional"],
                "Hedged": row["Hedged"],
                "Un-Hedged": unhedged,
                "Spread": spread_series.values,
                "Var Rate": var_rate,
                "Plan Rate": plan_rate,
                "Var Amount (€)": var_amount,
                "Var Cashflow (€)": var_cf,
                "Plan Amount (€)": plan_amount,
                "Plan Cashflow (€)": plan_cf,
                "Tranche": tranche_name
            }, index=forecast_tranche.index)
        
            results_var.append(df_var)
                
        # Concatenazione risultati
        final_var_df = pd.concat(results_var).reset_index()
        final_var_df["KRI Amount"] = final_var_df["Var Amount (€)"]- final_var_df["Plan Amount (€)"]
        final_var_df["KRI Cashflow"] = final_var_df["Var Cashflow (€)"]- final_var_df["Plan Cashflow (€)"]

        plan_series_plot = pd.Series(index=forecast_quarterly.index,data=[get_plan_euribor_for_date(d, plan_euribor_df) for d in forecast_quarterly.index])

        
        st.subheader("📊 Forecast Euribor 3M 📊 ")
        plt.figure(figsize=(15,6))
        # Serie storica
        plt.plot(df_dropped.index, df_dropped['euribor_3m'], label="Originale", color='black')
        
        # Forecast unico Monte Carlo (median e intervallo conformalizzato)
        plt.plot(forecast_quarterly.index, forecast_quarterly['median'], label='Mean Forecast', color='green', linestyle='--')
        plt.plot(plan_series_plot.index, plan_series_plot.values, label = 'Euribor 3m Piano', color = 'blue', linestyle= '-.')
        plt.plot(tassi_impliciti['Data'],tassi_impliciti['Tasso Forward'], label = 'Forward Euribor Fonte Bloomberg', color = 'red', linestyle= '-.')
        plt.fill_between(
            forecast_quarterly.index,
            forecast_quarterly['lower_adj'],
            forecast_quarterly['upper_adj'],
            color='red', alpha=0.2, label='Adjusted Interval (Conformal)'
        )
        
        plt.title("Serie storica + Forecast Monte Carlo EURIBOR 3M")
        plt.xlabel("Date")
        plt.ylabel("EURIBOR 3M")
        plt.ylim(0, 6)
        plt.legend()
        plt.grid(True)
        
        st.pyplot(plt.gcf())
        plt.close()

        plt.figure(figsize=(15,6))
        # Serie storica
        plt.plot(df_dropped.index, df_dropped['euribor_3m'], label="Originale", color='black')
        
        # Forecast unico Monte Carlo (median e intervallo conformalizzato)
        plt.plot(forecast_quarterly.index, forecast_quarterly['median'], label='Mean Forecast', color='green', linestyle='--')
        plt.plot(plan_series_plot.index, plan_series_plot.values, label = 'Euribor 3m Piano', color = 'blue', linestyle= '-.')
        plt.fill_between(
            forecast_quarterly.index,
            forecast_quarterly['lower_adj'],
            forecast_quarterly['upper_adj'],
            color='red', alpha=0.2, label='Adjusted Interval (Conformal)'
        )
        
        plt.title("Serie storica + Forecast Monte Carlo EURIBOR 3M")
        plt.xlabel("Date")
        plt.ylabel("EURIBOR 3M")
        plt.ylim(0, 6)
        plt.legend()
        plt.grid(True)
        
        st.pyplot(plt.gcf())
        plt.close()
        
        def to_millions(df, cols):
            df2 = df.copy()
            df2[cols] = df2[cols] / 1_000_000
            return df2
    
        cols_mln = ["Notional", "Hedged", "Un-Hedged", "Var Amount (€)", "Var Cashflow (€)", "Plan Amount (€)", 
                "Plan Cashflow (€)", "KRI Amount", "KRI Cashflow"]
    
        final_var_df_mln = to_millions(final_var_df, cols_mln)
    
        st.subheader("📊 Risultati VaR – per Tranche (in milioni €)")
        df_show = final_var_df_mln.copy()
        for c in cols_mln:
            df_show[c] = df_show[c].map(lambda x: f"{x:.3f}")
    
        st.dataframe(df_show)
    
        st.subheader("📊 Risultati VaR Annualizzati– per Tranche (in milioni €)")
        final_copy = final_var_df.copy()
        final_copy.rename(columns={'index': 'Date'}, inplace=True)
        final_copy['Date'] = pd.to_datetime(final_copy['Date'])
        final_copy = final_copy.set_index('Date')
        final_copy["Year"] = final_copy.index.year
        agg_rules = {
            "Var Cashflow (€)": "sum",
            "Plan Cashflow (€)": "sum",
            "KRI Cashflow": "sum",
            "Notional": "first",
            "Hedged": "first",
            "Un-Hedged": "first",
            "Var Rate": "mean",
            "Plan Rate": "mean",
            "Var Amount (€)": "mean",
            "Plan Amount (€)": "mean",
            "KRI Amount": "mean"}
        final_var_annual = final_copy.groupby(["Year", "Tranche"]).agg(agg_rules)
        final_var_annual = to_millions(final_var_annual, cols_mln)
        
    
        st.dataframe(final_var_annual)
        
        portfolio_var = final_var_df_mln.groupby('index')[[
            "Var Amount (€)", "Var Cashflow (€)", "KRI Amount", "KRI Cashflow", "Plan Cashflow (€)"
        ]].sum().reset_index()
    
        st.subheader("📈 VaR Cumulato di Portafoglio (in milioni €)")
        st.dataframe(portfolio_var)
    
        st.subheader("📊 Risultati VaR Annualizzati (in milioni €)")
        final_copy = final_var_df.copy()
        final_copy.rename(columns={'index': 'Date'}, inplace=True)
        final_copy['Date'] = pd.to_datetime(final_copy['Date'])
        final_copy = final_copy.set_index('Date')
        final_copy["Year"] = final_copy.index.year
        agg_rules = {
            "Var Cashflow (€)": "sum",
            "Plan Cashflow (€)": "sum",
            "KRI Cashflow": "sum",
            "Notional": "first",
            "Hedged": "first",
            "Un-Hedged": "first",
            "Var Rate": "mean",
            "Plan Rate": "mean",
            "Var Amount (€)": "mean",
            "Plan Amount (€)": "mean",
            "KRI Amount": "mean"}
        final_var_annual_no_tranche = final_copy.groupby(["Year"]).agg(agg_rules)
        final_var_annual_no_tranche = to_millions(final_var_annual_no_tranche, cols_mln)
        
        st.dataframe(final_var_annual_no_tranche)
    
        
        st.subheader("📉 Grafico VaR di Portafoglio (in milioni €)")
        st.line_chart(portfolio_var.set_index('index')[["Var Cashflow (€)", "Plan Cashflow (€)"]])
    
        st.subheader("💸⚠️ KRI Portafoglio💸⚠️ (in milioni €)")
        st.line_chart(portfolio_var.set_index('index')["KRI Cashflow"])
        
        # ============================================================
        # Perdita totale stimata su tutto l'orizzonte
        # ============================================================
        
        perdita_totale_mln = portfolio_var["KRI Cashflow"].sum()
        st.subheader("💸 Perdita Totale Stimata del Portafoglio (in milioni €)")
        st.metric(label="Perdita Totale (MLN €)", value=f"{perdita_totale_mln:.3f}")
        hedged_total = tranche_df['Hedged'].sum()
        notional_total = tranche_df['Notional'].sum()
        unhedged_total = notional_total-hedged_total
        perdita_totale_perc = np.round((perdita_totale_mln*1000000)/unhedged_total,3)
        st.metric(label="Perdita Totale % su Un-Hedged", value=f"{perdita_totale_perc*100} %")
    
        # Export Excel
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            final_var_df.to_excel(writer, index=True, sheet_name="Tranches")
            portfolio_var.to_excel(writer, index=True, sheet_name="Portfolio")
            final_var_annual_no_tranche.to_excel(writer, index=True, sheet_name="Portfolio-yearly")
        st.download_button(
            label="📥 Scarica risultati in Excel",
            data=output.getvalue(),
            file_name="VaR_multi_tranche.xlsx",
            mime="application/vnd.ms-excel"
        )
# -----------------------
# 💰Liquidity Risk
# -----------------------
elif selected_kri == "💰 Liquidity Risk":
    st.subheader("📊 Monthly KRI Liquidity 📊")
    run_liquidity  = st.button("🚀 Calcolo KRI Liquidity...")

    if uploaded_file and run_liquidity:
        # Caricamento dati
        input_df = pd.read_excel(uploaded_file, sheet_name='cash_monthly_data')
        input_plan_df = pd.read_excel(uploaded_file, sheet_name='bp_data')

        # Normalizza i nomi delle colonne: rimuove spazi invisibili e strip
        input_df.columns = [col.replace('\xa0', ' ').strip() for col in input_df.columns]
        input_plan_df.columns = [col.replace('\xa0', ' ').strip() for col in input_plan_df.columns]
        
        st.subheader("📋 Cash Flow data caricati dall’Excel")

        # Funzione per calcolo fonti finanziamento
        def fonti_finanziamento(row, check_col, sum_cols, alt_cols):
            if row[check_col] > 0:
                return row[sum_cols].sum()
            else:
                return row[alt_cols].sum()
        
        input_df['Escrow Account'] = pd.to_numeric(input_df['Escrow Account'], errors='coerce').fillna(0)
        input_df['Num1'] = input_df.apply(
            lambda row: fonti_finanziamento(
                row,
                'Escrow Account',
                ['Debt drawings (RCF, Loan, Bond)','Escrow Account','Cash avaible net Time Depo'],
                ['Debt drawings (RCF, Loan, Bond)','Cash avaible net Time Depo']
            ),
            axis=1
        )

        input_df['Deno1'] = input_df[['Loan Repayments','Derivative Settlements (CCS & IRS)','Coupon','EUR Interest Payments']].abs().sum(axis=1)

        cols_operativi = ['Suppliers -Opex/Capex', 'Others Cost', 'Factoring Suppliers Opex/Capex',
                          'Salaries/Payroll', 'HR Others Cost (Telemaco…)', 'Inps/Irpef Contributions',
                          'Rents and property costs', 'VAT', 'Corporate Taxes (IRES/IRAP)', 'Guarantees Cost']

        input_df['Deno2'] = input_df['Deno1'] + input_df[cols_operativi].abs().sum(axis=1)

        # Calcolo indicatori
        input_df['Indicatore 12m'] = np.where(input_df['Deno1'] > 0, input_df['Num1'] / input_df['Deno1'], np.nan)
        input_df['Liquidity Coverage Ratio (con spese operative)'] = np.where(input_df['Deno2'] > 0, input_df['Num1'] / input_df['Deno2'], np.nan)

        # Selezione sicura delle colonne per visualizzazione
        cols_to_show = ['Indicatore 12m', 'Liquidity Coverage Ratio (con spese operative)']
        existing_cols = [col for col in cols_to_show if col in input_df.columns]
        if existing_cols:
            st.dataframe(input_df[existing_cols])
        else:
            st.warning("⚠️ Le colonne richieste non sono presenti nel DataFrame.")

        import plotly.express as px

        # Controllo se la colonna 'M/€' esiste
        if 'M/€' in input_df.columns:
            fig = px.line(
                input_df,
                x='M/€',
                y=existing_cols,
                labels={'variable': 'KRI'},
                title="📈 Liquidity Risk KRI Monthly",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Mese",
                legend_title="KRI",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ La colonna 'M/€' non è presente nel DataFrame.")

        # -----------------------------
        # Copertura Opex/Capex e Liquidity Margin
        # -----------------------------


        # Calcoli EBITDA e Operating Free Cash Flow
        input_plan_df["EBITDA_org_cash@Risk"] = input_plan_df["EBITA@Risk"] + input_plan_df["Cash adjustments"]
        input_plan_df["EBITDA_org_cash"] = input_plan_df["EBITDAaL organic IFRS"] + input_plan_df["Cash adjustments"]  
        input_plan_df["EBITDAaL cash@Risk"] = input_plan_df["EBITDA_org_cash@Risk"] + input_plan_df["One-off cash"]
        input_plan_df["EBITDAaL cash"] = input_plan_df["EBITDA_org_cash"] + input_plan_df["One-off cash"]

        input_plan_df["Operating Free Cash Flow pre-tax@Risk"] = (
            input_plan_df["EBITDAaL cash@Risk"]
            + input_plan_df["Capex"]
            + input_plan_df['PNRR subsidies']
            + input_plan_df["Change in Working Capital"]
            + input_plan_df["Change in TFR"]
            + input_plan_df["Change in Commercial Basket"]
            + input_plan_df["Change in ARO fund"]
        )

        input_plan_df["Operating Free Cash Flow pre-tax"] = (
            input_plan_df["EBITDAaL cash"]
            + input_plan_df["Capex"]
            + input_plan_df['PNRR subsidies']
            + input_plan_df["Change in Working Capital"]
            + input_plan_df["Change in TFR"]
            + input_plan_df["Change in Commercial Basket"]
            + input_plan_df["Change in ARO fund"]
        )

        input_plan_df["Operating Free Cash Flow post-tax  (ultim 3 mesi per 2025)@Risk"] = input_plan_df["Operating Free Cash Flow pre-tax@Risk"] + input_plan_df["Cash Taxes"]
        input_plan_df["Operating Free Cash Flow post-tax  (ultim 3 mesi per 2025)"] = input_plan_df["Operating Free Cash Flow pre-tax"] + input_plan_df["Cash Taxes"]

        input_plan_df.loc[input_plan_df.index[0], 'Operating Free Cash Flow post-tax  (ultim 3 mesi per 2025)@Risk'] *= 3 / 12
        input_plan_df.loc[input_plan_df.index[0], 'Operating Free Cash Flow post-tax  (ultim 3 mesi per 2025)'] *= 3 / 12
        
        input_plan_df["Totale@Risk"] = (
            input_plan_df["Operating Free Cash Flow post-tax  (ultim 3 mesi per 2025)@Risk"]
            + input_plan_df["Interest expenses (fixed al 2025)"]
            + input_plan_df["Dividendi"]
        )

        input_plan_df["Totale"] = (
            input_plan_df["Operating Free Cash Flow post-tax  (ultim 3 mesi per 2025)"]
            + input_plan_df["Interest expenses (fixed al 2025)"]
            + input_plan_df["Dividendi"]
        )

        # Liquidity Margin con STOP
        initial_liquidity = st.number_input(
            "Initial Liquidity (€m)",
            min_value=0.0,
            value=5491.0,
            step=50.0,
            format="%.0f"
        )

        def liquidity_margin(series, initial):
            result = []
            current = initial
            for v in series:
                current += v
                if current <= 0:
                    result.append(np.nan)
                    break
                result.append(current)
            return result + [np.nan] * (len(series) - len(result))

        input_plan_df["Liquidity_Margin@Risk"] = liquidity_margin(input_plan_df["Totale@Risk"], initial_liquidity)
        input_plan_df["Liquidity_Margin"] = liquidity_margin(input_plan_df["Totale"], initial_liquidity)
        st.dataframe(input_plan_df)
        # Grafico Liquidity Margin
        y_base = input_plan_df["Liquidity_Margin"]
        y_risk = input_plan_df["Liquidity_Margin@Risk"]
        floor_value = 650
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = input_plan_df['Time']
        ax.plot(x, y_base, linewidth=3, label="Liquidity Margin")
        ax.plot(x, y_risk, linewidth=3, label="Liquidity Margin @Risk")
        ax.plot(x, [floor_value]*len(x), linestyle=":", linewidth=3, label="Floor (650 mln)")
        ax.set_title("Liquidity Scenario", fontsize=16)
        ax.set_xlabel("Time")
        ax.set_ylabel("€m")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Export Excel
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            input_df.to_excel(writer, index=True, sheet_name="Liquidity Coverage Ratio")
            input_plan_df.to_excel(writer, index=True, sheet_name="Liquidity Coverage Ratio2")
            st.download_button(
                label="📥 Scarica risultati in Excel",
                data=output.getvalue(),
                file_name="KRI_Liquidity_Risk.xlsx",
                mime="application/vnd.ms-excel"
            )
# -----------------------
# Ebitda @Risk 📊📈
# -----------------------
elif selected_kri == "📊📈 Ebitda @Risk":
    if uploaded_file:
        df = load_risk_factors(uploaded_file)
        #st.write("📄 Anteprima dati:", df.head())
    
        blocks = parse_factors(df)
    
        anni = sorted(df['anno'].dropna().unique())
    
        n_years = len(anni)
        start_year = anni[0]
    
        
        unique_blocks = {b["name"]: b for b in blocks}.values()
    
        activation_df = pd.DataFrame({
        "Blocco": [b["name"] for b in unique_blocks],
        **{str(anno): [True] * len(unique_blocks) for anno in anni}
        })
    
        st.subheader("Seleziona i fattori di rischio per anno")
        activation_df = st.data_editor(activation_df, use_container_width=True)
        # Questo dict sarà Blocco -> {anno: True/False}
        activation_dict = activation_df.set_index("Blocco").T.to_dict()
        
        
        st.subheader("Inserisci EBITDA a piano per ogni anno")
        
        ebitda_base_dict = {}
        ebitda_base_list = []
        for anno in anni:
            ebitda_anno = st.number_input(
                f"EBITDA a piano anno {anno}",
                value=1_000_000.0,
                step=10_000.0,
                format="%.2f",
                key=f"ebitda_{anno}"
            )
            ebitda_base_dict[anno] = ebitda_anno
            ebitda_base_list.append(ebitda_anno)
            
        n_sim = st.number_input("Numero di simulazioni", min_value=100, max_value=1000_000, value=1_000, step=100)
        # Selectbox più chiaro con gli anni reali
        anno_inizio_k_label = st.selectbox(
        "Anno da cui applicare l'incertezza sui fattori",
        options=anni,
        index=0
        )
    
        # Converti l’anno selezionato nel relativo indice (1-based)
        anno_inizio_k = anni.index(anno_inizio_k_label) + 1
    
        st.subheader("🔧 Imposta il trend di incertezza da applicare ai fattori")
        trend = st.selectbox(
            "Trend dell'incertezza sui fattori di rischio",
            options=["costante", "lineare", "moltiplicativo"],
            index=0
        )
        st.subheader("⚡ Eventi strategici legati ai fattori di rischio")
        
        # Estrai anni disponibili dai blocchi
        anni = sorted({b['anno'] for b in blocks})
        anni_str = [str(a) for a in anni]
        
        # Rimuovi duplicati per nome
        unique_blocks = {b['name']: b for b in blocks}.values()
        
        # Parametri globali default
        default_lambda = 0.2
        default_magnitudo = 0.2
        default_segno = "Negativo"
        
        # Parametri shock per ciascun fattore
        st.markdown("### Configura gli eventi strategici")
        
        shock_event_config = []
        
        for i, block in enumerate(unique_blocks):
            with st.expander(f"⚙️ {block['name']}", expanded=False):
                # Checkbox per abilitare lo shock
                attivo = st.checkbox("Abilita shock su questo fattore", key=f"shock_enable_{i}")
        
                # Anni in cui può essere attivo (solo se attivo)
                anni_attivi = {}
                if attivo:
                    st.markdown("**Anni in cui applicare lo shock:**")
                    col_check = st.columns(len(anni_str))
                    for j, anno in enumerate(anni_str):
                        anni_attivi[anno] = col_check[j].checkbox(anno, value=False, key=f"{block['name']}_{anno}")
        
                    # Parametri shock (solo se attivo)
                    lambda_poisson = st.slider("λ (frequenza evento shock)", min_value=0.0, max_value=1.0, value=default_lambda, step=0.01, key=f"lambda_{i}")
                    magnitudo = st.slider("Magnitudo dell'impatto", min_value=0.0, max_value=1.0, value=default_magnitudo, step=0.01, key=f"magnitudo_{i}")
                    segno = st.selectbox("Segno dell'impatto", ["Negativo", "Positivo"], index=0, key=f"segno_{i}")
                else:
                    lambda_poisson = 0.0
                    magnitudo = 0.0
                    segno = "Negativo"
        
                # Salva configurazione
                shock_event_config.append({
                    "name": block['name'],
                    "attivo": attivo,
                    "anni_attivi": anni_attivi,
                    "lambda": lambda_poisson,
                    "magnitudo": magnitudo,
                    "segno": segno
                })
        
        # Costruisci dizionario finale per uso nella simulazione
        shock_event_dict = {
            cfg["name"]: {
                "anni_attivi": cfg["anni_attivi"],
                "lambda": cfg["lambda"],
                "magnitudo": cfg["magnitudo"],
                "segno": cfg["segno"]
            }
            for cfg in shock_event_config if cfg["attivo"]
        }
    
        st.subheader("⚡ Eventi esogeni (Macroeconomici) su fattori di rischio")
        
        lambda_shock = st.slider("Frequenza media shock (λ)", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
        magnitudo_shock = st.slider("Magnitudo shock", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
        
        # Inizializza lo stato se non esiste
        if "shock_activated" not in st.session_state:
            st.session_state.shock_activated = False
        
        # Usa direttamente il valore del checkbox per aggiornare lo stato
        attiva_shock = st.checkbox("Attiva Shock esogeni sull'EBITDA", value=st.session_state.shock_activated)
        
        st.session_state.shock_activated = attiva_shock
        
        if st.button("▶️ Esegui simulazione"):
            #risultati = simulate_ebitda_multi_year(
            #    blocks= blocks,
            #    ebitda_base_list= ebitda_base_list,
            #    n_sim= n_sim,
            #    n_years= n_years,
            #    anno_inizio_k= anno_inizio_k,
            #    trend=trend
            #)
            
            #risultati = simulate_ebitda_multi_year(blocks, ebitda_base_list, n_sim=n_sim, n_years=n_years,
            #                               anno_inizio_k=anno_inizio_k, trend=trend,
            #                               attiva_shock=st.session_state.shock_activated, lambda_shock_annuo=lambda_shock, magnitudo_shock=magnitudo_shock
            
            
            
            print(f'blocchi {blocks}')
                                           
            risultati, cor_matrix_by_year, ricavi_negativi_records, df_parametri_simulati =  simulate_ebitda_multi_year_blocks_with_ricavi(
            blocks=blocks,
            ebitda_base_list=ebitda_base_list,
            n_sim=n_sim,
            anni = anni,
            anno_inizio_k=anno_inizio_k,
            trend=trend,
            activation_matrix=activation_dict,
            attiva_shock=st.session_state.shock_activated,
            lambda_shock_annuo=lambda_shock,
            magnitudo_shock= magnitudo_shock,
            shock_event_dict= shock_event_dict # 👈 NUOVO DIZIONARIO
                )
                
            # st.write("Simulazione completata!")
    
            #import io
        # Creo un buffer in memoria per salvare il dataframe Excel
            #output = io.BytesIO()
            #with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            #    df_parametri_simulati.to_excel(writer, sheet_name='ParametriSimulati')
            #    output.seek(0)
    
        # Bottone per il download
            #st.download_button(
            #    label="📥 Scarica Excel dei parametri simulati",
            #    data=output,
            #    file_name='parametri_simulati.xlsx',
            #   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            #)        
    
            for res in risultati:
                anno = res['anno']
                sim = res['ebitda_simulazioni']
                st.subheader(f"📈 Anno {anno}")
                st.write(f"Media EBITDA simulata: {np.mean(sim):,.2f}")
                st.write(f"Deviazione standard: {np.std(sim):,.2f}")
                fig_dist = px.histogram(sim, nbins=100, title=f"Distribuzione EBITDA simulata anno {anno}")
                st.plotly_chart(fig_dist, use_container_width=True)
    
            st.subheader("📊 Grafici riepilogativi delle simulazioni")
    
            anni = list(range(2025, 2025 + len(risultati)))
            mean_ebitda = [np.mean(r["ebitda_simulazioni"]) for r in risultati]
            p5_ebitda = [np.percentile(r["ebitda_simulazioni"], 5) for r in risultati]
            p95_ebitda = [np.percentile(r["ebitda_simulazioni"], 95) for r in risultati]
    
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=anni, y=mean_ebitda, mode='lines+markers', name='Media EBITDA', line=dict(color='navy')))
            fig1.add_trace(go.Scatter(x=anni + anni[::-1], y=p95_ebitda + p5_ebitda[::-1], fill='toself',
                                      fillcolor='rgba(135, 206, 250, 0.4)', line=dict(color='rgba(255,255,255,0)'),
                                      hoverinfo="skip", showlegend=True, name='Intervallo 5°-95°'))
            fig1.add_trace(go.Scatter(x=anni, y=ebitda_base_list, mode='lines+markers', name='EBITDA di Piano',
                                      line=dict(color='gray', dash='dash')))
            fig1.update_layout(title="Andamento medio EBITDA con intervallo di confidenza e piano",
                               xaxis_title="Anno", yaxis_title="EBITDA", xaxis=dict(tickmode='linear'), template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)
            
            anni = list(range(2025, 2025 + len(risultati)))
            mean_ebitda = [np.mean(r["ebitda_simulazioni"]) for r in risultati]
            p5_ebitda = [np.percentile(r["ebitda_simulazioni"], 5) for r in risultati]
            p95_ebitda = [np.percentile(r["ebitda_simulazioni"], 95) for r in risultati]
            ebitda_base = ebitda_base_list
            
            # Trova gli anni con shock (>=1)
            anni_con_shock = [anno for anno, r in zip(anni, risultati) if r.get("shock_ebitda", 0) > 0]
            media_con_shock = [np.mean(r["ebitda_simulazioni"]) for r in risultati if r.get("shock_ebitda", 0) > 0]
            
            fig_shock = go.Figure()
            
            # Linea media EBITDA
            fig_shock.add_trace(go.Scatter(
                x=anni,
                y=mean_ebitda,
                mode='lines+markers',
                name='Media EBITDA',
                line=dict(color='navy')
            ))
            
            # Banda di confidenza 5°-95°
            fig_shock.add_trace(go.Scatter(
                x=anni + anni[::-1],
                y=p95_ebitda + p5_ebitda[::-1],
                fill='toself',
                fillcolor='rgba(135, 206, 250, 0.4)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Intervallo 5°-95°'
            ))
            
            # EBITDA di piano
            fig_shock.add_trace(go.Scatter(
                x=anni,
                y=ebitda_base,
                mode='lines+markers',
                name='EBITDA di Piano',
                line=dict(color='gray', dash='dash')
            ))
            
            # Marker per gli shock esterni
            if anni_con_shock:
                fig_shock.add_trace(go.Scatter(
                    x=anni_con_shock,
                    y=media_con_shock,
                    mode='markers',
                    name='Shock Esterni',
                    marker=dict(color='red', size=12, symbol='x'),
                    hovertemplate='Anno %{x}<br>Shock Esterno<br>Media EBITDA: %{y:.2f}<extra></extra>'
                ))
            
            fig_shock.update_layout(
                title="Andamento medio EBITDA con intervallo di confidenza, piano e shock esterni",
                xaxis_title="Anno",
                yaxis_title="EBITDA",
                xaxis=dict(tickmode='linear'),
                template='plotly_white'
            )
            
            st.plotly_chart(fig_shock, use_container_width=True) 
            
            # Crea la tabella degli anni con shock esogeni
            tabella_shock = pd.DataFrame([
            {
            "Anno": str(r["anno"]),
            "Shock_EBITDA": r.get("shock_ebitda", 0),
            "Media EBITDA": f"€{np.mean(r['ebitda_simulazioni']):,.2f}"
            }
            for r in risultati if r.get("shock_ebitda", 0) > 0
            ])
    
            if not tabella_shock.empty:
                st.subheader("📉 Anni con shock esogeni")
                st.dataframe(tabella_shock)
            else:
                st.info("✅ Nessuno shock esogeno si è verificato nel modello probabilistico.")
            
            
    
            df_box = pd.DataFrame({
                "Anno": np.repeat(anni, n_sim),
                "EBITDA": np.concatenate([r["ebitda_simulazioni"] for r in risultati])
            })
            fig2 = px.box(df_box, x="Anno", y="EBITDA", title="Distribuzione EBITDA simulato per anno")
            fig2.update_layout(template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)
            
                   
    
            st.subheader("🔝 Top 10 correlazioni tra fattori di rischio per anno (grafico)")
            
            for anno, data in cor_matrix_by_year.items():
                corr_matrix = data["matrice"]
                fattori = data["fattori"]
            
                if not isinstance(corr_matrix, pd.DataFrame):
                    corr_matrix = pd.DataFrame(corr_matrix, index=fattori, columns=fattori)
            
                top_corr_pairs = get_top_correlations(corr_matrix, top_n=10)
                fig = plot_top_corr_bar(top_corr_pairs, anno)
                st.plotly_chart(fig, use_container_width=True)
            
    
            tornado_per_anno, importanza_totale = calcola_importanza_fattori(risultati)
    
            st.subheader("🌪️ Importanza fattori di rischio per anno")
            
            for entry in tornado_per_anno:
                anno = entry['anno']
                importanza = entry['importanza'].sort_values(ascending=True)
                fig_tornado = go.Figure(go.Bar(
                    x=importanza.values,
                    y=importanza.index,
                    orientation='h',
                    marker_color='salmon'
                ))
                fig_tornado.update_layout(
                    title=f"Anno {anno} - Importanza fattori di rischio",
                    xaxis_title="Importanza relativa",
                    yaxis_title="Fattori di rischio",
                    template='plotly_white'
                )
                st.plotly_chart(fig_tornado, use_container_width=True)
    
            st.subheader("📊 Importanza aggregata fattori di rischio")
            importanza_totale = importanza_totale.sort_values(ascending=True)
            fig_totale = go.Figure(go.Bar(
                x=importanza_totale.values,
                y=importanza_totale.index,
                orientation='h',
                marker_color='mediumseagreen'
            ))
            fig_totale.update_layout(
                title="Importanza aggregata fattori di rischio",
                xaxis_title="Importanza relativa",
                yaxis_title="Fattori di rischio",
                template='plotly_white'
            )
            st.plotly_chart(fig_totale, use_container_width=True)
    
            percentili = [5, 50, 95]
            color_map = {5: 'red', 50: 'blue', 95: 'green'}
    
            fig_percentili = go.Figure()
            for p in percentili:
                y_p = [np.percentile(r["ebitda_simulazioni"], p) for r in risultati]
                fig_percentili.add_trace(go.Scatter(
                    x=anni,
                    y=y_p,
                    mode='lines+markers',
                    name=f'Percentile {p}',
                    line=dict(width=2, dash='solid', color=color_map[p])
                ))
            fig_percentili.add_trace(go.Scatter(
                x=anni,
                y=ebitda_base_list,
                mode='lines+markers',
                name='EBITDA di Piano',
                line=dict(color='gray', dash='dash')
            ))
            fig_percentili.update_layout(
                title="Percentili 5° - 50° - 95° EBITDA per anno simulato",
                xaxis_title="Anno",
                yaxis_title="EBITDA",
                template='plotly_white'
            )
            st.plotly_chart(fig_percentili, use_container_width=True)
                        
            st.subheader("📊 Eventi rilevanti per fattore di rischio")
            
            # 🔧 Costruzione corretta dati flat
            shock_data = []
            
            for entry in risultati:
                anno = entry.get("anno")
                shock_dict = entry.get("shock_occorrenze", {})
            
                for fattore, shock in shock_dict.items():
                    shock_data.append({
                        "Anno": anno,
                        "Fattore": fattore,
                        "Shock": 1 if shock else 0
                    })
            
            # 📊 DataFrame
            df_shock = pd.DataFrame(shock_data)
            
            # 🚨 Se vuoto → stop
            if df_shock.empty:
                st.info("Nessun dato disponibile")
                st.stop()
            
            # 🔄 Pivot
            df_pivot = df_shock.pivot_table(
                index="Fattore",
                columns="Anno",
                values="Shock",
                aggfunc="max"
            ).fillna(0)
            
            # 🔤 ✓ / x
            def simbolo(x):
                return "✓" if x == 1 else "x"
            
            df_tabella = df_pivot.map(simbolo)
            
            # 🎨 Stile (il tuo)
            def stile_simbolo(val):
                if val == "✓":
                    return "color: green; font-weight: bold"
                elif val == "x":
                    return "color: red; font-weight: bold"
                return ""
            
            df_styled = df_tabella.style.map(stile_simbolo)
            
            # 📺 Output
            st.dataframe(df_styled)
               
            st.subheader("🌪️ Tornado chart per fattori di rischio (percentili 5°-95°)")
    
            fattori = sorted(set().union(*[e["fattori_simulati"].keys() for e in risultati]))
            
            data = []
            for entry in risultati:
                anno = entry["anno"]
                for fatt in fattori:
                    sim_vals = entry["fattori_simulati"].get(fatt)
                    if sim_vals is not None:
                        p5 = np.percentile(sim_vals, 5)
                        p95 = np.percentile(sim_vals, 95)
                        data.append({"Anno": anno, "Fattore": fatt, "Valore": p5, "Tipo": "5° Percentile"})
                        data.append({"Anno": anno, "Fattore": fatt, "Valore": p95, "Tipo": "95° Percentile"})
                    else:
                        st.info(f"ℹ️ Fattore '{fatt}' disattivato nell'anno {entry['anno']}")
                        
            df_tornado = pd.DataFrame(data)
            df_tornado["Valore_milioni"] = df_tornado["Valore"] / 1_000_000
            fig_tornado_ts = px.bar(
                df_tornado,
                x="Valore",
                y="Fattore",
                color="Tipo",
                facet_col="Anno",
                orientation='h',
                barmode="overlay",
                color_discrete_map={"5° Percentile": "red", "95° Percentile": "green"},
                title="Dispersione dei fattori di rischio (5° vs 95° percentile) per anno",
                text=df_tornado["Valore_milioni"].apply(lambda x: f"{x:.2f}Mln") 
            )
            fig_tornado_ts.update_layout(template="plotly_white", showlegend=True, height=800, width=1200)
            st.plotly_chart(fig_tornado_ts, use_container_width=True)
            
            #plot_k_min_max_plotly(blocks)
    
            excel_data = genera_output_excel(risultati, ebitda_base_dict, df_tabella)
    
            st.download_button(
                label="📥 Esporta risultati",
                data=excel_data,
                file_name="risultati_ebitda.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    else:
        st.info("Carica un file Excel per iniziare la simulazione.")
   
        
    
    

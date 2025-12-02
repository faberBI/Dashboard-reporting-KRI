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
import pickle
from datetime import datetime
from ecbdata import ecbdata
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf


# Library custom
from utils.data_loader import load_kri_excel, validate_kri_data
from functions.energy_risk import (historical_VaR, run_heston, analyze_simulation, compute_downside_upperside_risk, var_ebitda_risk)
from functions.copper import simulate_cb_egarch_outsample, get_forecast_plot
from functions.geospatial import (get_risk_area_frane, get_risk_area_idro, get_magnitudes_for_comune)

# -----------------------
# Configurazione Streamlit
# -----------------------

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
kri_options = ["⚡ Energy Risk", "🌪️ Natural Event Risk", "🟠 Copper Price", "💰🔑 Access to Funding", "🛡️💻 Cyber","💳 Credit risk" ,"📈 Interest Rate"]

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
                "Anno": [2025, 2026, 2027],
                "Fabbisogno": [1548, 1557, 1373],
                "Covered": [1408.6, 933.9, 619],
                "Solar": [0, 203, 422],
                "Forward Price": [115.99, 106.85, 94.00],
                "Budget Price": [115, 121, 120]
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
    st.subheader("📌 Parametri di simulazione Energy Risk")

    # Parametri input manuale
    def df_to_str(df, col_name, default):
        if col_name in df.columns:
            values = df[col_name].dropna().tolist()
            if len(values) > 0:
                return ",".join(map(str, values))
        return default

    fabbisogno = st.text_input("Fabbisogno (MWh)", df_to_str(df, "Fabbisogno", "1548,1557,1373"))
    covered = st.text_input("Covered (MWh)", df_to_str(df, "Covered", "1408.6,933.9,619"))
    solar = st.text_input("Solar (MWh)", df_to_str(df, "Solar", "0,203,422"))
    forward_price = st.text_input("Forward Price (€)", df_to_str(df, "Forward Price", "115.99,106.85,94.00"))
    budget_price = st.text_input("Budget Price (€)", df_to_str(df, "Budget Price", "115,121,120"))
    
    st.subheader("💰 Inserisci o modifica EBITDA per anno")
    # Verifica che il DataFrame non sia vuoto
    if df.empty:
        st.warning("⚠️ Nessun dato disponibile nel DataFrame!")
    else:
        # Se la colonna Ebitda non esiste, la aggiunge con un valore predefinito
        if "Ebitda" not in df.columns:
            df["Ebitda"] = [1_900_000_000] * len(df)

        # Dizionario per i valori inseriti
        ebitda_inputs = {}

        # Crea un campo numerico per ogni anno
        for i, row in df.iterrows():
            anno = int(row["Anno"]) if "Anno" in df.columns else (2025 + i)
            default_value = float(row["Ebitda"])

            ebitda_inputs[anno] = st.number_input(
                f"EBITDA per {anno} (€)",
                min_value=0.0,
                value=default_value,
                step=1_000_000.0,
                format="%.0f"
            )

    # Aggiorna la colonna Ebitda con i valori inseriti
        df["Ebitda"] = [ebitda_inputs[anno] for anno in df["Anno"]]

    # Mostra il DataFrame aggiornato
        st.dataframe(df.style.format({"Ebitda": "€{:,.0f}"}))
    
    # Parsing input
    try:
        fabbisogno = [float(x) for x in fabbisogno.split(",")]
        covered = [float(x) for x in covered.split(",")]
        solar = [float(x) for x in solar.split(",")]
        forward_price = [float(x) for x in forward_price.split(",")]
        budget_price = [float(x) for x in budget_price.split(",")]
        #ebitda = [float(x) for x in ebitda.split(",")]
    except Exception as e:
        st.error(f"❌ Errore nei parametri: {e}")
        st.stop()

    if not (len(fabbisogno) == len(covered) == len(solar) == len(forward_price) == len(budget_price)):
        st.error("⚠️ Tutti i parametri devono avere lo stesso numero di valori per anno.")
        st.stop()

    st.success("✅ Parametri validi, pronti per la simulazione!")

    # -----------------------
    # Parametri simulazione
    # -----------------------
    n_simulations = st.number_input("Numero di simulazioni", min_value=100, max_value=100_000, value=10_000, step=100)
    n_trials_heston = st.number_input("Numero di trial Heston", min_value=10, max_value=1000, value=100, step=10)
    end_date = st.date_input("Data finale simulazione", pd.to_datetime("2027-12-31"))
    start_date = st.date_input("Dati aggiornati al", pd.Timestamp.today().date())
    start_date_sim = pd.Timestamp.today().normalize()

    days_to_simulate = (pd.to_datetime(end_date) - pd.to_datetime(start_date_sim)).days
    future_dates = pd.date_range(start=start_date_sim, periods=days_to_simulate, freq='D')
    unique_years = sorted(future_dates.year.unique().tolist())

    # -----------------------
    # Pulsante Simulazione
    # -----------------------
    if st.button("💹 Esegui simulazione Energy Risk"):
        st.info("Simulazione in corso...")

        # Carica file Excel PUN
        data_path = "Data/Pun 10_04_2025.xlsx"
        df_excel = None
        if os.path.exists(data_path):
            df_excel = pd.read_excel(data_path)
            st.success("🔄 Dati Caricati")

        if df_excel is None or df_excel.empty:
            uploaded_file = st.file_uploader("Seleziona il file Excel PUN", type=["xlsx"])
            if uploaded_file:
                df_excel = pd.read_excel(uploaded_file)
            else:
                st.warning("Carica il file Excel per procedere con la simulazione.")
                st.stop()

        if 'Date' not in df_excel.columns or 'GMEPIT24 Index' not in df_excel.columns:
            st.error("Il file Excel deve contenere 'Date' e 'GMEPIT24 Index'.")
            st.stop()

        df_excel['Date'] = pd.to_datetime(df_excel['Date'])
        df_excel['Log_Returns'] = np.log(df_excel['GMEPIT24 Index'] / df_excel['GMEPIT24 Index'].shift(1))
        df_excel = df_excel.dropna(subset=['Log_Returns'])
        st.session_state.energy_df = df_excel

        df_filtered = df_excel
        if df_filtered.empty:
            st.error("Il filtro ha prodotto un DataFrame vuoto")
            st.stop()

        # ---------------------------
        # Simulazione Heston
        # ---------------------------
        _, simulated_prices = run_heston(
            df_filtered,
            n_trials=n_trials_heston,
            n_simulations=n_simulations,
            end_date=end_date
        )

        future_dates_sim = pd.date_range(
            start=df_filtered['Date'].max(),
            periods=(pd.to_datetime(end_date) - df_filtered['Date'].max()).days,
            freq='D'
        )
        simulated_df = pd.DataFrame(
            simulated_prices.T,
            index=future_dates_sim,
            columns=[f"Simulazione {i+1}" for i in range(n_simulations)]
        )
        simulated_df = simulated_df.mask((simulated_df < 40) | (simulated_df >= 350))

        monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig = analyze_simulation(
            simulated_df, unique_years, forward_prices=forward_price)
        st.pyplot(fig)

        # -----------------------
        # Forecast + storico
        # -----------------------
        forecast_price = pd.DataFrame.from_dict(yearly_percentiles, orient='index', columns=['5%', '50%', '95%'])
        st.markdown("### 📊 Forecast Output")
        st.info("Questi sono i valori previsionali basati sui percentili annuali.")
        def format_euro(x): return f"€ {x:.2f}" if pd.notnull(x) else ""
        st.dataframe(forecast_price.style.format({col: format_euro for col in forecast_price.columns}).background_gradient(cmap='Greens', low=0.1, high=0.4))

        anni_prezzi = [2020, 2021, 2022, 2023, 2024] + unique_years
        anni_prezzi = [int(y) for y in anni_prezzi]

        historical_price = df_filtered.groupby(df_filtered['Date'].dt.year)['GMEPIT24 Index'].mean().tail(6).values.tolist()
        df_historical = pd.DataFrame({"Historical Price": historical_price, "Year": anni_prezzi[:len(historical_price)]})
        df_hist_styled = df_historical.style.format({"Historical Price": format_euro}).background_gradient(cmap='Greens', low=0.1, high=0.4)
        st.session_state.df_historical = df_historical
        st.dataframe(df_hist_styled)

        predict_price = forecast_price['50%'].values.tolist()
        p95 = forecast_price['95%'].values.tolist()
        p5 = forecast_price['5%'].values.tolist()

        forward_price_full = forward_price.copy()
        budget_price_full = [0]*(len(anni_prezzi)-len(budget_price)) + budget_price

        # Allinea lunghezze
        historical_price = historical_price + [0]*(len(anni_prezzi)-len(historical_price))
        predict_price = [0]*(len(anni_prezzi)-len(predict_price)) + predict_price
        p95 = [0]*(len(anni_prezzi)-len(p95)) + p95
        p5 = [0]*(len(anni_prezzi)-len(p5)) + p5
        forward_price_full = [0]*(len(anni_prezzi)-len(forward_price_full)) + forward_price_full

        # -----------------------
        # Calcolo Open Position e Risk
        # -----------------------
        df_risk, df_open, df_prezzi, df_target_policy, fig = compute_downside_upperside_risk(
            anni=df["Anno"].tolist(),
            fabbisogno=fabbisogno,
            covered=covered,
            solar=solar,
            anni_prezzi=anni_prezzi,
            media_pun=historical_price,
            predictive=predict_price,
            p95=p95,
            p5=p5,
            frwd=forward_price_full,
            budget=budget_price_full,
            observation_period=start_date_sim.strftime("%d/%m/%Y")
        )
        #st.markdown("### ⚠️ Open position ")
        df_open['Open Position Value (€)'] = df_open['Open Position'].values * budget_price* 1000
        df_open['Open Position Value No Solar (€)'] = df_open['Open Position (no solar)'].values * budget_price* 1000
        #st.dataframe(df_open)
        
        st.pyplot(fig)
        st.markdown("### ⚠️ Analisi Rischio (Downside / Upside)")
        st.info("Valori di rischio calcolati in base alle differenze tra percentili, budget e open position.")

        # Copia del DataFrame per styling
        df_styled = df_risk.style
        
        # Colonne da escludere dalla formattazione in milioni
        exclude_cols = ["Year", "Anno", "year", "anno"]
        
        # Colonne da formattare in milioni di euro
        cols_to_format = [c for c in df_risk.columns if c not in exclude_cols]
        
        # Applica rosso tenue alle colonne "Downside"
        downside_cols = [c for c in df_risk.columns if c.startswith("Downside")]
        if downside_cols:
            df_styled = df_styled.background_gradient(
                cmap='Reds', low=0.1, high=0.4, subset=downside_cols
            )
        
        # Applica verde tenue alle colonne "Upside"
        upside_cols = [c for c in df_risk.columns if c.startswith("Upside")]
        if upside_cols:
            df_styled = df_styled.background_gradient(
                cmap='Greens', low=0.1, high=0.4, subset=upside_cols
            )
        
        # Funzione di formattazione in milioni di euro
        def format_mln_euro(x):
            return f"€ {x/1e6:,.2f} Mln" if pd.notnull(x) else ""
        
        # Applica la formattazione a tutte le colonne selezionate in **un unico passaggio**
        format_dict = {col: format_mln_euro for col in cols_to_format}
        df_styled = df_styled.format(format_dict)
        
        # Visualizza su Streamlit
        st.dataframe(df_styled)

        st.markdown("### ⚠️ Target Policy")
        st.info("Valori % di copertura del fabbisogno.")
        st.dataframe(df_target_policy)


        anni_comuni = set(df_risk['Year']).intersection(df_open['Anno'])
        df_risk = df_risk[df_risk['Year'].isin(anni_comuni)].reset_index(drop=True)

        
        fig_var = var_ebitda_risk(
        periodo_di_analisi= start_date.strftime("as of %d/%m/%Y"),
        df_risk=df_risk,
        df_open=df_open,
        df_ebitda=df,
        font_path="utils/TIMSans-Medium.ttf"
            )
        st.pyplot(fig_var, dpi=160)

        
        # Salvataggio in session_state
        st.session_state.update({
            "df_risk": df_risk,
            "df_open": df_open,
            "df_prezzi": df_prezzi,
            "df_target_policy": df_target_policy,
            "historical_price": historical_price,
            "predict_price": predict_price,
            "p95": p95,
            "p5": p5,
            "forward_price_full": forward_price_full,
            "budget_price_full": budget_price_full,
            "covered": covered,
            "fabbisogno": fabbisogno,
            "solar": solar,
            "unique_years": unique_years,
            "anni_prezzi": anni_prezzi,
            "start_date_sim": start_date_sim,
            "df_ebitda": df[["Anno", "Ebitda"]]
        })

    # -----------------------
    # Riacquisti Energia
    # -----------------------
    if "df_open" in st.session_state:
        st.subheader("📌 Acquisto energia aggiuntiva per anno")
    
        if "extra_purchase" not in st.session_state:
            st.session_state.extra_purchase = {anno: 0.0 for anno in st.session_state.unique_years}
    
        for anno in st.session_state.unique_years:
            qta = st.number_input(
                f"Anno {anno} - MWh da acquistare",
                min_value=0.0,
                value=st.session_state.extra_purchase.get(anno, 0.0),
                step=10.0,
                key=f"extra_{anno}"
            )
            st.session_state.extra_purchase[anno] = qta
    
        if st.button("🔄 Ricalcola Open Position con riacquisti", key="recalc_btn"):
            covered_adjusted = [c + st.session_state.extra_purchase[a] for c, a in zip(st.session_state.covered, st.session_state.unique_years)]
    
            df_risk_new, df_open_new, df_prezzi_new, df_target_policy_new, fig = compute_downside_upperside_risk(
                anni=st.session_state.unique_years,
                fabbisogno=st.session_state.fabbisogno,
                covered=covered_adjusted,
                solar=st.session_state.solar,
                anni_prezzi=st.session_state.anni_prezzi,
                media_pun=st.session_state.historical_price,
                predictive=st.session_state.predict_price,
                p95=st.session_state.p95,
                p5=st.session_state.p5,
                frwd=st.session_state.forward_price_full,
                budget=st.session_state.budget_price_full,
                observation_period=st.session_state.start_date_sim
            )
    
            # ✅ Mantiene sia le versioni vecchie che le nuove
            st.session_state.df_open_new = df_open_new
            st.session_state.df_risk_new = df_risk_new
            st.session_state.df_prezzi_new = df_prezzi_new
            st.session_state.df_target_policy_new = df_target_policy_new
    
            st.subheader("📋 Tabella Open Position (aggiornata)")
            st.dataframe(df_open_new)

            st.markdown("### ⚠️ Analisi Rischio (Downside / Upside)")

            # Copia del DataFrame per styling
            df_styled_new = df_risk_new.style
        
            # Colonne da escludere dalla formattazione in milioni
            exclude_cols = ["Year", "Anno", "year", "anno"]
        
            # Colonne da formattare in milioni di euro
            cols_to_format = [c for c in df_risk_new.columns if c not in exclude_cols]
        
            # Applica rosso tenue alle colonne "Downside"
            downside_cols = [c for c in df_risk_new.columns if c.startswith("Downside")]
            if downside_cols:
                df_styled_new = df_styled_new.background_gradient(
                    cmap='Reds', low=0.1, high=0.4, subset=downside_cols
                )
        
            # Applica verde tenue alle colonne "Upside"
            upside_cols = [c for c in df_risk_new.columns if c.startswith("Upside")]
            if upside_cols:
                df_styled_new = df_styled_new.background_gradient(
                    cmap='Greens', low=0.1, high=0.4, subset=upside_cols
                )
        
            # Funzione di formattazione in milioni di euro
            def format_mln_euro(x):
                return f"€ {x/1e6:,.2f} Mln" if pd.notnull(x) else ""
        
            # Applica la formattazione a tutte le colonne selezionate in **un unico passaggio**
            format_dict = {col: format_mln_euro for col in cols_to_format}
            df_styled_new = df_styled_new.format(format_dict)
        
            # Visualizza su Streamlit
            st.dataframe(df_styled_new)


            st.markdown("### ⚠️ Target Policy Aggiornato")
            st.info("Valori % di copertura del fabbisogno.")
            st.dataframe(df_target_policy_new)

            # --- Profit/Loss ---
            df_gain_loss = pd.DataFrame({
                "Anno": st.session_state.unique_years,
                "MWh Acquistati": [st.session_state.extra_purchase[a] for a in st.session_state.unique_years],
                "Prezzo Forward (€)": st.session_state.forward_price_full[-len(st.session_state.unique_years):],
                "Prezzo Budget (€)": st.session_state.budget_price_full[-len(st.session_state.unique_years):]
            })
            df_gain_loss["Δ Prezzo (Budget - Forward)"] = (
                df_gain_loss["Prezzo Budget (€)"] - df_gain_loss["Prezzo Forward (€)"]
            )
            df_gain_loss["Profit/Loss (€)"] = (
                df_gain_loss["MWh Acquistati"] * 1000 * df_gain_loss["Δ Prezzo (Budget - Forward)"]
            )
    
            df_gain_loss["Profit/Loss (€)"] = df_gain_loss["Profit/Loss (€)"].apply(lambda x: f"€ {x:,.0f}")
            df_gain_loss["Δ Prezzo (Budget - Forward)"] = df_gain_loss["Δ Prezzo (Budget - Forward)"].apply(lambda x: f"€ {x:,.2f}")
    
            st.session_state.df_gain_loss = df_gain_loss
    
            st.subheader("💰 Analisi Guadagno/Perdita Riacquisto")
            st.dataframe(df_gain_loss)
            st.success("✅ Open Position e Analisi Riacquisto aggiornate con successo!")
    
        # -----------------------
        # 💾 Esportazione in Excel
        # -----------------------
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            written = False
    
            # --- Versioni ORIGINALI ---
            if "df_open" in st.session_state and not st.session_state.df_open.empty:
                st.session_state.df_open.to_excel(writer, sheet_name='Open Position (orig)', index=False)
                written = True
            if "df_risk" in st.session_state and not st.session_state.df_risk.empty:
                st.session_state.df_risk.to_excel(writer, sheet_name='Analisi Rischio (orig)', index=False)
                written = True
            if "df_target_policy" in locals() and not df_target_policy.empty:
                df_target_policy.to_excel(writer, sheet_name='Target Policy (orig)', index=False)
                written = True
    
            # --- Versioni NUOVE (dopo riacquisto) ---
            if "df_open_new" in st.session_state and not st.session_state.df_open_new.empty:
                st.session_state.df_open_new.to_excel(writer, sheet_name='Open Position (new)', index=False)
                written = True
            if "df_risk_new" in st.session_state and not st.session_state.df_risk_new.empty:
                st.session_state.df_risk_new.to_excel(writer, sheet_name='Analisi Rischio (new)', index=False)
                written = True
            if "df_target_policy_new" in st.session_state and not st.session_state.df_target_policy_new.empty:
                st.session_state.df_target_policy_new.to_excel(writer, sheet_name='Target Policy (new)', index=False)
                written = True
    
            # --- Profit/Loss ---
            if "df_gain_loss" in st.session_state and not st.session_state.df_gain_loss.empty:
                st.session_state.df_gain_loss.to_excel(writer, sheet_name='Riacquisto Profit-Loss', index=False)
                written = True
    
            # --- Serie storiche e prezzi ---
            if "df_prezzi" in locals() and not df_prezzi.empty:
                df_prezzi.to_excel(writer, sheet_name='Prezzi PUN', index=False)
                written = True
            if "df_historical" in st.session_state and not st.session_state.df_historical.empty:
                st.session_state.df_historical.to_excel(writer, sheet_name='Historical Price', index=False)
                written = True
            if "energy_df" in st.session_state and not st.session_state.energy_df.empty:
                st.session_state.energy_df.to_excel(writer, sheet_name='Serie PUN', index=False)
                written = True
    
            if not written:
                pd.DataFrame({"Info": ["Nessun dato disponibile"]}).to_excel(writer, sheet_name="Empty", index=False)
    
            buffer.seek(0)
    
        st.download_button(
            label="💾 Scarica tutti i dati del Energy Risk in Excel ",
            data=buffer,
            file_name="KRI_Energy_Risk.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
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
# 🌪️ Copper Risk
# -----------------------
elif selected_kri == "🟠 Copper Price":
    st.subheader("🟠 Simulazione Future a 3 mesi su Copper")
    st.info("Esegui la simulazione multivariata sul future del copper a 3 mesi")

    df = pd.read_excel('Data/df.xlsx')
    df.set_index('Date', inplace=True)

    N = len(df)
    train_end = int(0.8 * N)
    val_end = train_end + int(0.1 * N)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    now = datetime.now()
    last_date = now.strftime("%d-%m-%Y")

    # Caricamento modelli
    with open('utils/catboost_model.pkl', 'rb') as file:
        model_cb = pickle.load(file)

    with open('utils/copula_model.pkl', 'rb') as file:
        copula_model = pickle.load(file)

    with open('utils/egarch_model.pkl', 'rb') as file:
        egarch_model = pickle.load(file)

    with open('utils/egarch_fit.pkl', 'rb') as file:
        egarch_fit = pickle.load(file)

    S0_test = float(df['PX_LAST'].iloc[-1])
    
    # -----------------------------------------------
    # 📅 Selezione data finale simulazione
    # -----------------------------------------------
    end_date = st.date_input(
        "📅 Seleziona la data di fine simulazione",
        value=datetime(2028, 12, 31),
        min_value=datetime.now()
    )

    # -----------------------------------------------
    # 📦 INPUT: quantità per anno + prezzo budget
    # -----------------------------------------------
    start_year = datetime.now().year
    end_year = end_date.year
    years = list(range(start_year, end_year + 1))

    st.subheader("📦 Quantità di Copper da vendere per anno")

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
        "💰 Prezzo di budget (USD per tonnellata)",
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

        result_df, sim_prices = simulate_cb_egarch_outsample(
            copula_model=copula_model,
            model_cb=model_cb,
            egarch_model=egarch_model,
            egarch_fit=egarch_fit,
            last_date=last_date,
            end_date=end_date.strftime("%Y-%m-%d"),
            S0=S0_test,
            n_sims=10
        )

        fig = get_forecast_plot(df, result_df)
        st.pyplot(fig)

        st.subheader("📊 Risultati Simulazione")
        result_df.index = pd.to_datetime(result_df.index)

        # -----------------------------------------------
        # 📅 Groupby per anno
        # -----------------------------------------------
        result_df["year"] = result_df.index.year
        result_df_yearly = result_df.groupby("year")[["median", "average", "lower", "upper"]].mean()

        # -----------------------------------------------
        # 💰 Aggiunta quantità e calcolo P&L vs budget
        # -----------------------------------------------
        result_df_yearly["qty"] = result_df_yearly.index.map(quantities)

        result_df_yearly["PnL_vs_budget"] = (
            (result_df_yearly["lower"] - budget_price) * result_df_yearly["qty"]
        )
        
        st.subheader("📘 Risultati per anno (medie + quantità + PnL)")
        st.dataframe(result_df_yearly)

        st.subheader("📘 Estratto risultati giornalieri")
        st.dataframe(result_df.head())

        # -----------------------------------------------
        # 💾 Download Excel
        # -----------------------------------------------
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=True, sheet_name='Giornaliero Simulazione')
            result_df_yearly.to_excel(writer, index=True, sheet_name='Annuale Aggregato')
            df.to_excel(writer, index=True, sheet_name='Copper Price')
            buffer.seek(0)

        st.download_button(
            label="💾 Scarica risultati in Excel",
            data=buffer,
            file_name="Simulazione_Copper_price.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
# -----------------------
# 🌪️ Access to Funding
# -----------------------
elif selected_kri == "💰🔑 Access to Funding":
    print('Access to Funding')
    
elif selected_kri == "💳 Credit risk":
    st.subheader("🏦 Credit Risk – Aging & Indicatori")

    uploaded_credit = st.file_uploader("📂 Carica il file Aging", type="xlsx")

    provision_t1 = st.number_input(
        "Provision (T-1)",
        min_value=0.0,
        step=1000.0,
        format="%.2f"
    )

    if uploaded_credit:
        df = pd.read_excel(uploaded_credit)
        df.columns = df.columns.str.strip()

        required_cols = [
            "Periodo",  # nuova colonna periodo
            "TRADE RECEIVABLES (NET)",
            "Not Overdue", "1-90", "91-180",
            "181-365", "Over 365", "PROVISION"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("⚠️ Il file deve contenere le colonne corrette.")
        else:
            st.success("File caricato correttamente!")
            
            df["Periodo"] = pd.to_datetime(df["Periodo"], format="%m-%Y")
            # --------------------------
            # Raggruppamento per Periodo
            # --------------------------
            grouped = df.groupby("Periodo").sum().reset_index()

            # --------------------------
            # KPI CALCULATION
            # --------------------------
            grouped["Over90"] = grouped["91-180"] + grouped["181-365"] + grouped["Over 365"]
            grouped["Pct_Over_90"] = grouped["Over90"] / grouped["TRADE RECEIVABLES (NET)"]

            grouped["Delta_Provision"] = grouped["PROVISION"].diff().fillna(0)

            grouped["Aging"] = (
                0   * grouped["Not Overdue"] +
                45  * grouped["1-90"] +
                135 * grouped["91-180"] +
                270 * grouped["181-365"] +
                365 * grouped["Over 365"]
            ) / grouped["TRADE RECEIVABLES (NET)"]

            # Dataframe indicatori principali
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
            # 1️⃣ Percentuale Over 90 giorni
            fig_pct = px.bar(
                kpi_df,
                x="Periodo",
                y="Pct_Over_90",
                text="Pct_Over_90",
                labels={"Pct_Over_90": "Pct Over 90"},
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
                labels={"Delta_Provision": "Delta Provision"},
                title="💰 Delta Provision vs T-1 per Periodo in €",
                color="Delta_Provision",
                color_continuous_scale="Oranges"
            )
            fig_delta.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_delta, use_container_width=True)
            
            # 3️⃣ Aging medio
            fig_aging = px.bar(
                kpi_df,
                x="Periodo",
                y="Aging",
                text="Aging",
                labels={"Aging": "Aging medio (giorni)"},
                title="⏳ Aging medio dei crediti per Periodo",
                color="Aging",
                color_continuous_scale="Greens"
            )
            fig_aging.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_aging, use_container_width=True)
            # -----------------------------------------------
            # 💾 Download Excel
            # -----------------------------------------------
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                
                df.to_excel(writer, index=False, sheet_name='Aging Raw')
                kpi_df.to_excel(writer, index=False, sheet_name='Indicatori KPI')

                # foglio riassunto KPI (solo valori medi)
                summary = kpi_df.mean(numeric_only=True).to_frame("Value")
                summary.to_excel(writer, sheet_name='Sintesi KPI')

                buffer.seek(0)

            st.download_button(
                label="💾 Scarica file Credit Risk (Excel)",
                data=buffer,
                file_name="Credit_Risk_Aging_Indicators.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
  
elif selected_kri == "🛡️💻 Cyber":
    print('Cyber')
elif selected_kri == "📈 Interest Rate":
    import matplotlib.pyplot as plt
    series = {
    
    # --- Euribor / Money Market ---
    "euribor_3m": "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",

    # --- Politica monetaria BCE ---
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
        plt.plot(df_dropped.index[:train_end], y_pred_train, label="Train Pred", color='blue')
        # Validation
        plt.plot(df_dropped.index[train_end:val_end], y_pred_val, label="Val Pred", color='orange')
        # Test
        plt.plot(df_dropped.index[val_end:], y_pred_test, label="Test Pred", color='green')
        plt.title("Serie originale vs Predizione completa")
        plt.legend()
        plt.grid(True)
        # Render in Streamlit
        st.pyplot(plt.gcf())
        # Chiudi figura per evitare problemi
        plt.close()
    
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
    
    df_ecb = download_ecb_series(series)
    df_yahoo = download_yahoo_series(yahoo_symbols)
    df_all = df_ecb.join(df_yahoo, how="outer")
    df_all = df_all.sort_index().ffill()
    df_dropped = df_all.dropna()
    st.write("Database pronto ☑️")
    
    stl = STL(df_dropped['euribor_3m'], period=30, robust=True)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    # --- Split train / val / test ---
    n = len(df_dropped)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    X = df_dropped.drop(columns='euribor_3m')
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    
    trend_train = trend.iloc[:train_end]
    trend_val = trend.iloc[train_end:val_end]
    trend_test = trend.iloc[val_end:]
    
    residual_train = residual.iloc[:train_end]
    residual_val = residual.iloc[train_end:val_end]
    residual_test = residual.iloc[val_end:]
    
    seasonal_train = seasonal.iloc[:train_end]
    seasonal_val = seasonal.iloc[train_end:val_end]
    seasonal_test = seasonal.iloc[val_end:]
    
    import pickle
    
    # === LOAD TREND MODEL ===
    with open("Dashboard-reporting-KRI/utils/trend_model.pkl", "rb") as f:
        trend_model = pickle.load(f)
    
    # === LOAD RESIDUAL MODEL ===
    with open("Dashboard-reporting-KRI/utils/residual_model.pkl", "rb") as f:
        residual_model = pickle.load(f)
    
    # === LOAD SARIMA SEASONAL MODEL ===
    with open("Dashboard-reporting-KRI/utils/sarima_seasonal.pkl", "rb") as f:
        sarima_fit = pickle.load(f)
    
    # Predizioni trend
    trend_pred_train = trend_model.predict(X_train)
    trend_pred_val = trend_model.predict(X_val)
    trend_pred_test = trend_model.predict(X_test)
    
    
    # Predizioni seasonal
    seasonal_pred_train = sarima_fit.predict(start=0, end=len(seasonal_train)-1)
    seasonal_pred_val = sarima_fit.predict(start=len(seasonal_train), end=len(seasonal_train)+len(seasonal_val)-1)
    seasonal_pred_test = sarima_fit.forecast(steps=len(seasonal_test))
    
    
    # Predizioni residual
    residual_pred_train = residual_model.predict(X_train)
    residual_pred_val = residual_model.predict(X_val)
    residual_pred_test = residual_model.predict(X_test)
    
    # --- Predizione finale ---
    y_pred_train = trend_pred_train + seasonal_pred_train + residual_pred_train
    y_pred_val = trend_pred_val + seasonal_pred_val + residual_pred_val
    y_pred_test = trend_pred_test + seasonal_pred_test + residual_pred_test
    
    st.subheader("📊 Trend analisi with Hybrid ML model 📊")
    plot_predictions_streamlit(df_dropped, y_pred_train, y_pred_val, y_pred_test, train_end, val_end)
        
# ============================================================
# FUNZIONE PER IL CALCOLO DEL VAR DI UNA SINGOLA TRANCHE
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

# ============================================================
# SIMULAZIONE UNICA EURIBOR MONTE CARLO + CONFORMAL
# ============================================================
def simulate_euribor(series, df_dropped, n_sims=1000, alpha=0.05, horizon_days=3*360):
    # Ottimizzazione parametri OU
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

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

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

    # Conformal adjustment
    calibration_y = series[-252:]
    samples_cal = np.random.choice(simulations.flatten(), size=(len(calibration_y), n_period))
    lower_cal = np.percentile(samples_cal, 2.5, axis=1)
    upper_cal = np.percentile(samples_cal, 97.5, axis=1)
    nonconformity = np.maximum(lower_cal - calibration_y, calibration_y - upper_cal)
    q_hat = np.quantile(np.append(nonconformity, np.inf), 0.95)

    lower_adj = lower_emp - q_hat
    upper_adj = upper_emp + q_hat

    while np.any(upper_adj <= lower_adj):
        mask = upper_adj <= lower_adj
        upper_adj[mask] = lower_adj[mask] + 0.2

    idx = pd.date_range(start=df_dropped.index[-1] + pd.Timedelta(days=1), periods=n_period, freq="D")
    forecast_df = pd.DataFrame({
        "lower_emp": lower_emp,
        "upper_emp": upper_emp,
        "median": median,
        "lower_adj": lower_adj,
        "upper_adj": upper_adj
    }, index=idx)
    forecast_quarterly = forecast_df.resample("Q").mean()

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

# ============================================================
# STREAMLIT INTERFACCIA
# ============================================================
st.subheader("📊 Calcolo VaR 95% su Simulazioni Euribor 3M 📊")
run_sim = st.button("🚀 Inizia Simulazione VaR su tutte le Tranche")

if uploaded_file and run_sim:
    tranche_df = pd.read_excel(uploaded_file, sheet_name="Tranches")
    st.subheader("📋 Tranche caricate dall’Excel")
    st.dataframe(tranche_df)

    series_df = df_dropped.loc["2023-01-01":]
    series = series_df["euribor_3m"].values
    last_date = pd.to_datetime(series_df.index[-1])

    max_horizon_days = (pd.to_datetime(tranche_df['Maturity']).max() - last_date).days

    # 1️⃣ Simulazione unica EURIBOR
    forecast_df, forecast_quarterly = simulate_euribor(series, df_dropped, n_sims= 10_000, horizon_days= max_horizon_days)

    results_var = []
    
    # 2️⃣ Ciclo su tranche usando la simulazione unica
    for idx, row in tranche_df.iterrows():
        tranche_name = row.get("Tranche", f"T{idx+1}")
        unhedged = (row["Notional"] - row["Hedged"])
        plan_rate = (row["Euribor"] + row["Spread"])
    
        # Taglio forecast fino alla maturità della tranche
        maturity_date = pd.to_datetime(row["Maturity"])
        forecast_tranche = forecast_quarterly[forecast_quarterly.index <= maturity_date]
        var_rate = forecast_tranche["upper_adj"] + row["Spread"]
    
        var_amount = (var_rate/100) * unhedged
        plan_amount = (plan_rate/100) * unhedged
        days = forecast_tranche.index.to_series().diff().dt.days.fillna(90)
        var_cf = var_amount * (days / 360)
        plan_cf = plan_amount * (days / 360)
    
        # DataFrame con indice corretto per la tranche
        df_var = pd.DataFrame({
            "Notional": row["Notional"],
            "Hedged": row["Hedged"],
            "Un-Hedged": unhedged,
            "Var Rate": var_rate,
            "Plan Rate": plan_rate,
            "Var Amount (€)": var_amount,
            "Var Cashflow (€)": var_cf,
            "Plan Amount (€)": plan_amount,
            "Plan Cashflow (€)": plan_cf,
            "KRI Amount": (var_amount - plan_amount),
            "KRI Cashflow": np.max(var_cf - plan_cf,0),
            "Tranche": tranche_name
        }, index=forecast_tranche.index)
    
        results_var.append(df_var)
    
    # Concatenazione risultati
    final_var_df = pd.concat(results_var).reset_index()
    
    st.subheader("📊 Forecast Euribor 3M - Tutte le Tranche")
    plt.figure(figsize=(15,6))
    # Serie storica
    plt.plot(df_dropped.index, df_dropped['euribor_3m'], label="Originale", color='black')
    # Forecast unico Monte Carlo (median e intervallo conformalizzato)
    plt.plot(forecast_quarterly.index, forecast_quarterly['median'], label='Mean Forecast', color='green', linestyle='--')
    plt.fill_between(
        forecast_quarterly.index,
        forecast_quarterly['lower_adj'],
        forecast_quarterly['upper_adj'],
        color='red', alpha=0.2, label='Adjusted Interval (Conformal)'
    )
    
    plt.title("Serie storica + Forecast Monte Carlo EURIBOR 3M")
    plt.xlabel("Date")
    plt.ylabel("EURIBOR 3M")
    plt.legend()
    plt.grid(True)
    
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("📊 Risultati VaR – per Tranche")
    st.dataframe(final_var_df)

    portfolio_var = final_var_df.groupby('index')[[
        "Var Amount (€)", "Var Cashflow (€)", "KRI Amount", "KRI Cashflow", "Plan Cashflow (€)"
    ]].sum().reset_index()

    st.subheader("📈 VaR Cumulato di Portafoglio (€)")
    st.dataframe(portfolio_var)

    st.subheader("📉 Grafico VaR di Portafoglio")
    st.line_chart(portfolio_var.set_index('index')[["Var Cashflow (€)", "Plan Cashflow (€)"]])

    st.subheader("💸⚠️ KRI Portafoglio💸⚠️")
    st.line_chart(portfolio_var.set_index('index')["KRI Cashflow"])

    # Export Excel
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        final_var_df.to_excel(writer, index=True, sheet_name="Tranches")
        portfolio_var.to_excel(writer, index=True, sheet_name="Portfolio")
    st.download_button(
        label="📥 Scarica risultati in Excel",
        data=output.getvalue(),
        file_name="VaR_multi_tranche.xlsx",
        mime="application/vnd.ms-excel"
    )

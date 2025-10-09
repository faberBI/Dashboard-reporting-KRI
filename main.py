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

# Library custom
from utils.data_loader import load_kri_excel, validate_kri_data
from functions.energy_risk import (historical_VaR, run_heston, analyze_simulation, compute_downside_upperside_risk, var_ebitda_risk)

# -----------------------
# Configurazione Streamlit
# -----------------------
st.set_page_config(page_title="KRI Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard KRI")

# -----------------------
# Selezione KRI
# -----------------------
kri_options = ["⚡ Energy Risk", "🌪️ Natural Event Risk", "📌 KRI 3"]

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
    ebitda = st.text_input("EBITDA (€)", df_to_str(df, "Ebitda", "1900000000"))

    # Parsing input
    try:
        fabbisogno = [float(x) for x in fabbisogno.split(",")]
        covered = [float(x) for x in covered.split(",")]
        solar = [float(x) for x in solar.split(",")]
        forward_price = [float(x) for x in forward_price.split(",")]
        budget_price = [float(x) for x in budget_price.split(",")]
        ebitda = float(ebitda.replace(",", ""))
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
            anni=unique_years,
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

        st.pyplot(fig)

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
            "ebitda": ebitda
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
            df_risk, df_open, df_prezzi, df_target_policy, fig = compute_downside_upperside_risk(
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

            st.session_state.df_open = df_open
            st.session_state.df_risk = df_risk

            st.subheader("📋 Tabella Open Position (aggiornata)")
            st.dataframe(df_open)

            # Profit/Loss
            df_gain_loss = pd.DataFrame({
                "Anno": st.session_state.unique_years,
                "MWh Acquistati": [st.session_state.extra_purchase[a] for a in st.session_state.unique_years],
                "Prezzo Forward (€)": st.session_state.forward_price_full[-len(st.session_state.unique_years):],
                "Prezzo Budget (€)": st.session_state.budget_price_full[-len(st.session_state.unique_years):]
            })
            df_gain_loss["Δ Prezzo (Budget - Forward)"] = df_gain_loss["Prezzo Budget (€)"] - df_gain_loss["Prezzo Forward (€)"]
            df_gain_loss["Profit/Loss (€)"] = df_gain_loss["MWh Acquistati"] * df_gain_loss["Δ Prezzo (Budget - Forward)"]
            df_gain_loss["Profit/Loss (€)"] = df_gain_loss["Profit/Loss (€)"].apply(lambda x: f"€ {x:,.0f}")
            df_gain_loss["Δ Prezzo (Budget - Forward)"] = df_gain_loss["Δ Prezzo (Budget - Forward)"].apply(lambda x: f"€ {x:,.2f}")
            st.subheader("💰 Analisi Guadagno/Perdita Riacquisto")
            st.dataframe(df_gain_loss)
            st.success("✅ Open Position e Analisi Riacquisto aggiornate con successo!")

        
       
        # Pulsante per scaricare Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # df_prezzi
            df_prezzi.to_excel(writer, sheet_name='Prezzi PUN', index=False)
    
            # df_risk
            df_risk.to_excel(writer, sheet_name='Analisi Rischio', index=False)
    
            # df_historical
            df_historical.to_excel(writer, sheet_name='Historical Price', index=False)
    
            # forecast_price
            forecast_price.to_excel(writer, sheet_name='Forecast PUN', index=True)
    
            # Serie PUN storica
            if 'energy_df' in st.session_state:
                st.session_state.energy_df.to_excel(writer, sheet_name='Serie PUN', index=False)
            
            # Target Policy 
            df_target_policy.to_excel(writer, sheet_name='Target Policy', index=False)
    
            if "df_gain_loss" in locals() and not df_gain_loss.empty:
                # Rimuove il simbolo € e converte per sicurezza a numerico
                df_gain_loss_clean = df_gain_loss.copy()
                for col in ["Profit/Loss (€)", "Δ Prezzo (Budget - Forward)"]:
                    df_gain_loss_clean[col] = (
                    df_gain_loss_clean[col]
                    .replace("€", "", regex=True)
                    .replace(",", "", regex=True)
                    .astype(float, errors='ignore'))
                df_gain_loss_clean.to_excel(writer, sheet_name='Riacquisto Profit-Loss', index=False)
            
            buffer.seek(0)

        st.download_button(
            label="💾 Scarica tutti i dati del Energy Risk in Excel ",
            data=buffer,
            file_name="KRI_Energy_Risk.xlsx",
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

        from functions.geospatial import (
            get_risk_area_idro,
            get_risk_area_frane,
            get_magnitudes_for_comune
        )

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
            try:
                results = simulazione_portafoglio_con_rischi_correlati(
                    df=df,
                    db_frane=db_frane,
                    db_idro=db_idro,
                    df_sismico=df_sismico,
                    n_simulazioni=int(n_simulazioni)
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




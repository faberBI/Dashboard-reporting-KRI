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
   

    # Parsing input
    try:
        fabbisogno = [float(x) for x in fabbisogno.split(",")]
        covered = [float(x) for x in covered.split(",")]
        solar = [float(x) for x in solar.split(",")]
        forward_price = [float(x) for x in forward_price.split(",")]
        budget_price = [float(x) for x in budget_price.split(",")]
    except Exception as e:
        st.error(f"❌ Errore nei parametri: {e}")
        st.stop()

    if not (len(fabbisogno) == len(covered) == len(solar) == len(forward_price) == len(budget_price)):
        st.error("⚠️ Tutti i parametri devono avere lo stesso numero di valori per anno.")
        st.stop()

    st.success("✅ Parametri validi, pronti per la simulazione!")

    # Simulazione
    n_simulations = st.number_input("Numero di simulazioni", min_value=100, max_value=100_000, value=10_000, step=100)
    n_trials_heston = st.number_input("Numero di trial Heston", min_value=10, max_value=1000, value=100, step=10)
    end_date = st.date_input("Data finale simulazione", pd.to_datetime("2027-12-31"))
    start_date = st.date_input("Dati aggiornati al", pd.Timestamp.today().date())
    ebitda = st.text_input("EBITDA (€)", df_to_str(df, "Ebitda", "1900000000"))
    
    start_date_sim = pd.Timestamp.today().normalize()
    days_to_simulate = (pd.to_datetime(end_date) - pd.to_datetime(start_date_sim)).days
    future_dates = pd.date_range(start=start_date_sim, periods=days_to_simulate, freq='D')
    unique_years = sorted(future_dates.year.unique().tolist())
    
    # -----------------------
    # Lancia simulazione
    # -----------------------
    if st.button("💹 Esegui simulazione Energy Risk"):
        st.info("Simulazione in corso...")

        # Prova a caricare il file Excel locale
        data_path = "Data/Pun 10_04_2025.xlsx"
        df_excel = None
        if os.path.exists(data_path):
            df_excel = pd.read_excel(data_path)
            st.success(f"🔄 Dati Caricati")

        # Se il file locale non esiste o è vuoto, richiedi upload
        if df_excel is None or df_excel.empty:
            uploaded_file = st.file_uploader("Seleziona il file Excel PUN", type=["xlsx"])
            if uploaded_file:
                df_excel = pd.read_excel(uploaded_file)
            else:
                st.warning("Carica il file Excel per procedere con la simulazione.")
                st.stop()

        # Controllo colonne obbligatorie
        if 'Date' not in df_excel.columns or 'GMEPIT24 Index' not in df_excel.columns:
            st.error("Il file Excel deve contenere 'Date' e 'GMEPIT24 Index'.")
            st.stop()

        # Converte Date e calcola Log_Returns
        df_excel['Date'] = pd.to_datetime(df_excel['Date'])
        df_excel['Log_Returns'] = np.log(df_excel['GMEPIT24 Index'] / df_excel['GMEPIT24 Index'].shift(1))
        df_excel = df_excel.dropna(subset=['Log_Returns'])

        # Salva in session_state
        st.session_state.energy_df = df_excel

        # Filtra per intervallo date selezionato
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

        # Analisi distribuzione
        monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig = analyze_simulation(simulated_df, unique_years, forward_prices= forward_price)  # <-- PASSA qui il forward price
        st.pyplot(fig)

        # --------------------------------------
        # Forecast + Storico
        # --------------------------------------

        forecast_price = df = pd.DataFrame.from_dict(yearly_percentiles, orient='index', columns=['5%', '50%', '95%'])
        st.markdown("### 📊 Forecast Output")  # titolo con icona
        st.info("Questi sono i valori previsionali basati sui percentili annuali.")  # box informativo
        # Funzione di formattazione in euro
        def format_euro(x):
            return f"€ {x:.2f}" if pd.notnull(x) else ""

        # Applica formattazione solo alle colonne numeriche
        cols_to_format = [c for c in forecast_price.columns]  # qui non c'è Year, quindi tutte le colonne
        st.dataframe(forecast_price.style.format({col: format_euro for col in cols_to_format}).background_gradient(cmap='Greens', low=0.1, high=0.4, subset=cols_to_format))

        # Combinazione anni storico + forecast
        anni_prezzi = [2020, 2021, 2022, 2023, 2024] + unique_years
        anni_prezzi = [int(y) for y in anni_prezzi]

        # Media storica PUN per gli anni storici
        historical_price = df_filtered.groupby(df_filtered['Date'].dt.year)['GMEPIT24 Index'].mean().tail(6).values.tolist()
        # historical_price = historical_price[:-1]

        st.markdown("### 📅 Historical Price")  # Titolo con icona
        st.info("Media storica PUN per gli anni disponibili.")  # Box informativo
        
        df_historical = pd.DataFrame({"Historical Price": historical_price, "Year": anni_prezzi[:len(historical_price)]})

        # Copia per styling
        df_hist_styled = df_historical.style

        # Colonne da escludere dalla formattazione
        exclude_cols = ["Year", "Anno", "year", "anno"]

        # Colonne da formattare in euro
        cols_to_format = [c for c in df_historical.columns if c not in exclude_cols]

        # Funzione di formattazione
        def format_euro(x):
            return f"€ {x:.2f}" if pd.notnull(x) else ""

        # Applica formattazione in un unico passaggio
        format_dict = {col: format_euro for col in cols_to_format}
        df_hist_styled = df_hist_styled.format(format_dict)

        # Applica gradiente solo sulle colonne numeriche
        df_hist_styled = df_hist_styled.background_gradient(cmap='Greens', low=0.1, high=0.4, subset=cols_to_format)

        # Mostra su Streamlit
        st.dataframe(df_hist_styled)

        predict_price = forecast_price['50%'].values.tolist()
        p95 = forecast_price['95%'].values.tolist()
        p5 = forecast_price['5%'].values.tolist()

        # Lista forward price corrispondente agli anni forecast
        forward_price_full = forward_price.copy()  # assume forward_price contiene solo i valori forecast

        # Budget price (allinea lunghezza aggiungendo zeri davanti)
        budget_price_full = [0] * (len(anni_prezzi) - len(budget_price)) + budget_price

        # Allineamento lunghezze di tutte le liste con anni_prezzi
        missing_len_hp = len(anni_prezzi) - len(historical_price)
        missing_len_pp = len(anni_prezzi) - len(predict_price)
        missing_len_p95 = len(anni_prezzi) - len(p95)
        missing_len_p5 = len(anni_prezzi) - len(p5)
        missing_len_f = len(anni_prezzi) - len(forward_price_full)

        historical_price = historical_price + [0] * missing_len_hp
        predict_price = [0]*missing_len_pp + predict_price
        p95 = [0]*missing_len_p95 + p95
        p5 = [0]*missing_len_p5 + p5
        forward_price_full = [0]*missing_len_f + forward_price_full

        # Chiamata alla funzione principale
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
        observation_period=start_date_sim.strftime("%d/%m/%Y"))

        # Visualizzazione su Streamlit
        st.pyplot(fig)
        
        # -------------------------------
        # 📋 Tabella Target Policy
        # -------------------------------
        st.markdown("### 🎯 Target Policy Analysis")
        st.info("Rapporto tra coperto e fabbisogno con e senza energia solare, confrontato con i livelli di Target Policy (95%, 85%, 50%).")

        # Copia per visualizzazione
        df_target_display = df_target_policy.copy()

        # Formattazione percentuali
        cols_to_format = ["% Purchased w/o Solar", "% Purchased with Solar", "Target Policy"]

        for col in cols_to_format:
            df_target_display[col] = df_target_display[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        # Mostra la tabella
        st.dataframe(df_target_display)

        st.markdown("### 📈 Analisi Prezzi PUN ")
        st.info("Tabella contenente media PUN, percentili, Forward e Budget per ogni anno.")
        import pandas as pd
        
        # Applica la formattazione a tutte le colonne tranne eventualmente l'anno
        cols_to_format = [c for c in df_prezzi.columns if c.lower()  not in ["year", "anno", "Year"]]
        
        df_display = df_prezzi.copy()
        # Aggiungi € e arrotonda a 2 decimali
        for col in cols_to_format:
            df_display[col] = df_display[col].apply(lambda x: f"€ {x:.2f}" if pd.notnull(x) else "")
        st.dataframe(df_display)
        
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
        
        from datetime import datetime
        # Grafico VaR EBITDA
        fig = var_ebitda_risk(
            periodo_di_analisi=start_date.strftime("as of %d/%m/%Y"),
            df_risk=df_risk,
            df_open=df_open,
            ebitda=ebitda,
            font_path="utils/TIMSans-Medium.ttf"
            )

        st.pyplot(fig)

        st.success("Simulazione completata!")
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




import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.data_loader import load_kri_excel, validate_kri_data
from functions.montecarlo import run_heston, analyze_simulation, compute_downside_upperside_risk, var_ebitda_risk

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

if uploaded_file:
    df = load_kri_excel(uploaded_file, selected_kri)
    if validate_kri_data(df, selected_kri):
        st.session_state.kri_data[selected_kri] = df
        st.success(f"✅ {selected_kri} aggiunto con successo!")

# -----------------------
# Mostra KRI caricati solo per il KRI selezionato
# -----------------------
if selected_kri in st.session_state.kri_data:
    st.subheader("📊 KRI caricati")
    df_to_show = st.session_state.kri_data[selected_kri]
    st.markdown(f"### 📌 **{selected_kri}**")
    st.dataframe(df_to_show.head())

# -----------------------
# Parametri Energy Risk
# -----------------------
if selected_kri == "⚡ Energy Risk":
    st.subheader("📌 Parametri di simulazione Energy Risk")

    # Inizializza session_state per parametri Energy Risk se non esiste
    if "energy_params" not in st.session_state:
        st.session_state.energy_params = {
            "fabbisogno": [1548, 1557, 1373],
            "covered": [1408.6, 933.9, 619],
            "solar": [0, 203, 422],
            "forward_price": [115.99, 106.85, 94.00],
            "budget_price": [115, 121, 120]
        }

    # Funzione per trasformare lista in stringa
    def list_to_str(lst):
        return ",".join(map(str, lst))

    # Input manuale con valori salvati in session_state
    col1, col2 = st.columns(2)
    with col1:
        fabbisogno_input = st.text_input("Fabbisogno (MWh)", list_to_str(st.session_state.energy_params["fabbisogno"]))
        covered_input = st.text_input("Covered (MWh)", list_to_str(st.session_state.energy_params["covered"]))
        solar_input = st.text_input("Solar (MWh)", list_to_str(st.session_state.energy_params["solar"]))
    with col2:
        forward_input = st.text_input("Forward Price (€)", list_to_str(st.session_state.energy_params["forward_price"]))
        budget_input = st.text_input("Budget Price (€)", list_to_str(st.session_state.energy_params["budget_price"]))

    # Parsing input e aggiornamento session_state
    try:
        st.session_state.energy_params["fabbisogno"] = [float(x) for x in fabbisogno_input.split(",")]
        st.session_state.energy_params["covered"] = [float(x) for x in covered_input.split(",")]
        st.session_state.energy_params["solar"] = [float(x) for x in solar_input.split(",")]
        st.session_state.energy_params["forward_price"] = [float(x) for x in forward_input.split(",")]
        st.session_state.energy_params["budget_price"] = [float(x) for x in budget_input.split(",")]
    except Exception as e:
        st.error(f"❌ Errore nei parametri: {e}")
        st.stop()

    # Controllo lunghezze coerenti
    vals = st.session_state.energy_params
    if not (len(vals["fabbisogno"]) == len(vals["covered"]) == len(vals["solar"]) == len(vals["forward_price"]) == len(vals["budget_price"])):
        st.error("⚠️ Tutti i parametri devono avere lo stesso numero di valori per anno.")
        st.stop()

    st.success("✅ Parametri validi!")

    # Date e simulazioni
    end_date = st.date_input("Data finale simulazione", pd.to_datetime("2027-12-31"))
    start_date = pd.Timestamp.today().normalize()
    days_to_simulate = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    future_dates = pd.date_range(start=start_date, periods=days_to_simulate, freq='D')
    unique_years = sorted(future_dates.year.unique().tolist())

    n_simulations = st.number_input("Numero di simulazioni", min_value=100, max_value=100_000, value=10_000, step=100)
    n_trials_heston = st.number_input("Numero di trial Heston", min_value=10, max_value=1000, value=100, step=10)

    # Bottone simulazione
    if st.button("💹 Esegui simulazione Energy Risk"):
        st.info("Simulazione in corso...")

        # Caricamento file PUN (locale o uploader)
        data_path = "Data/Pun.xlsx"
        df_excel = None
        if os.path.exists(data_path):
            df_excel = pd.read_excel(data_path)
            st.success("🔄 Dati PUN caricati dal file locale")

        if df_excel is None or df_excel.empty:
            uploaded_file = st.file_uploader("Seleziona file Excel PUN", type=["xlsx"])
            if uploaded_file:
                df_excel = pd.read_excel(uploaded_file)
            else:
                st.warning("Carica il file Excel per procedere con la simulazione.")
                st.stop()

        # Controllo colonne
        if 'Date' not in df_excel.columns or 'GMEPIT24 Index' not in df_excel.columns:
            st.error("Il file deve contenere 'Date' e 'GMEPIT24 Index'.")
            st.stop()

        # Log Returns
        df_excel['Date'] = pd.to_datetime(df_excel['Date'])
        df_excel['Log_Returns'] = np.log(df_excel['GMEPIT24 Index'] / df_excel['GMEPIT24 Index'].shift(1))
        df_excel = df_excel.dropna(subset=['Log_Returns'])

        # Salva in session_state
        st.session_state.energy_df = df_excel

        # Simulazione Heston
        best_params, simulated_prices = run_heston(
            df_excel,
            n_trials=n_trials_heston,
            n_simulations=n_simulations,
            end_date=end_date
        )

        # Analisi distribuzione
        future_dates_sim = pd.date_range(
            start=df_excel['Date'].max(),
            periods=(pd.to_datetime(end_date) - df_excel['Date'].max()).days,
            freq='D'
        )
        simulated_df = pd.DataFrame(
            simulated_prices.T,
            index=future_dates_sim,
            columns=[f"Simulazione {i+1}" for i in range(n_simulations)]
        )

        monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig = analyze_simulation(simulated_df, unique_years)
        st.pyplot(fig)

        # Forecast e storico
        forecast_price = pd.DataFrame.from_dict(yearly_percentiles, orient='index', columns=['5%', '50%', '95%'])
        st.markdown("### 📊 Forecast Output")
        st.dataframe(forecast_price.style.background_gradient(cmap='Blues').format("{:.2f}"))

        historical_price = df_excel.groupby(df_excel['Date'].dt.year)['GMEPIT24 Index'].mean().tail(6).values.tolist()[:-1]
        df_historical = pd.DataFrame({"Year": [2020,2021,2022,2023,2024][:len(historical_price)], "Historical Price": historical_price})
        st.markdown("### 📅 Historical Price")
        st.dataframe(df_historical.style.background_gradient(cmap='Oranges').format("{:.2f}"))

        # Allineamento anni
        anni_prezzi = [2020,2021,2022,2023,2024] + unique_years
        predict_price = forecast_price['50%'].values.tolist()
        p95 = forecast_price['95%'].values.tolist()
        p5 = forecast_price['5%'].values.tolist()

        # Richiamo funzione principale
        df_risk = compute_downside_upperside_risk(
            unique_years,
            vals["fabbisogno"],
            vals["covered"],
            vals["solar"],
            anni_prezzi,
            historical_price,
            predict_price,
            p95,
            p5,
            vals["forward_price"],
            vals["budget_price"],
            observation_period=start_date.strftime("%d/%m/%Y")
        )

        st.markdown("### ⚠️ Analisi Rischio (Downside / Upside)")
        st.dataframe(df_risk.style.background_gradient(cmap='Reds'))


        # Grafico VaR EBITDA
        fig = var_ebitda_risk(periodo_di_analisi=end_date.strftime("as of %d/%m/%Y"), df_risk=df_risk, font_path="utils/TIMSans-Medium.ttf")
        st.pyplot(fig)

        st.success("Simulazione completata!")

        st.dataframe(df_risk.head())
        st.download_button("Scarica risultati Excel",
                           data=open("Simulation_VaR_results.xlsx", "rb").read(),
                           file_name="Simulation_VaR_results.xlsx")

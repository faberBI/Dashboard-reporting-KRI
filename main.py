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

# Library custom
from utils.data_loader import load_kri_excel, validate_kri_data
from functions.montecarlo import historical_VaR, run_heston, analyze_simulation, compute_downside_upperside_risk, var_ebitda_risk


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
    df = load_kri_excel(uploaded_file, selected_kri)  # funzione che definisci tu
    if validate_kri_data(df, selected_kri):          # funzione che definisci tu
        st.session_state.kri_data[selected_kri] = df
        st.success(f"✅ {selected_kri} aggiunto con successo!")

# -----------------------
# Mostra KRI caricati solo per il KRI selezionato
# -----------------------
if selected_kri in st.session_state.kri_data:
    df = st.session_state.kri_data[selected_kri]
    st.subheader(f"📌 {selected_kri}")  # Usa subheader o markdown una sola volta
    st.dataframe(df.head())



# -----------------------
# Analisi specifica per ENERGY RISK
# -----------------------
if selected_kri == "⚡ Energy Risk":
    st.subheader("📌 Parametri di simulazione Energy Risk")

    # Se esiste df già in session_state e c'è un file caricato
    if selected_kri in st.session_state.kri_data and uploaded_file:
        df = st.session_state.kri_data[selected_kri]
    else:
        # Se non ci sono dati, crea un DataFrame vuoto con valori di default
        st.warning("⚠️ Nessun file Excel caricato: usare i valori di default o inserire manualmente i dati")
        df = pd.DataFrame({
            "Anno": [2025, 2026, 2027],
            "Fabbisogno": [1548, 1557, 1373],
            "Covered": [1408.6, 933.9, 619],
            "Solar": [0, 203, 422],
            "Forward Price": [115.99, 106.85, 94.00],
            "Budget Price": [115, 121, 120]
        })
        st.session_state.kri_data[selected_kri] = df

    # -----------------------
    # Date e simulazioni
    # -----------------------
    end_date = st.date_input("Data finale simulazione", pd.to_datetime("2027-12-31"))
    start_date = pd.Timestamp.today().normalize()
    days_to_simulate = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    future_dates = pd.date_range(start=start_date, periods=days_to_simulate, freq='D')
    unique_years = sorted(future_dates.year.unique().tolist())

    n_simulations = st.number_input("Numero di simulazioni", min_value=100, max_value=100_000, value=10_000, step=100)
    n_trials_heston = st.number_input("Numero di trial Heston", min_value=10, max_value=1000, value=100, step=10)

    # -----------------------
    # Parametri aggiuntivi (da Excel se presenti, altrimenti input manuali)
    # -----------------------
    def df_to_str(df, col_name, default):
        if col_name in df.columns:
            values = df[col_name].dropna().tolist()
            if len(values) > 0:
                return ",".join(map(str, values))
        return default

    default_fabbisogno = df_to_str(df, "Fabbisogno", "1548,1557,1373")
    default_covered = df_to_str(df, "Covered", "1408.6,933.9,619")
    default_solar = df_to_str(df, "Solar", "0,203,422")
    default_forward_price = df_to_str(df, "Forward Price", "115.99,106.85,94.00")
    default_budget_price = df_to_str(df, "Budget Price", "115,121,120")

    st.markdown("### Inserimento manuale parametri ⚡")
    col1, col2 = st.columns(2)
    with col1:
        fabbisogno = st.text_input("Fabbisogno (MWh)", default_fabbisogno)
        covered = st.text_input("Covered (MWh)", default_covered)
        solar = st.text_input("Solar (MWh)", default_solar)
    with col2:
        forward_price = st.text_input("Forward Price (€)", default_forward_price)
        budget_price = st.text_input("Budget Price (€)", default_budget_price)

    # -----------------------
    # Parsing input
    # -----------------------
    try:
        fabbisogno = [float(x) for x in fabbisogno.split(",")]
        covered = [float(x) for x in covered.split(",")]
        solar = [float(x) for x in solar.split(",")]
        forward_price = [float(x) for x in forward_price.split(",")]
        budget_price = [float(x) for x in budget_price.split(",")]
    except Exception as e:
        st.error(f"❌ Errore nei parametri: {e}")
        st.stop()

    # Controllo lunghezze coerenti
    if not (len(fabbisogno) == len(covered) == len(solar) == len(forward_price) == len(budget_price)):
        st.error("⚠️ Tutti i parametri devono avere lo stesso numero di valori per anno.")
        st.stop()

    st.success("✅ Parametri validi, pronti per la simulazione!")

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
        best_params, simulated_prices = run_heston(
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

        # simulated_df = simulated_df.mask((simulated_df < 35) | (simulated_df >= 200))

        # Analisi distribuzione
        monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig = analyze_simulation(simulated_df, unique_years)
        st.pyplot(fig)

        # --------------------------------------
        # Forecast + Storico
        # --------------------------------------

        forecast_price = df = pd.DataFrame.from_dict(yearly_percentiles, orient='index', columns=['5%', '50%', '95%'])
        st.markdown("### 📊 Forecast Output")  # titolo con icona
        st.info("Questi sono i valori previsionali basati sui percentili annuali.")  # box informativo

        # Mostra il DataFrame con stile
        st.dataframe(forecast_price.style.background_gradient(cmap='Greens', low=0.1, high=0.4).format("{:.2f}"))

        # Combinazione anni storico + forecast
        anni_prezzi = [2020, 2021, 2022, 2023, 2024] + unique_years
        anni_prezzi = [int(y) for y in anni_prezzi]

        # Media storica PUN per gli anni storici
        historical_price = df_filtered.groupby(df_filtered['Date'].dt.year)['GMEPIT24 Index'].mean().tail(6).values.tolist()
        historical_price = historical_price[:-1]

        st.markdown("### 📅 Historical Price")  # Titolo con icona
        st.info("Media storica PUN per gli anni disponibili.")  # Box informativo
        df_historical = pd.DataFrame({"Historical Price": historical_price, "Year": anni_prezzi[:len(historical_price)]})
        # Mostra il DataFrame con sfumatura di colori
        st.dataframe(df_historical.style.background_gradient(cmap='Greens', low=0.1, high=0.4).format("{:.2f}"))

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
        df_risk, df_open, df_prezzi, fig = compute_downside_upperside_risk(
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
        observation_period=start_date.strftime("%d/%m/%Y"))

        # Visualizzazione su Streamlit
        st.pyplot(fig)
        st.markdown("### 📈 Analisi Prezzi PUN ")
        st.info("Tabella contenente media PUN, percentili, Forward e Budget per ogni anno.")

        # Colori per evidenziare i valori più alti
        st.dataframe(df_prezzi)
        st.markdown("### ⚠️ Analisi Rischio (Downside / Upside)")
        st.info("Valori di rischio calcolati in base alle differenze tra percentili, budget e open position.")

        # Colori per evidenziare i valori più alti
        st.dataframe(
        df_risk.style.background_gradient(cmap='Greens', low=0.1, high=0.4,subset=df_risk.columns[1:]).format("{:.0f}")  # Senza decimali, più leggibile per valori grandi
        )

        # Grafico VaR EBITDA
        fig = var_ebitda_risk(periodo_di_analisi=end_date.strftime("as of %d/%m/%Y"), df_risk=df_risk, font_path="utils/TIMSans-Medium.ttf")
        st.pyplot(fig)

        st.success("Simulazione completata!")

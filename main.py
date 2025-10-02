import streamlit as st
import pandas as pd
from utils.data_loader import load_kri_excel, validate_kri_data
from montecarlo import (
    historical_VaR, run_heston, analyze_simulation,
    compute_downside_upperside_risk, var_ebitda_risk
)

st.set_page_config(page_title="Dashboard KRI", layout="wide")
st.title("Dashboard KRI")

# -----------------------
# Selezione KRI
# -----------------------
kri_options = ["Energy Risk", "Natural Event Risk", "KRI 3"]

if "kri_data" not in st.session_state:
    st.session_state.kri_data = {}

selected_kri = st.sidebar.selectbox("Seleziona KRI", kri_options)

uploaded_file = st.sidebar.file_uploader(f"Carica Excel per {selected_kri}", type="xlsx", key=selected_kri)

if uploaded_file:
    df = load_kri_excel(uploaded_file, selected_kri)
    if validate_kri_data(df, selected_kri):
        st.session_state.kri_data[selected_kri] = df
        st.success(f"{selected_kri} aggiunto con successo!")

# Mostra KRI caricati
if st.session_state.kri_data:
    st.subheader("KRI caricati")
    for kri_name, df in st.session_state.kri_data.items():
        st.write(f"**{kri_name}**")
        st.dataframe(df.head())

# -----------------------
# Parametri di simulazione
# -----------------------
st.subheader("Parametri di simulazione")

if uploaded_file:
    # Date configurabili
    start_date = st.date_input("Data iniziale storico", df['Date'].min())
    end_date = st.date_input("Data finale simulazione", pd.to_datetime("2027-12-31"))

    days_to_simulate = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    future_dates = pd.date_range(start=df['Date'].max(), periods=days_to_simulate, freq='D')
    unique_years = sorted(future_dates.year.unique().tolist())

    # Parametri Monte Carlo
    n_simulations = st.number_input("Numero di simulazioni", min_value=100, max_value=100_000, value=10_000, step=100)
    n_trials_heston = st.number_input("Numero di trial Heston", min_value=10, max_value=1000, value=100, step=10)

    # Parametri aggiuntivi
    fabbisogno = st.text_input("Fabbisogno", "1548,1557,1373")
    covered = st.text_input("Covered", "1408.6,933.9,619")
    solar = st.text_input("Solar", "0,203,422")
    forward_price = st.text_input("Forward Price", "115.99,106.85,94.00")
    budget_price = st.text_input("Budget Price", "115,121,120")

    # Conversione string -> lista numerica
    fabbisogno = [float(x) for x in fabbisogno.split(",")]
    covered = [float(x) for x in covered.split(",")]
    solar = [float(x) for x in solar.split(",")]
    forward_price = [float(x) for x in forward_price.split(",")]
    budget_price = [float(x) for x in budget_price.split(",")]

    # Bottone per lanciare simulazione
    if st.button("Esegui simulazione"):
        st.info("Simulazione in corso...")

        # Filtra storico
        df_filtered = df[df['Date'] >= pd.to_datetime(start_date)]

        # 1. Calcolo VaR storico
        rendimenti_giornalieri = df_filtered['Rendimenti']  # Assumendo tu abbia la colonna "Rendimenti"
        results_df = historical_VaR(rendimenti_giornalieri, n_simulations=n_simulations, csv_file="VaR_results.csv")

        # 2. Simulazione Heston
        best_params, simulated_prices = run_heston(df_filtered, n_trials=n_trials_heston,
                                                   n_simulations=n_simulations, end_date=end_date)
        simulated_df = pd.DataFrame(simulated_prices.T, index=future_dates,
                                    columns=[f"Simulazione {i+1}" for i in range(n_simulations)])
        simulated_df = simulated_df.mask((simulated_df < 35) | (simulated_df >= 200))

        # 3. Analisi distribuzione
        monthly_percentiles, monthly_means, yearly_percentiles, yearly_means = analyze_simulation(
            simulated_df, years=unique_years,
            output_file="distribution_plot.png",
            csv_file_m="monthly_percentiles.csv",
            csv_file_y="yearly_percentiles.csv"
        )

        forecast_price = pd.read_csv("yearly_percentiles.csv")

        # Storico + forecast
        anni_prezzi = sorted(df_filtered['Year'].unique().tolist()) + unique_years
        historical_price = df_filtered.groupby('Year')['GMEPIT24 Index'].mean().tail(len(anni_prezzi)).tolist()
        predict_price = forecast_price['50%'].tolist()
        p95 = forecast_price['95%'].tolist()
        p5 = forecast_price['5%'].tolist()

        # Adeguamento lunghezza
        missing_len_hp = len(anni_prezzi) - len(historical_price)
        missing_len_b = len(anni_prezzi) - len(budget_price)
        missing_len_f = len(anni_prezzi) - len(forward_price)
        missing_len_pp = len(anni_prezzi) - len(predict_price)
        missing_len_p95 = len(anni_prezzi) - len(p95)
        missing_len_p5 = len(anni_prezzi) - len(p5)

        historical_price += [0] * missing_len_hp
        budget_price = [0] * missing_len_b + budget_price
        forward_price = [0] * missing_len_f + forward_price
        predict_price += [0] * missing_len_pp
        p95 += [0] * missing_len_p95
        p5 += [0] * missing_len_p5

        # 4. Calcolo rischio
        df_risk = compute_downside_upperside_risk(
            unique_years, fabbisogno, covered, solar,
            anni_prezzi, historical_price, predict_price, p95, p5, forward_price, budget_price,
            observation_period=start_date.strftime("%d/%m/%Y"),
            chart_path="simulation_chart.png",
            output_path=f"Simulation_VaR_results.xlsx"
        )

        # 5. Grafico VaR EBITDA
        var_ebitda_risk(periodo_di_analisi=end_date.strftime("as of %d/%m/%Y"),
                        df_risk=df_risk,
                        font_path="TIMSans-Medium.ttf",
                        output_file="var_ebitda_risk.png")

        st.success("Simulazione completata!")
        st.image("var_ebitda_risk.png", caption="VaR EBITDA Risk")
        st.dataframe(df_risk.head())

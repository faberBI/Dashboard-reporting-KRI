import streamlit as st
import pandas as pd
import sys
import os
from utils.data_loader import load_kri_excel, validate_kri_data
from functions.montecarlo import historical_VaR, run_heston, analyze_simulation, compute_downside_upperside_risk, var_ebitda_risk



# -----------------------
# Configurazione Streamlit
# -----------------------

st.set_page_config(page_title="Dashboard KRI", layout="wide")
st.title("Dashboard KRI")

# -----------------------
# Selezione KRI
# -----------------------
kri_options = ["Energy Risk", "Natural Event Risk", "KRI 3"]

if "kri_data" not in st.session_state:
    st.session_state.kri_data = {}

selected_kri = st.sidebar.selectbox("Seleziona KRI", kri_options)

uploaded_file = st.sidebar.file_uploader(
    f"Carica Excel per {selected_kri}", type="xlsx", key=selected_kri
)

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
# Analisi specifica per ENERGY RISK
# -----------------------
if selected_kri == "Energy Risk":
    st.subheader("Parametri di simulazione Energy Risk")

    # Controlla se i dati KRI sono caricati
    if "kri_data" not in st.session_state:
        st.session_state.kri_data = {}

    if selected_kri in st.session_state.kri_data and uploaded_file:
        df = st.session_state.kri_data[selected_kri]
    else:
        # Se non ci sono dati, crea un DataFrame vuoto con colonne minime
        st.warning("Nessun file Excel caricato: usare i valori di default o inserire manualmente i dati")
        df = pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=3, freq="Y"),
            "Fabbisogno": [1548, 1557, 1373],
            "Covered": [1408.6, 933.9, 619],
            "Solar": [0, 203, 422],
            "Forward Price": [115.99, 106.85, 94.00],
            "Budget Price": [115, 121, 120]
        })
        st.session_state.kri_data[selected_kri] = df

    # Date configurabili
    start_date = st.date_input("Data iniziale storico", df['Date'].min())
    end_date = st.date_input("Data finale simulazione", pd.to_datetime("2027-12-31"))

    days_to_simulate = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    future_dates = pd.date_range(start=df['Date'].max(), periods=days_to_simulate, freq='D')
    unique_years = sorted(future_dates.year.unique().tolist())

    # Parametri Monte Carlo
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

    fabbisogno = st.text_input("Fabbisogno", default_fabbisogno)
    covered = st.text_input("Covered", default_covered)
    solar = st.text_input("Solar", default_solar)
    forward_price = st.text_input("Forward Price", default_forward_price)
    budget_price = st.text_input("Budget Price", default_budget_price)

    try:
        fabbisogno = [float(x) for x in fabbisogno.split(",")]
        covered = [float(x) for x in covered.split(",")]
        solar = [float(x) for x in solar.split(",")]
        forward_price = [float(x) for x in forward_price.split(",")]
        budget_price = [float(x) for x in budget_price.split(",")]
    except Exception as e:
        st.error(f"Errore nei parametri: {e}")
        st.stop()


    # -----------------------
    # Lancia simulazione
    # -----------------------
    if st.button("Esegui simulazione Energy Risk"):
        st.info("Simulazione in corso...")
    # ---------------------------
    # Caricamento file Excel
    # ---------------------------
        uploaded_file = st.file_uploader("Seleziona il file Excel PUN", type=["xlsx"])
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                
                # Controllo colonne obbligatorie
                if 'Date' not in df.columns or 'GMEPIT24 Index' not in df.columns:
                    st.error("Il file Excel deve contenere le colonne 'Date' e 'GMEPIT24 Index'.")
                    st.stop()
                
                # Assicurati che la colonna Date sia in formato datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Calcolo Log_Returns
                df['Log_Returns'] = np.log(df['GMEPIT24 Index'] / df['GMEPIT24 Index'].shift(1))
                
                st.success("Dati caricati correttamente!")
                st.dataframe(df.head())
                
                # Adesso df può essere passato a run_heston
                # best_params, simulated_prices = run_heston(df, ...)
        
            except Exception as e:
                st.error(f"Errore caricamento Excel: {e}")
                st.stop()
        else:
            st.warning("Carica il file Excel per procedere.")

        df = df_filtered 
        # 3. Simulazione Heston
        best_params, simulated_prices = run_heston(df_filtered,
                                                   n_trials=n_trials_heston,
                                                   n_simulations=n_simulations,
                                                   end_date=end_date)
        simulated_df = pd.DataFrame(simulated_prices.T, index=future_dates,
                                    columns=[f"Simulazione {i+1}" for i in range(n_simulations)])
        simulated_df = simulated_df.mask((simulated_df < 35) | (simulated_df >= 200))

        # 4. Analisi distribuzione
        monthly_percentiles, monthly_means, yearly_percentiles, yearly_means, fig = analyze_simulation(simulated_df, unique_years)
        st.pyplot(fig)

        forecast_price = yearly_percentiles

        # Storico + forecast
        anni_prezzi = sorted(df_filtered['Year'].unique().tolist()) + unique_years
        historical_price = df_filtered.groupby('Year')['GMEPIT24 Index'].mean().tail(len(anni_prezzi)).tolist()
        predict_price = forecast_price['50%'].tolist()
        p95 = forecast_price['95%'].tolist()
        p5 = forecast_price['5%'].tolist()

        # Adeguamento lunghezze
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

        # 5. Calcolo rischio
        df_risk, df_open, df_prezzi, fig = compute_downside_upperside_risk(
            unique_years, fabbisogno, covered, solar,
            anni_prezzi, historical_price, predict_price, p95, p5,
            forward_price, budget_price,
            observation_period=start_date.strftime("%d/%m/%Y")
            )

        st.pyplot(fig)
        st.dataframe(df_risk)
        st.dataframe(df_open)
        st.dataframe(df_prezzi)

        # 6. Grafico VaR EBITDA
        fig = var_ebitda_risk(periodo_di_analisi=end_date.strftime("as of %d/%m/%Y"), df_risk=df_risk, font_path="TIMSans-Medium.ttf")
        st.pyplot(fig)

        # -----------------------
        # Output a video
        # -----------------------
        st.success("Simulazione completata!")
        st.image("var_ebitda_risk.png", caption="VaR EBITDA Risk")
        st.dataframe(df_risk.head())
        st.download_button("Scarica risultati Excel",
                           data=open("Simulation_VaR_results.xlsx", "rb").read(),
                           file_name="Simulation_VaR_results.xlsx")

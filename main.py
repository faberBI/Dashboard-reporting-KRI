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
    plot_cvar_reduction_over_iterations
)
from functions.copper import (make_lag_df, monte_carlo_forecast_cp_from_disk, plot_copper_forecast, plot_var_vs_budget, full_copper_forecast)
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
kri_options = ["⚡ Energy Risk", "🌪️ Natural Event Risk", "🟠 Copper Price", "🛡️💻 Cyber","💳 Credit risk" ,"📈 Interest Rate", "Liquidity Risk💰"]

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
    # Verifica che il DataFrame non sia vuoto
    if df.empty:
        st.warning("⚠️ Nessun dato disponibile nel DataFrame!")
    else:
        # Se la colonna Ebitda non esiste, la aggiunge con un valore predefinito
        agg_dict = {
            "Fabbisogno": "sum",
            "PPA Erg": "sum",
            "Forward": "sum",
            "Solar": "sum",
            "Prezzo Forward": "mean",
            "Prezzo Budget": "mean"}

        # Group by per anno
        df_grouped = df.groupby("Anno").agg(agg_dict).reset_index()
        if "Ebitda" not in df_grouped.columns:
            df_grouped["Ebitda"] = 1_900_000_000

        # Dizionario per i valori inseriti
        ebitda_inputs = {}

        # Crea un campo numerico per ogni anno
        for i, row in df_grouped.iterrows():
            anno = int(row["Anno"]) if "Anno" in df_grouped.columns else (2025 + i)
            default_value = float(row["Ebitda"])

            ebitda_inputs[anno] = st.number_input(
                f"EBITDA per {anno} (€)",
                min_value=0.0,
                value=default_value,
                step=1_000_000.0,
                format="%.0f"
            )

    # Aggiorna la colonna Ebitda con i valori inseriti
        df_grouped["Ebitda"] = [ebitda_inputs[anno] for anno in df_grouped["Anno"]]

    # Mostra il DataFrame aggiornato
        st.dataframe(df)
        st.dataframe(df_grouped.style.format({"Ebitda": "€{:,.0f}"}))

    st.success("✅ Parametri validi, pronti per la simulazione!")

    # -----------------------
    # Parametri simulazione
    # -----------------------
    import random
    random.seed(42)
    
    n_simulations = st.number_input("Numero di simulazioni", min_value=1000, max_value=1000_000, value=10_000, step=1000)
    risk_appetite = st.number_input("Risk appetite - Max loss in % of EBIDA", min_value=0.005, max_value=0.2, value=0.01, step=0.001)
    step = st.number_input("MWh to buy each step of Optmization", min_value= 1000, max_value=10000, value=5000, step=1000)
    alpha = st.number_input("% minima di copertura del fabbisogno", min_value= 0.5, max_value=1.0, value=0.85, step=0.05)
    
    n_year = len(df['Anno'].unique())
    st.metric(label="Numero di anni da simulare", value=n_year)
    
    start_date = st.date_input("Dati aggiornati al", pd.Timestamp.today().date())
    start_date_sim = pd.Timestamp.today().normalize()

    # -----------------------------------------------------------
    # PULSANTE SIMULAZIONE
    # -----------------------------------------------------------
    
    if st.button("💹 Esegui simulazione Energy Risk"):
        
        st.info("⏳ Simulazione in corso...")
    
        # ---------------------------------------
        # LETTURA FILE EXCEL
        # ---------------------------------------
        data_path = "Data/Pun_ts_price.xlsx"
        df_excel = None
    
        if os.path.exists(data_path):
            df_excel = pd.read_excel(data_path)
            st.success("📊 Dati PUN caricati dal percorso predefinito.")
        else:
            uploaded_file = st.file_uploader("Carica il file Excel PUN", type=["xlsx"])
            if uploaded_file is None:
                st.warning("⚠️ Carica un file per procedere.")
                st.stop()
            df_excel = pd.read_excel(uploaded_file)
    
        # Controllo colonne richieste
        if "Date" not in df_excel.columns or "GMEPIT24 Index" not in df_excel.columns:
            st.error("❌ Il file deve contenere le colonne 'Date' e 'GMEPIT24 Index'.")
            st.stop()
    
        # Preprocessing
        df_excel["Date"] = pd.to_datetime(df_excel["Date"])
        st.session_state.energy_df = df_excel
    
        if df_excel.empty:
            st.error("❌ Il dataset filtrato è vuoto.")
            st.stop()

        np.random.seed(42)
        last_5y, monthly_std, monthly_price = get_return(data_path)
        st.success("✅ Statistiche calcolate!")
        L, rho_hat = apply_cholesky(last_5y)
        hist_pun = pd.read_excel(data_path)
        hist_pun["log_return"] = np.log(hist_pun["GMEPIT24 Index"] / hist_pun["GMEPIT24 Index"].shift(1))
        hist_pun["Month"] = hist_pun["Date"].dt.month
        hist_filter = hist_pun[hist_pun['Year']>2015]
        series = hist_filter.set_index('Date')['GMEPIT24 Index'].astype(float).dropna()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series.index, series.values, linewidth=2)
        ax.set_title("Serie Storica PUN mensile", fontsize=13)
        ax.set_xlabel("Data")
        ax.set_ylabel("Prezzo")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        PUN_monthly_forecast = forecast_monthly_prices(series, n_years=n_year)
        st.success("✅ Modello ibrido allenato!")
        st.subheader("📈 Prezzo PUN e Volatilità")
        monthly_sigma, rolling_std, sigma_t = get_garch(last_5y)
        st.success("✅ Volatilità stimata!")
        plot_volatility(rolling_std, sigma_t)
        st.subheader("📈 Forecast Hybrid Model")
        PUN_paths, shocks = simulate_prices(PUN_monthly_forecast, monthly_price['avg_price'].values,
                                        monthly_sigma, monthly_std, L, n_sims=n_simulations)
        VaR_95_monthly = np.percentile(PUN_paths, 95, axis=0)
        st.success("✅ VaR al 95 percentile calcolato!")
        dati_fibercop = compute_VaR(df, VaR_95_monthly)
        dati_fibercop['Anno_Mese'] = dati_fibercop['Anno'].astype(str) + "-" + dati_fibercop['Month']
        st.dataframe(dati_fibercop.drop(['Anno_Mese'], axis=1))
        st.subheader("📈 Grafico VaR mensile")
        plot_monthly_VaR(VaR_95_monthly, start_year=2026)
        
        fig = plot_energy_stack_with_var(dati_fibercop)
        st.pyplot(fig, use_container_width=True)
        fig_plot_var = plot_var_bars(dati_fibercop)
        
        st.metric( label="Yearly Value@Risk with Solar",value=f"€ {np.round(dati_fibercop['Var_monthly_95_w_solar'].sum(), 0):,.0f}")
        st.metric(label="Yearly Value@Risk w/o Solar",value=f"€ {np.round(dati_fibercop['Var_monthly_95_w/o_solar'].sum(), 0):,.0f}")

        st.subheader("📈 Hedging Optimization Model")
        df = dati_fibercop.copy()
        df["Fabbisogno"] *= 1000
        df["Copertura_base"] = df['PPA Erg']+df['Forward']+ df['Solar']
        df["Copertura_base"] *= 1000
        df["Scoperto_base"] = df['scoperto_w_solar']
        df["Scoperto_base"] *= 1000
        CVaR_limit = df_grouped["Ebitda"] * risk_appetite
        
        # fixed for algorithm optimization
        df["hedge_cost"] = df["Prezzo Forward"] - df["Prezzo Budget"]
        hedge = np.zeros(12, dtype=float)
        
        total_fabbisogno = df["Fabbisogno"].sum()
        max_copertura_totale = total_fabbisogno * alpha
        coperto_attuale = df["Copertura_base"].sum()
        
        max_needed = max(max_copertura_totale - coperto_attuale, 0)
        
        weights = df["Scoperto_base"].values / df["Scoperto_base"].sum()
        hedge += max_needed * weights
        
        max_hedge_mensile = df["Fabbisogno"].values * 0.85 - df["Copertura_base"].values
        hedge = np.minimum(hedge, max_hedge_mensile)
        
        # saturazione residua
        residuo = max_copertura_totale - (coperto_attuale + hedge.sum())
        
        while residuo > 0:
            spazio = np.maximum(max_hedge_mensile - hedge, 0)
            if spazio.sum() == 0:
                break
            
            incremento = residuo * spazio / spazio.sum()
            hedge += incremento
            hedge = np.minimum(hedge, max_hedge_mensile)
            residuo = max_copertura_totale - (coperto_attuale + hedge.sum())
        
        CVaR_current = compute_CVaR(hedge)
        iteration = 0
        log = []
        
        while CVaR_current > CVaR_limit:
            iteration += 1
        
            spazio = max_hedge_mensile - hedge
            admissible = (spazio >= step) & (df["hedge_cost"].values > 0)
        
            if not admissible.any():
                st.warning("⚠️ Nessun ulteriore miglioramento possibile.")
                break
            
            CVaR_test = np.full(12, np.nan)
        
            for m in np.where(admissible)[0]:
                hedge_test = hedge.copy()
                hedge_test[m] += step
        
                copertura_annua = (coperto_attuale + hedge_test.sum()) / total_fabbisogno
                if copertura_annua <= alpha:
                    CVaR_test[m] = compute_CVaR(hedge_test)
        
            delta_CVaR = CVaR_current - CVaR_test
            efficiency = delta_CVaR / (step * df["hedge_cost"].values)
        
            best_month = np.nanargmax(efficiency)
        
            if not np.isfinite(efficiency[best_month]) or efficiency[best_month] <= 0:
                break
            
            hedge[best_month] += step
            CVaR_current = CVaR_test[best_month]
        
            hedge_tot = hedge.sum()
        
            log.append({
                "Iterazione": iteration,
                "Mese": df.loc[best_month, "Month"],
                "Hedge_tot_MWh": hedge_tot,
                "CVaR_€": CVaR_current,
                "CVaR_%_EBITDA": CVaR_current / EBITDA * 100,
                "Copertura_annua_%": (coperto_attuale + hedge_tot) / total_fabbisogno * 100
            })
        
        df["Hedge_addizionale_MWh"] = hedge
        df["Copertura_totale"] = df["Copertura_base"] + hedge
        df["Scoperto_finale"] = df["Fabbisogno"] - df["Copertura_totale"]
        
        total_hedge_cost = np.sum(hedge * df["hedge_cost"].values)
        copertura_annua_pct = df["Copertura_totale"].sum() / total_fabbisogno * 100
        
        st.metric("CVaR finale (€)", f"{CVaR_current:,.0f}")
        st.metric("CVaR / EBITDA (%)", f"{CVaR_current / EBITDA * 100:.2f}%")
        st.metric("Costo hedge totale (€)", f"{total_hedge_cost:,.0f}")
        st.metric("Copertura annua (%)", f"{copertura_annua_pct:.2f}%")
        
        st.subheader("📊 Dettaglio mensile hedge")
        st.dataframe(
            df[[
                "Month", "Fabbisogno", "Copertura_base",
                "Hedge_addizionale_MWh", "Copertura_totale",
                "Scoperto_finale", "hedge_cost"
            ]]
        )
        st.subheader("📊 Copertura energetica mensile")
        plot_monthly_coverage_stack(df)
        st.subheader("📈 Hedge addizionale mensile")
        plot_monthly_additional_hedge(df)
        st.subheader("💰 Riduzione del rischio (CVaR) per iterazione")
        plot_cvar_reduction_over_iterations(log)
        
        # Esportazione Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            last_5y.to_excel(writer, sheet_name="last_5y", index=False)
            monthly_std.to_excel(writer, sheet_name="monthly_std", index=False)
            monthly_price.to_excel(writer, sheet_name="monthly_price", index=False)
            pd.DataFrame(PUN_monthly_forecast).to_excel(writer, sheet_name="PUN_forecast", index=False)
            pd.DataFrame(monthly_sigma, columns=['monthly_sigma']).to_excel(writer, sheet_name="monthly_sigma", index=False)
            pd.DataFrame(PUN_paths).to_excel(writer, sheet_name="PUN_paths", index=False)
            dati_fibercop.to_excel(writer, sheet_name="dati_var", index=False)
            pd.DataFrame(VaR_95_monthly, columns=['VaR_95']).to_excel(writer, sheet_name="VaR_95_monthly", index=False)
            pd.DataFrame(shocks).to_excel(writer, sheet_name="shocks", index=False)
            pd.DataFrame(df).to_excel(writer, sheet_name="Hedging", index=False)
        buffer.seek(0)
    
        st.download_button("💾 Scarica tutti i dati in Excel", data=buffer,
                           file_name="Energy_Risk_VaR.xlsx",
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

    fig = full_copper_forecast(link_df="Data/copper_price.xlsx", price_col='Copper', N_SIM=10000, alpha=0.05, DIST="ged", calibration_size_pct=0.05)
    st.pyplot(fig)
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

        result_df , result_df_annual = monte_carlo_forecast_cp_from_disk(series,
                                      cat_model_path="utils/catboost_model.cbm",
                                      garch_model_path="utils/garch_model.pkl",
                                      params_path="utils/model_params.pkl",
                                      N_SIM=n_sims, alpha=0.05,
                                      end_date=end_date, random_seed=42)

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
        train_idx = df_dropped.index[:len(y_pred_train)]
        plt.plot(train_idx, y_pred_train, label="Train Pred", color='blue')
        # Validation
        val_start = len(y_pred_train)
        val_end_idx = val_start + len(y_pred_val)
        val_idx = df_dropped.index[val_start:val_end_idx]
        plt.plot(val_idx, y_pred_val, label="Val Pred", color='orange')
        # Test
        test_idx = df_dropped.index[-len(y_pred_test):]
        plt.plot(test_idx, y_pred_test, label="Test Pred", color='green')
        plt.title("Serie originale vs Predizione completa")
        plt.legend()
        plt.grid(True)
        # Render in Streamlit
        st.pyplot(plt.gcf())
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
    df_yahoo = download_yahoo_series(yahoo_symbols, start = '2021-01-01')
    df_all = df_ecb.join(df_yahoo, how="outer")
    df_all = df_all.sort_index().ffill()
    df_dropped = df_all.dropna()
    
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
    def simulate_euribor(series, df_dropped, n_sims=1000, alpha=0.05, horizon_days=3*360, plan_euribor_df=None):
        np.random.seed(234)
        
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
    
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=234))
        study.optimize(objective, n_trials=1000)
    
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
        mean = np.mean(simulations, axis=0)
    
        # Conformal adjustment
        calibration_y = series[-252:]
        np.random.seed(234)
        samples_cal = np.random.choice(simulations.flatten(), size=(len(calibration_y), n_period))
        lower_cal = np.percentile(samples_cal, 2.5, axis=1)
        upper_cal = np.percentile(samples_cal, 97.5, axis=1)
        nonconformity = np.maximum(lower_cal - calibration_y, calibration_y - upper_cal)
        q_hat = np.quantile(np.append(nonconformity, np.inf), 0.95)
    
        lower_adj = lower_emp - q_hat
        upper_adj = upper_emp + q_hat
    
        while np.any(upper_adj <= lower_adj):
            mask = upper_adj <= lower_adj
            upper_adj[mask] = lower_adj[mask] + 0.05
    
        
        idx = pd.date_range(start=df_dropped.index[-1] + pd.Timedelta(days=1), periods=n_period, freq="D")
    
        if plan_euribor_df is not None:
            plan_euribor_df['Tasso'] = plan_euribor_df['Tasso'].astype(float)

        # Serie giornaliera basata sul tasso fisso annuale
        plan_rate_series = pd.Series(
            index=idx,
            data=[plan_euribor_df.loc[plan_euribor_df['Anno'] == d.year, 'Tasso'].values[0] for d in idx]
        )

    
        forecast_df = pd.DataFrame({
            "lower_emp": lower_emp,
            "upper_emp": upper_emp,
            "median": median,
            'mean': mean,
            "lower_adj": lower_adj,
            "upper_adj": upper_adj
        }, index=idx)
        
        forecast_quarterly = forecast_df.resample("Q").mean()
        # Media ponderata solo sulla colonna 'median'
        forecast_quarterly['median'] = (forecast_quarterly['median'] * 0.5+ plan_rate_series.resample("Q").mean() * 0.5)
        plan_q = plan_rate_series.resample("Q").mean()
        forecast_quarterly['lower_adj'] = (forecast_quarterly['lower_adj']*0.85)
    
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
    
    def get_spread_for_date(date, spread_df):
        row = spread_df[(spread_df["From"] <= date) & (spread_df["To"] > date)]
        if not row.empty:
            return float(row["Spread"].iloc[0])
        else:
            return float(spread_df["Spread"].iloc[-1]) 

    def get_plan_euribor_for_date(date, plan_df):
        year = date.year

        # Ordiniamo per sicurezza
        plan_df_sorted = plan_df.sort_values("Anno")

        # Se l'anno è presente → uso diretto
        if year in plan_df_sorted["Anno"].values:
            return plan_df_sorted.loc[plan_df_sorted["Anno"] == year, "Tasso"].values[0]

        # Se l'anno è successivo all'ultimo disponibile → ultimo valore noto
        if year > plan_df_sorted["Anno"].max():
            return plan_df_sorted.iloc[-1]["Tasso"]

        # Se l'anno è precedente al primo disponibile → primo valore noto
        if year < plan_df_sorted["Anno"].min():
            return plan_df_sorted.iloc[0]["Tasso"]

        # Fallback 
        return np.nan

    
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
        perdita_totale_perc = perdita_totale_mln/unhedged_total
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
elif selected_kri == "Liquidity Risk💰":
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

        
    
    

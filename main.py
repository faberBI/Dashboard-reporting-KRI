import streamlit as st
from utils.data_loader import load_kri_excel, validate_kri_data

st.title("Dashboard KRI")

# Lista dei KRI pre-impostati
kri_options = ["Energy Risk", "Natural Event Risk", "KRI 3"]

# Inizializza session_state per salvare dati caricati
if "kri_data" not in st.session_state:
    st.session_state.kri_data = {}

# Selezione KRI dal menu a tendina
selected_kri = st.sidebar.selectbox("Seleziona KRI", kri_options)

# Caricamento Excel per il KRI selezionato
uploaded_file = st.sidebar.file_uploader(f"Carica Excel per {selected_kri}", type="xlsx", key=selected_kri)

if uploaded_file:
    df = load_kri_excel(uploaded_file, selected_kri)
    if validate_kri_data(df, selected_kri):
        # Salva nel session_state
        st.session_state.kri_data[selected_kri] = df
        st.success(f"{selected_kri} aggiunto con successo!")

# Mostra tutti i KRI caricati finora
if st.session_state.kri_data:
    st.subheader("KRI caricati")
    for kri_name, df in st.session_state.kri_data.items():
        st.write(f"**{kri_name}**")
        st.dataframe(df.head())


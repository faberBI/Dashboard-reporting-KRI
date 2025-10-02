import pandas as pd
import streamlit as st

# Definizione delle colonne richieste per ogni KRI
KRI_COLUMNS = {
    "KRI 1": ["Parametro1", "Parametro2", "Parametro3"],
    "KRI 2": ["ParametroA", "ParametroB"],
    "KRI 3": ["Value", "Weight", "Frequency"]
}

def load_kri_excel(file, kri_name):
    """
    Carica il file Excel specifico per un KRI e valida le colonne richieste.
    
    Parameters
    ----------
    file : UploadedFile o percorso locale
        File Excel caricato dall'utente tramite Streamlit o locale.
    kri_name : str
        Nome del KRI selezionato
    
    Returns
    -------
    pd.DataFrame o None
    """
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        return None

    # Validazione colonne
    required_columns = KRI_COLUMNS.get(kri_name, [])
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Errore: mancano le seguenti colonne per {kri_name}: {missing}")
        return None
    
    return df


def validate_kri_data(df, kri_name):
    """
    Valida che il DataFrame contenga le colonne corrette per il KRI.
    
    Parameters
    ----------
    df : pd.DataFrame
    kri_name : str
    
    Returns
    -------
    bool
    """
    if df is None or df.empty:
        st.error(f"I dati per {kri_name} sono vuoti o non validi.")
        return False
    
    required_columns = KRI_COLUMNS.get(kri_name, [])
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Mancano le colonne obbligatorie per {kri_name}: {missing}")
        return False
    
    return True

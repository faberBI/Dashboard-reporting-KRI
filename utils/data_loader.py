import pandas as pd
import streamlit as st

def load_excel(file, sheet_name=0):
    """
    Carica un file Excel e restituisce un DataFrame.
    
    Parameters
    ----------
    file : UploadedFile o percorso locale
        File Excel caricato dall'utente tramite Streamlit o locale.
    sheet_name : str o int, default=0
        Nome o indice del foglio da caricare.

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_excel(file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        return None


def validate_data(df, required_columns=None):
    """
    Valida il DataFrame in base alle colonne richieste.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame da validare
    required_columns : list, optional
        Lista di colonne obbligatorie

    Returns
    -------
    bool, list
        True/False se valido, lista di errori
    """
    errors = []
    
    if df is None or df.empty:
        errors.append("Il dataset è vuoto o non valido.")
    else:
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                errors.append(f"Mancano le seguenti colonne: {missing}")
    
    return len(errors) == 0, errors


def load_from_manual_input(data_dict):
    """
    Crea un DataFrame dai dati inseriti manualmente via interfaccia.
    
    Parameters
    ----------
    data_dict : dict
        Dizionario {colonna: lista_valori} es. {"KRI1": [10,20,30]}
    
    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.DataFrame(data_dict)
        return df
    except Exception as e:
        st.error(f"Errore nella creazione del DataFrame: {e}")
        return None


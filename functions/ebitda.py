import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import io
from scipy.stats import norm, multivariate_normal
from copulas.multivariate import GaussianMultivariate
from scipy.stats import rankdata, norm
from sklearn.covariance import LedoitWolf
import plotly.graph_objects as go

@st.cache_data

def ensure_dataframe(df):
    """
    Garantisce che df sia un DataFrame.
    Se non lo è o è None, ritorna DataFrame vuoto.
    """
    if isinstance(df, pd.DataFrame):
        return df.copy()
    elif df is None:
        return pd.DataFrame()
    try:
        # prova a convertire dict, list, Series, ndarray
        df_conv = pd.DataFrame(df)
        if isinstance(df_conv, pd.DataFrame):
            return df_conv
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def safe_pivot(df, index, columns, values):
    if df.empty:
        return pd.DataFrame()
    try:
        pivoted = df.pivot(index=index, columns=columns, values=values)
        return pivoted.fillna(0)
    except Exception as e:
        print("Errore pivot:", e)
        return pd.DataFrame()

def plot_top_corr_bar(top_corr_pairs, anno):
    df_corr = pd.DataFrame(top_corr_pairs, columns=["Feature 1", "Feature 2", "Correlazione"])
    # Crea una stringa per ogni coppia
    df_corr["Coppia"] = df_corr["Feature 1"] + " ↔ " + df_corr["Feature 2"]
    # Ordina per valore assoluto di correlazione
    df_corr = df_corr.reindex(df_corr["Correlazione"].abs().sort_values(ascending=True).index)
    
    fig = px.bar(
        df_corr,
        x="Correlazione",
        y="Coppia",
        orientation='h',
        title=f"Top 10 correlazioni più alte - Anno {anno}",
        color="Correlazione",
        color_continuous_scale=px.colors.diverging.RdBu,
        range_color=[-1, 1]
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig
    
import numpy as np
import pandas as pd

def get_top_correlations(corr_matrix, top_n=10):
    corr_matrix = corr_matrix.copy()
    
    # 🔹 Imposta la diagonale a NaN senza usare np.fill_diagonal
    n_rows = corr_matrix.shape[0]
    n_cols = corr_matrix.shape[1]
    for i in range(min(n_rows, n_cols)):
        corr_matrix.iloc[i, i] = np.nan  # funziona anche se non quadrata

    # Trasforma in serie di correlazioni assolute
    corr_unstacked = corr_matrix.abs().unstack()

    # Rimuovi duplicati (A,B) e (B,A)
    corr_unstacked = corr_unstacked[
        corr_unstacked.index.get_level_values(0) < corr_unstacked.index.get_level_values(1)
    ]

    # Prendi le top N correlazioni
    top_corr = corr_unstacked.sort_values(ascending=False).head(top_n)

    # Ricostruisci coppie con valore originale (segno incluso)
    pairs = []
    for (f1, f2), val_abs in top_corr.items():
        val_orig = corr_matrix.loc[f1, f2]
        pairs.append((f1, f2, val_orig))

    return pairs
    
def simula_fattori_empiricamente(fattori_simulati, n_sim):
    if isinstance(fattori_simulati, dict):
        fattori = list(fattori_simulati.keys())
        data = np.vstack([fattori_simulati[f] for f in fattori])
    else:
        data = np.array(fattori_simulati)
        fattori = [f"Fattore_{i}" for i in range(data.shape[0])]

    n_fattori, n_obs = data.shape

    # Stima matrice di covarianza con Ledoit-Wolf
    lw = LedoitWolf().fit(data.T)  # attenzione: sklearn vuole (n_samples, n_features)
    cov_shrink = lw.covariance_

    # Deriva la matrice di correlazione dalla covarianza
    d = np.sqrt(np.diag(cov_shrink))
    cor_matrix = cov_shrink / np.outer(d, d)
    
    print(f'\nMatrice di correlazione stimata (Ledoit-Wolf):\n{cor_matrix}\n')

    # Simula da gaussiana multivariata con matrice corretta
    Z_sim = np.random.multivariate_normal(mean=np.zeros(n_fattori), cov=cor_matrix, size=n_sim).T

    # Trasformazione copula: normal -> uniform -> margini empirici
    U = norm.cdf(Z_sim)
    sim_matrix_corr = np.zeros_like(U)

    for i in range(n_fattori):
        sorted_data = np.sort(data[i])
        quantiles = np.linspace(1/(n_obs+1), n_obs/(n_obs+1), n_obs)
        sim_matrix_corr[i] = np.interp(U[i], quantiles, sorted_data)

    return sim_matrix_corr, cor_matrix, fattori


def genera_template_input():
    data = {
        "Fattore di rischio": ["Esempio Fattore 1", "Esempio Fattore 2"],
        "Distribuzione": ['triangolare','normale'],
        "variabile": ["prezzo", "quantità"],
        "anno": [2025, 2026],
        "valore  a piano": [1000, 2000],
        "min": [2, np.nan],
        "moda":[2.5, np.nan],
        "max":[4, np.nan],
        "incertezza":[np.nan, np.nan],
        "q":[450, np.nan ],
        "p":[np.nan,20],
        "costo variabile":["no", "si"],
        "perc":[np.nan, "6%"],
        "variabile dipendente":[np.nan,'Esempio Fattore 1'],
        "tipo_variabile":['ricavo','costo'],
        "k_min":[0.01, 0.015],
        "k_max":[0.05, 0.055],
        "mu":[np.nan,100],
        "sigma":[np.nan,50]
       
    }
    df_template = pd.DataFrame(data)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Template Input')
    buffer.seek(0)
    return buffer



def load_risk_factors(file):
    df = pd.read_excel(file)
    return df

def parse_factors(df):
    blocks = []
    for _, row in df.iterrows():
        dist = row['Distribuzione']
        varia = row['variabile']  # prezzo, quantità, o NaN
        anno = row['anno']  # aggiunto anno

        # Prendo k_min, k_max
        k_min = row['k_min'] if not pd.isna(row['k_min']) else 0
        k_max = row['k_max'] if not pd.isna(row['k_max']) else 0

        # Prendo valore base a piano
        base_val = row['valore a piano'] if not pd.isna(row['valore a piano']) else 0
        perc = row['perc'] if not pd.isna(row['perc']) else 0
        dipendente= row['variabile dipendente']
        block = {
            'name': row['Fattore di rischio'],
            'anno': anno,  # aggiunto anno
            'varia': varia,
            'costo_variabile': (str(row['costo variabile']).lower() == 'si') if isinstance(row['costo variabile'], str) else False,
            'perc': perc,
            'dipendente': dipendente,
            'incertezza': row['incertezza'] if not pd.isna(row['incertezza']) else 1,
            'valore_a_piano': base_val,
            'tipo_variabile': row['tipo_variabile'] if not pd.isna(row['tipo_variabile']) else 'ricavo',
            'k_min': k_min,
            'k_max': k_max
        }

        # Gestione percentuale
        perc_raw = row['perc']
        if isinstance(perc_raw, str) and perc_raw.strip().endswith('%'):
            perc = float(perc_raw.strip().strip('%')) / 100
        else:
            perc = perc_raw if not pd.isna(perc_raw) else 0
        block['perc'] = perc

        # Valori fissi per p e q (in caso di nessuna variazione)
        p_fixed = row['p'] if not pd.isna(row['p']) else 1
        q_fixed = row['q'] if not pd.isna(row['q']) else 1

        if varia == 'prezzo':
            block['varia'] = 'Solo Prezzo'
            if dist == 'triangolare':
                block['p'] = {'a': row['min'], 'b': row['moda'], 'c': row['max'], 'dist': 'triangolare'}
            elif dist == 'normale':
                block['p'] = {'mu': row['mu'], 'sigma': row['sigma'], 'dist': 'normale'}
            else:
                block['p'] = {'b': p_fixed, 'dist': None}
            block['q'] = {'b': q_fixed, 'dist': None}

        elif varia == 'quantità':
            block['varia'] = 'Solo Quantità'
            if dist == 'triangolare':
                block['q'] = {'a': row['min'], 'b': row['moda'], 'c': row['max'], 'dist': 'triangolare'}
                block['p'] = {'b': p_fixed, 'dist': None}
            elif dist == 'normale':
                block['q'] = {'mu': row['mu'], 'sigma': row['sigma'], 'dist': 'normale'}
                block['p'] = {'b': p_fixed, 'dist': None}
            else:
                block['q'] = {'b': q_fixed, 'dist': None}
                block['p'] = {'b': p_fixed, 'dist': None}
                
        else:
            block['varia'] = None
            block['p'] = {'b': p_fixed, 'dist': None}
            block['q'] = {'b': q_fixed, 'dist': None}

        block['dipendente'] = row['variabile dipendente'] if not pd.isna(row['variabile dipendente']) else None
        
        if 'p' not in block:
            print(f"Attenzione: block senza chiave 'p' trovato! Riga:\n{row}")

        blocks.append(block)
    return blocks



def sample_distribution(dist_type, params, size):
    if dist_type == 'triangolare':
        a = params['a']
        b = params['b']
        c = params['c']
        return np.random.triangular(a, b, c, size)
    elif dist_type is None:
        return np.full(size, params['b'])
        
    elif dist_type=='normale':
        mu = params.get('mu')
        sigma = params.get('sigma')   
        return np.random.normal(mu, sigma, size)
    else:
        return np.full(size, params.get('b', 0))


def apply_uncertainty_to_params(params, dist_type, k_min, k_max):
    params_mod = params.copy()

    if dist_type == "triangolare":
        a = params.get('a')
        b = params.get('b')
        c = params.get('c')

        if a is not None:
            new_a = a * (1 - k_min)
            params_mod['a'] = max(0, new_a)
        if b is not None:
            params_mod['b'] = b 
        if c is not None:
            params_mod['c'] = c * (1 + k_max)
    
    elif dist_type =="normale":
        mu = params.get('mu')
        sigma = params.get('sigma')
        params_mod['sigma'] = sigma * (1 + k_max) 
    
    elif dist_type is None:
        b = params.get('b')
    return params_mod


def simulate_ebitda_multi_year_blocks(
    blocks, 
    ebitda_base_list, 
    n_sim, 
    anni,  
    anno_inizio_k, 
    trend,
    attiva_shock=False, 
    lambda_shock_annuo=0.1, 
    magnitudo_shock=0.2, 
    activation_matrix=None,
    shock_event_dict=None
):
    risultati_anni = []
    anni_str = [str(a) for a in anni]
    cor_matrix_by_year = {}

    if attiva_shock:
        total_shocks = np.random.poisson(lam=lambda_shock_annuo * len(anni))

    # 👉 Se la Poisson genera almeno 1 evento, ne consentiamo solo uno
        if total_shocks > 1:
            total_shocks = 1

    # 👉 Scegli al massimo un anno
        anni_shock = np.random.choice(anni, size=total_shocks, replace=False) if total_shocks > 0 else []

        print(f"[INFO] Shock esogeni totali simulati su EBITDA: {total_shocks}")
    else:
        anni_shock = []

    # --- Nuova logica: per ogni fattore, estraiamo al massimo 1 anno con shock ---
    shock_event_schedule = {}
    if shock_event_dict:
        for nome_fattore, cfg in shock_event_dict.items():
            anni_attivi = [anno for anno, attivo in cfg["anni_attivi"].items() if attivo]
            if anni_attivi:
                lambda_totale = cfg["lambda"]
                num_eventi = np.random.poisson(lambda_totale)
    
                # 🔁 Se non è stato estratto nessuno shock ma lambda ≥ 0.9, forziamo 1 evento
                if num_eventi == 0 and lambda_totale >= 0.9:
                    num_eventi = 1
    
                if num_eventi > 0:
                    anno_shock = np.random.choice(anni_attivi)
                    shock_event_schedule[nome_fattore] = int(anno_shock)
                    
    print("[DEBUG] shock_event_schedule:", shock_event_schedule)                
    # ----------------------------------------------------------

    # Mappa per effetto a cascata degli shock sui fattori
    shock_start_year = {}

    for i, anno in enumerate(anni):
        shock_occorrenze = {}
        anno_label = str(anno)
        ebitda_sim = np.zeros(n_sim)
        blocchi_simulati = {}
        fattori_simulati = {}

        blocks_per_anno = [b for b in blocks if b['anno'] == anno]

        if activation_matrix is not None:
            blocks_attivi = [
                b for b in blocks_per_anno
                if activation_matrix.get(b['name'], {}).get(anno_label, True)
            ]
        else:
            blocks_attivi = blocks_per_anno

        blocks_attivi = sorted(blocks_attivi, key=lambda x: 1 if x.get('varia') == 'Derivato' else 0)

        for block in blocks_attivi:
            nome = block['name']
            k_min_orig = block.get('k_min', 0)
            k_max_orig = block.get('k_max', 0)

            if i + 1 < anno_inizio_k:
                k_min = k_max = 0
            else:
                anni_passati = i + 1 - (anno_inizio_k - 1)
                if trend == 'lineare':
                    k_min = k_min_orig * anni_passati
                    k_max = k_max_orig * anni_passati
                elif trend == 'moltiplicativo':
                    k_min = k_min_orig * (1.5 ** (anni_passati - 1))
                    k_max = k_max_orig * (1.5 ** (anni_passati - 1))
                else:
                    k_min = k_min_orig
                    k_max = k_max_orig

            varia = block.get('varia')

            if varia in ["Solo Prezzo", "Entrambi"]:
                p_dict = block['p']
                p_dist = p_dict.get('dist')
            else:
                p_dict = {'b': block['p']['b'], 'dist': None}
                p_dist = None

            if varia in ["Solo Quantità", "Entrambi"]:
                q_dict = block['q']
                q_dist = q_dict.get('dist')
            else:
                q_dict = {'b': block['q']['b'], 'dist': None}
                q_dist = None

            if 'param_log' not in block:
                block['param_log'] = {}

            params_mod_p = apply_uncertainty_to_params(p_dict, p_dist, k_min, k_max) if p_dist else p_dict
            params_mod_q = apply_uncertainty_to_params(q_dict, q_dist, k_min, k_max) if q_dist else q_dict

            block['param_log'][anno] = {
                "k_min": k_min,
                "k_max": k_max,
                "prezzo": params_mod_p,
                "quantità": params_mod_q
            }

            p = sample_distribution(p_dist, params_mod_p, n_sim) if p_dist else np.full(n_sim, params_mod_p['b'])
            q = sample_distribution(q_dist, params_mod_q, n_sim) if q_dist else np.full(n_sim, params_mod_q['b'])

            base = block.get('valore_a_piano', 0)
            sim = p * q

            dipendente_sim = None
            if block.get('costo_variabile', False) and block.get('dipendente'):
                dipendente_sim = blocchi_simulati.get(block['dipendente'])

            if dipendente_sim is not None:
                perc = block.get('perc', 1)
                tipo_var = block.get('tipo_variabile', 'ricavo').lower()
                sim = dipendente_sim['sim'] * perc
                if tipo_var == "costo":
                    sim = -sim

            # --- Applica shock solo nell'anno schedulato e propagalo negli anni successivi ---
            if shock_event_dict:
                shock_cfg = shock_event_dict.get(nome)
                anno_shock_fattore = shock_event_schedule.get(nome, None)
                if shock_cfg and anno_shock_fattore is not None:
                    if isinstance(anno_shock_fattore, int) and anno >= anno_shock_fattore:
                        impatto = shock_cfg["magnitudo"]
                        segno = shock_cfg["segno"]
                        
                        # Applica in modo coerente con il segno del valore simulato (costo vs ricavo)
                        if segno == "Negativo":
                            if np.all(sim < 0):
                                sim *= (1 + impatto)  # costo aumenta = peggiora
                            else:
                                sim *= (1 - impatto)  # ricavo diminuisce = peggiora
                        else:  # segno "Positivo"
                            if np.all(sim < 0):
                                sim *= (1 - impatto)  # costo diminuisce = migliora
                            else:
                                sim *= (1 + impatto)  # ricavo aumenta = migliora        
            
            shock_occorrenze[nome] = shock_event_schedule.get(nome) == anno
            
            
            blocchi_simulati[nome] = {"sim": sim, "base": base}
            fattori_simulati[nome] = sim - base

        # Applica copula per correlazione tra fattori
        df_simulazioni, cor_matrix, fattori = simula_fattori_empiricamente(fattori_simulati, n_sim)
        cor_matrix_by_year[anno] = {
            "matrice": cor_matrix,
            "fattori": fattori
        }

        delta_totale_per_simulazione = df_simulazioni.sum(axis=0)
        ebitda_sim = ebitda_base_list[i] + delta_totale_per_simulazione

        # Shock su EBITDA complessivo (non sui fattori)
        if attiva_shock and len(anni_shock) > 0:
            anno_shock = min(anni_shock)  # lo shock esogeno può capitare una sola volta
            if anno >= anno_shock:
                ebitda_sim *= (1 - magnitudo_shock)


        df_sim = pd.DataFrame(df_simulazioni.T, columns=fattori)

        risultati_anni.append({
            "anno": anno,
            "ebitda_simulazioni": ebitda_sim,
            "fattori_simulati": df_sim.to_dict(orient='list'),
            "shock_ebitda": int(np.sum(np.array(anni_shock) == anno)),
            "matrice_correlazione": cor_matrix_by_year[anno],
            "nomi_fattori": fattori,
            "shock_occorrenze": shock_occorrenze
        })

    return risultati_anni, cor_matrix_by_year

import pandas as pd
import numpy as np

def simulate_ebitda_multi_year_blocks_with_ricavi(
    blocks, 
    ebitda_base_list, 
    n_sim, 
    anni,  
    anno_inizio_k, 
    trend,
    attiva_shock=False, 
    lambda_shock_annuo=0.1, 
    magnitudo_shock=0.2, 
    activation_matrix=None,
    shock_event_dict=None
):
    risultati_anni = []
    anni_str = [str(a) for a in anni]
    cor_matrix_by_year = {}

    # Per raccogliere ricavi negativi: lista di dict
    ricavi_negativi_records = []

    # Per salvare le simulazioni dei fattori per ogni anno in DataFrame
    df_parametri_simulati_per_anno = {}

    if attiva_shock:
        if lambda_shock_annuo > 0.9:
        # Forzo 1 evento shock, sicuro
            total_shocks = 1
        else:
        # Numero di eventi shock da Poisson (può essere anche 0)
            total_shocks = np.random.poisson(lam=lambda_shock_annuo * len(anni))
        # Limito a massimo 1 evento
            if total_shocks > 1:
                total_shocks = 1
    
    # Se c'è almeno un evento, scelgo un anno casuale (un solo anno)
        anni_shock = np.random.choice(anni, size=total_shocks, replace=False) if total_shocks > 0 else []
        print(f"[INFO] Shock totali simulati su EBITDA: {total_shocks}")
    else:
        anni_shock = []

    # Nuova logica per shock sui fattori
    shock_event_schedule = {}
    if shock_event_dict:
        for nome_fattore, cfg in shock_event_dict.items():
            anni_attivi = [anno for anno, attivo in cfg["anni_attivi"].items() if attivo]
            if anni_attivi:
                lambda_totale = cfg["lambda"]
                num_eventi = np.random.poisson(lambda_totale)

                if num_eventi == 0 and lambda_totale >= 0.9:
                    num_eventi = 1

                if num_eventi > 0:
                    anno_shock = np.random.choice(anni_attivi)
                    shock_event_schedule[nome_fattore] = int(anno_shock)

    print("[DEBUG] shock_event_schedule:", shock_event_schedule)                
    
    fattore_cumulativo_shock = 1.0
    
    for i, anno in enumerate(anni):
        shock_occorrenze = {}
        anno_label = str(anno)
        ebitda_sim = np.zeros(n_sim)
        blocchi_simulati = {}
        fattori_simulati = {}

        blocks_per_anno = [b for b in blocks if b['anno'] == anno]

        if activation_matrix is not None:
            blocks_attivi = [
                b for b in blocks_per_anno
                if activation_matrix.get(b['name'], {}).get(anno_label, True)
            ]
        else:
            blocks_attivi = blocks_per_anno

        blocks_attivi = sorted(blocks_attivi, key=lambda x: 1 if x.get('varia') == 'Derivato' else 0)

        for block in blocks_attivi:
            nome = block['name']
            k_min_orig = block.get('k_min', 0)
            k_max_orig = block.get('k_max', 0)

            if i + 1 < anno_inizio_k:
                k_min = k_max = 0
            else:
                anni_passati = i + 1 - (anno_inizio_k - 1)
                if trend == 'lineare':
                    k_min = k_min_orig * anni_passati
                    k_max = k_max_orig * anni_passati
                elif trend == 'moltiplicativo':
                    k_min = k_min_orig * (1.5 ** (anni_passati - 1))
                    k_max = k_max_orig * (1.5 ** (anni_passati - 1))
                else:
                    k_min = k_min_orig
                    k_max = k_max_orig

            varia = block.get('varia')

            if varia in ["Solo Prezzo", "Entrambi"]:
                p_dict = block['p']
                p_dist = p_dict.get('dist')
            else:
                p_dict = {'b': block['p']['b'], 'dist': None}
                p_dist = None

            if varia in ["Solo Quantità", "Entrambi"]:
                q_dict = block['q']
                q_dist = q_dict.get('dist')
            else:
                q_dict = {'b': block['q']['b'], 'dist': None}
                q_dist = None

            if 'param_log' not in block:
                block['param_log'] = {}

            params_mod_p = apply_uncertainty_to_params(p_dict, p_dist, k_min, k_max) if p_dist else p_dict
            params_mod_q = apply_uncertainty_to_params(q_dict, q_dist, k_min, k_max) if q_dist else q_dict

            block['param_log'][anno] = {
                "k_min": k_min,
                "k_max": k_max,
                "prezzo": params_mod_p,
                "quantità": params_mod_q
            }

            p = sample_distribution(p_dist, params_mod_p, n_sim) if p_dist else np.full(n_sim, params_mod_p['b'])
            q = sample_distribution(q_dist, params_mod_q, n_sim) if q_dist else np.full(n_sim, params_mod_q['b'])

            base = block.get('valore_a_piano', 0)
            
            # 🔍 Controllo se il fattore dipende da un altro
            dipendente_sim = None
            if block.get('costo_variabile', False) and block.get('dipendente'):
                dipendente_sim = blocchi_simulati.get(block['dipendente'])
            
            if dipendente_sim is not None:
                # ✅ CASO 1: fattore dipendente
                perc = block.get('perc', 1)
                sim = dipendente_sim['sim'] * perc
            
            else:
                # ✅ CASO 2: fattore autonomo
                sim = p * q
            
            # 👉 Dopo aver calcolato sim, gestiamo il segno
            tipo_var = block.get('tipo_variabile', 'ricavo').lower()
            if tipo_var == "costo":
                sim = -sim
            
            # 📌 Rileva ricavi negativi (solo se tipo_variabile NON è costo)
            if tipo_var != 'costo':
                ind_negativi = np.where(sim < 0)[0]
                for idx in ind_negativi:
                    ricavi_negativi_records.append({
                        "anno": anno,
                        "nome_fattore": nome,
                        "indice_simulazione": idx,
                        "valore_simulazione": sim[idx]
                    })
                # Forza valori negativi a zero
                sim = np.maximum(sim, 0)

            # Applica shock fattori
            if shock_event_dict:
                shock_cfg = shock_event_dict.get(nome)
                anno_shock_fattore = shock_event_schedule.get(nome, None)
                if shock_cfg and anno_shock_fattore is not None:
                    if isinstance(anno_shock_fattore, int) and anno >= anno_shock_fattore:
                        impatto = shock_cfg["magnitudo"]
                        segno = shock_cfg["segno"]
                        
                        if segno == "Negativo":
                            if np.all(sim < 0):
                                sim *= (1 + impatto)
                            else:
                                sim *= (1 - impatto)
                        else:
                            if np.all(sim < 0):
                                sim *= (1 - impatto)
                            else:
                                sim *= (1 + impatto)

            shock_occorrenze[nome] = shock_event_schedule.get(nome) == anno

            blocchi_simulati[nome] = {"sim": sim, "base": base}
            fattori_simulati[nome] = sim - base

        # Creo DataFrame parametri simulati per questo anno (righe: simulazioni, colonne: fattori)
        df_parametri_simulati_per_anno[anno] = pd.DataFrame(
            {nome: blocchi_simulati[nome]['sim'] for nome in blocchi_simulati}
        )

        # Copula e correlazioni
        df_simulazioni, cor_matrix, fattori = simula_fattori_empiricamente(fattori_simulati, n_sim)
        cor_matrix_by_year[anno] = {
            "matrice": cor_matrix,
            "fattori": fattori
        }

        delta_totale_per_simulazione = df_simulazioni.sum(axis=0)
        ebitda_sim = ebitda_base_list[i] + delta_totale_per_simulazione


        # Shock EBITDA complessivo
        #if attiva_shock:
        #    n_shocks_anno = np.sum(np.array(anni_shock) == anno)
        #    if n_shocks_anno > 0:
        #        ebitda_sim *= (1 - magnitudo_shock) ** n_shocks_anno
        
        
        # Se è attivo lo shock e lo shock è previsto per questo anno
        # if attiva_shock:
        #    n_shocks_anno = np.sum(np.array(anni_shock) == anno)
        #    if n_shocks_anno > 0:
                # Applico shock complessivo
        #        ebitda_sim *= (1 - magnitudo_shock) ** n_shocks_anno
                
                # Calcolo trend a partire dalla lista storica ebitda_base_list (escluso anno corrente)
                # (ad esempio calcolo crescita media annuale o trend lineare)
        #        if i > 0:
        #            trend_growth = (ebitda_base_list[i] - ebitda_base_list[i-1]) / ebitda_base_list[i-1]
        #        else:
        #            trend_growth = 0
                
                # Applico trend sul valore medio simulato con shock
        #        media_ebitda = np.mean(ebitda_sim)
        #        print(f'media ebitda a seguito di uno shock esogeno: {media_ebitda}')
        #        ebitda_sim = media_ebitda * (1 + trend_growth) + (ebitda_sim - np.mean(ebitda_sim))

        # if attiva_shock:
        #    n_shocks_anno = np.sum(np.array(anni_shock) == anno)
        #    if n_shocks_anno > 0:
        #        fattore_cumulativo_shock *= (1 - magnitudo_shock) ** n_shocks_anno
        
        #    if fattore_cumulativo_shock < 1:  # significa che almeno uno shock è successo
        #        if i > 0:
        #            trend_growth = (ebitda_base_list[i] - ebitda_base_list[i-1]) / ebitda_base_list[i-1]
        #        else:
        #            trend_growth = 0
        
        #        media_ebitda = np.mean(ebitda_sim)
        #        ebitda_sim = media_ebitda * (1 + trend_growth) + (ebitda_sim - np.mean(ebitda_sim))
        #        ebitda_sim = ebitda_sim * fattore_cumulativo_shock
       
        if attiva_shock:
            n_shocks_anno = np.sum(np.array(anni_shock) == anno)
        
            # 🔴 1️⃣ Se c’è shock quest’anno: abbatti EBITDA e imposta il fattore cumulativo
            if n_shocks_anno > 0:
                fattore_cumulativo_shock *= (1 - magnitudo_shock) ** n_shocks_anno
                shock_avvenuto = True  # Flag per dire che da qui in poi c'è uno shock da gestire
        
            # 🟠 2️⃣ Se lo shock è già avvenuto in anni precedenti (o in questo anno), lavora su trend e fattore
            if 'shock_avvenuto' in locals() and shock_avvenuto:
                # Applico decadimento del fattore shock (attenuazione)
                if i > 0:
                    fattore_cumulativo_shock += (1 - fattore_cumulativo_shock) * 0.20
                    # In alternativa: fattore_cumulativo_shock *= (1 - tasso_decadimento_annuo)
        
                # Calcolo il trend sugli EBITDA di piano (sempre su base_list)
                if i > 0:
                    trend_growth = (ebitda_base_list[i] - ebitda_base_list[i-1]) / ebitda_base_list[i-1]
                else:
                    trend_growth = 0
        
                # Applico il trend al valore simulato tenendo conto dello shock
                media_ebitda = np.mean(ebitda_sim)
                ebitda_sim = media_ebitda * (1 + trend_growth) + (ebitda_sim - media_ebitda)
        
                # Applico l’effetto cumulativo dello shock (decaduto)
                ebitda_sim = ebitda_sim * fattore_cumulativo_shock
        
        
        df_sim = pd.DataFrame(df_simulazioni.T, columns=fattori)

        risultati_anni.append({
            "anno": anno,
            "ebitda_simulazioni": ebitda_sim,
            "fattori_simulati": df_sim.to_dict(orient='list'),
            "shock_ebitda": int(np.sum(np.array(anni_shock) == anno)),
            "matrice_correlazione": cor_matrix_by_year[anno],
            "nomi_fattori": fattori,
            "shock_occorrenze": shock_occorrenze
        })

    # DataFrame finale con tutte le simulazioni negative su ricavi (prima del clipping)
    df_ricavi_negativi = pd.DataFrame(ricavi_negativi_records)

    # Unisci i DataFrame parametri simulati in uno solo multi-anno, con MultiIndex (anno, simulazione)
    df_parametri_simulati = pd.concat(
        {anno: df_parametri_simulati_per_anno[anno] for anno in df_parametri_simulati_per_anno},
        names=['anno', 'simulazione']
    )

    return risultati_anni, cor_matrix_by_year, df_ricavi_negativi, df_parametri_simulati




def simulate_ebitda_multi_year_blocks_old(
    blocks, 
    ebitda_base_list, 
    n_sim, 
    anni,  
    anno_inizio_k, 
    trend,
    attiva_shock=False, 
    lambda_shock_annuo=0.1, 
    magnitudo_shock=0.2, 
    activation_matrix=None,
    shock_event_dict=None
):
    risultati_anni = []
    anni_str = [str(a) for a in anni]
    cor_matrix_by_year = {}

    # Shock su EBITDA
    if attiva_shock:
        total_shocks = np.random.poisson(lam=lambda_shock_annuo * len(anni))
        anni_shock = np.random.choice(anni, size=total_shocks) if total_shocks > 0 else []
        print(f"[INFO] Shock totali simulati su EBITDA: {total_shocks}")
    else:
        anni_shock = []

    shock_start_year = {}
    for i, anno in enumerate(anni):
        anno_label = str(anno)
        ebitda_sim = np.zeros(n_sim)
        blocchi_simulati = {}
        fattori_simulati = {}

        blocks_per_anno = [b for b in blocks if b['anno'] == anno]

        if activation_matrix is not None:
            blocks_attivi = [
                b for b in blocks_per_anno
                if activation_matrix.get(b['name'], {}).get(anno_label, True)
            ]
        else:
            blocks_attivi = blocks_per_anno

        blocks_attivi = sorted(blocks_attivi, key=lambda x: 1 if x.get('varia') == 'Derivato' else 0)

        for block in blocks_attivi:
            nome = block['name']
            k_min_orig = block.get('k_min', 0)
            k_max_orig = block.get('k_max', 0)

            if i + 1 < anno_inizio_k:
                k_min = k_max = 0
            else:
                anni_passati = i + 1 - (anno_inizio_k - 1)
                if trend == 'lineare':
                    k_min = k_min_orig * anni_passati
                    k_max = k_max_orig * anni_passati
                elif trend == 'moltiplicativo':
                    k_min = k_min_orig * (1.5 ** (anni_passati - 1))
                    k_max = k_max_orig * (1.5 ** (anni_passati - 1))
                else:
                    k_min = k_min_orig
                    k_max = k_max_orig

            varia = block.get('varia')

            if varia in ["Solo Prezzo", "Entrambi"]:
                p_dict = block['p']
                p_dist = p_dict.get('dist')
            else:
                p_dict = {'b': block['p']['b'], 'dist': None}
                p_dist = None

            if varia in ["Solo Quantità", "Entrambi"]:
                q_dict = block['q']
                q_dist = q_dict.get('dist')
            else:
                q_dict = {'b': block['q']['b'], 'dist': None}
                q_dist = None

            if 'param_log' not in block:
                block['param_log'] = {}

            params_mod_p = apply_uncertainty_to_params(p_dict, p_dist, k_min, k_max) if p_dist else p_dict
            params_mod_q = apply_uncertainty_to_params(q_dict, q_dist, k_min, k_max) if q_dist else q_dict

            block['param_log'][anno] = {
                "k_min": k_min,
                "k_max": k_max,
                "prezzo": params_mod_p,
                "quantità": params_mod_q
            }

            p = sample_distribution(p_dist, params_mod_p, n_sim) if p_dist else np.full(n_sim, params_mod_p['b'])
            q = sample_distribution(q_dist, params_mod_q, n_sim) if q_dist else np.full(n_sim, params_mod_q['b'])

            base = block.get('valore_a_piano', 0)
            sim = p * q

            dipendente_sim = None
            if block.get('costo_variabile', False) and block.get('dipendente'):
                dipendente_sim = blocchi_simulati.get(block['dipendente'])

            if dipendente_sim is not None:
                perc = block.get('perc', 1)
                tipo_var = block.get('tipo_variabile', 'ricavo').lower()
                sim = dipendente_sim['sim'] * perc
                if tipo_var == "costo":
                    sim = -sim

            if shock_event_dict:
                shock_cfg = shock_event_dict.get(nome)
                if shock_cfg and shock_cfg["anni_attivi"].get(anno_label, False):
                    shock_count = np.random.poisson(shock_cfg["lambda"])
                    if shock_count > 0:
                        impatto = shock_cfg["magnitudo"]
                        if shock_cfg["segno"] == "Negativo":
                            sim *= (1 - impatto) ** shock_count
                        else:
                            sim *= (1 + impatto) ** shock_count

            blocchi_simulati[nome] = {"sim": sim, "base": base}
            fattori_simulati[nome] = sim - base
    
    
        #print(f'fattori simulati: {fattori_simulati}')
        # Applica copula per correlazione tra fattori
        df_simulazioni, cor_matrix, fattori = simula_fattori_empiricamente(fattori_simulati, n_sim)
        cor_matrix_by_year[anno] = {
            "matrice": cor_matrix,
            "fattori": fattori
        }
        
        delta_totale_per_simulazione = df_simulazioni.sum(axis=0)
        ebitda_sim = ebitda_base_list[i] + delta_totale_per_simulazione

        #if attiva_shock:
        #    n_shocks_anno = np.sum(np.array(anni_shock) == anno)
        #    if n_shocks_anno > 0:
        #        ebitda_sim *= (1 - magnitudo_shock) ** n_shocks_anno
        
        # Se è attivo lo shock e lo shock è previsto per questo anno
        if attiva_shock:
            n_shocks_anno = np.sum(np.array(anni_shock) == anno)
            if n_shocks_anno > 0:
                # Applico shock complessivo
                ebitda_sim *= (1 - magnitudo_shock) ** n_shocks_anno
                
                # Calcolo trend a partire dalla lista storica ebitda_base_list (escluso anno corrente)
                # (ad esempio calcolo crescita media annuale o trend lineare)
                if i > 0:
                    trend_growth = (ebitda_base_list[i] - ebitda_base_list[i-1]) / ebitda_base_list[i-1]
                else:
                    trend_growth = 0
                
                # Applico trend sul valore medio simulato con shock
                media_ebitda = np.mean(ebitda_sim)
                ebitda_sim = media_ebitda * (1 + trend_growth) + (ebitda_sim - np.mean(ebitda_sim))

        df_sim = pd.DataFrame(df_simulazioni.T, columns=fattori)
        
        risultati_anni.append({
            "anno": anno,
            "ebitda_simulazioni": ebitda_sim,
            "fattori_simulati": df_sim.to_dict(orient='list'),
            "shock_ebitda": int(np.sum(np.array(anni_shock) == anno)),
            "matrice_correlazione": cor_matrix_by_year[anno],
            "nomi_fattori": fattori
        })

    return risultati_anni, cor_matrix_by_year

def plot_k_min_max_plotly(blocks):
    for block in blocks:
        nome = block['name']
        param_log = block.get('param_log', {})

        if not param_log:
            continue

        anni = sorted(param_log.keys())
        k_min_vals = [param_log[anno]['k_min'] for anno in anni]
        k_max_vals = [param_log[anno]['k_max'] for anno in anni]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anni, y=k_min_vals,
            mode='lines+markers',
            name='k_min',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=anni, y=k_max_vals,
            mode='lines+markers',
            name='k_max',
            line=dict(color='green')
        ))

        fig.update_layout(
            title=f"Andamento k_min / k_max per '{nome}'",
            xaxis_title="Anno",
            yaxis_title="Valore k",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_white"
        )
        fig.show()



def calcola_importanza_fattori(risultati_raw):
    tornado_per_anno = []
    importanza_totale = {}

    for anno_data in risultati_raw:
        anno = anno_data["anno"]
        y = anno_data["ebitda_simulazioni"]
        X = pd.DataFrame(anno_data["fattori_simulati"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        importanza = pd.Series(np.abs(model.coef_), index=X.columns)
        importanza /= importanza.sum()
        tornado_per_anno.append({
            "anno": anno,
            "importanza": importanza
        })

        for f, v in importanza.items():
            importanza_totale[f] = importanza_totale.get(f, 0) + v

    importanza_totale = pd.Series(importanza_totale).sort_values(ascending=False)
    return tornado_per_anno, importanza_totale



def safe_applymap(df, func):
    """
    Applica applymap a df se è DataFrame. 
    Se non lo è, crea DataFrame vuoto.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)  # forza la conversione, o diventa vuoto se non compatibile
    if df.empty:
        return df
    return df.applymap(func)

def genera_output_excel(risultati_anni, ebitda_base_dict, df_styled):
    dati_output = []
    fattori_data = []
    shock_data = []

    # Estrazione fattori presenti
    fattori = sorted(set().union(*[e.get("fattori_simulati", {}).keys() for e in risultati_anni]))

    for entry in risultati_anni:
        anno = entry["anno"]
        ebitda_sim = entry["ebitda_simulazioni"]
        base = ebitda_base_dict.get(anno, 0)

        # EBITDA - percentili e statistiche
        percentili = np.percentile(ebitda_sim, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        media = np.mean(ebitda_sim)
        sigma = np.std(ebitda_sim)

        delta_5 = (percentili[1] - base) / base * 100 if base != 0 else None
        delta_95 = (percentili[7] - base) / base * 100 if base != 0 else None
        delta_1 = (percentili[0] - base) / base * 100 if base != 0 else None
        delta_99 = (percentili[8] - base) / base * 100 if base != 0 else None

        dati_output.append({
            "Anno": anno,
            "Valore a piano": base,
            "Media EBITDA": media,
            "Deviazione standard (σ)": sigma,
            "Percentile 1%": percentili[0],
            "Percentile 5%": percentili[1],
            "Percentile 10%": percentili[2],
            "Percentile 25%": percentili[3],
            "Mediana": percentili[4],
            "Percentile 75%": percentili[5],
            "Percentile 90%": percentili[6],
            "Percentile 95%": percentili[7],
            "Percentile 99%": percentili[8],
            "Δ% 1% vs Piano": delta_1,
            "Δ% 5% vs Piano": delta_5,
            "Δ% 95% vs Piano": delta_95,
            "Δ% 99% vs Piano": delta_99,
            "Shock esogeni applicati": entry.get("shock_ebitda", 0)
        })

        # Percentili fattori
        for fatt in fattori:
            sim_vals = entry.get("fattori_simulati", {}).get(fatt)
            if sim_vals is not None:
                p5 = np.percentile(sim_vals, 5)
                p95 = np.percentile(sim_vals, 95)
                fattori_data.append({
                    "Anno": anno,
                    "Fattore": fatt,
                    "Tipo": "5° Percentile",
                    "Valore": p5
                })
                fattori_data.append({
                    "Anno": anno,
                    "Fattore": fatt,
                    "Tipo": "95° Percentile",
                    "Valore": p95
                })

    # --- DataFrame per ciascuna sheet ---
    df_output = pd.DataFrame(dati_output)
    df_fattori = pd.DataFrame(fattori_data)
  
    # --- Esportazione Excel ---
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=False, sheet_name='Risultati EBITDA')
        df_fattori.to_excel(writer, index=False, sheet_name='Fattori di Rischio')
        df_styled.to_excel(writer, sheet_name='Shock Fattori')  
    buffer.seek(0)
    return buffer

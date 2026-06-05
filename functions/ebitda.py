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

def normalize_dist(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x in ["", "none", "nan"]:
        return None
    return x


def normalize_varia(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if x in ["prezzo", "solo prezzo"]:
        return "Solo Prezzo"
    elif x in ["quantità", "quantita", "solo quantità", "solo quantita"]:
        return "Solo Quantità"
    elif x in ["entrambi", "prezzo e quantità", "prezzo e quantita"]:
        return "Entrambi"
    return None


def build_dist_legacy(row, side):
    """
    side = 'p' oppure 'q'
    Usa il tracciato legacy:
    - Distribuzione, min, moda, max, mu, sigma, prob, value
    - colonna fissa opposta: p o q
    """
    dist = normalize_dist(row.get("Distribuzione"))

    if side == "p":
        fixed_val = row.get("p", 1)
    else:
        fixed_val = row.get("q", 1)

    fixed_val = 1 if pd.isna(fixed_val) else fixed_val

    if dist == "triangolare":
        return {
            "dist": "triangolare",
            "a": row.get("min", np.nan),
            "b": row.get("moda", np.nan),
            "c": row.get("max", np.nan)
        }

    elif dist == "normale":
        return {
            "dist": "normale",
            "mu": row.get("mu", np.nan),
            "sigma": row.get("sigma", np.nan)
        }

    elif dist == "bernoulli":
        return {
            "dist": "bernoulli",
            "prob": row.get("prob", np.nan),
            "value": row.get("value", np.nan)
        }

    else:
        return {
            "dist": None,
            "b": fixed_val
        }


def build_dist_specific(row, prefix):
    """
    prefix = 'prezzo' oppure 'quantità'
    Usa il nuovo tracciato per il caso 'Entrambi'
    """
    dist = normalize_dist(row.get(f"Distribuzione_{prefix}"))

    if dist == "triangolare":
        return {
            "dist": "triangolare",
            "a": row.get(f"min_{prefix}", np.nan),
            "b": row.get(f"moda_{prefix}", np.nan),
            "c": row.get(f"max_{prefix}", np.nan)
        }

    elif dist == "normale":
        return {
            "dist": "normale",
            "mu": row.get(f"mu_{prefix}", np.nan),
            "sigma": row.get(f"sigma_{prefix}", np.nan)
        }

    elif dist == "bernoulli":
        return {
            "dist": "bernoulli",
            "prob": row.get(f"prob_{prefix}", np.nan),
            "value": row.get(f"value_{prefix}", np.nan)
        }

    else:
        base_col = "p" if prefix == "prezzo" else "q"
        base_val = row.get(base_col, 1)
        base_val = 1 if pd.isna(base_val) else base_val
        return {
            "dist": None,
            "b": base_val
        }

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
        "Fattore di rischio": ["Prezzo Energia", "Quantità Venduta", "Bonus straordinario", "Costo straordinario"],
        "variabile": ["prezzo", "quantità", "prezzo", "prezzo"],
        "anno": [2025, 2025, 2026, 2026],
        "valore a piano": [100000, 200000, 0, 0],
        "costo variabile": ["no", "no", "no", "no"],
        "perc": [np.nan, np.nan, np.nan, np.nan],
        "variabile dipendente": [np.nan, np.nan, np.nan, np.nan],
        "tipo_variabile": ["ricavo", "ricavo", "ricavo", "costo"],
        "k_min": [0.01, 0.01, 0.00, 0.00],
        "k_max": [0.05, 0.05, 0.00, 0.00],

        # PREZZO
        "distribuzione_prezzo": ["normale", None, "bernoulli", "bernoulli"],
        "p_base": [np.nan, 50, np.nan, np.nan],
        "p_min": [np.nan, np.nan, np.nan, np.nan],
        "p_moda": [np.nan, np.nan, np.nan, np.nan],
        "p_max": [np.nan, np.nan, np.nan, np.nan],
        "p_mu": [120, np.nan, np.nan, np.nan],
        "p_sigma": [15, np.nan, np.nan, np.nan],
        "p_prob": [np.nan, np.nan, 0.30, 0.15],
        "p_value": [np.nan, np.nan, 1000000, 500000],

        # QUANTITA'
        "distribuzione_quantità": [None, "triangolare", None, None],
        "q_base": [1000, np.nan, 1, 1],
        "q_min": [np.nan, 800, np.nan, np.nan],
        "q_moda": [np.nan, 1000, np.nan, np.nan],
        "q_max": [np.nan, 1300, np.nan, np.nan],
        "q_mu": [np.nan, np.nan, np.nan, np.nan],
        "q_sigma": [np.nan, np.nan, np.nan, np.nan],
        "q_prob": [np.nan, np.nan, np.nan, np.nan],
        "q_value": [np.nan, np.nan, np.nan, np.nan],
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
        varia = normalize_varia(row.get('variabile'))
        anno = row.get('anno')

        k_min = row['k_min'] if not pd.isna(row.get('k_min')) else 0
        k_max = row['k_max'] if not pd.isna(row.get('k_max')) else 0

        base_val = row['valore a piano'] if not pd.isna(row.get('valore a piano')) else 0

        perc_raw = row.get('perc', 0)
        if isinstance(perc_raw, str) and perc_raw.strip().endswith('%'):
            perc = float(perc_raw.strip().replace('%', '')) / 100
        else:
            perc = perc_raw if not pd.isna(perc_raw) else 0

        block = {
            'name': row['Fattore di rischio'],
            'anno': anno,
            'varia': varia,
            'costo_variabile': (str(row.get('costo variabile', '')).strip().lower() == 'si'),
            'perc': perc,
            'dipendente': row.get('variabile dipendente') if not pd.isna(row.get('variabile dipendente')) else None,
            'incertezza': row.get('incertezza', 1) if not pd.isna(row.get('incertezza', 1)) else 1,
            'valore_a_piano': base_val,
            'tipo_variabile': row.get('tipo_variabile', 'ricavo') if not pd.isna(row.get('tipo_variabile', 'ricavo')) else 'ricavo',
            'k_min': k_min,
            'k_max': k_max
        }

        # default
        p_dict = {"dist": None, "b": row.get("p", 1) if not pd.isna(row.get("p", 1)) else 1}
        q_dict = {"dist": None, "b": row.get("q", 1) if not pd.isna(row.get("q", 1)) else 1}

        if varia == "Solo Prezzo":
            p_dict = build_dist_legacy(row, "p")
            q_dict = {"dist": None, "b": row.get("q", 1) if not pd.isna(row.get("q", 1)) else 1}

        elif varia == "Solo Quantità":
            q_dict = build_dist_legacy(row, "q")
            p_dict = {"dist": None, "b": row.get("p", 1) if not pd.isna(row.get("p", 1)) else 1}

        elif varia == "Entrambi":
            # usa le nuove colonne opzionali
            p_dict = build_dist_specific(row, "prezzo")
            q_dict = build_dist_specific(row, "quantità")

        block['p'] = p_dict
        block['q'] = q_dict

        blocks.append(block)

    return blocks


def sample_distribution(dist_type, params, size):
    if dist_type == 'triangolare':
        a = params['a']
        b = params['b']
        c = params['c']
        return np.random.triangular(a, b, c, size)

    elif dist_type == 'normale':
        mu = params.get('mu')
        sigma = params.get('sigma')
        return np.random.normal(mu, sigma, size)

    elif dist_type == 'bernoulli':
        prob = params.get('prob', 0)
        value = params.get('value', 1)
        eventi = np.random.binomial(1, prob, size)
        return eventi * value

    elif dist_type is None:
        return np.full(size, params.get('b', 0))

    else:
        return np.full(size, params.get('b', 0))
    

def apply_uncertainty_to_params(params, dist_type, k_min, k_max):
    params_mod = params.copy()

    if dist_type == "triangolare":
        a = params.get('a')
        b = params.get('b')
        c = params.get('c')

        if a is not None and not pd.isna(a):
            params_mod['a'] = max(0, a * (1 - k_min))
        if b is not None and not pd.isna(b):
            params_mod['b'] = b
        if c is not None and not pd.isna(c):
            params_mod['c'] = c * (1 + k_max)

    elif dist_type == "normale":
        mu = params.get('mu')
        sigma = params.get('sigma')
        params_mod['mu'] = mu
        params_mod['sigma'] = sigma * (1 + k_max) if sigma is not None and not pd.isna(sigma) else sigma

    elif dist_type == "bernoulli":
        prob = params.get('prob')
        value = params.get('value')

        params_mod['prob'] = prob  # di solito non toccherei la probabilità con k
        if value is not None and not pd.isna(value):
            params_mod['value'] = value * (1 + k_max)

    elif dist_type is None:
        b = params.get('b')
        params_mod['b'] = b

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


def build_distribution_from_row(row, prefix):
    """
    prefix = 'p' oppure 'q'
    Supporta sia:
    - nuovo formato (distribuzione_prezzo)
    - legacy (Distribuzione, prob, value)
    """

    # --- 1) PRIORITA': NUOVO FORMATO ---
    dist_col = f"distribuzione_{'prezzo' if prefix == 'p' else 'quantità'}"
    dist_type = row.get(dist_col, None)

    if isinstance(dist_type, str):
        dist_type = dist_type.strip().lower()

    # --- 2) FALLBACK SU LEGACY ---
    if not dist_type or str(dist_type) == "nan":
        dist_type = row.get("Distribuzione", None)
        if isinstance(dist_type, str):
            dist_type = dist_type.strip().lower()

    base_val = row.get(f"{prefix}_base", 1)
    base_val = 1 if pd.isna(base_val) else base_val

    # --- TRIANGOLARE ---
    if dist_type == "triangolare":
        return {
            "dist": "triangolare",
            "a": row.get(f"{prefix}_min", row.get("min", np.nan)),
            "b": row.get(f"{prefix}_moda", row.get("moda", np.nan)),
            "c": row.get(f"{prefix}_max", row.get("max", np.nan)),
        }

    # --- NORMALE ---
    elif dist_type == "normale":
        return {
            "dist": "normale",
            "mu": row.get(f"{prefix}_mu", row.get("mu", np.nan)),
            "sigma": row.get(f"{prefix}_sigma", row.get("sigma", np.nan)),
        }

    # --- BERNOULLI (QUI È IL FIX) ---
    elif dist_type == "bernoulli":
        return {
            "dist": "bernoulli",
            "prob": row.get(f"{prefix}_prob", row.get("prob", np.nan)),
            "value": row.get(f"{prefix}_value", row.get("value", np.nan)),
        }

    else:
        return {
            "dist": None,
            "b": base_val
        }

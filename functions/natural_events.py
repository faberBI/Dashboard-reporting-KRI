import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import multivariate_normal, pearsonr, t
from scipy.stats import lognorm, poisson
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import ParametricType, Univariate


def get_index_costo_costruzione(data_inizio, data_valutazione, tipo_indicatore, database):
    # Converte le date e crea anno-mese
    data_inizio = pd.to_datetime(data_inizio)
    data_valutazione = pd.to_datetime(data_valutazione)
    anno_mese_inizio = data_inizio.to_period("M").strftime("%Y-%m")
    anno_mese_valutazione = data_valutazione.to_period("M").strftime("%Y-%m")

    # Estrai valore indice alla data_inizio
    riga_inizio = database[
        (database['indicatore'] == tipo_indicatore) &
        (database['time'] == anno_mese_inizio)
    ]
    indice_num = riga_inizio['value'].iloc[0] if not riga_inizio.empty else 100

    # Estrai valore indice alla data_valutazione
    riga_valutazione = database[
        (database['indicatore'] == tipo_indicatore) &
        (database['time'] == anno_mese_valutazione)
    ]
    indice_denom = riga_valutazione['value'].iloc[0] if not riga_valutazione.empty else 100

    # Calcola e ritorna il rapporto degli indici
    rapporto = indice_denom / indice_num
    return np.round(rapporto, 3)


classi_rischio = {
    "Frane": {
        "Molto elevata P4": (0.05, 20),
        "Elevata P3": (0.05, 50),
        "Media P2": (0.02, 100),
        "Moderata P1": (0.02, 200),
        "Aree di Attenzione AA": (0.01, 500),
        "Molto bassa": (0.01, 500)
    },
    "Idro": {
        "Pericolosità idraulica elevata - HighProbabilityHazard": (0.2, 20, 50),
        "Pericolosità idraulica media - MediumProbabilityHazard": (0.1, 100, 200),
        "Pericolosità idraulica bassa - LowProbabilityHazard": (0.04, 200, 500)
    }
}

alpha_tilde_classi_frane = {
    "Molto elevata P4": 8,
    "Elevata P3": 6,
    "Media P2": 4,
    "Moderata P1": 2,
    "Aree di Attenzione AA": 2,
    "Molto bassa": 2
}


# frane

def calcola_vulnerabilita_intrinseca_frane(classe_rischio):
    yi, _ = classi_rischio["Frane"][classe_rischio]
    alpha_tilde = alpha_tilde_classi_frane[classe_rischio]
    numeratore = (1 - math.exp(-alpha_tilde * yi**2))
    denominatore = (1 - math.exp(-alpha_tilde))
    x0_i = (1 - 0.1 * alpha_tilde) * (numeratore / denominatore) + 0.1 * alpha_tilde
    return yi, x0_i

def calcola_perdita_attesa_frane(classe_rischio, valore_esposto, n_simulazioni=10000, sigma_alpha=0.2):
    # Estrazione dei parametri dal dizionario delle classi di rischio
    yi, periodo = classi_rischio["Frane"][classe_rischio]
    alpha_tilde = alpha_tilde_classi_frane[classe_rischio]

    # Coefficienti per limitare la perdita in base alla classe di rischio
    coefficienti_limitazione = {
        "Molto elevata P4": 0.01,
        "Elevata P3": 0.009,
        "Media P2": 0.006,
        "Moderata P1": 0.005,
        "Aree di Attenzione AA": 0.001,
        "Molto bassa": 0.0006
    }

    coeff_limit = coefficienti_limitazione.get(classe_rischio, 0.0006)  # Default a 0.05 per classi non specificate

    # Lista per memorizzare le perdite simulate
    perdite_simulate = []

    # Simulazione delle perdite per n_simulazioni iterazioni
    for _ in range(n_simulazioni):
        # Simulazione del parametro alpha
        alpha_sim = max(np.random.normal(alpha_tilde, sigma_alpha * alpha_tilde), 0.1)

        # Calcolo della perdita basata su una formula specifica
        numeratore = (1 - math.exp(-alpha_sim * yi**2))
        denominatore = (1 - math.exp(-alpha_sim))
        x0_i_sim = (1 - 0.1 * alpha_sim) * (numeratore / denominatore) + 0.1 * alpha_sim

        # Simulazione degli eventi tramite una distribuzione di Poisson
        n_eventi = np.random.poisson(lam=yi * periodo)

        # Calcolo della perdita (senza cap inizialmente)
        perdita = valore_esposto * x0_i_sim * n_eventi

        # Limitiamo la perdita secondo il coefficiente per la classe di rischio
        perdita = perdita * coeff_limit

        # Applichiamo il cap: non può essere maggiore del valore esposto
        perdita = np.clip(perdita, 0, valore_esposto)

        # Aggiungiamo la perdita simulata alla lista
        perdite_simulate.append(perdita)

    # Convertiamo la lista in un array numpy
    perdite_simulate = np.array(perdite_simulate)

    # Calcoliamo la perdita media e i percentili
    perdita_media = perdite_simulate.mean()
    percentili = {
        "p_5": np.quantile(perdite_simulate, 0.05),
        "p_95": np.quantile(perdite_simulate, 0.95),
        "p_15": np.quantile(perdite_simulate, 0.15),
        "p_25": np.quantile(perdite_simulate, 0.25),
        "p_50": np.quantile(perdite_simulate, 0.5),
        "p_75": np.quantile(perdite_simulate, 0.75),
        "p_99_5": np.quantile(perdite_simulate, 0.995),
        "p_99_6": np.quantile(perdite_simulate, 0.996),
        "p_99_7": np.quantile(perdite_simulate, 0.997),
        "p_99_8": np.quantile(perdite_simulate, 0.998),
        "p_99_9": np.quantile(perdite_simulate, 0.999)
    }

    # Restituiamo i risultati: perdita media, percentili, deviazione standard, yi, periodo
    return perdite_simulate ,np.quantile(perdite_simulate, 0.995),np.quantile(perdite_simulate, 0.997),np.quantile(perdite_simulate, 0.999) ,percentili


# idro

def vulnerabilita_profondita_pol(h):
    v_relativa = 4.02 * h**2 + 17.33 * h
    return min(v_relativa, 100) / 100


def simulazione_perdita_attesa_idro(classe_rischio, valore_esposto, h, n_simulazioni=1000, sigma_vuln=0.2, confidenza=(0.05, 0.95)):
    # Estrazione dei parametri dal dizionario delle classi di rischio
    lambda_idro, periodo_min, periodo_max = classi_rischio["Idro"][classe_rischio]

    # Coefficienti per limitare la perdita in base alla classe di rischio
    coefficienti_limitazione_idro = {
        "Pericolosità idraulica elevata - HighProbabilityHazard": 0.0075,
        "Pericolosità idraulica media - MediumProbabilityHazard": 0.003,
        "Pericolosità idraulica bassa - LowProbabilityHazard": 0.00125
    }

    coeff_limit = coefficienti_limitazione_idro.get(classe_rischio, 0.0025)  # Default a 0.05 per classi non specificate

    # Calcolo del periodo medio e della probabilità dell'evento
    periodo_avg = np.mean([periodo_min, periodo_max])
    probabilita_evento = 1 - math.exp(-lambda_idro * periodo_avg)

    # Lista per memorizzare le perdite simulate
    perdite_simulate = []

    # Simulazione delle perdite per n_simulazioni iterazioni
    for _ in range(n_simulazioni):
        coeff1 = np.random.normal(4.02, sigma_vuln)
        coeff2 = np.random.normal(17.33, sigma_vuln)

        # Calcolo della vulnerabilità
        vuln = coeff1 * h**2 + coeff2 * h
        vuln = np.clip(vuln / 100, 0, 1)

        # Simulazione degli eventi tramite una distribuzione di Poisson
        eventi = np.random.poisson(lambda_idro * periodo_avg)

        # Calcolo della perdita (senza cap inizialmente)
        perdita = valore_esposto * probabilita_evento * vuln * eventi

        # Limitiamo la perdita secondo il coefficiente per la classe di rischio
        perdita = perdita * coeff_limit

        # Applichiamo il cap: non può essere maggiore del valore esposto
        perdita = np.clip(perdita, 0, valore_esposto)

        # Aggiungiamo la perdita simulata alla lista
        perdite_simulate.append(perdita)

    # Convertiamo la lista in un array numpy
    perdite_simulate = np.array(perdite_simulate)

    # Calcoliamo i percentili delle perdite simulate
    percentili = {
        "p_5": np.quantile(perdite_simulate, confidenza[0]),
        "p_95": np.quantile(perdite_simulate, confidenza[1]),
        "p_15": np.quantile(perdite_simulate, 0.15),
        "p_25": np.quantile(perdite_simulate, 0.25),
        "p_50": np.quantile(perdite_simulate, 0.5),
        "p_75": np.quantile(perdite_simulate, 0.75),
        "p_99_5": np.quantile(perdite_simulate, 0.995),
        "p_99_6": np.quantile(perdite_simulate, 0.996),
        "p_99_7": np.quantile(perdite_simulate, 0.997),
        "p_99_8": np.quantile(perdite_simulate, 0.998),
        "p_99_9": np.quantile(perdite_simulate, 0.999)
    }

    # Restituiamo la perdita media, i percentili, la probabilità dell'evento e la deviazione standard delle perdite
    return perdite_simulate ,np.quantile(perdite_simulate, 0.995), np.quantile(perdite_simulate, 0.997),np.quantile(perdite_simulate, 0.999), percentili, probabilita_evento, perdite_simulate.std()

# sismico

def calculate_IEMS(Mw, D):
    return 1.45 * Mw - 2.46 * np.log(D) + 8.166


def calculate_mu_D(IEMS, VI, Q=2.3):
    return 2.5 * (1 + np.tanh((IEMS + 6.25 * VI - 13.1) / Q))


def generate_damage_probability(mu_D):
    damage_levels = np.arange(0, 5)
    sigma_D = 0.7
    probabilities = np.exp(-0.5 * ((damage_levels - mu_D) / sigma_D)**2)
    return probabilities / np.sum(probabilities)


def calculate_value_loss(damage_probabilities, loss_values):
    return np.sum(damage_probabilities * loss_values)


def simulazione_perdita_attesa_sismica(m_min, m_max, b, risk_factor, VI, valore_esposto, n_simulazioni=10000, Q=2.3,loss_values = np.array([0.0, 0.05, 0.1, 0.15, 0.2]), confidenza=(0.05, 0.95)):
    if m_min == m_max:
      perdite = np.zeros(n_simulazioni)
      percentili = {k: 0.0 for k in ["p_5", "p_95", "p_15", "p_25", "p_50", "p_75", "p_99_5", "p_99_6", "p_99_7", "p_99_8", "p_99_9"]}
      return perdite, 0.0, 0.0, 0.0, percentili

    perdite = []

    for _ in range(n_simulazioni):
        u = np.random.uniform()
        Mw = m_min - np.log10(1 - u * (1 - 10 ** (-b * (m_max - m_min) * risk_factor))) / (b * risk_factor)
        D = np.random.uniform(1.0, 20.0)
        IEMS = calculate_IEMS(Mw, D)
        mu_D = calculate_mu_D(IEMS, VI, Q)
        dpm = generate_damage_probability(mu_D)
        perdita = calculate_value_loss(dpm, loss_values) * valore_esposto
        perdite.append(perdita)

    perdite = np.array(perdite)
    percentili = {
        "p_5": np.quantile(perdite, confidenza[0]),
        "p_95": np.quantile(perdite, confidenza[1]),
        "p_15": np.quantile(perdite, 0.15),
        "p_25": np.quantile(perdite, 0.25),
        "p_50": np.quantile(perdite, 0.5),
        "p_75": np.quantile(perdite, 0.75),
        "p_99_5": np.quantile(perdite, 0.995),
        "p_99_6": np.quantile(perdite, 0.996),
        "p_99_7": np.quantile(perdite, 0.997),
        "p_99_8": np.quantile(perdite, 0.998),
        "p_99_9": np.quantile(perdite, 0.999)
    }

    return perdite, np.quantile(perdite, 0.995), np.quantile(perdite, 0.997),np.quantile(perdite, 0.999), percentili

# tempeste

def simula_danno_tempesta(valore_mercato, n_simulazioni=1000, eventi_in_25_anni=9, s=0.6541, loc=0, scale=0.4511, confidenza=(0.05, 0.95)):
    lambda_annuo = eventi_in_25_anni / 25
    danni = []

    for _ in range(n_simulazioni):
        numero_tempeste = np.random.poisson(lambda_annuo)

        if numero_tempeste > 0:
            lognorm_losses = lognorm.rvs(s, loc=loc, scale=scale, size=numero_tempeste)
            lognorm_losses = np.clip(lognorm_losses, 0, 0.003)  # o anche rimuovi il clip se troppo limitante
            danno_tempeste = np.sum(lognorm_losses) * valore_mercato
        else:
            danno_tempeste = 0.0

        danni.append(danno_tempeste)

    danni = np.array(danni)
    percentili = {
        "p_5": np.quantile(danni, confidenza[0]),
        "p_95": np.quantile(danni, confidenza[1]),
        "p_15": np.quantile(danni, 0.15),
        "p_25": np.quantile(danni, 0.25),
        "p_50": np.quantile(danni, 0.5),
        "p_75": np.quantile(danni, 0.75),
        "p_99_5": np.quantile(danni, 0.995),
        "p_99_6": np.quantile(danni, 0.996),
        "p_99_7": np.quantile(danni, 0.997),
        "p_99_8": np.quantile(danni, 0.998),
        "p_99_9": np.quantile(danni, 0.999)
    }

    return danni, np.quantile(danni, 0.995), np.quantile(danni, 0.997), np.quantile(danni, 0.999), percentili, np.std(danni)

# funzioni geospaziali

def get_risk_area_idro(lat, long, database):
    punto = gpd.GeoDataFrame({'geometry': [Point(long, lat)]}, crs=database.crs)
    punto_m = punto.to_crs(epsg=32632).buffer(600).to_crs(epsg=4326)
    zone = database[database.intersects(punto_m.iloc[0])]
    return "Pericolosità idraulica bassa - LowProbabilityHazard" if zone.empty else zone.iloc[0]['scenario']


def get_risk_area_frane(lat, long, database):
    punto = gpd.GeoDataFrame({'geometry': [Point(long, lat)]}, crs=database.crs)
    punto_m = punto.to_crs(epsg=32632).buffer(600).to_crs(epsg=4326)
    zone = database[database.intersects(punto_m.iloc[0])]
    return "Molto bassa" if zone.empty else zone.iloc[0]['per_fr_ita']


def get_magnitudes_for_comune(codice_comune, df, default_values={"MwMin": 0, "MwMax": 0, "MwMed": 0}):
    row = df[df["codice_com"] == codice_comune]
    return (default_values["MwMin"], default_values["MwMax"], default_values["MwMed"]) if row.empty else (
        row["MwMin"].values[0], row["MwMax"].values[0], row["MwMed"].values[0])

classi_rischio = {
    "Frane": {
        "Molto elevata P4": (0.05, 20),
        "Elevata P3": (0.05, 50),
        "Media P2": (0.02, 100),
        "Moderata P1": (0.02, 200),
        "Aree di Attenzione AA": (0.01, 500),
        "Molto bassa": (0.01, 500)
    },
    "Idro": {
        "Pericolosità idraulica elevata - HighProbabilityHazard": (0.2, 20, 50),
        "Pericolosità idraulica media - MediumProbabilityHazard": (0.1, 100, 200),
        "Pericolosità idraulica bassa - LowProbabilityHazard": (0.04, 200, 500)
    }
}

alpha_tilde_classi_frane = {
    "Molto elevata P4": 8,
    "Elevata P3": 6,
    "Media P2": 4,
    "Moderata P1": 2,
    "Aree di Attenzione AA": 2,
    "Molto bassa": 2
}


# frane

def calcola_vulnerabilita_intrinseca_frane(classe_rischio):
    yi, _ = classi_rischio["Frane"][classe_rischio]
    alpha_tilde = alpha_tilde_classi_frane[classe_rischio]
    numeratore = (1 - math.exp(-alpha_tilde * yi**2))
    denominatore = (1 - math.exp(-alpha_tilde))
    x0_i = (1 - 0.1 * alpha_tilde) * (numeratore / denominatore) + 0.1 * alpha_tilde
    return yi, x0_i

def calcola_perdita_attesa_frane(classe_rischio, valore_esposto, n_simulazioni=1000, sigma_alpha=0.2):
    # Estrazione dei parametri dal dizionario delle classi di rischio
    yi, periodo = classi_rischio["Frane"][classe_rischio]
    alpha_tilde = alpha_tilde_classi_frane[classe_rischio]

    # Coefficienti per limitare la perdita in base alla classe di rischio
    coefficienti_limitazione = {
        "Molto elevata P4": 0.01,
        "Elevata P3": 0.009,
        "Media P2": 0.006,
        "Moderata P1": 0.005,
        "Aree di Attenzione AA": 0.001,
        "Molto bassa": 0.0006
    }

    coeff_limit = coefficienti_limitazione.get(classe_rischio, 0.0006)  # Default a 0.05 per classi non specificate

    # Lista per memorizzare le perdite simulate
    perdite_simulate = []

    # Simulazione delle perdite per n_simulazioni iterazioni
    for _ in range(n_simulazioni):
        # Simulazione del parametro alpha
        alpha_sim = max(np.random.normal(alpha_tilde, sigma_alpha * alpha_tilde), 0.1)

        # Calcolo della perdita basata su una formula specifica
        numeratore = (1 - math.exp(-alpha_sim * yi**2))
        denominatore = (1 - math.exp(-alpha_sim))
        x0_i_sim = (1 - 0.1 * alpha_sim) * (numeratore / denominatore) + 0.1 * alpha_sim

        # Simulazione degli eventi tramite una distribuzione di Poisson
        n_eventi = np.random.poisson(lam=yi * periodo)

        # Calcolo della perdita (senza cap inizialmente)
        perdita = valore_esposto * x0_i_sim * n_eventi

        # Limitiamo la perdita secondo il coefficiente per la classe di rischio
        perdita = perdita * coeff_limit

        # Applichiamo il cap: non può essere maggiore del valore esposto
        perdita = np.clip(perdita, 0, valore_esposto)

        # Aggiungiamo la perdita simulata alla lista
        perdite_simulate.append(perdita)

    # Convertiamo la lista in un array numpy
    perdite_simulate = np.array(perdite_simulate)

    # Calcoliamo la perdita media e i percentili
    perdita_media = perdite_simulate.mean()
    percentili = {
        "p_5": np.quantile(perdite_simulate, 0.05),
        "p_95": np.quantile(perdite_simulate, 0.95),
        "p_15": np.quantile(perdite_simulate, 0.15),
        "p_25": np.quantile(perdite_simulate, 0.25),
        "p_50": np.quantile(perdite_simulate, 0.5),
        "p_75": np.quantile(perdite_simulate, 0.75),
        "p_99_5": np.quantile(perdite_simulate, 0.995),
        "p_99_6": np.quantile(perdite_simulate, 0.996),
        "p_99_7": np.quantile(perdite_simulate, 0.997),
        "p_99_8": np.quantile(perdite_simulate, 0.998),
        "p_99_9": np.quantile(perdite_simulate, 0.999)
    }

    # Restituiamo i risultati: perdita media, percentili, deviazione standard, yi, periodo
    return perdite_simulate ,np.quantile(perdite_simulate, 0.995),np.quantile(perdite_simulate, 0.997),np.quantile(perdite_simulate, 0.999) ,percentili


# idro

def vulnerabilita_profondita_pol(h):
    v_relativa = 4.02 * h**2 + 17.33 * h
    return min(v_relativa, 100) / 100


def simulazione_perdita_attesa_idro(classe_rischio, valore_esposto, h, n_simulazioni=1000, sigma_vuln=0.2, confidenza=(0.05, 0.95)):
    # Estrazione dei parametri dal dizionario delle classi di rischio
    lambda_idro, periodo_min, periodo_max = classi_rischio["Idro"][classe_rischio]

    # Coefficienti per limitare la perdita in base alla classe di rischio
    coefficienti_limitazione_idro = {
        "Pericolosità idraulica elevata - HighProbabilityHazard": 0.0075,
        "Pericolosità idraulica media - MediumProbabilityHazard": 0.003,
        "Pericolosità idraulica bassa - LowProbabilityHazard": 0.00125
    }

    coeff_limit = coefficienti_limitazione_idro.get(classe_rischio, 0.0025)  # Default a 0.05 per classi non specificate

    # Calcolo del periodo medio e della probabilità dell'evento
    periodo_avg = np.mean([periodo_min, periodo_max])
    probabilita_evento = 1 - math.exp(-lambda_idro * periodo_avg)

    # Lista per memorizzare le perdite simulate
    perdite_simulate = []

    # Simulazione delle perdite per n_simulazioni iterazioni
    for _ in range(n_simulazioni):
        coeff1 = np.random.normal(4.02, sigma_vuln)
        coeff2 = np.random.normal(17.33, sigma_vuln)

        # Calcolo della vulnerabilità
        vuln = coeff1 * h**2 + coeff2 * h
        vuln = np.clip(vuln / 100, 0, 1)

        # Simulazione degli eventi tramite una distribuzione di Poisson
        eventi = np.random.poisson(lambda_idro * periodo_avg)

        # Calcolo della perdita (senza cap inizialmente)
        perdita = valore_esposto * probabilita_evento * vuln * eventi

        # Limitiamo la perdita secondo il coefficiente per la classe di rischio
        perdita = perdita * coeff_limit

        # Applichiamo il cap: non può essere maggiore del valore esposto
        perdita = np.clip(perdita, 0, valore_esposto)

        # Aggiungiamo la perdita simulata alla lista
        perdite_simulate.append(perdita)

    # Convertiamo la lista in un array numpy
    perdite_simulate = np.array(perdite_simulate)

    # Calcoliamo i percentili delle perdite simulate
    percentili = {
        "p_5": np.quantile(perdite_simulate, confidenza[0]),
        "p_95": np.quantile(perdite_simulate, confidenza[1]),
        "p_15": np.quantile(perdite_simulate, 0.15),
        "p_25": np.quantile(perdite_simulate, 0.25),
        "p_50": np.quantile(perdite_simulate, 0.5),
        "p_75": np.quantile(perdite_simulate, 0.75),
        "p_99_5": np.quantile(perdite_simulate, 0.995),
        "p_99_6": np.quantile(perdite_simulate, 0.996),
        "p_99_7": np.quantile(perdite_simulate, 0.997),
        "p_99_8": np.quantile(perdite_simulate, 0.998),
        "p_99_9": np.quantile(perdite_simulate, 0.999)
    }

    # Restituiamo la perdita media, i percentili, la probabilità dell'evento e la deviazione standard delle perdite
    return perdite_simulate ,np.quantile(perdite_simulate, 0.995), np.quantile(perdite_simulate, 0.997),np.quantile(perdite_simulate, 0.999), percentili, probabilita_evento, perdite_simulate.std()

# sismico

def calculate_IEMS(Mw, D):
    return 1.45 * Mw - 2.46 * np.log(D) + 8.166


def calculate_mu_D(IEMS, VI, Q=2.3):
    return 2.5 * (1 + np.tanh((IEMS + 6.25 * VI - 13.1) / Q))


def generate_damage_probability(mu_D):
    damage_levels = np.arange(0, 5)
    sigma_D = 0.7
    probabilities = np.exp(-0.5 * ((damage_levels - mu_D) / sigma_D)**2)
    return probabilities / np.sum(probabilities)


def calculate_value_loss(damage_probabilities, loss_values):
    return np.sum(damage_probabilities * loss_values)


def simulazione_perdita_attesa_sismica(m_min, m_max, b, risk_factor, VI, valore_esposto, n_simulazioni=1000, Q=2.3,loss_values = np.array([0.0, 0.05, 0.1, 0.15, 0.2]), confidenza=(0.05, 0.95)):
    if m_min == m_max:
      perdite = np.zeros(n_simulazioni)
      percentili = {k: 0.0 for k in ["p_5", "p_95", "p_15", "p_25", "p_50", "p_75", "p_99_5", "p_99_6", "p_99_7", "p_99_8", "p_99_9"]}
      return perdite, 0.0, 0.0, 0.0, percentili

    perdite = []

    for _ in range(n_simulazioni):
        u = np.random.uniform()
        Mw = m_min - np.log10(1 - u * (1 - 10 ** (-b * (m_max - m_min) * risk_factor))) / (b * risk_factor)
        D = np.random.uniform(1.0, 20.0)
        IEMS = calculate_IEMS(Mw, D)
        mu_D = calculate_mu_D(IEMS, VI, Q)
        dpm = generate_damage_probability(mu_D)
        perdita = calculate_value_loss(dpm, loss_values) * valore_esposto
        perdite.append(perdita)

    perdite = np.array(perdite)
    percentili = {
        "p_5": np.quantile(perdite, confidenza[0]),
        "p_95": np.quantile(perdite, confidenza[1]),
        "p_15": np.quantile(perdite, 0.15),
        "p_25": np.quantile(perdite, 0.25),
        "p_50": np.quantile(perdite, 0.5),
        "p_75": np.quantile(perdite, 0.75),
        "p_99_5": np.quantile(perdite, 0.995),
        "p_99_6": np.quantile(perdite, 0.996),
        "p_99_7": np.quantile(perdite, 0.997),
        "p_99_8": np.quantile(perdite, 0.998),
        "p_99_9": np.quantile(perdite, 0.999)
    }

    return perdite, np.quantile(perdite, 0.995), np.quantile(perdite, 0.997),np.quantile(perdite, 0.999), percentili

# tempeste

def simula_danno_tempesta(valore_mercato, n_simulazioni=10000, eventi_in_25_anni=9, s=0.6541, loc=0, scale=0.4511, confidenza=(0.05, 0.95)):
    lambda_annuo = eventi_in_25_anni / 25
    danni = []

    for _ in range(n_simulazioni):
        numero_tempeste = np.random.poisson(lambda_annuo)

        if numero_tempeste > 0:
            lognorm_losses = lognorm.rvs(s, loc=loc, scale=scale, size=numero_tempeste)
            lognorm_losses = np.clip(lognorm_losses, 0, 0.003)  # o anche rimuovi il clip se troppo limitante
            danno_tempeste = np.sum(lognorm_losses) * valore_mercato
        else:
            danno_tempeste = 0.0

        danni.append(danno_tempeste)

    danni = np.array(danni)
    percentili = {
        "p_5": np.quantile(danni, confidenza[0]),
        "p_95": np.quantile(danni, confidenza[1]),
        "p_15": np.quantile(danni, 0.15),
        "p_25": np.quantile(danni, 0.25),
        "p_50": np.quantile(danni, 0.5),
        "p_75": np.quantile(danni, 0.75),
        "p_99_5": np.quantile(danni, 0.995),
        "p_99_6": np.quantile(danni, 0.996),
        "p_99_7": np.quantile(danni, 0.997),
        "p_99_8": np.quantile(danni, 0.998),
        "p_99_9": np.quantile(danni, 0.999)
    }

    return danni, np.quantile(danni, 0.995), np.quantile(danni, 0.997), np.quantile(danni, 0.999), percentili, np.std(danni)


def simulazione_portafoglio_con_rischi_correlati(df, n_simulazioni=100_000, database_frane=None, database_idro=None, db_sismico=None):
    results = []

    df["key_zona"] = df["comune"].str.strip() + "_" + df["zona"].str.strip()

    for key_zona, gruppo in df.groupby("key_zona"):
        immobili_zona = gruppo.to_dict(orient="records")

        for immobile in immobili_zona:

            id_immobile = immobile["id"]
            valore_esposto = immobile['building']
            valore_mercato = immobile['building']
            valore_content = immobile['content']
            lat, long = immobile["lat"], immobile["long"]
            h_idraulico = 1  # Valore ipotetico o da estrarre

            area_frane = get_risk_area_frane(lat, long, db_frane)
            area_idro = get_risk_area_idro(lat, long, db_idro)

            print('immobile numero: ', id_immobile)
            print('classe rischio idro: ', area_idro)
            print('classe rischio frane: ', area_frane)

            # building
            danno_frane, frane_perc_995,frane_perc_997,frane_perc_999, percentili_frane, = calcola_perdita_attesa_frane(area_frane, valore_esposto, n_simulazioni)
            danno_idro, idro_perc_995,idro_perc_997,idro_perc_999, percentili_idro, _, _ = simulazione_perdita_attesa_idro(area_idro, valore_esposto, h_idraulico, n_simulazioni)

            MwMin, MwMax, MwMed = get_magnitudes_for_comune(immobile["codice_comune"], df_sismico)
            b = 1
            risk_factor = 1
            VI = 0.1
            print('classe rischio sismico: ', MwMin, MwMax, MwMed)

            danno_sismico, terremoto_perc_995,terremoto_perc_997,terremoto_perc_999, _ = simulazione_perdita_attesa_sismica(MwMin, MwMax, b, risk_factor, VI, valore_esposto, n_simulazioni)

            danno_tempeste, tempesta_perc_995,tempesta_perc_997,tempesta_perc_999,  _, _= simula_danno_tempesta(valore_mercato, n_simulazioni)

            # content
            danno_frane_content, frane_perc_995_content,_,_, _, = calcola_perdita_attesa_frane(area_frane, valore_content, n_simulazioni)
            danno_idro_content, idro_perc_995_content,_,_, _, _, _ = simulazione_perdita_attesa_idro(area_idro, valore_content, h_idraulico, n_simulazioni)
            danno_sismico_content, terremoto_perc_995_content,_,_, _ = simulazione_perdita_attesa_sismica(MwMin, MwMax, b, risk_factor, VI, valore_content, n_simulazioni)
            danno_tempeste_content, tempesta_perc_995_content,_,_,  _, _= simula_danno_tempesta(valore_content, n_simulazioni)

            # X = np.vstack([danno_frane, danno_idro, danno_sismico, danno_tempeste]).T
            # mu = X.mean(axis=0)
            # cov = np.cov(X.T)
            #
            # # Simula perdite aggregate correlate
            # Z = np.random.multivariate_normal(mean=mu, cov=cov, size=n_simulazioni)
            # Z = np.clip(Z, a_min=0, a_max=None)
            # aggregated_losses = Z.sum(axis=1)

            # Pesi per ripartizione tra building e content
            pesi_danno_per_rischio = {
                "frane": {"building": 1, "content": 0},
                "idro": {"building": 1, "content": 0},
                "sismico": {"building": 1, "content": 0},
                "tempeste": {"building": 1, "content": 0},
            }

            # Suddividi i danni
            X = pd.DataFrame({
                'frane_building': danno_frane * pesi_danno_per_rischio['frane']['building'],
                'frane_content': danno_frane_content * pesi_danno_per_rischio['frane']['content'],
                'idro_building': danno_idro * pesi_danno_per_rischio['idro']['building'],
                'idro_content': danno_idro_content * pesi_danno_per_rischio['idro']['content'],
                'sismico_building': danno_sismico * pesi_danno_per_rischio['sismico']['building'],
                'sismico_content': danno_sismico_content * pesi_danno_per_rischio['sismico']['content'],
                'tempeste_building': danno_tempeste * pesi_danno_per_rischio['tempeste']['building'],
                'tempeste_content': danno_tempeste_content * pesi_danno_per_rischio['tempeste']['content'],
            })
            # Fit modello copula
            model = GaussianMultivariate(distribution=Univariate)
            model.fit(X)

            # Simula osservazioni
            Z = model.sample(n_simulazioni)
            Z["total_loss"] = Z.sum(axis=1)

            # Applica capping se eccede il valore di mercato
            exceeding = Z["total_loss"] > valore_mercato
            Z.loc[exceeding, X.columns] = (
                Z.loc[exceeding, X.columns]
                .div(Z.loc[exceeding, "total_loss"], axis=0)
                .mul(valore_mercato)
            )

            # Calcola le perdite aggregate totali
            aggregated_losses = Z[X.columns].sum(axis=1).values
            # Estrai percentili aggregati
            perdita_aggregata_25 = np.quantile(aggregated_losses, 0.25)
            perdita_aggregata_50 = np.quantile(aggregated_losses, 0.5)
            perdita_aggregata_75 = np.quantile(aggregated_losses, 0.75)
            perdita_aggregata_90 = np.quantile(aggregated_losses, 0.90)
            perdita_aggregata_95 = np.quantile(aggregated_losses, 0.95)
            perdita_aggregata_97 = np.quantile(aggregated_losses, 0.97)
            perdita_aggregata_99 = np.quantile(aggregated_losses, 0.99)
            perdita_aggregata_995 = np.quantile(aggregated_losses, 0.995)
            perdita_aggregata_995 = np.quantile(aggregated_losses, 0.995)
            perdita_aggregata_997 = np.quantile(aggregated_losses, 0.997)
            perdita_aggregata_999 = np.quantile(aggregated_losses, 0.999)


            immobile_results = {
                "id_immobile": id_immobile,
                'rischio_frane': area_frane,
                'zona_geografica':key_zona,
                'rischio_idro': area_idro,
                'Magnitudo_min': MwMin,
                'Magnitudo_max': MwMax,
                "Perdita_995_Frane": (frane_perc_995* pesi_danno_per_rischio['frane']['building'] )+ (frane_perc_995_content* pesi_danno_per_rischio['frane']['content']),
                "Perdita_995_Idro": (idro_perc_995* pesi_danno_per_rischio['idro']['building'] )+(idro_perc_995_content* pesi_danno_per_rischio['idro']['content'] ),
                "Perdita_995_Sismico": (terremoto_perc_995* pesi_danno_per_rischio['sismico']['building'] )+(terremoto_perc_995_content* pesi_danno_per_rischio['sismico']['content'] ),
                "Perdita_995_Tempesta": (tempesta_perc_995* pesi_danno_per_rischio['tempeste']['building'] )+(tempesta_perc_995_content* pesi_danno_per_rischio['tempeste']['content'] ),
                "Perdita_aggregata_25": perdita_aggregata_25,
                "Perdita_aggregata_50": perdita_aggregata_50,
                "Perdita_aggregata_75": perdita_aggregata_75,
                "Perdita_aggregata_90": perdita_aggregata_90,
                "Perdita_aggregata_95": perdita_aggregata_95,
                "Perdita_aggregata_97": perdita_aggregata_97,
                "Perdita_aggregata_99": perdita_aggregata_99,
                "Perdite_aggregata_995":perdita_aggregata_995,
                "Perdite_aggregata_997":perdita_aggregata_997,
                "Perdite_aggregata_999":perdita_aggregata_999

            }

            results.append(immobile_results)

    df_perdite = pd.DataFrame(results)
    return df_perdite


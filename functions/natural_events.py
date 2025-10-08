import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import multivariate_normal, pearsonr, t
from scipy.stats import lognorm, poisson
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import ParametricType, Univariate

# ==========================
# Funzione di utilità percentili
# ==========================
def calcola_percentili(array, confidenza=(0.05, 0.95)):
    return {
        "p_5": np.quantile(array, confidenza[0]),
        "p_95": np.quantile(array, confidenza[1]),
        "p_15": np.quantile(array, 0.15),
        "p_25": np.quantile(array, 0.25),
        "p_50": np.quantile(array, 0.5),
        "p_75": np.quantile(array, 0.75),
        "p_99_5": np.quantile(array, 0.995),
        "p_99_6": np.quantile(array, 0.996),
        "p_99_7": np.quantile(array, 0.997),
        "p_99_8": np.quantile(array, 0.998),
        "p_99_9": np.quantile(array, 0.999)
    }

# ==========================
# Frane
# ==========================
def calcola_vulnerabilita_intrinseca_frane(classe):
    yi, _ = classi_rischio["Frane"][classe]
    alpha_tilde = alpha_tilde_classi_frane[classe]
    numeratore = 1 - math.exp(-alpha_tilde * yi**2)
    denominatore = 1 - math.exp(-alpha_tilde)
    x0_i = (1 - 0.1 * alpha_tilde) * (numeratore / denominatore) + 0.1 * alpha_tilde
    return yi, x0_i

def calcola_perdita_attesa_frane(classe, valore, n_simulazioni=10000, sigma_alpha=0.2):
    yi, periodo = classi_rischio["Frane"][classe]
    alpha_tilde = alpha_tilde_classi_frane[classe]
    coeff_limit = {
        "Molto elevata P4": 0.01,
        "Elevata P3": 0.009,
        "Media P2": 0.006,
        "Moderata P1": 0.005,
        "Aree di Attenzione AA": 0.001,
        "Molto bassa": 0.0006
    }.get(classe, 0.0006)

    perdite = []
    for _ in range(n_simulazioni):
        alpha_sim = max(np.random.normal(alpha_tilde, sigma_alpha * alpha_tilde), 0.1)
        numeratore = 1 - math.exp(-alpha_sim * yi**2)
        denominatore = 1 - math.exp(-alpha_sim)
        x0_i_sim = (1 - 0.1 * alpha_sim) * (numeratore / denominatore) + 0.1 * alpha_sim
        n_eventi = np.random.poisson(lam=yi * periodo)
        perdita = np.clip(valore * x0_i_sim * n_eventi * coeff_limit, 0, valore)
        perdite.append(perdita)

    perdite = np.array(perdite)
    return perdite,np.quantile(perdite, 0.95),np.quantile(perdite, 0.995), np.quantile(perdite, 0.997), np.quantile(perdite, 0.999), calcola_percentili(perdite)

# ==========================
# Idro
# ==========================
def vulnerabilita_profondita_pol(h):
    return min(4.02*h**2 + 17.33*h, 100)/100

def simulazione_perdita_attesa_idro(classe, valore, h, n_simulazioni=1000, sigma_vuln=0.2, confidenza=(0.05, 0.95)):
    lambda_idro, periodo_min, periodo_max = classi_rischio["Idro"][classe]
    coeff_limit = {
        "Pericolosità idraulica elevata - HighProbabilityHazard": 0.0075,
        "Pericolosità idraulica media - MediumProbabilityHazard": 0.003,
        "Pericolosità idraulica bassa - LowProbabilityHazard": 0.00125
    }.get(classe, 0.0025)
    periodo_avg = np.mean([periodo_min, periodo_max])
    p_evento = 1 - math.exp(-lambda_idro * periodo_avg)

    perdite = []
    for _ in range(n_simulazioni):
        coeff1, coeff2 = np.random.normal(4.02, sigma_vuln), np.random.normal(17.33, sigma_vuln)
        vuln = np.clip((coeff1*h**2 + coeff2*h)/100, 0, 1)
        eventi = np.random.poisson(lambda_idro * periodo_avg)
        perdita = np.clip(valore * p_evento * vuln * eventi * coeff_limit, 0, valore)
        perdite.append(perdita)

    perdite = np.array(perdite)
    return perdite, np.quantile(perdite, 0.95), np.quantile(perdite, 0.995), np.quantile(perdite, 0.997), np.quantile(perdite, 0.999), calcola_percentili(perdite), p_evento, perdite.std()

# ==========================
# Sismico
# ==========================
def calculate_IEMS(Mw, D):
    return 1.45 * Mw - 2.46 * np.log(D) + 8.166

def calculate_mu_D(IEMS, VI, Q=2.3):
    return 2.5 * (1 + np.tanh((IEMS + 6.25 * VI - 13.1)/Q))

def generate_damage_probability(mu_D):
    damage_levels = np.arange(5)
    sigma_D = 0.7
    probs = np.exp(-0.5 * ((damage_levels - mu_D)/sigma_D)**2)
    return probs/np.sum(probs)

def calculate_value_loss(damage_probabilities, loss_values):
    return np.sum(damage_probabilities*loss_values)

def simulazione_perdita_attesa_sismica(m_min, m_max, b, risk_factor, VI, valore, n_simulazioni=10000, Q=2.3,
                                        loss_values=np.array([0.0,0.05,0.1,0.15,0.2]), confidenza=(0.05,0.95)):
    if m_min == m_max:
        perdite = np.zeros(n_simulazioni)
        return perdite, 0.0, 0.0, 0.0, {k:0.0 for k in ["p_5","p_95","p_15","p_25","p_50","p_75","p_99_5","p_99_6","p_99_7","p_99_8","p_99_9"]}

    perdite = []
    for _ in range(n_simulazioni):
        u = np.random.uniform()
        Mw = m_min - np.log10(1 - u*(1 - 10**(-b*(m_max-m_min)*risk_factor)))/(b*risk_factor)
        D = np.random.uniform(1.0,20.0)
        IEMS = calculate_IEMS(Mw,D)
        mu_D = calculate_mu_D(IEMS,VI,Q)
        dpm = generate_damage_probability(mu_D)
        perdite.append(calculate_value_loss(dpm, loss_values)*valore)

    perdite = np.array(perdite)
    return perdite, np.quantile(perdite, 0.95), np.quantile(perdite, 0.995), np.quantile(perdite, 0.997), np.quantile(perdite, 0.999), calcola_percentili(perdite)

# ==========================
# Tempeste
# ==========================
def simula_danno_tempesta(valore, n_simulazioni=10000, eventi_in_25_anni=9, s=0.6541, loc=0, scale=0.4511, confidenza=(0.05,0.95)):
    lambda_annuo = eventi_in_25_anni / 25
    danni = []
    for _ in range(n_simulazioni):
        n_tempeste = np.random.poisson(lambda_annuo)
        danno = np.sum(np.clip(lognorm.rvs(s, loc=loc, scale=scale, size=n_tempeste), 0, 0.003))*valore if n_tempeste>0 else 0.0
        danni.append(danno)
    danni = np.array(danni)
    return danni,np.quantile(danni, 0.95), np.quantile(danni, 0.995), np.quantile(danni, 0.997), np.quantile(danni, 0.999), calcola_percentili(danni), np.std(danni)

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
            danno_frane,frane_perc_95,  frane_perc_995,frane_perc_997,frane_perc_999, percentili_frane, = calcola_perdita_attesa_frane(area_frane, valore_esposto, n_simulazioni)
            danno_idro, idro_perc_95, idro_perc_995,idro_perc_997,idro_perc_999, percentili_idro, _, _ = simulazione_perdita_attesa_idro(area_idro, valore_esposto, h_idraulico, n_simulazioni)

            MwMin, MwMax, MwMed = get_magnitudes_for_comune(immobile["codice_comune"], df_sismico)
            b = 1
            risk_factor = 1
            VI = 0.1
            print('classe rischio sismico: ', MwMin, MwMax, MwMed)

            danno_sismico,terremoto_perc_95,  terremoto_perc_995,terremoto_perc_997,terremoto_perc_999, _ = simulazione_perdita_attesa_sismica(MwMin, MwMax, b, risk_factor, VI, valore_esposto, n_simulazioni)

            danno_tempeste, tempesta_perc_95, tempesta_perc_995,tempesta_perc_997,tempesta_perc_999,  _, _= simula_danno_tempesta(valore_mercato, n_simulazioni)
 
            # content
            danno_frane_content, frane_perc_95_content, frane_perc_995_content,_,_, _, = calcola_perdita_attesa_frane(area_frane, valore_content, n_simulazioni)
            danno_idro_content, idro_perc_95_content, idro_perc_995_content,_,_, _, _, _ = simulazione_perdita_attesa_idro(area_idro, valore_content, h_idraulico, n_simulazioni)
            danno_sismico_content, terremoto_perc_95_content , terremoto_perc_995_content,_,_, _ = simulazione_perdita_attesa_sismica(MwMin, MwMax, b, risk_factor, VI, valore_content, n_simulazioni)
            danno_tempeste_content, tempesta_perc_95_content, tempesta_perc_995_content,_,_,  _, _= simula_danno_tempesta(valore_content, n_simulazioni)

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
                "Perdita_95_Frane": (frane_perc_95* pesi_danno_per_rischio['frane']['building'] )+ (frane_perc_95_content* pesi_danno_per_rischio['frane']['content']),
                "Perdita_95_Idro": (idro_perc_95* pesi_danno_per_rischio['idro']['building'] )+(idro_perc_95_content* pesi_danno_per_rischio['idro']['content'] ),
                "Perdita_95_Sismico": (terremoto_perc_95* pesi_danno_per_rischio['sismico']['building'] )+(terremoto_perc_95_content* pesi_danno_per_rischio['sismico']['content'] ),
                "Perdita_95_Tempesta": (tempesta_perc_95* pesi_danno_per_rischio['tempeste']['building'] )+(tempesta_perc_95_content* pesi_danno_per_rischio['tempeste']['content'] ),
                "Perdita_aggregata_25": perdita_aggregata_25,
                "Perdita_aggregata_50": perdita_aggregata_50,
                "Perdita_aggregata_75": perdita_aggregata_75,
                "Perdita_aggregata_90": perdita_aggregata_90,
                "Perdita_aggregata_95": perdita_aggregata_95,
                "Perdita_aggregata_97": perdita_aggregata_97,
                "Perdita_aggregata_99": perdita_aggregata_99,
                "Perdite_aggregata_995":perdita_aggregata_995,
                "Perdite_aggregata_997":perdita_aggregata_997,
                "Perdite_aggregata_999":perdita_aggregata_999,
                "KRI_95" :perdita_aggregata_95 / valore_mercato

            }

            results.append(immobile_results)

    df_perdite = pd.DataFrame(results)

    df_perdite['Peso_frane'] = df['Perdita_95_Frane']/(df_perdite['Perdita_95_Frane']+df_perdite['Perdita_95_Idro']+df_perdite['Perdita_95_Tempesta']+df_perdite['Perdita_95_Sismico'])
    df_perdite['Peso_idro'] = df['Perdita_95_Idro']/(df_perdite['Perdita_95_Frane']+df_perdite['Perdita_95_Idro']+df_perdite['Perdita_95_Tempesta']+df_perdite['Perdita_95_Sismico'])
    df_perdite['Peso_tempeste']=df['Perdita_95_Tempesta']/(df_perdite['Perdita_95_Frane']+df_perdite['Perdita_95_Idro']+df_perdite['Perdita_95_Tempesta']+df_perdite['Perdita_95_Sismico'])
    df_perdite['Peso_terremoto'] = df['Perdita_95_Sismico']/(df_perdite['Perdita_95_Frane']+df_perdite['Perdita_95_Idro']+df_perdite['Perdita_95_Tempesta']+df_perdite['Perdita_95_Sismico'])

    df_perdite['Perdita_95_Frane_new'] = df_perdite['Peso_frane'] * df['Perdita_aggregata_95']
    df_perdite['Perdita_95_Idro_new'] = df_perdite['Peso_idro'] * df['Perdita_aggregata_95']
    df_perdite['Perdita_95_Tempesta_new'] = df_perdite['Peso_tempeste'] * df['Perdita_aggregata_95']
    df_perdite['Perdita_95_Sismico_new'] = df_perdite['Peso_terremoto'] * df['Perdita_aggregata_95']
    
    return df_perdite


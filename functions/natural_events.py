import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import Univariate

def simulazione_portafoglio_con_rischi_correlati(
    excel_path, 
    n_simulazioni=100_000, 
    db_frane=None, 
    db_idro=None, 
    df_sismico=None
):
    # Leggi dati immobili
    df = pd.read_excel(excel_path)
    results = []

    # Chiave unica per zona + comune
    df["key_zona"] = df["comune"].str.strip() + "_" + df["zona"].str.strip()

    # Ciclo per ogni zona
    for key_zona, gruppo in df.groupby("key_zona"):
        immobili_zona = gruppo.to_dict(orient="records")

        for immobile in immobili_zona:
            id_immobile = immobile["id"]
            valore_building = immobile["building"]
            valore_content = immobile["content"]
            valore_mercato = immobile["building"]
            lat, long = immobile["lat"], immobile["long"]
            codice_comune = immobile["codice_comune"]
            h_idraulico = 1  # Valore di esempio, da aggiornare se disponibile

            # --- Determina classe rischio ---
            area_frane = get_risk_area_frane(lat, long, db_frane)
            area_idro = get_risk_area_idro(lat, long, db_idro)
            MwMin, MwMax, MwMed = get_magnitudes_for_comune(codice_comune, df_sismico)
            b = 1
            risk_factor = 1
            VI = 0.1

            # --- Simulazione danni building ---
            danno_frane, frane_995, frane_997, frane_999, _ = calcola_perdita_attesa_frane(area_frane, valore_building, n_simulazioni)
            danno_idro, idro_995, idro_997, idro_999, _, _, _ = simulazione_perdita_attesa_idro(area_idro, valore_building, h_idraulico, n_simulazioni)
            danno_sismico, sismico_995, sismico_997, sismico_999, _ = simulazione_perdita_attesa_sismica(MwMin, MwMax, b, risk_factor, VI, valore_building, n_simulazioni)
            danno_tempeste, tempesta_995, tempesta_997, tempesta_999, _, _ = simula_danno_tempesta(valore_mercato, n_simulazioni)

            # --- Simulazione danni content ---
            danno_frane_c, _, _, _, _ = calcola_perdita_attesa_frane(area_frane, valore_content, n_simulazioni)
            danno_idro_c, _, _, _, _, _, _ = simulazione_perdita_attesa_idro(area_idro, valore_content, h_idraulico, n_simulazioni)
            danno_sismico_c, _, _, _, _ = simulazione_perdita_attesa_sismica(MwMin, MwMax, b, risk_factor, VI, valore_content, n_simulazioni)
            danno_tempeste_c, _, _, _, _, _ = simula_danno_tempesta(valore_content, n_simulazioni)

            # --- Costruzione DataFrame per copula ---
            X = pd.DataFrame({
                'frane_building': danno_frane,
                'frane_content': danno_frane_c,
                'idro_building': danno_idro,
                'idro_content': danno_idro_c,
                'sismico_building': danno_sismico,
                'sismico_content': danno_sismico_c,
                'tempeste_building': danno_tempeste,
                'tempeste_content': danno_tempeste_c,
            })

            # --- Fit copula gaussiana ---
            model = GaussianMultivariate(distribution=Univariate)
            model.fit(X)
            Z = model.sample(n_simulazioni)
            Z["total_loss"] = Z.sum(axis=1)

            # Applica capping sul valore di mercato
            exceeding = Z["total_loss"] > valore_mercato
            Z.loc[exceeding, X.columns] = (
                Z.loc[exceeding, X.columns]
                .div(Z.loc[exceeding, "total_loss"], axis=0)
                .mul(valore_mercato)
            )
            aggregated_losses = Z[X.columns].sum(axis=1).values

            # --- Estrazione percentili aggregati ---
            percentili_agg = {
                f'perdite_aggregata_{p}': np.quantile(aggregated_losses, p) 
                for p in [0.25, 0.5, 0.75, 0.90, 0.95, 0.97, 0.99, 0.995, 0.997, 0.999]
            }

            # --- Salvataggio risultati per immobile ---
            results.append({
                "id_immobile": id_immobile,
                'rischio_frane': area_frane,
                'rischio_idro': area_idro,
                'Magnitudo_min': MwMin,
                'Magnitudo_max': MwMax,
                'zona_geografica': key_zona,
                "Perdita_995_Frane": frane_995,
                "Perdita_995_Idro": idro_995,
                "Perdita_995_Sismico": sismico_995,
                "Perdita_995_Tempesta": tempesta_995,
                **percentili_agg
            })

    return pd.DataFrame(results)





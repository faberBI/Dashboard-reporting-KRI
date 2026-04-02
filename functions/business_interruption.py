import pandas as pd
import numpy as np
from scipy.stats import gamma

def get_kri_bi(df, n_sim=10000):

    # --- DATE E DURATA ---
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['End Date'] = pd.to_datetime(df['End Date'])
    df['durata in giorni'] = (df['End Date'] - df['Start Date']).dt.days

    # =========================
    # 📌 KRI PER CAUSA
    # =========================
    result = df.groupby('CAUSA').agg(
        count=('CAUSA', 'count'),
        durata_sum=('durata in giorni', 'sum')
    )

    total_count = result['count'].sum()
    total_durata = result['durata_sum'].sum()

    result['freq_rel'] = result['count'] / total_count
    result['durata_rel'] = result['durata_sum'] / total_durata

    result['TFRI'] = result['freq_rel'] * result['durata_rel']

    # Normalizzazione safe
    min_val = result['TFRI'].min()
    max_val = result['TFRI'].max()

    if max_val != min_val:
        result['TFRI_norm'] = (result['TFRI'] - min_val) / (max_val - min_val)
    else:
        result['TFRI_norm'] = 0

    # =========================
    # 📌 GEO - REGIONI
    # =========================
    result_GEO_REG = df.groupby('Regioni').agg(
        count=('CAUSA', 'count'),
        durata_sum=('durata in giorni', 'sum')
    )

    total_count_GEO_REG = result_GEO_REG['count'].sum()
    total_durata_GEO_REG = result_GEO_REG['durata_sum'].sum()

    result_GEO_REG['freq_rel'] = result_GEO_REG['count'] / total_count_GEO_REG
    result_GEO_REG['durata_rel'] = result_GEO_REG['durata_sum'] / total_durata_GEO_REG

    result_GEO_REG['WGHI_reg'] = result_GEO_REG['freq_rel'] * result_GEO_REG['durata_rel']

    min_val_reg = result_GEO_REG['WGHI_reg'].min()
    max_val_reg = result_GEO_REG['WGHI_reg'].max()

    if max_val_reg != min_val_reg:
        result_GEO_REG['WGHI_reg_norm'] = (
            result_GEO_REG['WGHI_reg'] - min_val_reg
        ) / (max_val_reg - min_val_reg)
    else:
        result_GEO_REG['WGHI_reg_norm'] = 0

    # =========================
    # 📌 GEO - PROVINCE
    # =========================
    result_GEO_PROV = df.groupby('AREA (PROVINCIA)').agg(
        count=('CAUSA', 'count'),
        durata_sum=('durata in giorni', 'sum')
    )

    total_count_GEO_PROV = result_GEO_PROV['count'].sum()
    total_durata_GEO_PROV = result_GEO_PROV['durata_sum'].sum()

    result_GEO_PROV['freq_rel'] = result_GEO_PROV['count'] / total_count_GEO_PROV
    result_GEO_PROV['durata_rel'] = result_GEO_PROV['durata_sum'] / total_durata_GEO_PROV

    result_GEO_PROV['WGHI_prov'] = result_GEO_PROV['freq_rel'] * result_GEO_PROV['durata_rel']

    min_val_prov = result_GEO_PROV['WGHI_prov'].min()
    max_val_prov = result_GEO_PROV['WGHI_prov'].max()

    if max_val_prov != min_val_prov:
        result_GEO_PROV['WGHI_prov_norm'] = (
            result_GEO_PROV['WGHI_prov'] - min_val_prov
        ) / (max_val_prov - min_val_prov)
    else:
        result_GEO_PROV['WGHI_prov_norm'] = 0

    # =========================
    # 📌 SIMULAZIONE PER CAUSA
    # =========================
    risultati = []

    for causa in df['CAUSA'].unique():
        subset = df[df['CAUSA'] == causa]['durata in giorni'].dropna()

        if len(subset) < 2:
            continue

        shape, loc, scale = gamma.fit(subset, floc=0)

        sims = gamma.rvs(shape, loc=loc, scale=scale, size=n_sim)

        risultati.append({
            'CAUSA': causa,
            'mean_sim': sims.mean(),
            'std_sim': sims.std(),
            'p95': np.percentile(sims, 95),
            'count': len(subset)
        })

    risultati_df = pd.DataFrame(risultati)

    # =========================
    # 📌 KRI BASATO SU P95
    # =========================
    # ✔️ sì: usare p95 ha molto senso (risk-oriented)
    risultati_df['Expected Severe Outage Rate'] = risultati_df['p95']

    # Normalizzazione KRI
    min_kri = risultati_df['Expected Severe Outage Rate'].min()
    max_kri = risultati_df['Expected Severe Outage Rate'].max()

    if max_kri != min_kri:
        risultati_df['Expected Severe Outage Rate_norm'] = (
            risultati_df['Expected Severe Outage Rate'] - min_kri
        ) / (max_kri - min_kri)
    else:
        risultati_df['Expected Severe Outage Rate_norm'] = 0

    # =========================
    return result.reset_index(), result_GEO_REG.reset_index(), result_GEO_PROV.reset_index(), risultati_df

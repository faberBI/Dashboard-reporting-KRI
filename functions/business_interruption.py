import pandas as pd
import numpy as np
from scipy.stats import gamma

def get_kri_bi(df, n_sim=10000):
    import pandas as pd
    import numpy as np
    from scipy.stats import gamma

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
    result['TFRI_norm'] = (result['TFRI'] - min_val) / (max_val - min_val) if max_val != min_val else 0

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
    result_GEO_REG['WGHI_reg_norm'] = (
        (result_GEO_REG['WGHI_reg'] - min_val_reg) / (max_val_reg - min_val_reg)
        if max_val_reg != min_val_reg else 0
    )

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
    result_GEO_PROV['WGHI_prov_norm'] = (
        (result_GEO_PROV['WGHI_prov'] - min_val_prov) / (max_val_prov - min_val_prov)
        if max_val_prov != min_val_prov else 0
    )

    # =========================
    # 📌 SIMULAZIONE PER CAUSA
    # =========================
    risultati = []
    for causa in df['CAUSA'].unique():
        subset = df[df['CAUSA'] == causa]['durata in giorni'].dropna()
        subset = subset[subset > 0]  # solo valori positivi

        if len(subset) < 2:
            continue  # troppo pochi dati

        try:
            # Fit distribuzione Gamma
            shape, loc, scale = gamma.fit(subset, floc=0)

            # Simulazione
            sims = gamma.rvs(shape, loc=loc, scale=scale, size=n_sim)

            risultati.append({
                'CAUSA': causa,
                'mean_sim': sims.mean(),
                'std_sim': sims.std(),
                'p95': np.percentile(sims, 95),
                'count': len(subset)
            })
        except Exception as e:
            print(f"Errore fitting gamma per {causa}: {e}")
            continue

    risultati_df = pd.DataFrame(risultati)

    # =========================
    # 📌 KRI BASATO SU P95
    # =========================
    risultati_df['Expected Severe Outage Rate'] = risultati_df['p95']
    min_kri = risultati_df['Expected Severe Outage Rate'].min()
    max_kri = risultati_df['Expected Severe Outage Rate'].max()
    risultati_df['Expected Severe Outage Rate_norm'] = (
        (risultati_df['Expected Severe Outage Rate'] - min_kri) / (max_kri - min_kri)
        if max_kri != min_kri else 0
    )

    return (
        result.reset_index(),
        result_GEO_REG.reset_index(),
        result_GEO_PROV.reset_index(),
        risultati_df
    )


import matplotlib.pyplot as plt
import seaborn as sns

def plot_kri(result, result_GEO_REG, result_GEO_PROV, risultati_df, top_n=10):
    
    sns.set(style="whitegrid")

    # =========================
    # 📊 1. TOP CAUSE - TFRI
    # =========================
    plt.figure(figsize=(10, 6))
    top = result.sort_values('TFRI_norm', ascending=False).head(top_n)
    
    sns.barplot(data=top, x='TFRI_norm', y='CAUSA', palette='Reds_r')
    plt.title(f"Top {top_n} Cause per TFRI")
    plt.xlabel("TFRI")
    plt.ylabel("Causa")
    plt.tight_layout()
    plt.show()

    # =========================
    # 📊 2. KRI SIMULATO (P95)
    # =========================
    plt.figure(figsize=(10, 6))
    top_kri = risultati_df.sort_values('Expected Severe Outage Rate_norm', ascending=False).head(top_n)

    sns.barplot(data=top_kri, x='Expected Severe Outage Rate_norm', y='CAUSA', palette='Blues_r')
    plt.title(f"Top {top_n} Cause per 'Expected Severe Outage Rate")
    plt.xlabel("'Expected Severe Outage Rate")
    plt.ylabel("Causa")
    plt.tight_layout()
    plt.show()

    # =========================
    # 📊 3. HEATMAP RISCHIO (Freq vs Durata)
    # =========================
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=result,
        x='freq_rel',
        y='durata_rel',
        size='TFRI_norm',
        hue='TFRI_norm',
        palette='coolwarm',
        sizes=(50, 400)
    )

    for i in range(len(result)):
        plt.text(
            result['freq_rel'].iloc[i],
            result['durata_rel'].iloc[i],
            result['CAUSA'].iloc[i],
            fontsize=8
        )

    plt.title("Mappa Rischio: Frequenza vs Durata")
    plt.xlabel("Frequenza Relativa")
    plt.ylabel("Durata Relativa")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # =========================
    # 📊 4. REGIONI
    # =========================
    plt.figure(figsize=(10, 6))
    reg_sorted = result_GEO_REG.sort_values('WGHI_reg_norm', ascending=False).head(top_n)

    sns.barplot(data=reg_sorted, x='WGHI_reg_norm', y='Regioni', palette='Greens_r')
    plt.title(f"Top {top_n} Regioni per WGHI")
    plt.xlabel("WGHI")
    plt.ylabel("Regione")
    plt.tight_layout()
    plt.show()

    # =========================
    # 📊 5. PROVINCE
    # =========================
    plt.figure(figsize=(10, 6))
    prov_sorted = result_GEO_PROV.sort_values('WGHI_prov_norm', ascending=False).head(top_n)

    sns.barplot(data=prov_sorted, x='WGHI_prov_norm', y='AREA (PROVINCIA)', palette='Purples_r')
    plt.title(f"Top {top_n} Province per WGHI")
    plt.xlabel("WGHI")
    plt.ylabel("Provincia")
    plt.tight_layout()
    plt.show()


import unicodedata
import plotly.express as px
import unicodedata

def plot_kri_map_regioni_interattivo(result_GEO_REG, value_col='WGHI_reg_norm'):
    """
    Mappa interattiva con Plotly per regioni italiane
    """
    # Normalizzazione nomi
    def normalize(text):
        if text is None:
            return None
        return unicodedata.normalize('NFKD', str(text)).encode('ascii', errors='ignore').decode('utf-8').title().strip()

    df = result_GEO_REG.copy()
    df['Regioni_clean'] = df['Regioni'].apply(normalize)

    fig = px.choropleth(
        df,
        locations='Regioni_clean',
        locationmode='country names',  # riconosce le regioni italiane
        color=value_col,
        scope="europe",
        color_continuous_scale="RdYlGn_r",
        labels={value_col: "KRI Normalizzato"},
        hover_name='Regioni_clean',
        hover_data={value_col: True}
    )

    fig.update_layout(
        title_text="Mappa Interattiva KRI per Regione",
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='mercator',
            lataxis_range=[35, 47],  # limita a Italia
            lonaxis_range=[6, 19]
        )
    )

    fig.show()
    return fig


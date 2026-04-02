import pandas as pd
import numpy as np
from scipy.stats import gamma
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import streamlit as st
import geopandas as gpd
import json
import openai


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
    # 📌 IMPATTO CLIENTE
    # =========================
    result_Impatto_Cliente = df.groupby('Impatto Cliente').agg(
        count=('CAUSA', 'count'),
        durata_sum=('durata in giorni', 'sum')
    )

    total_count_Impatto_Cliente = result_Impatto_Cliente['count'].sum()
    total_durata_Impatto_Cliente = result_Impatto_Cliente['durata_sum'].sum()

    result_Impatto_Cliente['freq_rel'] = result_Impatto_Cliente['count'] / total_count_Impatto_Cliente
    result_Impatto_Cliente['durata_rel'] = result_Impatto_Cliente['durata_sum'] / total_durata_Impatto_Cliente
    result_Impatto_Cliente['WGHI_ic'] = result_Impatto_Cliente['freq_rel'] * result_Impatto_Cliente['durata_rel']

    min_val_prov = result_Impatto_Cliente['WGHI_ic'].min()
    max_val_prov = result_Impatto_Cliente['WGHI_ic'].max()
    result_Impatto_Cliente['WGHI_ic_norm'] = (
        (result_Impatto_Cliente['WGHI_ic'] - min_val_prov) / (max_val_prov - min_val_prov)
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
        result_Impatto_Cliente.reset_index(),
        risultati_df
    )


def plot_kri(result, result_GEO_REG, result_GEO_PROV, result_Impatto_Cliente, risultati_df, top_n=20):
    # =========================
    # 1️⃣ TOP CAUSE - TFRI
    # =========================
    top = result.sort_values('TFRI_norm', ascending=False).head(top_n)
    fig = px.bar(
        top,
        x='TFRI_norm',
        y='CAUSA',
        orientation='h',
        color='TFRI_norm',
        color_continuous_scale='Reds',
        title=f"Top {top_n} Cause per TFRI"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 2️⃣ KRI SIMULATO (P95)
    # =========================
    top_kri = risultati_df.sort_values('Expected Severe Outage Rate_norm', ascending=False).head(top_n)
    fig = px.bar(
        top_kri,
        x='Expected Severe Outage Rate_norm',
        y='CAUSA',
        orientation='h',
        color='Expected Severe Outage Rate_norm',
        color_continuous_scale='Blues',
        title=f"Top {top_n} Cause per 'Expected Severe Outage Rate'"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 3️⃣ HEATMAP RISCHIO (Freq vs Durata)
    # =========================
    fig = px.scatter(
        result,
        x='freq_rel',
        y='durata_rel',
        size='TFRI_norm',
        color='TFRI_norm',
        hover_data=['CAUSA'],
        color_continuous_scale='RdBu',
        size_max=40,
        title="Mappa Rischio: Frequenza vs Durata"
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 4️⃣ REGIONI
    # =========================
    reg_sorted = result_GEO_REG.sort_values('WGHI_reg_norm', ascending=False).head(top_n)
    fig = px.bar(
        reg_sorted,
        x='WGHI_reg_norm',
        y='Regioni',
        orientation='h',
        color='WGHI_reg_norm',
        color_continuous_scale='Greens',
        title=f"Top {top_n} Regioni per WGHI"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 5️⃣ PROVINCE
    # =========================
    prov_sorted = result_GEO_PROV.sort_values('WGHI_prov_norm', ascending=False).head(top_n)
    fig = px.bar(
        prov_sorted,
        x='WGHI_prov_norm',
        y='AREA (PROVINCIA)',
        orientation='h',
        color='WGHI_prov_norm',
        color_continuous_scale='Purples',
        title=f"Top {top_n} Province per WGHI"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 6️⃣ IMPATTO CLIENTE
    # =========================
    ic_sorted = result_Impatto_Cliente.sort_values('WGHI_ic_norm', ascending=False).head(top_n)
    fig = px.bar(
        ic_sorted,
        x='WGHI_ic_norm',
        y='Impatto Cliente',
        orientation='h',
        color='WGHI_ic_norm',
        color_continuous_scale='Oranges',
        title=f"Top {top_n} Impatti Cliente per WGHI"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def plot_kri_map_regioni_interattivo(result_GEO_REG, shapefile_path='Data/Reg01012026_g_WGS84.shp', value_col='WGHI_reg_norm'):
    """
    Mappa interattiva KRI per regioni italiane usando poligoni reali.
    """

    # --- Carica shapefile regioni italiane ---
    regioni = gpd.read_file(shapefile_path)

    # --- Assicurati che sia in lat/lon ---
    regioni = regioni.to_crs(epsg=4326)

    # --- Normalizzazione nomi per merge ---
    def normalize(text):
        if text is None:
            return None
        return unicodedata.normalize('NFKD', str(text)).encode('ascii', errors='ignore').decode('utf-8').title().strip()

    regioni['Regioni_clean'] = regioni['DEN_REG'].apply(normalize)
    result_GEO_REG['Regioni_clean'] = result_GEO_REG['Regioni'].apply(normalize)

    # --- Merge dati KRI con geometrie ---
    gdf = regioni.merge(result_GEO_REG, on='Regioni_clean', how='left')

    # --- Converti GeoDataFrame in GeoJSON per Plotly ---
    gdf_json = gdf.__geo_interface__

    # --- Crea mappa choropleth ---
    fig = px.choropleth(
        gdf,
        geojson=gdf_json,
        locations=gdf.index,
        color=value_col,
        hover_name='Regioni_clean',
        color_continuous_scale="RdYlGn_r",
        projection="mercator",
        title="Mappa Interattiva KRI per Regione (WGHI)"
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

    return fig


def get_gpt_insights_kri(
    result,
    result_GEO_REG,
    result_GEO_PROV,
    result_Impatto_Cliente,
    risultati_df,
    model="gpt-4"
):
    """
    Genera insight sui KRI di Business Interruption in stile Risk Analyst (discorsivo + data-driven)
    """

    # =========================
    # 📊 PREPARAZIONE INPUT
    # =========================
    summary_text = "KRI Business Interruption Dataset:\n\n"

    # --- CAUSE ---
    top_cause = result.sort_values('TFRI_norm', ascending=False).head(10)
    summary_text += "Top 10 cause per TFRI:\n"
    for _, row in top_cause.iterrows():
        summary_text += (
            f"- {row['CAUSA']}: "
            f"freq_rel={row['freq_rel']:.2f}, "
            f"durata_rel={row['durata_rel']:.2f}, "
            f"TFRI_norm={row['TFRI_norm']:.2f}\n"
        )

    # --- REGIONI ---
    top_regioni = result_GEO_REG.sort_values('WGHI_reg_norm', ascending=False).head(10)
    summary_text += "\nTop 10 regioni per WGHI:\n"
    for _, row in top_regioni.iterrows():
        summary_text += f"- {row['Regioni']}: WGHI_reg_norm={row['WGHI_reg_norm']:.2f}\n"

    # --- PROVINCE ---
    top_prov = result_GEO_PROV.sort_values('WGHI_prov_norm', ascending=False).head(10)
    summary_text += "\nTop 10 province per WGHI:\n"
    for _, row in top_prov.iterrows():
        summary_text += f"- {row['AREA (PROVINCIA)']}: WGHI_prov_norm={row['WGHI_prov_norm']:.2f}\n"

    # --- IMPATTO CLIENTE ---
    top_ic = result_Impatto_Cliente.sort_values('WGHI_ic_norm', ascending=False).head(10)
    summary_text += "\nTop 10 impatti cliente:\n"
    for _, row in top_ic.iterrows():
        summary_text += f"- {row['Impatto Cliente']}: WGHI_ic_norm={row['WGHI_ic_norm']:.2f}\n"

    # --- SIMULAZIONI ---
    top_sim = risultati_df.sort_values('Expected Severe Outage Rate_norm', ascending=False).head(10)
    summary_text += "\nTop 10 cause per severità (simulazione):\n"
    for _, row in top_sim.iterrows():
        summary_text += (
            f"- {row['CAUSA']}: "
            f"p95={row['p95']:.2f}, "
            f"mean={row['mean_sim']:.2f}, "
            f"std={row['std_sim']:.2f}\n"
        )

    # =========================
    # 🧠 PROMPT (ANALYST DISCORSIVO)
    # =========================
    system_prompt = """
Sei un Senior Risk Analyst esperto in Business Interruption.

Analizza i KRI forniti e produci un commento strutturato, chiaro e data-driven.

Linee guida:
- Usa solo i dati forniti
- Mantieni un tono professionale ma discorsivo
- Evita descrizioni lunghe o generiche
- Inserisci sempre i valori numerici a supporto
- Evidenzia ciò che conta davvero (concentrazione, anomalie, gap)

Output richiesto:

1. Cause critiche
- Top 3 per TFRI_norm
- Top 3 per p95
- Evidenziare eventuali mismatch tra frequenza e severità

2. Concentrazione geografica
- Top 3 regioni e province
- Evidenziare eventuali cluster di rischio

3. Impatto cliente
- Top 3 categorie
- Evidenziare eventuale dominanza

4. Variabilità e tail risk
- Cause con deviazione standard più alta
- Gap tra mean e p95
- Identificazione di possibili heavy tail

5. Risk signals
- 2–3 osservazioni sintetiche basate sui dati
- Inserire brevi raccomandazioni pratiche (non generiche)

Formato:
- Testo discorsivo ma sintetico
- Frasi brevi e incisive
- Linguaggio da risk analyst senior
"""

    # =========================
    # 🤖 CHIAMATA GPT
    # =========================
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_text}
        ],
        max_tokens=1000,
        temperature=0.4  # 🎯 bilanciato: naturale ma non inventa
    )

    return response.choices[0].message.content
    

import geopandas as gpd
import requests
import io
import zipfile
import os
import json
import subprocess


def load_shapefiles_from_dropbox(frane_url, idro_url, extract_folder="./data"):
    """
    Scarica gli shapefile DB-Frane e DB-Idro da Dropbox, li estrae e restituisce due GeoDataFrame.
    
    Parametri:
    - frane_url: str, link Dropbox DB-Frane.zip (modifica dl=0 → dl=1)
    - idro_url: str, link Dropbox DB-Idro.zip (modifica dl=0 → dl=1)
    - extract_folder: cartella dove estrarre i file
    
    Ritorna:
    - db_frane: GeoDataFrame
    - db_idro: GeoDataFrame
    """
    os.makedirs(extract_folder, exist_ok=True)
    
    def _download_and_load(url, name):
        # Assicurati che il link Dropbox sia dl=1 per download diretto
        url = url.replace("dl=0", "dl=1")
        zip_path = os.path.join(extract_folder, f"{name}.zip")
        
        # Scarica il file
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Estrai lo ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        # Trova lo shapefile
        shp_files = [f for f in os.listdir(extract_folder) if f.endswith(".shp") and name.lower() in f.lower()]
        if not shp_files:
            raise FileNotFoundError(f"Nessun file .shp trovato nello ZIP di {name}")
        
        shp_path = os.path.join(extract_folder, shp_files[0])
        gdf = gpd.read_file(shp_path)
        
        # Rimuovi ZIP temporaneo
        os.remove(zip_path)
        return gdf
    
    db_frane = _download_and_load(frane_url, "Frane")
    db_idro = _download_and_load(idro_url, "Idro")
    
    return db_frane, db_idro
    
# ==========================
# Dizionari centralizzati
# ==========================
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

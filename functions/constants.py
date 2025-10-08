import geopandas as gpd
import requests
import io
import zipfile
import os
import json
import subprocess


def load_kaggle_shapefiles(username, key, datasets_folder="./datasets"):
    """
    Scarica i dataset da Kaggle (db-frane e db-idro-full), estrae e carica gli shapefile in GeoDataFrame.

    Parametri:
    - username: str, username Kaggle
    - key: str, Kaggle API key
    - datasets_folder: str, cartella dove salvare i dataset

    Ritorna:
    - db_frane: GeoDataFrame
    - db_idro: GeoDataFrame
    """
    
    # --- 1. Crea il file kaggle.json ---
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_credentials = {"username": username, "key": key}
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        json.dump(kaggle_credentials, f)
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    
    # --- 2. Installa Kaggle API ---
    subprocess.run(["pip", "install", "--quiet", "kaggle"], check=True)
    
    # --- 3. Scarica ed estrai i dataset ---
    os.makedirs(datasets_folder, exist_ok=True)
    
    datasets = {
        "db-frane": "faberbi/db-frane",
        "db-idro_full": "faberbi/db-idro-full"
    }
    
    for folder_name, kaggle_id in datasets.items():
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", kaggle_id,
            "-p", datasets_folder,
            "--unzip"
        ], check=True)
    
    # --- 4. Funzione interna per caricare shapefile ---
    def _load_shapefile(folder_path, name_contains):
        shp_files = [f for f in os.listdir(folder_path) if f.endswith('.shp') and name_contains in f]
        if not shp_files:
            raise FileNotFoundError(f"Nessun file .shp contenente '{name_contains}' trovato in {folder_path}")
        shp_path = os.path.join(folder_path, shp_files[0])
        return gpd.read_file(shp_path)
    
    # --- 5. Carica shapefile ---
    db_frane = _load_shapefile(datasets_folder, "db_frane")
    db_idro = _load_shapefile(datasets_folder, "db_idro_full")
    
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

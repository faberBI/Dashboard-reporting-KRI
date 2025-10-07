import geopandas as gpd
import requests
import io
import zipfile

def load_shapefile_from_drive(drive_url):
    # Scarica lo zip da Google Drive
    r = requests.get(drive_url)
    r.raise_for_status()

    # Leggi zip in memoria
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Cerca il file .shp dentro lo zip
    shp_file = [f for f in z.namelist() if f.endswith(".shp")][0]
    
    # Carica shapefile con geopandas
    gdf = gpd.read_file(f"zip://{drive_url}!{shp_file}")
    return gdf
    
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

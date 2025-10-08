import geopandas as gpd
import requests
import io
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def load_shapefile_from_drive(file_id, zip_name="data.zip", extract_folder="data"):
    """
    Scarica un file ZIP da Google Drive tramite ID, estrae il contenuto
    e restituisce il GeoDataFrame del primo shapefile trovato.
    
    Args:
        file_id (str): ID del file Google Drive.
        zip_name (str): Nome del file ZIP locale.
        extract_folder (str): Cartella dove estrarre i file.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame dello shapefile, oppure None se non trovato.
    """
    url = f"https://drive.google.com/uc?id={file_id}"

    # Scarica il file ZIP solo se non esiste già
    if not os.path.exists(zip_name):
        gdown.download(url, zip_name, quiet=False)

    # Estrai il contenuto del file ZIP
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

    # Trova il primo file .shp nella cartella estratta
    shp_files = [f for f in os.listdir(extract_folder) if f.endswith('.shp')]

    if shp_files:
        shp_path = os.path.join(extract_folder, shp_files[0])
        gdf = gpd.read_file(shp_path)
        return gdf
    else:
        print("Nessun file .shp trovato nello ZIP.")
        return None
    
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

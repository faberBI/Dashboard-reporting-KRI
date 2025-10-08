import geopandas as gpd
import requests
import io
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def load_all_shapefiles_from_drive_folder(folder_id, local_folder="shapefiles_temp"):
    """
    Scarica tutti gli shapefile da una cartella Google Drive e ritorna un dizionario di GeoDataFrame.
    
    Args:
        folder_id (str): ID della cartella di Google Drive.
        local_folder (str): Cartella locale temporanea dove salvare i file.
    
    Returns:
        dict: chiavi = nome shapefile, valori = GeoDataFrame
    """
    # Autenticazione
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Crea cartella locale
    os.makedirs(local_folder, exist_ok=True)

    # Lista file nella cartella
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

    # Raggruppa i file per shapefile (nome base senza estensione)
    shapefiles = {}
    for file in file_list:
        name = file['title']
        if '.' in name:
            base, ext = name.rsplit('.', 1)
            if ext.lower() in ['shp', 'shx', 'dbf', 'prj']:
                if base not in shapefiles:
                    shapefiles[base] = []
                shapefiles[base].append(file)

    if not shapefiles:
        raise ValueError("Nessuno shapefile trovato nella cartella!")

    # Scarica tutti i file di ogni shapefile
    gdfs = {}
    for base_name, files in shapefiles.items():
        for f in files:
            f.GetContentFile(os.path.join(local_folder, f['title']))
        # Trova il file .shp
        shp_file = [f for f in os.listdir(local_folder) if f.startswith(base_name) and f.endswith(".shp")][0]
        shp_path = os.path.join(local_folder, shp_file)
        gdfs[base_name] = gpd.read_file(shp_path)
    
    return gdfs
    
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

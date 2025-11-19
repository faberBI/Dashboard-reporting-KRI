import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def get_risk_area_idro(lat, long, database):
    punto = gpd.GeoDataFrame({'geometry': [Point(long, lat)]}, crs=database.crs)
    punto_m = punto.to_crs(epsg=32632).buffer(600).to_crs(epsg=4326)
    zone = database[database.intersects(punto_m.iloc[0])]
    return "Pericolosità idraulica bassa - LowProbabilityHazard" if zone.empty else zone.iloc[0]['scenario']


def get_risk_area_frane(lat, long, database):
    punto = gpd.GeoDataFrame({'geometry': [Point(long, lat)]}, crs=database.crs)
    punto_m = punto.to_crs(epsg=32632).buffer(600).to_crs(epsg=4326)
    zone = database[database.intersects(punto_m.iloc[0])]
    return "Molto bassa" if zone.empty else zone.iloc[0]['per_fr_ita']


def get_magnitudes_for_comune(codice_comune, df, default_values={"MwMin": 0, "MwMax": 0, "MwMed": 0}):
    row = df[df["codice_com"] == codice_comune]
    return (default_values["MwMin"], default_values["MwMax"], default_values["MwMed"]) if row.empty else (
        row["MwMin"].values[0], row["MwMax"].values[0], row["MwMed"].values[0])

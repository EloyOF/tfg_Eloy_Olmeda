#!/usr/bin/env python
# coding: utf-8

import numpy as np
import geopandas as gpd
import os


# Se carga el fichero de espacios verdes candidatos
gdf_final = gpd.read_file("localizaciones_final.geojson")

# Media y desviación típica asignado a cada tipo de espacio verde
parametros_coste = {
    "jardín de bolsillo": (54.5, 10.9),
    "parque de barrio": (44, 8.8),
    "gran parque": (57.5, 11.5),
    "cubierta verde": (115, 23)}

def calcular_coste_fila(row):
    """
    Función que calcula el coste de un espacio verde dada una fila
    """
    tipo = row["tipo"]
    area = row["area"]
    if tipo not in parametros_coste:
        return np.nan
    media, desviacion = parametros_coste[tipo]
    coste_unitario = np.random.normal(media, desviacion)
    return int(round(coste_unitario * area))

def generar_x_instancias(gdf_base, carpeta_salida, n=20, seed=1):
    """
    Función que genera instancias del fichero (20 por defecto), añadiéndole una 
    semilla diferente a cada fichero. Esta función recibe un fichero y genera una
    carpeta donde almacena las distintas instancias.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    for i in range(1, n+1):
        np.random.seed(seed+i)  
        gdf_inst = gdf_base.copy()
        gdf_inst["coste"] = gdf_inst.apply(calcular_coste_fila, axis=1)
        ruta = os.path.join(carpeta_salida, f"instancia_costes_{i:02d}.geojson")
        gdf_inst.to_file(ruta, driver="GeoJSON")

# Se llama a la función de generar instancias sobre el fichero de candidatos y se le
# pasa el nombre de la carpeta a crear para que guarde ahí las 20 instancias
generar_x_instancias(gdf_final, "instancias_costes", n=20)

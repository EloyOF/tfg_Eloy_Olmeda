#!/usr/bin/env python
# coding: utf-8

import numpy as np
import geopandas as gpd
import pandas as pd

# SOLARES
# Se carga el fichero, se excluyen las pedanías y se mantienen aquellos solares con un
# área de mínimo 150 metros cuadrados
solares = gpd.read_file("solares_valencia_geo.geojson")
excluir = ["POBLATS DEL NORD", "POBLATS DEL SUD", "POBLATS DE L'OEST", "POBLATS OEST"]
gdf_solares = solares[~solares["nomdistrit"].isin(excluir)]
gdf_solares = gdf_solares[gdf_solares["st_area_shape"] >= 150]

# PLANIFICACIÓN ESPACIOS VERDES
# Se carga el fichero de espacios verdes planificados y original 
planificados = gpd.read_file("planificacion_zonas_verdes.geojson").to_crs(epsg=25830)
gdf_existentes = gpd.read_file("espacios_verdes_final_geo.geojson").to_crs(epsg=25830)

# Se eliminan los espacios verdes de tipo red viaria y se mantienen aquellos de más 
# de una hectárea
gdf_planificado= planificados[(planificados["nivel2"] != "Red viaria") &
                              (planificados["m2_poligon"] >= 10000)]

# Se aplica un filtrado para que aquellos espacios verdes planificados que solapen con 
# alguno de los que ya hay no se utilicen
buffer_size = 5  # en metros
gdf_existentes_buffered = gdf_existentes.buffer(buffer_size)
geom_existente_unida = gdf_existentes_buffered.unary_union

gdf_planificados_exclusivos = gdf_planificado[
    ~gdf_planificado.intersects(geom_existente_unida)]

# Se excluyen algunos espacios verdes que no resultan adecuados por diferentes motivos
ids_a_eliminar = [304, 323, 374, 378, 881, 460, 1141, 1263, 62, 1325, 1329, 1330,
                  1353, 1221, 142, 156, 237, 914, 844, 1434, 132, 1418, 1419, 493, 885]
gdf_planificados = gdf_planificados_exclusivos[~gdf_planificados_exclusivos["id"].isin(ids_a_eliminar)]

# TERRENOS EN DESUSO O ABANDONADOS
# Se carga el fichero y se añade una columna con el área en metros cuadrados de
# cada localización
brownfield = gpd.read_file("brownfield.geojson").to_crs(epsg=25830)
brownfield['area_m2'] = brownfield.geometry.area

# Se excluyen todos aquellos terrenos fuera del área metropolitana de Valencia y
# menores de 150 metros cuadrados
distritos = gpd.read_file("distritos_final.geojson") 
valencia_area = distritos.unary_union
brownfields_recortados = brownfield.clip(valencia_area)
brownfields_recortados = brownfields_recortados[brownfields_recortados["area_m2"] > 150] 

# Se añade el distrito a cada terreno en desuso
gdf_distritos = gpd.read_file("distritos_filtrado.geojson")
gdf_distritos = gdf_distritos.to_crs(epsg=25830)
brownfields_recortados = gpd.sjoin(brownfields_recortados,
                                   gdf_distritos[["nombre", "geometry"]],
                                   how="left", predicate="intersects")
brownfields_recortados["distrito"] = brownfields_recortados["nombre"]
gdf_brownfields = brownfields_recortados.drop(columns=["nombre", "index_right"])

# CUBIERTAS VERDES
# Se carga el fichero y se renombra una columna
cubiertas = gpd.read_file("cubiertas_verdes.geojson").to_crs(epsg=25830)
gdf_cubiertas = cubiertas.rename(columns={"superficie": "area"})

# A la columna creada se le calcula para cada fila el área en metros cuadrado
gdf_cubiertas["area"] = gdf_cubiertas.geometry.area
gdf_cubiertas["tipo"] = "cubierta verde"

# Se les añade también a cada cubierta su distrito de pertenencia
distritos = [
    "L'OLIVERETA",
    "CIUTAT VELLA",
    "CIUTAT VELLA",
    "QUATRE CARRERES",
    "RASCANYA",
    "PATRAIX",
    "L'OLIVERETA",
    "L'OLIVERETA",
    "QUATRE CARRERES",
    "L'OLIVERETA",
    "BENICALAP",
    "QUATRE CARRERES"]

gdf_cubiertas["distrito"] = distritos

# UNIFICACIÓN
# Se renombran columnas para que sean todas iguales y se crean algunas necesarias
gdf_solares = gdf_solares.rename(columns={"nomdistrit": "distrito"})
gdf_solares = gdf_solares.rename(columns={"refcat": "id"})
gdf_solares = gdf_solares.rename(columns={"st_area_shape": "area"})
gdf_solares["tipo"] = None  

gdf_planificados = gdf_planificados.rename(columns={"m2_poligon": "area"})
gdf_planificados["distrito"] = None  
gdf_planificados["tipo"] = None  

gdf_brownfields = gdf_brownfields.rename(columns={"area_m2": "area"})
gdf_brownfields["distrito"] = None 
gdf_brownfields["tipo"] = None  

# Se crea el fichero unificado que contiene las columnas que se le pide
gdfs = [gdf_solares, gdf_planificados, gdf_brownfields, gdf_cubiertas]
columnas_conservar = ["geometry", "area", "distrito", "id", "tipo"]
gdfs_filtrados = [gdf[columnas_conservar] for gdf in gdfs]
gdf_unificado = gpd.GeoDataFrame(pd.concat(gdfs_filtrados, 
                                           ignore_index=True), crs=gdfs[0].crs)

# Se crea una función para clasificar cada zona en un tipo
def clasificar_zona(row):
    """
    Función que establece el tipo de espacio de verde que es cada terreno
    al pasarle una fila correspondiente a su tamaño en metros cuadrados
    """
    if row["tipo"] == "cubierta verde":
        return "cubierta verde"
    elif row["area"] <= 4000:
        return "jardín de bolsillo"
    elif row["area"] <= 10000:
        return "parque de barrio"
    else:
        return "gran parque"

gdf_unificado["tipo"] = gdf_unificado.apply(clasificar_zona, axis=1)

# Se le asigna un identificador a cada espacio verde y se excluye uno que daba problemas
gdf_unificado["id"] = range(len(gdf_unificado))
gdf_unificado = gdf_unificado[gdf_unificado["id"] != 61]

# Se le asigna finalmente el distrito correspondiente a cada localización mediante
# la columna "geometry", ya que había ficheros sin sus distrito o mal
gdf_unificado = gpd.sjoin(gdf_unificado, gdf_distritos[["nombre", "geometry"]],
                          how="left", predicate="intersects")
gdf_unificado["distrito"] = gdf_unificado["nombre"]
gdf_unificado = gdf_unificado.drop(columns=["nombre", "index_right"])

# Se le asigna un coste a cada espacio verde según una media y desviación típica
parametros_coste = {
    "jardín de bolsillo": (54.5, 10.9),
    "parque de barrio": (44, 8.8),
    "gran parque": (57.5, 11.5),
    "cubierta verde": (115, 23)}

np.random.seed(33)
def calcular_coste_fila(row):
    """
    Función para calcular el coste, mediante una distribucion normal, de un
    espacio verde dada una fila del fichero.
    """
    tipo = row["tipo"]
    area = row["area"]
    if tipo not in parametros_coste:
        return np.nan  
    media, desviacion = parametros_coste[tipo]
    coste_unitario = np.random.normal(media, desviacion)    
    return int(round(coste_unitario * area))


np.random.seed(33)
gdf_unificado["coste"] = gdf_unificado.apply(calcular_coste_fila, axis=1)

# Se guarda el fichero final unificado
gdf_unificado.to_file("localizaciones_final.geojson", driver="GeoJSON")

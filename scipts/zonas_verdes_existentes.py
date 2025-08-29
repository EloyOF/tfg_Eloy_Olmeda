#!/usr/bin/env python
# coding: utf-8

from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import openrouteservice as ors
from shapely.geometry import shape, mapping
import ast
# Se lee el fichero de datos con los espacios verdes originales
gdf = gpd.read_file("espacios_verdes_existentes.geojson")

# Se renombran sus columnas para mayor comodidad
gdf.columns = ["id", "id_jardin", "nombre", "barrio", "tipo", "area",
               "elementos_fitness", "superficie_huerto_urbano",
               "zona", "distrito", "unidad_gestion", "geo_point_2d", "geometry"]

def extraer_coord(geo_str):
    """
    Función para extraer las coordenadas de una geometría.
    """
    if isinstance(geo_str, str):
        try:
            return ast.literal_eval(geo_str)
        except (ValueError, SyntaxError):
            return None
    return None

# Se crean las columnas de latitud y longitud
gdf["geo_point_2d"] = gdf["geo_point_2d"].apply(extraer_coord)
gdf["lat"] = gdf["geo_point_2d"].apply(lambda x: x["lat"] if x else None)
gdf["lon"] = gdf["geo_point_2d"].apply(lambda x: x["lon"] if x else None)

# Se renombran los distritos que están mal
gdf["distrito"] = gdf["distrito"].replace("SAIDIA", "LA SAIDIA")
gdf["distrito"] = gdf["distrito"].replace("POBLATS MARÍTIMS", "POBLATS MARITIMS")
gdf["distrito"] = gdf["distrito"].replace("SANT PAU", "CAMPANAR")

# Se renombran los barrios que están mal
gdf["barrio"] = gdf["barrio"].replace("RAIOSA", "LA RAIOSA")
gdf["barrio"] = gdf["barrio"].replace("BOTANIC", "EL BOTANIC")
gdf["barrio"] = gdf["barrio"].replace("PETXINA", "LA PETXINA")
gdf["barrio"] = gdf["barrio"].replace("CARME", "EL CARME")
gdf["barrio"] = gdf["barrio"].replace("PUNTA", "LA PUNTA")
gdf["barrio"] = gdf["barrio"].replace("EXPOSICION", "EXPOSICIO")
gdf["barrio"] = gdf["barrio"].replace("TORRE", "LA TORRE")
gdf["barrio"] = gdf["barrio"].replace("SANT LLORENÇ", "SANT LLORENS")
gdf["barrio"] = gdf["barrio"].replace("TENDETES", "LES TENDETES")
gdf["barrio"] = gdf["barrio"].replace("EL GRAO", "EL GRAU")
gdf["barrio"] = gdf["barrio"].replace("LA CREU DEL GRAO", "LA CREU DEL GRAU")
gdf["barrio"] = gdf["barrio"].replace("FORN D'ALCEDO", "EL FORN D'ALCEDO")
gdf["barrio"] = gdf["barrio"].replace("ROIG DE CORELLA", "LA RAIOSA")

# Se excluyen las pedanías
excluir = ["POBLATS DEL NORD", "POBLES DEL SUD", "POBLATS DE L`OEST"]
filtrado = gdf[~gdf["distrito"].isin(excluir)]

# Se excluyen dichos espacios de tipo acompañamiento viario y menores a una hectárea
datos_filtrado = gdf[(gdf["tipo"] != "Acompañamiento Viario")]
datos_filtrado = datos_filtrado[datos_filtrado["area"] >= 10000]

# Se guarda el fichero resultante de lo anterior y se carga
gdf.to_file("espacios_verdes_existentes_final_geo.geojson", driver="GeoJSON")
gdf_zonas_verdes = gpd.read_file("espacios_verdes_existentes_final_geo.geojson")

# Se carga el fichero de los distritos de Valencia, donde se excluyen las pedanías
# También se guarda
distritos = gpd.read_file("distritos.geojson")
excluir = ['POBLATS DEL NORD', 'POBLATS DEL SUD', "POBLATS DE L'OEST"]
gdf_distritos = distritos[~distritos["nombre"].isin(excluir)]
gdf_distritos.to_file("distritos_filtrado.geojson", driver="GeoJSON")

# Ahora se crea el conjunto que unifica ambos ficheros

# Se unifican todas los polígonos de los distritos en uno solo y se guarda
area_valencia = unary_union(gdf_distritos.geometry)
area_valencia = gpd.GeoDataFrame(geometry=[area_valencia], crs="EPSG:4326")
area_valencia.to_file("area_valencia.geojson", driver="GeoJSON")

# API de OpenRouteService
client = ors.Client(key='5b3ce3597851110001cf62482f481ac876494fd88581390976ae7b39')

# Coordenadas de los centroides de cada espacio verde existente
coordenadas = []
for i, row in gdf_zonas_verdes.iterrows():
    lat_str, lon_str = row["geo_point_2d"].split(",")
    lat = float(lat_str.strip())
    lon = float(lon_str.strip())
    coordenadas.append([lon, lat])

# Almacenamiento de las isócronas de 7 minutos a pie de cada espacio verde
lista_isocronas = []
for coord in coordenadas:
    response = client.isochrones(
        locations=[coord],
        range_type="time",
        profile='foot-walking',
        range=[60*7],    # isócronas de 7 minutos a pie
        validate=False,
        attributes=["total_pop"]) 
    for feature in response["features"]:
        geom = shape(feature["geometry"])
        lista_isocronas.append(geom)

# Se unen las isócronas, se recortan para que no salgan del área metropolitana
# de Valencia, se guardan y se cargan
union_isocronas = unary_union(lista_isocronas) 
isocronas_recortadas = union_isocronas.intersection(area_valencia)
gdf_isocronas = gpd.GeoDataFrame(geometry=[isocronas_recortadas], crs="EPSG:4326")
gdf_isocronas.to_file("area_isocronas.geojson", driver="GeoJSON")
union_isocronas = gpd.read_file("area_isocronas.geojson")

# Se pone la ruta al ficher de WorldPop con las estimaciones de población
ruta_raster = "esp_ppp_2020.tif"

# Se cra una función para calcular la población
def calcular_poblacion(poligono, tif_path=ruta_raster):
    """
    Función para calcular la población dado un polígono y un fichero .tif que 
    contenga información sobre población
    """
    with rasterio.open(tif_path) as src:
        out_image, i =  mask(src, [mapping(poligono.geometry.iloc[0])], crop=True)
        data = out_image[0]
        return np.sum(data[data > 0])

# Se calcula la zona de Valencia no cubierta por las isócronas
zona_no_cubierta = area_valencia.difference(isocronas_recortadas)

# Se calcula la población perteneciente a la zona de Valencia cubierta por las 
# isócronas y la que no, además de la total
poblacion_total = round(calcular_poblacion(area_valencia))
poblacion_cubierta = round(calcular_poblacion(union_isocronas))
poblacion_descubierta = round(calcular_poblacion(zona_no_cubierta))

# Se transforman los ficheros de espacios verdes y distritos a epsg=25830 para poder
# aplicarles operaciones en sistema métrico
gdf_zonas_verdes = gdf_zonas_verdes.to_crs(epsg=25830)
gdf_distritos = gdf_distritos.to_crs(gdf_zonas_verdes.crs)

# Se calculan los porcentajes de población cubiertos por zona verde en cada distrito y 
# se añade al fichero de distritos columnas con el valor para cada distrito, de su
# población total, cantitad de población cubierta por la superficie de las isócronas
# y el porcentaje de población cubierto por isócronas 
totales = []
cubiertas = []
porcentajes = []
for i, row in gdf_distritos.iterrows():
    # Área del distrito 
    distrito_geom = row['geometry']
    # Área de las isócronas que está únicamente dentro de el distrito 
    interseccion = distrito_geom.intersection(isocronas_recortadas)
    # Población total del distrito
    poblacion_total = round(calcular_poblacion(distrito_geom))
    totales.append(poblacion_total)
    # Población cubierta por las isócronas en el distrito
    poblacion_cubierta = round(calcular_poblacion(interseccion))
    cubiertas.append(poblacion_cubierta)
    # Porcentaje de población del distrito cubierta por las isócronas
    porcentaje = (poblacion_cubierta / poblacion_total) *100
    porcentajes.append(porcentaje)

gdf_distritos['poblacion_total'] = totales
gdf_distritos['poblacion_cubierta'] = cubiertas
gdf_distritos['porcentaje_cubierto'] = porcentajes

# Se calculan ahora los metros cuadrados de espacio verde para cada distrito y se 
# añade el valor al fichero de distritos, junto al valor de metros cuadrados de 
# espacio verde en cada distrito
gdf_distritos["area_zona_verde_m2"] = 0.0

# Calcular m² de espacio verde por distrito
for idx, distrito in gdf_distritos.iterrows():
    zonas_en_distrito = gdf_zonas_verdes[gdf_zonas_verdes.intersects(distrito.geometry)]
    area_total = zonas_en_distrito.intersection(distrito.geometry).area.sum()
    gdf_distritos.at[idx, "area_zona_verde_m2"] = area_total

# Calcular m² por habitante por distrito
gdf_distritos["m2_por_habitante"] = (gdf_distritos["area_zona_verde_m2"] / 
                                     gdf_distritos["poblacion_total"]).round(2)

# Calcular m² de espacio verde en Valencia ciudad
union_ciudad = gdf_distritos.unary_union
zonas_en_ciudad = gdf_zonas_verdes[gdf_zonas_verdes.intersects(union_ciudad)]
area_verde_ciudad = zonas_en_ciudad.intersection(union_ciudad).area.sum()

# Calcular m² por habitante en Valencia
poblacion_total = gdf_distritos["poblacion_total"].sum()
m2_por_habitante_valencia = round(area_verde_ciudad / poblacion_total, 2)

# Se añade también el porcentaje de zona verde de cada distrito
gdf_distritos["%_zona_verde"] = (gdf_distritos["area_zona_verde_m2"]/
                                 gdf_distritos["area_total_distrito_m2"]*100).round(2)

# Se guarda el fichero final, qu ees el fichero con la información (por distrito)
# de los espacios verdes existentes y que se usará en el modelo
gdf_distritos.to_file("fichero_existentes_final.geojson", driver="GeoJSON")

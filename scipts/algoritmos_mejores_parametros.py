#!/usr/bin/env python
# coding: utf-8

import numpy as np
import geopandas as gpd
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.tchebicheff import Tchebicheff


# Se cargan los ficheros de espacios verdes existentes y candidatos, además de el
# de distritos y se precalculan algunas variables
gdf_zonas_verdes = gpd.read_file("espacios_existentes_final_geo.geojson")
gdf_candidatos = gpd.read_file("candidatos_final.geojson")
gdf_distritos = gpd.read_file("fichero_existentes_final.geojson")
poblacion_total_total = gdf_distritos["poblacion_total"].sum()

def coeficiente_variacion_ponderado(x, pesos):
    """
    Función para calcular el coeficiente de variación ponderado como la relación
    entre la desviación estándar ponderada y la media ponderada. Esto lo hace a 
    partir de un vector sobre el que calcular el coeficiente (x) y otro de pesos.
    """
    x = np.array(x)
    pesos = np.array(pesos)
    media = np.average(x, weights=pesos)
    if media == 0:
        return 0.0
    varianza = np.average((x - media) ** 2, weights=pesos)
    std = np.sqrt(varianza)
    return std / media



def evaluar_solucion(vector_binario, gdf_candidatos, gdf_distritos,
                     valores_maximos, poblacion_total_total):
    """
    Función que evalúa una solución binaria calculando los siguiente objetivos:
    metros cuadrados de espacio verde por habitante, coste total y coeficiente de 
    variación ponderado por población.
    """
    seleccionados = gdf_candidatos[np.array(vector_binario) == 1]
    if not seleccionados.empty:
        area_nueva_dict = seleccionados.groupby("distrito")["area"].sum().to_dict()
        coste_total = seleccionados["coste"].sum()
    else:
        area_nueva_dict = {}
        coste_total = 0.0
    area_nueva_series = gdf_distritos["distrito"].map(area_nueva_dict).fillna(0).values
    area_total_zv = gdf_distritos["area_zona_verde_m2"].values + area_nueva_series
    poblacion_total = gdf_distritos["poblacion_total"].values
    m2_por_hab = area_total_zv / poblacion_total
    m2_hab_total = np.sum(m2_por_hab * poblacion_total) / poblacion_total_total
    cv_ponderado = coeficiente_variacion_ponderado(m2_por_hab, poblacion_total)
    m2_hab_esc = (m2_hab_total - 
                  valores_maximos["min_m2_hab"]) / (valores_maximos["max_m2_hab"] - 
                                                    valores_maximos["min_m2_hab"])
    coste_esc = (coste_total - 
                 valores_maximos["min_coste"]) / (valores_maximos["max_coste"] - 
                                                  valores_maximos["min_coste"])
    cv_esc = (cv_ponderado - 
              valores_maximos["min_cv"]) / (valores_maximos["max_cv"] - 
                                            valores_maximos["min_cv"])
    m2_hab_esc = np.clip(m2_hab_esc, 0, 1)
    coste_esc = np.clip(coste_esc, 0, 1)
    cv_esc = np.clip(cv_esc, 0, 1)
    return [1 - m2_hab_esc, coste_esc, cv_esc]

class MyProblem(Problem):
    """
    Definición del problema de optimización multiobjetivo para la planificación de 
    espacios verdes en Valencia con tres objetivos: maximizar los m² de zona verde por 
    habitante, minimizar el coste y minimizar la desigualdad
    """
    def __init__(self, gdf_distritos, gdf_candidatos, valores_maximos):
        super().__init__(n_var=len(gdf_candidatos), n_obj=3, n_constr=0, xl=0,
                         xu=1, type_var=np.bool_)
        self.gdf_distritos = gdf_distritos
        self.gdf_candidatos = gdf_candidatos
        self.valores_maximos = valores_maximos
        self.poblacion_total_total = gdf_distritos["poblacion_total"].sum()
    def _evaluate(self, X, out, *args, **kwargs):
        resultados = [ evaluar_solucion(x, self.gdf_candidatos, self.gdf_distritos,
                     self.valores_maximos,
                     self.poblacion_total_total) for x in X ]
        out["F"] = np.array(resultados)

# NSGA-II
# Mejores parámetros de NSGA-II según la experimentación en Optuna
mejores_params_nsga2 = {"pop_size": 100, "n_gen": 298, "crossover": "uniform",
    "prob_mut": 0.19437135808000208}

def ejecutar_nsga2_sobre_instancia(path_geojson, gdf_distritos, best_params):
    """
    Función que ejecuta el algoritmo NSGA-III sobre una instancia concreta del problema.
    Carga los candidatos desde el fichero indicado, construye el problema con los 
    valors máximos para el escalado y aplica el algoritmo con los mejores hiperparámetros
    obtenidos en Optuna. Devuelve el resultado del algoritmo.
    """
    gdf_candidatos = gpd.read_file(path_geojson)

    valores_maximos = {
        "min_m2_hab": 4.2261,
        "max_m2_hab": 6.0886,
        "min_coste": 0.0,
        "max_coste": gdf_candidatos["coste"].sum(),
        "min_cv": 0.1,
        "max_cv": 0.7}

    problem = MyProblem(
        gdf_distritos=gdf_distritos,
        gdf_candidatos=gdf_candidatos,
        valores_maximos=valores_maximos)

    crossover = UniformCrossover()
    mutation = BitflipMutation(prob=best_params["prob_mut"])

    algorithm = NSGA2(
        pop_size=best_params["pop_size"],
        sampling=BinaryRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True)

    termination = get_termination("n_gen", best_params["n_gen"])

    res = minimize(problem, algorithm, termination, verbose=True)
    
    return res

# Guardar los resultados en ficheros para acceder a ellos posteriormente
hv_nsga2 = []
ref_point = np.array([1.0, 1.0, 1.0])

# Aplicar algoritmo a las 10 instancias finales
for i in range(11, 21):
    path = f"instancias_costes/instancia_costes_{i:02}.geojson"
    res, gdf_candidatos = ejecutar_nsga2_sobre_instancia(path, gdf_distritos, 
                                                         mejores_params_nsga2)
    # Si hay solución
    if res.F.shape[0] > 0:
        hv = HV(ref_point=ref_point)(res.F)
        hv_nsga2.append(hv)
        # Guardar resultados
        np.savez(f"resultados_finales_finales/nsga2/ejecucion_5/instancia_{i:02}.npz", 
                 F=res.F, X=res.X)
        print(f"Instancia {i}: hipervolumen = {hv:.4f}")
    else:
        hv_nsga2.append(0.0)
        print(f"Instancia {i}: sin soluciones válidas")

# NSGA-III
# Mejores parámetros de NSGA-III según la experimentación en Optuna
mejores_params_nsga3 = {"pop_size": 100, "n_gen": 300, "crossover": "uniform",
    "prob_mut": 0.34675995516802116}

def ejecutar_nsga3_sobre_instancia(path_geojson, gdf_distritos, best_params):
    """
    Función que ejecuta el algoritmo NSGA-III sobre una instancia concreta del problema.
    Carga los candidatos desde el fichero indicado, construye el problema con los 
    valors máximos para el escalado y aplica el algoritmo con los mejores hiperparámetros
    obtenidos en Optuna. Devuelve el resultado del algoritmo.
    """
    gdf_candidatos = gpd.read_file(path_geojson)

    valores_maximos = {
        "min_m2_hab": 4.2261,
        "max_m2_hab": 6.0886,
        "min_coste": 0.0,
        "max_coste": gdf_candidatos["coste"].sum(),
        "min_cv": 0.1,
        "max_cv": 0.7}

    problem = MyProblem(
        gdf_distritos=gdf_distritos,
        gdf_candidatos=gdf_candidatos,
        valores_maximos=valores_maximos)

    ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)

    crossover = UniformCrossover()
    mutation = BitflipMutation(prob=best_params["prob_mut"])

    algorithm = NSGA3(
        pop_size=best_params["pop_size"],
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True)

    termination = get_termination("n_gen", best_params["n_gen"])

    res = minimize(problem, algorithm, termination, verbose=True)
    
    return res, gdf_candidatos

# Guardar los resultados en ficheros para acceder a ellos posteriormente
hv_nsga3 = []
ref_point = np.array([1.0, 1.0, 1.0])

# Aplicar algoritmo a las 10 instancias finales
for i in range(11, 21):
    path = f"instancias_costes/instancia_costes_{i:02}.geojson"
    res, gdf_candidatos = ejecutar_nsga3_sobre_instancia(path, gdf_distritos,
                                                         mejores_params_nsga3)
    # Si hay solución
    if res.F.shape[0] > 0:
        hv = HV(ref_point=ref_point)(res.F)
        hv_nsga3.append(hv)
        # Guardar resultados
        np.savez(f"resultados_finales_finales/nsga3/ejecucion_5/instancia_{i:02}.npz",
                 F=res.F, X=res.X)
        print(f"Instancia {i}: hipervolumen = {hv:.4f}")
    else:
        hv_nsga3.append(0.0)
        print(f"Instancia {i}: sin soluciones válidas")

# MOEA/D
# Mejores parámetros de MOEA/D según la experimentación en Optuna
mejores_params_moead = {"n_partitions": 12, "n_gen": 293, "crossover": "uniform",
    "prob_mut": 0.5630597058318464}

def ejecutar_moead_sobre_instancia(path_geojson, gdf_distritos, best_params):
    """
    Función que ejecuta el algoritmo MOEA/D sobre una instancia concreta del problema.
    Carga los candidatos desde el fichero indicado, construye el problema con los 
    valors máximos para el escalado y aplica el algoritmo con los mejores hiperparámetros
    obtenidos en Optuna. Devuelve el resultado del algoritmo.
    """
    gdf_candidatos = gpd.read_file(path_geojson)

    valores_maximos = {
        "min_m2_hab": 4.2261,
        "max_m2_hab": 6.0886,
        "min_coste": 0.0,
        "max_coste": gdf_candidatos["coste"].sum(),
        "min_cv": 0.1,
        "max_cv": 0.7}

    problem = MyProblem(
        gdf_distritos=gdf_distritos,
        gdf_candidatos=gdf_candidatos,
        valores_maximos=valores_maximos)

    ref_dirs = get_reference_directions("das-dennis", n_dim=3,
                                        n_partitions=best_params["n_partitions"])

    crossover = UniformCrossover()
    mutation = BitflipMutation(prob=best_params["prob_mut"])

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        decomposition=Tchebicheff())

    termination = get_termination("n_gen", best_params["n_gen"])

    res = minimize(problem, algorithm, termination, verbose=True)

    return res, gdf_candidatos

# Guardar los resultados en ficheros para acceder a ellos posteriormente
hv_moead = []
ref_point = np.array([1.0, 1.0, 1.0])

# Aplicar algoritmo a las 10 instancias finales
for i in range(11, 21):
    path = f"instancias_costes/instancia_costes_{i:02}.geojson"
    res, gdf_candidatos = ejecutar_moead_sobre_instancia(path, gdf_distritos,
                                                         mejores_params_moead)
    # Si hay resultado
    if res.F.shape[0] > 0:
        hv = HV(ref_point=ref_point)(res.F)
        hv_moead.append(hv)
        # Guardar resultados
        np.savez(f"resultados_finales_finales/moead/ejecucion_5/instancia_{i:02}.npz", 
                 F=res.F, X=res.X)

        print(f"Instancia {i}: hipervolumen = {hv:.4f}")
    else:
        hv_moead.append(0.0)
        print(f"Instancia {i}: sin soluciones válidas")

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import geopandas as gpd
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
import optuna
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.tchebicheff import Tchebicheff

# Se cargan los ficheros de espacios verdes existentes y candidatos, además de el
# de distritos y se precalculan algunas variables
gdf_zonas_verdes = gpd.read_file("espacios_verdes_final_geo.geojson")
gdf_candidatos = gpd.read_file("localizaciones_final.geojson")
gdf_distritos = gpd.read_file("distritos_final.geojson")
poblacion_total_total = gdf_distritos["poblacion_total"].sum()
m2_por_hab_distrito = gdf_distritos.set_index("distrito")["m2_por_habitante"].to_dict()
distritos_por_candidato = gdf_candidatos["distrito"].values

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

class EquidadDirectedMutation(Mutation):
    """
    Operador de mutación personalizado dirigido que favorece la equidad espacial.
    Activa candidatos en distritos de baja cobertura (<=3m²/hab), desactiva los
    de alta cobertura (>=7m²/hab) y aplica mutación estándar en los de cobertura
    media.
    """
    def __init__(self, prob, m2_por_hab_distrito, distritos_por_candidato):
        super().__init__()
        self.prob = prob
        self.distritos = distritos_por_candidato
        self.m2_array = np.array([m2_por_hab_distrito.get(d, 0) 
                                  for d in distritos_por_candidato])
    def _do(self, problem, X, **kwargs):
        n_ind, n_genes = X.shape
        X_mut = X.copy()
        mutation_mask = np.random.rand(n_ind, n_genes) < self.prob
        low_mask = self.m2_array <= 3
        high_mask = self.m2_array >= 7
        mid_mask = ~(low_mask | high_mask)  # entre 3 y 7
        for i in range(n_ind):
            mask = mutation_mask[i]
            X_mut[i, mask & low_mask] = 1
            X_mut[i, mask & high_mask] = 0
            mid_idx = mask & mid_mask
            X_mut[i, mid_idx] = 1 - X_mut[i, mid_idx]
        return X_mut

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
def objective(trial):
    """
    Función objetivo para Optuna que entrena NSGA-II con un conjunto de hiperparámetros
    propuestos y devuelve el hipervolumen medio obtenido en cinco instancias de prueba.
    """
    # Hiperparámetros elegidos
    pop_size = trial.suggest_categorical("pop_size", [50, 75, 100])  
    n_gen = trial.suggest_int("n_gen", 50, 200)

    # Cruce
    crossover_type = trial.suggest_categorical("crossover", ["uniform", "two_point"])
    if crossover_type == "uniform":
        crossover = UniformCrossover()
    else:
       crossover = TwoPointCrossover()

    # Mutación
    prob_mut = trial.suggest_float("prob_mut", 0.01, 0.6)
    mutation_type = trial.suggest_categorical("mutation", ["bitflip", "equidad"])

    # Punto de referencia (para el hipervolumen)
    ref_point = np.array([1.0, 1.0, 1.0])  # porque todos los objetivos están escalados y es el caso peor

    hv_scores = [] 
    
    # Evaluación sobre las instancias elegidas (las 5 primeras)
    for i in range(1, 6):
        path = f"instancias_costes/instancia_costes_{i:02}.geojson"
        gdf_candidatos = gpd.read_file(path)

        # Valores máximos y mínimos para el escalado
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

        if mutation_type == "equidad":
            distritos_por_candidato = gdf_candidatos["distrito"].tolist()
            mutation = EquidadDirectedMutation(
                prob=prob_mut,
                m2_por_hab_distrito=m2_por_hab_distrito,     
                distritos_por_candidato=distritos_por_candidato)
        else:
            mutation = BitflipMutation(prob=prob_mut)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True)

        termination = get_termination("n_gen", n_gen)
        
        # Ejecución del algoritmo
        res = minimize(problem, algorithm, termination, verbose=False)

        # Si hay solución, almacena el hipervolumen
        if res.F.shape[0] > 0:
            hv = HV(ref_point=ref_point)(res.F)
            hv_scores.append(hv)
        else:
            print(f"Instancia {i} no generó soluciones válidas.")
    
    # Si falla alguna instancia, su valor pasa a 0
    if len(hv_scores) < 5:
        return 0.0
    
    return np.mean(hv_scores)

# Creación de un estudio en Optuna donde hacer la experimentación en NSGA-II
study = optuna.create_study(
    direction="maximize",
    study_name="nsga2_final_final",
    storage="sqlite:///resultado_final.db",
    load_if_exists=True)

# Ejecución de Optuna en NSGA-II
study.optimize(objective, n_trials=100, n_jobs=4) # 100

# NSGA-III
def objective(trial):
    # Hiperparámetros elegidos
    n_partitions = trial.suggest_categorical("n_partitions", [8, 10, 12]) 
    n_gen = trial.suggest_int("n_gen", 50, 200)
    # Cruce
    crossover_type = trial.suggest_categorical("crossover", ["uniform", "two_point"])
    if crossover_type == "uniform":
        crossover = UniformCrossover()
    else:
       crossover = TwoPointCrossover()

    # Mutación
    prob_mut = trial.suggest_float("prob_mut", 0.01, 0.6)
    mutation_type = trial.suggest_categorical("mutation", ["bitflip", "equidad"])
    
    # Vectores de referencia
    ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=n_partitions)
    pop_size = len(ref_dirs)

    # Punto de referencia (para el hipervolumen)
    ref_point = np.array([1.0, 1.0, 1.0])  # porque todos los objetivos están escalados y es el caso peor

    hv_scores = []
    
    # Evaluación sobre las 10 primeras instancias
    for i in range(1, 6):
        path = f"instancias_costes/instancia_costes_{i:02}.geojson"
        gdf_candidatos = gpd.read_file(path)
        
         # Valores máximos y mínimos para el escalado
        valores_maximos = {
            "min_m2_hab": 4.2261,
            "max_m2_hab": 6.0886,
            "min_coste": 0.0,
            "max_coste": gdf_candidatos["coste"].sum(),
            "min_cv": 0.1,
            "max_cv": 0.7}
        
        if mutation_type == "equidad":
            distritos_por_candidato = gdf_candidatos["distrito"].tolist()
            mutation = EquidadDirectedMutation(
                prob=prob_mut,
                m2_por_hab_distrito=m2_por_hab_distrito,     
                distritos_por_candidato=distritos_por_candidato)
        else:
            mutation = BitflipMutation(prob=prob_mut)
        
        problem = MyProblem(
            gdf_distritos=gdf_distritos,
            gdf_candidatos=gdf_candidatos,
            valores_maximos=valores_maximos)

        ref_dirs = get_reference_directions("das-dennis", n_dim=3,
                                            n_partitions=n_partitions) 

        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True)

        termination = get_termination("n_gen", n_gen)
        
        #  Ejecución del algoritmo
        res = minimize(problem, algorithm, termination, verbose=False)

        # Si hay solución, almacena el hipervolumem
        if res.F.shape[0] > 0:
            hv = HV(ref_point=ref_point)(res.F)
            hv_scores.append(hv)

    # Si falla alguna instancia, su valor pasa a 0
    if len(hv_scores) < 5:
        return 0.0

    return np.mean(hv_scores)

# Creación de un estudio en Optuna donde hacer la experimentación en NSGA-III
study = optuna.create_study(
    direction="maximize",
    study_name="nsga3_final_final",
    storage="sqlite:///resultado_final.db",
    load_if_exists=True)

# Ejecución de Optuna en NSGA-III
study.optimize(objective, n_trials=100, n_jobs=4)

# MOEA/D
def objective(trial):
    # Direcciones de referencia para NSGA-III
    n_partitions = trial.suggest_categorical("n_partitions", [8, 10, 12])
    ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=n_partitions)
    
    # Hiperparámetros elegidos
    n_gen = trial.suggest_int("n_gen", 150, 300)

    # Cruce
    crossover_type = trial.suggest_categorical("crossover", ["uniform", "two_point"])
    if crossover_type == "uniform":
        crossover = UniformCrossover()
    else:
       crossover = TwoPointCrossover()
        
    # Mutación
    prob_mut = trial.suggest_float("prob_mut", 0.01, 0.6)
    mutation_type = trial.suggest_categorical("mutation", ["bitflip", "equidad"])
    
    # Punto de referencia (para el hipervolumen)
    ref_point = np.array([1.0, 1.0, 1.0])  # porque todos los objetivos están escalados y es el caso peor
    hv_scores = []

    # Evaluación sobre las 10 primeras instancias
    for i in range(1, 6):  
        path = f"instancias_costes/instancia_costes_{i:02}.geojson"
        gdf_candidatos = gpd.read_file(path)

        # Valores máximos y mínimos para el escalado
        valores_maximos = {
            "min_m2_hab": 4.2261,
            "max_m2_hab": 6.0886,
            "min_coste": 0.0,
            "max_coste": gdf_candidatos["coste"].sum(),
            "min_cv": 0.1,
            "max_cv": 0.7}
        
        if mutation_type == "equidad":
            distritos_por_candidato = gdf_candidatos["distrito"].tolist()
            mutation = EquidadDirectedMutation(
                prob=prob_mut,
                m2_por_hab_distrito=m2_por_hab_distrito,     
                distritos_por_candidato=distritos_por_candidato)
        else:
            mutation = BitflipMutation(prob=prob_mut)

        problem = MyProblem(
            gdf_distritos=gdf_distritos,
            gdf_candidatos=gdf_candidatos,
            valores_maximos=valores_maximos)

        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            sampling=BinaryRandomSampling(),
            crossover=crossover,
            mutation=mutation,
            decomposition=Tchebicheff())

        termination = get_termination("n_gen", n_gen)

        #  Ejecución del algoritmo
        res = minimize(problem, algorithm, termination, verbose=False)

        # Si hay solución, almacena el hipervolumem
        if res.F.shape[0] > 0:
            hv = HV(ref_point=ref_point)(res.F)
            hv_scores.append(hv)

    # Si falla alguna instancia, su valor pasa a 0
    if len(hv_scores) < 5:
        return 0.0

    return np.mean(hv_scores)

# Creación de un estudio en Optuna donde hacer la experimentación en MOEA/D
study = optuna.create_study(
    direction="maximize",
    study_name="moead_final_final",
    storage="sqlite:///resultado_final.db",
    load_if_exists=True)

# Ejecución de Optuna en MOEA/D
study.optimize(objective, n_trials=100, n_jobs=4)


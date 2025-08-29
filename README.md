# Planificación del acceso a espacios verdes en la ciudad de Valencia mediante metaheurística multiobjetivo

Repositorio con el código utilizado en el Trabajo Fin de Grado (TFG) para la planificación de zonas verdes en la ciudad de Valencia aplicando algoritmos evolutivos multiobjetivo.

## Contenido

- **data/** → ficheros de entrada (`.geojson`) con espacios verdes existentes, distritos y posibles localizaciones.  
- **src/** → scripts principales en Python:  
  - `preprocesamiento.py` → preparación y limpieza de datos.  
  - `instancias.py` → generación de instancias con costes distintos.  
  - `nsga2.py`, `nsga3.py`, `moead.py` → ejecución de algoritmos evolutivos.  
  - `optuna_experimentos.py` → optimización de hiperparámetros con Optuna.  
  - `resultados.py` → visualización y análisis de resultados.  

## Descripción breve

El trabajo implementa y compara tres algoritmos evolutivos multiobjetivo (NSGA-II, NSGA-III y MOEA/D), optimizando:  
- Maximizar los m² de zonas verdes por habitante,  
- Minimizar el coste de construcción,  
- Minimizar la desigualdad espacial (coeficiente de variación).  

Los resultados permiten evaluar qué localizaciones nuevas mejoran la accesibilidad a zonas verdes en la ciudad.

---

✍️ **Nota**: Este repositorio acompaña a la memoria del TFG. No está pensado como librería de uso general, sino como apoyo documental.


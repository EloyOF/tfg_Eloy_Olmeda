# Planificación del acceso a espacios verdes en la ciudad de Valencia mediante metaheurística multiobjetivo

Repositorio con el código utilizado en el Trabajo Fin de Grado (TFG) para la planificación de zonas verdes en la ciudad de Valencia aplicando algoritmos evolutivos multiobjetivo.

## Contenido

- **data/** → ficheros de datos utilizados para la realización del trabajo. Están tanto los ficheros originales como los filtrados.  
- **scripts/** → ficheros de código principales del trabajo en Python:  
  - `zonas_verdes_existentes.py` → generación del fichero de espacios verdes existentes junto con su tratado.  
  - `candidatos.py` → generación del fichero de espacios verdes candidatos junto con la obtención y tratado de cada fuente utilizada.  
  - `optuna_experimentos.py` → optimización de los hiperparámetros con Optuna.
  - `algoritmos_mejores_parametros` → ejecución de los algoritmos evolutivos con sus mejores hiperparámetros junto con su almacenaje de resultados.  

Este trabajo ha implementado y comparado tres algoritmos evolutivos multiobjetivo (NSGA-II, NSGA-III y MOEA/D), optimizando:  
- Maximizar los m² de zonas verdes por habitante.
- Minimizar el coste de construcción.  
- Minimizar la desigualdad espacial.  

Los resultados permiten evaluar qué localizaciones nuevas mejoran la accesibilidad a zonas verdes en la ciudad.

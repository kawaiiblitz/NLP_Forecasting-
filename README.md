# **Data Science Challenge**
Este repositorio contiene la solución al desafío de Data Science, dividido en dos partes principales: **clasificación de sentencias (NLP)** y **forecasting (predicción de ventas)**. Ambas soluciones fueron desarrolladas utilizando enfoques avanzados de Machine Learning y AI.

## **Estructura de las carpetas**
#### **Parte 1: Clasificación de Sentencias (NLP)**
- **`nlp/data`**: Contiene el dataset con las sentencias y sus etiquetas.
  - `sentences.txt`: Dataset de sentencias y sus etiquetas correspondientes.

- **`nlp/results`**: Almacena los resultados y métricas de los modelos de NLP.
  - `results.txt`: Salida con predicciones de los modelos y métricas obtenidas.

- **`nlp/scripts`**: Scripts en Python para la clasificación de sentencias.
  - `open_ai.py`: Clasificación utilizando OpenAI GPT.
  - `similitud.py`: Clasificación utilizando embeddings y similitud.
  - `zero_shot.py`: Clasificación utilizando técnicas de Zero-shot.

#### **Parte 2: Forecasting de ventas**
- **`forecasting/data`**: Contiene el dataset de ventas históricas y factores exógenos.
  - `forecasting.txt`: Dataset de ventas, promociones y días festivos.

- **`forecasting/notebooks`**: Notebooks de Jupyter para visualizaciones y análisis de series de tiempo.
  - `graphs.ipynb`: 

- **`forecasting/scripts`**: Scripts en Python para los modelos de forecasting.
  - `LSTM.py`: Predicción utilizando modelos LSTM.
  - `prophet.py`: Predicción utilizando Prophet.
  - `SARIMAX_ARIMAX.py`: Predicción utilizando SARIMAX y ARIMA.

- **`requirements.txt`**: Especifica las dependencias necesarias para ejecución del proyecto.


## **Ejemplo de Uso**
- **Zero-shot classification**  
   Ejecuta el siguiente comando desde la carpeta `nlp/scripts`:
   ```bash
   python zero_shot.py

## **Requerimientos**
Este proyecto requiere **Python 3.8.20**. Las dependencias necesarias están detalladas en el archivo `requirements.txt`. Para instalarlas, utiliza el siguiente comando:
```bash
pip install -r requirements.txt


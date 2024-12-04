import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Cargar datos
data = {
    'Week': list(range(1, 31)),
    'Sales': [152, 485, 398, 320, 156, 121, 238, 70, 152, 171, 264, 380, 137, 422, 149, 409, 201, 180, 199, 358,
              307, 393, 463, 343, 435, 241, 493, 326, None, None],
    'Promotion': [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'Holiday': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

###################### Ajustar modelo SARIMAX ######################


# Separar datos
train = df.iloc[:28]  # Semanas 1–28 para entrenamiento
validation = df.iloc[24:28]  # Semanas 25–28 para validación
test = df.iloc[28:]  # Semanas 29–30 para predicción

# Variables exógenas
exog_train = train[['Promotion', 'Holiday']]
exog_validation = exog_train.iloc[24:28]
exog_test = test[['Promotion', 'Holiday']]

# Modelo
model = SARIMAX(train['Sales'], exog=exog_train, order=(2, 1, 2))  # Análisis de estacionalidad
results = model.fit()

# Resumen del modelo
print(results.summary())

# Predicciones para semanas 25–28 (validación)
forecast_validation = results.predict(start=24, end=27, exog=exog_validation)

# Evaluar el rendimiento en validación
validation_real = validation['Sales']
mae_val = mean_absolute_error(validation_real, forecast_validation)
rmse_val = np.sqrt(mean_squared_error(validation_real, forecast_validation))

print("\nMétricas de Validación (Semanas 25–28):")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")

# Predicciones para semanas 29–30
forecast_test = results.predict(start=28, end=29, exog=exog_test)

# Mostrar predicciones para semanas 29 y 30
print("\nPredicciones para Semanas 29 y 30:")
print(forecast_test)

# Crear una tabla de resultados
df['yhat'] = list(results.predict(start=0, end=27, exog=exog_train)) + list(forecast_test)
df['Error'] = df['Sales'] - df['yhat']

# Mostrar tabla de predicciones vs reales
print("\nTabla completa de predicciones vs reales:")
print(df[['Week', 'Sales', 'yhat', 'Error']])

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(df['Week'], df['Sales'], label='Ventas Reales', marker='o', color='#F7C600', linewidth=2)
plt.plot(df['Week'], df['yhat'], label='Predicciones', linestyle='--', color='#1F3040', linewidth=2)
plt.axvline(x=24, color='gray', linestyle='--', label='Inicio de Validación')
plt.axvline(x=28, color='gray', linestyle='--', label='Inicio de Predicciones')
plt.title('Modelo SARIMAX: Predicción de Ventas', fontsize=14, weight='bold')
plt.xlabel('Semana', fontsize=12)
plt.ylabel('Ventas', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()



###################### Ajustar modelo ARIMA ######################

# Cargar datos
data = {
    'Week': list(range(1, 31)),
    'Sales': [152, 485, 398, 320, 156, 121, 238, 70, 152, 171, 264, 380, 137, 422, 149, 409, 201, 180, 199, 358,
              307, 393, 463, 343, 435, 241, 493, 326, None, None],
    'Promotion': [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'Holiday': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)


# Separar datos
train = df.iloc[:28]  # Semanas 1–28 para entrenamiento
validation = df.iloc[24:28]  # Semanas 25–28 para validación
test = df.iloc[28:]  # Semanas 29–30 para predicción

model_arima = ARIMA(train['Sales'], order=(2, 1, 2))  # Orden basado en el análisis previo
results_arima = model_arima.fit()

# Resumen del modelo
print(results_arima.summary())

# Predicciones para semanas 25–28 (validación)
forecast_validation_arima = results_arima.predict(start=24, end=27)

# Evaluar el rendimiento en validación
validation_real = validation['Sales']
mae_val_arima = mean_absolute_error(validation_real, forecast_validation_arima)
rmse_val_arima = np.sqrt(mean_squared_error(validation_real, forecast_validation_arima))

print("\nMétricas de Validación (Semanas 25–28):")
print(f"MAE: {mae_val_arima:.2f}")
print(f"RMSE: {rmse_val_arima:.2f}")

# Predicciones para semanas 29–30
forecast_test_arima = results_arima.predict(start=28, end=29)

# Mostrar predicciones para semanas 29 y 30
print("\nPredicciones para Semanas 29 y 30:")
print(forecast_test_arima)

# Crear una tabla de resultados
df['yhat'] = list(results_arima.predict(start=0, end=27)) + list(forecast_test_arima)
df['Error'] = df['Sales'] - df['yhat']

# Mostrar tabla de predicciones vs reales
print("\nTabla completa de predicciones vs reales:")
print(df[['Week', 'Sales', 'yhat', 'Error']])

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(df['Week'], df['Sales'], label='Ventas Reales', marker='o', color='#F7C600', linewidth=2)
plt.plot(df['Week'], df['yhat'], label='Predicciones', linestyle='--', color='#1F3040', linewidth=2)
plt.title('Modelo ARIMA: Predicción de Ventas', fontsize=14, weight='bold')
plt.xlabel('Semana', fontsize=12)
plt.ylabel('Ventas', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
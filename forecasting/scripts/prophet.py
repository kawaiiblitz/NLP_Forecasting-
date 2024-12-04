import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

###################### Prophet (Facebook)  ######################

# Cargar datos
file_path = r"C:\Users\rache\OneDrive\Documentos\MeLi\ML_challenge\DS_challenge\forecasting.txt"
df = pd.read_csv(file_path, delimiter='|')

# Renombrar columnas para Prophet
df = df.rename(columns={
    'Week': 'ds',
    'Sales': 'y'
})

# Crear fechas ficticias para 'ds' basadas en semanas
df['ds'] = pd.date_range(start='2000-01-01', periods=len(df), freq='W')

# Separar semanas conocidas y futuras
known_weeks = df.iloc[:28].copy()  # Semanas 1–28 (entrenamiento y validación)
future_weeks = df.iloc[28:].copy()  # Semanas 29 y 30 (predicción)

# Crear y configurar el modelo Prophet
model = Prophet()
model.add_regressor('Promotion')
model.add_regressor('Holiday')

# Entrenar el modelo con semanas 1–28
model.fit(known_weeks)

# Crear un DataFrame extendido para predecir semanas 29 y 30
future = model.make_future_dataframe(periods=2, freq='W')

# Agregar variables exógenas
future['Promotion'] = df['Promotion']
future['Holiday'] = df['Holiday']

# Realizar predicciones
forecast = model.predict(future)

# Calcular errores para semanas 24–28
validation_24_28 = forecast.iloc[23:28]  # Filtrar predicciones para semanas 24–28
real_24_28 = known_weeks.iloc[23:28]['y']  # Valores reales para semanas 24–28
mae_24_28 = mean_absolute_error(real_24_28, validation_24_28['yhat'])
rmse_24_28 = np.sqrt(mean_squared_error(real_24_28, validation_24_28['yhat']))

print(f"Métricas de validación (semanas 24–28):")
print(f"MAE: {mae_24_28:.2f}")
print(f"RMSE: {rmse_24_28:.2f}")

# Resultados para semanas 29 y 30
predictions_29_30 = forecast.iloc[28:30][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print("\nPredicciones para semanas 29 y 30:")
print(predictions_29_30)

# Crear tabla de predicción vs real para semanas 1–30
df['yhat'] = forecast['yhat']  # Agregar las predicciones al DataFrame original
df['Error'] = df['y'] - df['yhat']  # Calcular el error

# Mostrar tabla predicciones vs reales
print("\nTabla completa de predicciones vs reales:")
print(df[['ds', 'y', 'yhat', 'Error']])

# Graficar resultados con bandas de incertidumbre
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Ventas Reales', marker='o', color='#F7C600', linewidth=2)
plt.plot(df['ds'], df['yhat'], label='Predicciones', linestyle='--', color='#1F3040', linewidth=2)

# Banda de incertidumbre (yhat_lower y yhat_upper)
plt.fill_between(df['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3)

# Etiquetas y título
plt.title("Predicción de ventas con Prophet", fontsize=14, weight='bold')
plt.xlabel("Semana", fontsize=12)
plt.ylabel("Ventas", fontsize=12)
plt.grid(True)
plt.legend(["Ventas reales", "Predicciones"])
plt.show()

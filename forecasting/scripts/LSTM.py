import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Datos originales
data = {
    'Week': list(range(1, 31)),
    'Sales': [152, 485, 398, 320, 156, 121, 238, 70, 152, 171, 264, 380, 137, 422, 149, 409, 201, 180, 199, 358,
              307, 393, 463, 343, 435, 241, 493, 326, None, None],
    'Promotion': [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'Holiday': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
}

df_original = pd.DataFrame(data)
df = df_original.copy()

# Reemplazar valores nulos en ventas
df['Sales'].fillna(0, inplace=True)

# Escalar los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Sales', 'Promotion', 'Holiday']])

# Crear secuencias
def create_sequences(data, look_back=6):
    """
    Crea secuencias para el entrenamiento del modelo.

    Args:
        data (np.array): Datos escalados.
        look_back (int): Número de pasos anteriores a usar como entrada.

    Returns:
        tuple: (X, y) con las secuencias de entrada y salida.
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, :])  # Todas las columnas (Ventas + Exógenas)
        y.append(data[i + look_back, 0])  # Solo columna de Ventas
    return np.array(X), np.array(y)

look_back = 6
X_train, y_train = create_sequences(scaled_data[:23], look_back=look_back)

# Convertir a tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
4
# Definir el modelo LSTM
class LSTMModel(nn.Module):
    """
    Clase para construir un modelo LSTM (Long Short-Term Memory) para predicción de series temporales.

    Args:
        input_size (int): Número de características de entrada (dimensión de cada vector de entrada).
        hidden_size (int): Número de unidades ocultas en cada capa LSTM.
        num_layers (int): Número de capas LSTM apiladas.
        output_size (int): Dimensión de la salida del modelo (en este caso, predicción escalar).
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        # Capa LSTM: Captura las dependencias temporales de los datos
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Capa completamente conectada: Transforma las salidas de LSTM a la dimensión de salida deseada
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define el paso hacia adelante del modelo (cálculo de predicciones).

        Args:
            x (tensor): Tensor de entrada con forma (batch_size, sequence_length, input_size).

        Returns:
            tensor: Predicciones con forma (batch_size, output_size).
        """
        # out: Salida del LSTM para todas las secuencias (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x)
        
        # Seleccionar la salida del último paso temporal y pasarla a la capa totalmente conectada
        out = self.fc(out[:, -1, :])  
        return out


# Inicializar el modelo
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 1
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Configurar el entrenamiento
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenar el modelo
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Predicción iterativa para semanas 24–30
predictions = []
input_seq = scaled_data[17:23]  # Ventana inicial (semanas 18–23 reales)

for i in range(7):  # Predicciones para semanas 24–30
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    pred = model(input_tensor).item()
    predictions.append(pred)

    # Actualizar ventana con predicciones dinámicas
    next_features = np.array([[pred, df.iloc[23 + i]['Promotion'], df.iloc[23 + i]['Holiday']]])
    input_seq = np.vstack([input_seq[1:], next_features])

# Reescalar predicciones
predictions_combined = np.hstack([np.array(predictions).reshape(-1, 1), np.zeros((7, 2))])
predictions_rescaled = scaler.inverse_transform(predictions_combined)[:, 0]

# Calcular métricas de validación (semanas 24–28)
real_values = df['Sales'].iloc[23:28].values
val_predictions = predictions_rescaled[:5]
mae_val = mean_absolute_error(real_values, val_predictions)
rmse_val = np.sqrt(mean_squared_error(real_values, val_predictions))

print(f"\nMétricas de validación (semanas 24–28):")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")

# Predicciones para semanas 29 y 30
print("\nPredicciones para semanas 29 y 30:")
for week, pred in zip(range(29, 31), predictions_rescaled[5:]):
    print(f"Semana {week}: {pred:.2f}")

# Crear una tabla completa de predicciones
df['yhat'] = [None] * 23 + list(val_predictions) + list(predictions_rescaled[5:])
df['Error'] = df['Sales'] - df['yhat']

print("\nTabla completa de predicciones vs reales:")
print(df[['Week', 'Sales', 'yhat', 'Error']])

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(df['Week'][:28], df['Sales'][:28], label='Ventas Reales', marker='o', color='#F7C600', linewidth=2)
plt.plot(df['Week'], df['yhat'], label='Predicciones', linestyle='--', color='#1F3040', linewidth=2)
plt.title("Predicción de ventas con LSTM", fontsize=14, weight='bold')
plt.xlabel("Semana")
plt.ylabel("Ventas")
plt.grid(True)
plt.legend()
plt.show()

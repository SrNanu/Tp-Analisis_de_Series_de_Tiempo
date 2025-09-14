# -*- coding: utf-8 -*-
"""
Demostración de Modelos de Series de Tiempo: Naive vs. Prophet vs. Deep Learning (LSTM)
Dataset: AirPassengers
"""

# =============================================================================
# PASO 1: INSTALACIÓN Y CARGA DE LIBRERÍAS
# =============================================================================
# Asegúrate de tener las librerías instaladas.
# Si no las tienes, ejecuta en tu terminal:
# pip install pandas matplotlib scikit-learn prophet tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Librerías para los modelos
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Configuración de Matplotlib para gráficos más grandes y legibles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 7)

# =============================================================================
# PASO 2: CARGA Y ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

# Cargar el dataset
# El dataset se puede encontrar en línea, por ejemplo, en Kaggle o repositorios de datos.
# Para facilitar la demo, lo cargamos desde una URL conocida.
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, parse_dates=['Month'], index_col='Month')

# Renombrar la columna para mayor claridad
df = df.rename(columns={'Passengers': 'y'})

print("Primeros 5 registros del dataset:")
print(df.head())
print("\nÚltimos 5 registros del dataset:")
print(df.tail())

# Visualizar la serie de tiempo completa
df['y'].plot(title='Número de Pasajeros de Aerolínea Mensuales (1949-1960)')
plt.ylabel('Número de Pasajeros')
plt.xlabel('Fecha')
plt.show()

# =============================================================================
# PASO 3: DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA
# =============================================================================

# Vamos a entrenar con los datos hasta finales de 1958 y probar con 1959 y 1960
train_size = len(df) - 24
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
print(f"Tamaño del conjunto de prueba: {len(test)}")

# =============================================================================
# MODELO 1: NAIVE FORECAST (LÍNEA BASE)
# =============================================================================
print("\n--- Entrenando Modelo Naive ---")

# La predicción es simplemente el último valor conocido en el conjunto de entrenamiento
last_value = train['y'][-1]
naive_predictions = pd.Series(np.repeat(last_value, len(test)), index=test.index)

# Calcular el error
rmse_naive = np.sqrt(mean_squared_error(test['y'], naive_predictions))
print(f"RMSE del Modelo Naive: {rmse_naive:.2f}")

# =============================================================================
# MODELO 2: PROPHET (MODELO AUTOMATIZADO)
# =============================================================================
print("\n--- Entrenando Modelo Prophet ---")

# Prophet requiere un formato específico: columnas 'ds' (fecha) y 'y' (valor)
prophet_train = train.reset_index().rename(columns={'Month': 'ds'})

# Crear y entrenar el modelo
model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_prophet.fit(prophet_train)

# Crear un dataframe futuro para realizar las predicciones (para los próximos 24 meses)
future = model_prophet.make_future_dataframe(periods=len(test), freq='MS')
forecast_prophet = model_prophet.predict(future)

# Extraer las predicciones que corresponden al período de prueba
prophet_predictions = forecast_prophet.set_index('ds')['yhat'][-len(test):]

# Calcular el error
rmse_prophet = np.sqrt(mean_squared_error(test['y'], prophet_predictions))
print(f"RMSE del Modelo Prophet: {rmse_prophet:.2f}")

# Prophet también puede graficar sus componentes (¡muy útil para la presentación!)
fig = model_prophet.plot_components(forecast_prophet)
plt.show()


# =============================================================================
# MODELO 3: DEEP LEARNING (LSTM)
# =============================================================================
print("\n--- Entrenando Modelo LSTM ---")

# 1. Preprocesamiento de datos para LSTM
# Las redes neuronales son sensibles a la escala de los datos, por lo que escalamos a [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

# 2. Función para crear secuencias de datos
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Usaremos los últimos 12 meses para predecir el siguiente
n_steps = 12
X_train, y_train = create_sequences(scaled_train, n_steps)

# 3. Construcción del Modelo LSTM con Keras
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

print(model_lstm.summary())

# 4. Entrenamiento del Modelo
model_lstm.fit(X_train, y_train, epochs=200, verbose=0) # verbose=0 para no llenar la salida

# 5. Proceso de Predicción Iterativa
lstm_predictions_scaled = []
# Tomamos la última secuencia del set de entrenamiento como punto de partida
last_sequence = scaled_train[-n_steps:]
current_batch = last_sequence.reshape((1, n_steps, 1))

for i in range(len(test)):
    # Obtener la predicción
    current_pred = model_lstm.predict(current_batch, verbose=0)[0]
    # Guardar la predicción
    lstm_predictions_scaled.append(current_pred)
    # Actualizar el batch para la siguiente predicción
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# 6. Invertir la escala de las predicciones para compararlas
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

# Calcular el error
rmse_lstm = np.sqrt(mean_squared_error(test['y'], lstm_predictions))
print(f"RMSE del Modelo LSTM: {rmse_lstm:.2f}")

# =============================================================================
# VISUALIZACIÓN INDIVIDUAL DE CADA MODELO
# =============================================================================
# Crear un DataFrame con las predicciones para facilitar el ploteo
test_predictions = test.copy()
test_predictions['Naive'] = naive_predictions.values
test_predictions['Prophet'] = prophet_predictions.values
test_predictions['LSTM'] = lstm_predictions

# --- Gráfico del Modelo Naive ---
plt.figure(figsize=(15, 7))
plt.plot(train['y'], label='Datos de Entrenamiento')
plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
plt.plot(test_predictions['Naive'], label='Predicción Naive', color='orange', linestyle='--')
plt.title('Resultado del Modelo Naive', fontsize=20)
plt.legend()
plt.show()

# --- Gráfico del Modelo Prophet ---
plt.figure(figsize=(15, 7))
plt.plot(train['y'], label='Datos de Entrenamiento')
plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
plt.plot(test_predictions['Prophet'], label='Predicción Prophet', color='green', linestyle='--')
plt.title('Resultado del Modelo Prophet', fontsize=20)
plt.legend()
plt.show()

# --- Gráfico del Modelo LSTM ---
plt.figure(figsize=(15, 7))
plt.plot(train['y'], label='Datos de Entrenamiento')
plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
plt.plot(test_predictions['LSTM'], label='Predicción LSTM', color='red', linestyle='--')
plt.title('Resultado del Modelo LSTM', fontsize=20)
plt.legend()
plt.show()


# =============================================================================
# PASO FINAL: COMPARACIÓN VISUAL Y MÉTRICAS
# =============================================================================
# (Aquí va el código que ya tenías para la gráfica final comparativa)
# ...

# =============================================================================
# PASO FINAL: COMPARACIÓN VISUAL Y MÉTRICAS
# =============================================================================



# Graficar todo junto
plt.figure(figsize=(16, 8))
plt.plot(train['y'], label='Datos de Entrenamiento')
plt.plot(test['y'], label='Datos Reales (Prueba)', color='black', lw=2)
plt.plot(test_predictions['Naive'], label='Predicción Naive', linestyle='--')
plt.plot(test_predictions['Prophet'], label='Predicción Prophet', linestyle='--')
plt.plot(test_predictions['LSTM'], label='Predicción LSTM', linestyle='--')

plt.title('Comparación de Modelos de Pronóstico', fontsize=20)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Número de Pasajeros', fontsize=14)
plt.legend()
plt.show()

# Imprimir tabla de resumen de errores
print("\n--- Resumen de Resultados (RMSE) ---")
print("=======================================")
print(f"| Modelo Naive  | {rmse_naive:15.2f} |")
print(f"| Modelo Prophet| {rmse_prophet:15.2f} |")
print(f"| Modelo LSTM   | {rmse_lstm:15.2f} |")
print("=======================================")
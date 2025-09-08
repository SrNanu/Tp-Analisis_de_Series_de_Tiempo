import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Carga y Preparación de Datos ---
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
df = df.rename(columns={'Passengers': 'y'})

# --- 2. Ingeniería de Características (El paso más importante aquí) ---
# Creamos características a partir de la fecha (índice)
df['month'] = df.index.month
df['year'] = df.index.year
df['quarter'] = df.index.quarter

# Creamos características de "lag" (valores de meses anteriores)
# El lag de 12 meses es crucial para capturar la estacionalidad anual
df['lag_1'] = df['y'].shift(1)
df['lag_12'] = df['y'].shift(12)

# Creamos una característica de ventana móvil para capturar la tendencia local
df['rolling_mean_12'] = df['y'].rolling(window=12).mean()

# Eliminamos las filas con valores NaN que se generaron por shift() y rolling()
df.dropna(inplace=True)

print("DataFrame con las nuevas características (features):")
print(df.head())

# --- 3. División de Datos en Entrenamiento y Prueba ---
# ¡IMPORTANTE! En series de tiempo, NO se puede hacer una división aleatoria.
# El conjunto de prueba DEBE ser el futuro.
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

# Separamos las características (X) del objetivo (y)
features = ['month', 'year', 'quarter', 'lag_1', 'lag_12', 'rolling_mean_12']
X_train = train[features]
y_train = train['y']
X_test = test[features]
y_test = test['y']

# --- 4. Entrenamiento del Modelo de Machine Learning ---
# Usamos un Random Forest, un modelo potente y versátil
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 5. Generación de Predicciones ---
predictions = model.predict(X_test)

# Creamos una serie de pandas con las predicciones para graficarla fácilmente
test['predictions'] = predictions

# --- 6. Visualización de Resultados ---
plt.figure(figsize=(14, 7))
plt.plot(train['y'], label='Datos de Entrenamiento')
plt.plot(test['y'], label='Datos Reales (Test)')
plt.plot(test['predictions'], label='Predicciones (Random Forest)', linestyle='--')
plt.title('Pronóstico con Random Forest (Machine Learning)')
plt.xlabel('Fecha')
plt.ylabel('Número de Pasajeros')
plt.legend()
plt.grid(True)
plt.show()
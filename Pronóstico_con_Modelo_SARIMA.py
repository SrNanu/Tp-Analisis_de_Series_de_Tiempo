import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# --- 1. Carga y Preparación de Datos ---
# Usamos un dataset clásico disponible online
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
df.index.freq = 'MS' # 'MS' indica que la frecuencia es inicio de mes (Month Start)

# Renombramos la columna para mayor claridad
df = df.rename(columns={'Passengers': 'y'})

print("Primeros datos de la serie:")
print(df.head())
print("\nInformación del DataFrame:")
df.info()

# --- 2. Visualización de la Serie ---
# Graficamos para observar tendencia y estacionalidad
df['y'].plot(figsize=(12, 5), title='Pasajeros de Aerolíneas Internacionales')
plt.ylabel('Número de Pasajeros')
plt.xlabel('Fecha')
plt.grid(True)
plt.show()

# --- 3. Verificación de Estacionariedad (paso clave en ARIMA) ---
# La serie claramente no es estacionaria (tiene tendencia y estacionalidad creciente)
# El modelo SARIMA puede manejar esto internamente con sus parámetros de diferenciación (d y D)
# Por ejemplo, una prueba de Dickey-Fuller lo confirmaría
result = adfuller(df['y'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}') # Un p-value alto (> 0.05) indica no estacionariedad

# --- 4. Ajuste del Modelo SARIMA ---
# SARIMA(p,d,q)(P,D,Q,m)
# p, d, q: Orden del componente no estacional (AR, I, MA)
# P, D, Q: Orden del componente estacional
# m: Periodicidad de la estacionalidad (12 para datos mensuales)

# Los parámetros (1,1,1)(1,1,1,12) son un punto de partida común para este tipo de series
model = SARIMAX(df['y'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()
print(results.summary())

# --- 5. Generación de Pronósticos ---
# Pronosticamos los próximos 36 meses (3 años)
forecast_steps = 36
forecast = results.get_forecast(steps=forecast_steps)

# Obtenemos los valores predichos y los intervalos de confianza
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()

# --- 6. Visualización de Resultados ---
ax = df['y'].plot(label='Observado', figsize=(14, 7))
pred_mean.plot(ax=ax, label='Pronóstico SARIMA', style='--')

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.15)

ax.set_xlabel('Fecha')
ax.set_ylabel('Número de Pasajeros')
plt.title('Pronóstico con Modelo SARIMA')
plt.legend()
plt.grid(True)
plt.show()
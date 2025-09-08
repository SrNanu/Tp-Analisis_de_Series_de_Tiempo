import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. Carga y Preparación de Datos ---
# Usamos el mismo dataset para comparar
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)

# Prophet requiere un formato específico de columnas: 'ds' para la fecha y 'y' para el valor
df = df.rename(columns={'Month': 'ds', 'Passengers': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

print("Primeros datos para Prophet:")
print(df.head())

# --- 2. Creación y Ajuste del Modelo Prophet ---
# Prophet detecta la estacionalidad anual por defecto en datos mensuales.
# Podemos especificar una estacionalidad multiplicativa, ya que la variación crece con la tendencia.
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)

# --- 3. Creación de un DataFrame Futuro para el Pronóstico ---
# Creamos un dataframe que se extienda 36 meses en el futuro
future = model.make_future_dataframe(periods=36, freq='MS')
print("\nDataFrame para el futuro:")
print(future.tail())

# --- 4. Generación de Pronósticos ---
forecast = model.predict(future)

# El forecast es un DataFrame con mucha información útil (tendencia, estacionalidad, etc.)
print("\nColumnas del pronóstico:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# --- 5. Visualización de Resultados ---
# Prophet tiene funciones de ploteo integradas y muy informativas

# Gráfico del pronóstico principal
fig1 = model.plot(forecast)
plt.title('Pronóstico con Prophet')
plt.xlabel('Fecha')
plt.ylabel('Número de Pasajeros')
plt.show()

# Gráfico de los componentes del modelo (tendencia y estacionalidad)
# Esto es muy útil para entender cómo el modelo ve los datos
fig2 = model.plot_components(forecast)
plt.show()
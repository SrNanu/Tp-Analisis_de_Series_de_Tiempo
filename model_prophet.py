import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet import Prophet

def run(train, test):

    print("\n--- Entrenando y Visualizando Modelo Prophet ---")

    #Lógica del Modelo
    prophet_train = train.reset_index().rename(columns={'Month': 'ds'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=len(test), freq='MS') #Crea el "template"
    forecast = model.predict(future) #Calcula la prediccion pero tiene un monton de datos mas
    predictions = forecast.set_index('ds')['yhat'][-len(test):] # Filtra solo los datos que nos importan

    #Cálculo del Error
    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo Prophet: {rmse:.2f}")

    #Visualización de Componentes
    model.plot_components(forecast)
    plt.show() 

    #Visualización de la Predicción
    plt.figure(figsize=(15, 7))
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
    plt.plot(predictions, label='Predicción Prophet', color='green', linestyle='--')
    plt.title('Resultado del Modelo Prophet', fontsize=20)
    plt.legend()
    plt.show() # Muestra la predicción y espera

    return predictions, rmse
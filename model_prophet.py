import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet import Prophet

def run(train, test):

    print("\n--- Entrenando y Visualizando Modelo Prophet ---")

    prophet_train = train.reset_index().rename(columns={'Month': 'ds'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    # Tendencia, Estacionalidad y Holidays


    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(test), freq='MS')
    forecast = model.predict(future) 
    predictions = forecast.set_index('ds')['yhat'][-len(test):] 

    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo Prophet: {rmse:.2f}")


    model.plot_components(forecast)
    plt.show() 

    plt.figure(figsize=(14, 6))
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
    plt.plot(predictions, label='Predicción Prophet', color='green', linestyle='--')
    plt.title('Resultado del Modelo Prophet', fontsize=20)
    plt.legend()
    plt.show() 

    return predictions, rmse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def run(train, test):

    print("\n--- Entrenando y Visualizando Modelo Naive ---")
    
    last_value = train['y'].iloc[-1]
    predictions = pd.Series(np.repeat(last_value, len(test)), index=test.index)
    
    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo Naive: {rmse:.2f}")
    
    plt.figure(figsize=(15, 7))
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
    plt.plot(predictions, label='Predicción Naive', color='orange', linestyle='--')
    plt.title('Resultado del Modelo Naive', fontsize=20)
    plt.legend()
    plt.show() 
    
    return predictions, rmse
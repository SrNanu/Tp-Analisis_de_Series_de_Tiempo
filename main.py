import matplotlib.pyplot as plt
import data_loader
import model_naive
import model_prophet
import model_lstm

# Configuración global de estilo para los gráficos
plt.style.use('seaborn-v0_8-whitegrid')

def main():

    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df, train, test = data_loader.cargar_y_preparar_datos(url)
    
    # Datos iniciales
    df['y'].plot(title='Número de Pasajeros de Aerolínea Mensuales (1949-1960) en miles', figsize=(15,7))
    plt.show() 

    # Modelos
    predicciones_naive, rmse_naive = model_naive.run(train, test)
    predicciones_prophet, rmse_prophet = model_prophet.run(train, test)
    predicciones_lstm, rmse_lstm = model_lstm.run(train, test)
    
    # Comparación final
    print("\n--- Mostrando Comparación Final de Modelos ---")
    predicciones = test.copy()
    predicciones['Naive'] = predicciones_naive
    predicciones['Prophet'] = predicciones_prophet
    predicciones['LSTM'] = predicciones_lstm
    
    plt.figure(figsize=(14, 6))
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(predicciones['y'], label='Datos Reales (Prueba)', color='black', lw=2)
    plt.plot(predicciones['Naive'], label='Predicción Naive', linestyle='--')
    plt.plot(predicciones['Prophet'], label='Predicción Prophet', linestyle='--')
    plt.plot(predicciones['LSTM'], label='Predicción LSTM', linestyle='--')
    plt.title('Comparación de Modelos de Pronóstico', fontsize=20)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Número de Pasajeros', fontsize=14)
    plt.legend()
    plt.show()

    # Errores
    print("\n--- Resumen de Resultados (RMSE) ---")
    print("=======================================")
    print(f"| Modelo Naive  | {rmse_naive:15.2f} |")
    print(f"| Modelo Prophet| {rmse_prophet:15.2f} |")
    print(f"| Modelo LSTM   | {rmse_lstm:15.2f} |")
    print("=======================================")

if __name__ == '__main__':
    main()
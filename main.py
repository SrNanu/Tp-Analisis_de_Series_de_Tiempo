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
    df['y'].plot(title='Número de Pasajeros de Aerolínea Mensuales (1949-1960)', figsize=(15,7))
    plt.show() #Los dataset ya tienen el plot?

    # Ejecución de los modelos
    # Cada uno calcula el modelo, muestra el gráfico y devuelve las predicciones y el RMSE
    naive_preds, rmse_naive = model_naive.run(train, test)
    prophet_preds, rmse_prophet = model_prophet.run(train, test)
    lstm_preds, rmse_lstm = model_lstm.run(train, test)
    
    # 3. Visualizar la comparación final
    print("\n--- Mostrando Comparación Final de Modelos ---")
    test_predictions = test.copy()
    test_predictions['Naive'] = naive_preds
    test_predictions['Prophet'] = prophet_preds
    test_predictions['LSTM'] = lstm_preds
    
    plt.figure(figsize=(16, 8))
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(test_predictions['y'], label='Datos Reales (Prueba)', color='black', lw=2)
    plt.plot(test_predictions['Naive'], label='Predicción Naive', linestyle='--')
    plt.plot(test_predictions['Prophet'], label='Predicción Prophet', linestyle='--')
    plt.plot(test_predictions['LSTM'], label='Predicción LSTM', linestyle='--')
    plt.title('Comparación de Modelos de Pronóstico', fontsize=20)
    plt.xlabel('Fecha', fontsize=14)
    plt.ylabel('Número de Pasajeros', fontsize=14)
    plt.legend()
    plt.show()
    
    # 4. Imprimir el resumen de errores
    print("\n--- Resumen de Resultados (RMSE) ---")
    print("=======================================")
    print(f"| Modelo Naive  | {rmse_naive:15.2f} |")
    print(f"| Modelo Prophet| {rmse_prophet:15.2f} |")
    print(f"| Modelo LSTM   | {rmse_lstm:15.2f} |")
    print("=======================================")
    print("\n--- Demostración Finalizada ---")

if __name__ == '__main__':
    main()
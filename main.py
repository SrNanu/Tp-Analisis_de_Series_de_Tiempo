# main.py

# Importar nuestras funciones desde los otros archivos
import data_loader
import models
import plotting

def main():
    """Función principal que orquesta todo el proceso."""
    
    # 1. Cargar y preparar los datos
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df, train, test = data_loader.load_and_prepare_data(url)
    
    # 2. Visualizar los datos iniciales
    plotting.plot_initial_data(df)
    
    # 3. Ejecutar cada modelo
    naive_preds, rmse_naive = models.run_naive_model(train, test)
    prophet_preds, rmse_prophet = models.run_prophet_model(train, test)
    lstm_preds, rmse_lstm = models.run_lstm_model(train, test)
    
    # 4. Preparar DataFrame para visualizaciones
    test_predictions = test.copy()
    test_predictions['Naive'] = naive_preds
    test_predictions['Prophet'] = prophet_preds
    test_predictions['LSTM'] = lstm_preds
    
    # 5. Visualizar resultados individuales
    plotting.plot_individual_prediction(train, test, test_predictions['Naive'], "Naive", "orange")
    plotting.plot_individual_prediction(train, test, test_predictions['Prophet'], "Prophet", "green")
    plotting.plot_individual_prediction(train, test, test_predictions['LSTM'], "LSTM", "red")
    
    # 6. Visualizar la comparación final
    plotting.plot_comparison(train, test_predictions)
    
    # 7. Imprimir el resumen de errores
    results = {
        "Naive": rmse_naive,
        "Prophet": rmse_prophet,
        "LSTM": rmse_lstm
    }
    plotting.print_summary_table(results)

if __name__ == '__main__':
    main()
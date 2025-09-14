# plotting.py
import matplotlib.pyplot as plt

# Configuración global para los gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 7)

def plot_initial_data(df):
    """Visualiza la serie de tiempo completa."""
    df['y'].plot(title='Número de Pasajeros de Aerolínea Mensuales (1949-1960)')
    plt.ylabel('Número de Pasajeros')
    plt.xlabel('Fecha')
    plt.show()

def plot_individual_prediction(train, test, predictions, model_name, color):
    """Grafica el resultado de un modelo individual."""
    plt.figure()
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
    plt.plot(predictions, label=f'Predicción {model_name}', color=color, linestyle='--')
    plt.title(f'Resultado del Modelo {model_name}', fontsize=20)
    plt.legend()
    plt.show()

def plot_comparison(train, test_predictions):
    """Grafica la comparación final de todos los modelos."""
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

def print_summary_table(results):
    """Imprime la tabla de resumen con los errores RMSE."""
    print("\n--- Resumen de Resultados (RMSE) ---")
    print("=======================================")
    for model, rmse in results.items():
        print(f"| Modelo {model:<8}| {rmse:15.2f} |")
    print("=======================================")
# model_lstm.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run(train, test, n_steps=12, epochs=200):
    """
    Ejecuta el modelo LSTM, muestra su predicción
    y devuelve las predicciones y el RMSE.
    """
    print("\n--- Entrenando y Visualizando Modelo LSTM ---")
    
    # 1. Preparación de datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    X_train, y_train = create_sequences(scaled_train, n_steps)
    
    # 2. Construcción y entrenamiento del modelo
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    # 3. Predicción
    lstm_predictions_scaled = []
    last_sequence = scaled_train[-n_steps:]
    current_batch = last_sequence.reshape((1, n_steps, 1))
    for _ in range(len(test)):
        current_pred = model.predict(current_batch, verbose=0)[0]
        lstm_predictions_scaled.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    predictions = scaler.inverse_transform(lstm_predictions_scaled)
    
    # 4. Cálculo del Error
    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo LSTM: {rmse:.2f}")

    # 5. Visualización
    plt.figure(figsize=(15, 7))
    plt.plot(train['y'], label='Datos de Entrenamiento')
    plt.plot(test['y'], label='Datos Reales (Prueba)', color='black')
    plt.plot(test.index, predictions, label='Predicción LSTM', color='red', linestyle='--')
    plt.title('Resultado del Modelo LSTM', fontsize=20)
    plt.legend()
    plt.show()

    return predictions.flatten(), rmse
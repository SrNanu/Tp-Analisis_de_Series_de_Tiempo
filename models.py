# models.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Librerías específicas de los modelos
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run_naive_model(train, test):
    """Entrena y predice usando el modelo Naive."""
    print("\n--- Entrenando Modelo Naive ---")
    last_value = train['y'].iloc[-1]
    predictions = pd.Series(np.repeat(last_value, len(test)), index=test.index)
    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo Naive: {rmse:.2f}")
    return predictions, rmse

def run_prophet_model(train, test):
    """Entrena y predice usando Prophet."""
    print("\n--- Entrenando Modelo Prophet ---")
    prophet_train = train.reset_index().rename(columns={'Month': 'ds'})
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_train)
    
    future = model.make_future_dataframe(periods=len(test), freq='MS')
    forecast = model.predict(future)
    
    # Graficar componentes es parte del análisis de Prophet
    model.plot_components(forecast).show()
    
    predictions = forecast.set_index('ds')['yhat'][-len(test):]
    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo Prophet: {rmse:.2f}")
    return predictions, rmse

def create_sequences(data, n_steps):
    """Función auxiliar para crear secuencias para la LSTM."""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def run_lstm_model(train, test, n_steps=12, epochs=200):
    """Entrena y predice usando una red LSTM."""
    print("\n--- Entrenando Modelo LSTM ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    
    X_train, y_train = create_sequences(scaled_train, n_steps)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())
    
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    # Proceso de predicción iterativa
    lstm_predictions_scaled = []
    last_sequence = scaled_train[-n_steps:]
    current_batch = last_sequence.reshape((1, n_steps, 1))
    
    for _ in range(len(test)):
        current_pred = model.predict(current_batch, verbose=0)[0]
        lstm_predictions_scaled.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
    predictions = scaler.inverse_transform(lstm_predictions_scaled)
    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    print(f"RMSE del Modelo LSTM: {rmse:.2f}")
    return predictions.flatten(), rmse
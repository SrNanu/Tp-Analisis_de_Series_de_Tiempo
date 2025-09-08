import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# --- 1. Carga y Preprocesamiento de Datos ---
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)
dataset = df['Passengers'].values.astype('float32').reshape(-1, 1)

# Normalizamos los datos. Las redes neuronales funcionan mejor con datos escalados (ej. entre 0 y 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# --- 2. División en Entrenamiento y Prueba (Cronológica) ---
train_size = int(len(dataset) * 0.8)
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# --- 3. Función para Crear Secuencias ---
# Esta función convierte la serie en pares (secuencia_de_entrada, valor_de_salida)
# ej: look_back=12 -> usa 12 meses para predecir el mes 13
def create_dataset(dataset, look_back=12):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# --- 4. Reshape para la LSTM ---
# La LSTM espera una entrada 3D: [muestras, pasos_de_tiempo, características]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# --- 5. Construcción y Entrenamiento del Modelo LSTM ---
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1))) # 50 neuronas en la capa LSTM
model.add(Dense(1)) # Capa de salida con 1 neurona (la predicción)
model.compile(loss='mean_squared_error', optimizer='adam')

print("Entrenando el modelo LSTM...")
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# --- 6. Generación de Predicciones ---
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# --- 7. Invertir la Normalización ---
# Devolvemos las predicciones a la escala original para poder interpretarlas
trainPredict = scaler.inverse_transform(trainPredict)
trainY_orig = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_orig = scaler.inverse_transform([testY])

# --- 8. Visualización de Resultados ---
# Preparamos los datos para graficar correctamente
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

plt.figure(figsize=(14, 7))
plt.plot(scaler.inverse_transform(dataset), label='Datos Originales')
plt.plot(trainPredictPlot, label='Predicciones de Entrenamiento (LSTM)')
plt.plot(testPredictPlot, label='Predicciones de Prueba (LSTM)', linestyle='--')
plt.title('Pronóstico con LSTM (Deep Learning)')
plt.xlabel('Tiempo (Meses)')
plt.ylabel('Número de Pasajeros')
plt.legend()
plt.grid(True)
plt.show()
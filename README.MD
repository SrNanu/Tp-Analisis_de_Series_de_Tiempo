
### Proyecto: Análisis Comparativo de Modelos de Series de Tiempo

Este proyecto realiza un análisis comparativo de tres modelos diferentes para el pronóstico de series de tiempo, utilizando el clásico dataset de "Pasajeros de Aerolínea".

---

#### Librerías Utilizadas

*   **Pandas:** Para la carga, manipulación y división de los datos.
*   **NumPy:** Para operaciones numéricas eficientes, especialmente en el cálculo de métricas y la preparación de datos para la LSTM.
*   **Matplotlib:** Para la creación de todas las visualizaciones y gráficos del proyecto.
*   **Scikit-learn:** Para el cálculo de la métrica de error (MSE) y para el escalado de datos (`MinMaxScaler`) en el modelo LSTM.
*   **Prophet:** Para implementar el modelo de pronóstico especializado de Facebook.
*   **Keras (TensorFlow):** Para construir, entrenar y ejecutar el modelo de red neuronal profunda (LSTM).

---

#### Modelos Implementados

Se implementaron y compararon los siguientes tres modelos, desde el más simple hasta el más complejo:

1.  **Modelo Naive (Ingenuo)**
    *   **¿Qué hace?** Es nuestro modelo de referencia. Predice que el valor futuro será simplemente el último valor conocido de la serie. Sirve como una línea base para evaluar la efectividad de los otros modelos.

2.  **Modelo Prophet**
    *   **¿Qué hace?** Es un modelo especializado que descompone la serie de tiempo en tres componentes principales: tendencia, estacionalidad y el efecto de días festivos. Los modela por separado y luego los suma para generar un pronóstico robusto y detallado.

3.  **Modelo LSTM (Long Short-Term Memory)**
    *   **¿Qué hace?** Es un modelo de Deep Learning (Red Neuronal Recurrente) diseñado para aprender de secuencias de datos. Se le enseñó a reconocer patrones en los últimos 12 meses para predecir el valor del mes siguiente. Representa el enfoque más avanzado y potente de los tres.
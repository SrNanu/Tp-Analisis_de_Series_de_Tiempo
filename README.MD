
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


Lo que hicimos fue seleccionar 3 modelos de analisis de series de tiempo para predecir los datos futuros a partir de un conjunto de datos pasados. 

Los datos que elegimos son la cantidad en millones de pasajeros en aerolinas desde los años 1949 a 1960, entonces para cada modelo lo vamos a entrenar con los datos hasta 1958 y despues va a predecir los dos años siguientes, finalmente lo vamos a comparar con los datos reales para ver que tan bien lo hizo

El primer modelo es el Naive o Ingenuo. Es el más simple de los tres y nos sirve como punto de referencia. Se basa en la premisa de que el próximo valor será igual al último valor conocido. Por lo tanto, toma el último dato del entrenamiento y predice que ese valor se repetirá constantemente en el futuro.

EEl segundo modelo es Prophet, creado por Meta. Es un modelo especializado que descompone la serie en componentes: una tendencia general, patrones estacionales (como el ciclo anual) y el efecto de feriados o eventos especiales. Para modelar la compleja forma de la estacionalidad, utiliza Series de Fourier. Finalmente, suma estos componentes para generar el pronóstico. Como veremos, este enfoque da una muy buena aproximación en nuestro problema.

El ultimo metodo es usando machine learning con una red neuronal LSTM o memoria a largo corto plazo, este tipo de red neuronal se usa bastante para predecir series de tiempo ya tiene memoria de largo y corto plazo, a difrenecia de una red neuronal recurrente simple que tiene memoria pero solo a corto plazo. En nuestro caso, creamos una red con una capa LSTM de 100 neuronas y una capa de salida. Para el entrenamiento, usamos el optimizador 'adam' y le indicamos que su objetivo era minimizar el Error Cuadrático Medio para ajustar sus parámetros internos

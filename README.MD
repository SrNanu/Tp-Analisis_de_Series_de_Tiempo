# Análisis Comparativo de Series de Tiempo con Python

Este repositorio contiene un análisis práctico y comparativo de cuatro enfoques diferentes para el pronóstico de series de tiempo utilizando Python. El objetivo es demostrar y contrastar un método estadístico clásico, una librería automatizada moderna, un enfoque de machine learning tradicional y un modelo de deep learning.

Todos los ejemplos utilizan el clásico conjunto de datos **"International Airline Passengers"** para permitir una comparación directa y visual de los resultados de cada modelo.

---

## Modelos Implementados

Aquí se encuentra un resumen de cada script y el enfoque que utiliza:

### 1. `Pronóstico_con_Modelo_SARIMA.py`
*   **Enfoque:** Estadístico Clásico.
*   **Librería Principal:** `statsmodels`.
*   **Descripción:** Implementación del modelo **SARIMA** (Seasonal AutoRegressive Integrated Moving Average). Este es un enfoque robusto que modela la tendencia, la estacionalidad y la autocorrelación de la serie. Requiere un análisis previo para determinar sus parámetros, ofreciendo un alto grado de control y interpretabilidad estadística.

### 2. `Pronóstico_con_Prophet_de_Facebook.py`
*   **Enfoque:** Modelo Aditivo Automatizado.
*   **Librería Principal:** `prophet`.
*   **Descripción:** Utiliza **Prophet**, la librería de Facebook diseñada para ser rápida, automática y fácil de usar. Descompone la serie de tiempo en tendencia, estacionalidad (anual, semanal, diaria) y efectos de festivos. Es especialmente potente para series de tiempo de negocio con patrones estacionales claros y datos faltantes.

### 3. `Scikit_learn_ML.py`
*   **Enfoque:** Machine Learning Clásico (Regresión Supervisada).
*   **Librería Principal:** `scikit-learn`.
*   **Descripción:** Transforma el problema de series de tiempo en un problema de regresión. Se realiza **ingeniería de características** (feature engineering) para crear variables a partir de la fecha (mes, año) y de valores pasados (lags). Se utiliza un modelo `RandomForestRegressor` para aprender la relación entre estas características y el valor a predecir. Este enfoque es muy flexible y permite añadir fácilmente variables externas.

### 4. `TensorFlow_y_Keras_DP.py`
*   **Enfoque:** Deep Learning.
*   **Librerías Principales:** `tensorflow` y `keras`.
*   **Descripción:** Aplica un modelo avanzado de Deep Learning utilizando una **Red Neuronal Recurrente LSTM** (Long Short-Term Memory). Las LSTMs están diseñadas específicamente para aprender de secuencias de datos, lo que las hace ideales para capturar patrones temporales complejos y no lineales. Este es el enfoque más potente para datasets grandes y complejos.

---

## Cómo Empezar

Sigue estos pasos para ejecutar los scripts en tu máquina local.

### Prerrequisitos
*   Python 3.x
*   Git

### Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone <URL-de-tu-repositorio>
    cd <nombre-del-repositorio>
    ```

2.  **Crea y activa un entorno virtual (recomendado):**
    ```bash
    # Crear el entorno
    python -m venv venv

    # Activar en Windows
    .\venv\Scripts\activate

    # Activar en macOS/Linux
    source venv/bin/activate
    ```

3.  **Instala las dependencias:**
    El archivo `requirements.txt` contiene todas las librerías necesarias.
    ```bash
    pip install -r requirements.txt
    ```

### Uso

Una vez configurado el entorno, puedes ejecutar cada script de forma individual desde la terminal. Cada script generará y mostrará un gráfico con los resultados del pronóstico.

```bash
python Pronóstico_con_Modelo_SARIMA.py
python Pronóstico_con_Prophet_de_Facebook.py
python Scikit_learn_ML.py
python TensorFlow_y_Keras_DP.py
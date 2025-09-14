# data_loader.py
import pandas as pd

def load_and_prepare_data(url):
    """
    Carga los datos desde una URL, los formatea y los divide en 
    conjuntos de entrenamiento y prueba.
    """
    df = pd.read_csv(url, header=0, parse_dates=['Month'], index_col='Month')
    df = df.rename(columns={'Passengers': 'y'})
    
    print("Primeros 5 registros del dataset:")
    print(df.head())
    print("\nÚltimos 5 registros del dataset:")
    print(df.tail())
    
    # Dividir los datos
    train_size = len(df) - 24
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    
    print(f"\nTamaño del conjunto de entrenamiento: {len(train)}")
    print(f"Tamaño del conjunto de prueba: {len(test)}")
    
    return df, train, test
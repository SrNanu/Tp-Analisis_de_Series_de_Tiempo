import pandas as pd

def cargar_y_preparar_datos(url):

    df = pd.read_csv(url, header=0, parse_dates=['Month'], index_col='Month')

    df = df.rename(columns={'Passengers': 'y'}) 

    train_size = len(df) - 24
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    
    print(f"\nTamaño del conjunto de entrenamiento (en meses): {len(train)}")
    print(f"Tamaño del conjunto de prueba (en meses): {len(test)}")
    
    return df, train, test
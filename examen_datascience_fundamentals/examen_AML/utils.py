import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
def cargar_datos(ruta_archivo):
    # Carga los datos del archivo CSV
    datos = pd.read_csv(ruta_archivo, delimiter=',')
    return datos

def exploracion_inicial(df):
    print("Informacion del dataset")
    print(df.info())
    print(df.head())
    ##Imprime ultimas 5 filas
    print("nÚltimas 5 filas del DataFrame:")
    print(df.tail(5))
    # Imprimir cantidad de filas y columnas del dataset
    print("Cantidad de filas: ", df.shape[0])
    print("Cantidad de columnas: ", df.shape[1])
# Identificar valores faltantes
def identificar_valores_faltantes(df):
    print("Valores faltantes en el dataset")
    qsna = df.shape[0] - df.isnull().sum(axis=0)
    qna = df.isnull().sum(axis=0)
    ppna = round(100 * (df.isnull().sum(axis=0) / df.shape[0]), 2)

    # Crear DataFrame con los resultados
    aux = {'datos sin NAs en q': qsna, 'Na en q': qna, 'Na en %': ppna}
    na_df = pd.DataFrame(data=aux)

    # Ordenar el DataFrame por el porcentaje de valores faltantes
    return na_df.sort_values(by='Na en %', ascending=False)
def verificacion_elementos_duplicados(df):
    ##Verificacion de elementos duplicados
    ###
    print("Cantidad de filas duplicadas: ",df.duplicated().sum())

### Verificar inconsistencias
#### Verificacion de inconsistencias en datos categoricos
def contar_valores_categoricos(df):
    # Recorre las columnas categóricas del DataFrame
    for col in df.select_dtypes(include=['object', 'category']):
        print(f'Valores únicos en la columna: {col}')
        print(df[col].value_counts())
        print('-' * 50)

##Se cambiar str to upper y strip utilizando metodo apply.
def modificar_valores_categoricos(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].str.upper().str.strip()
    return df
###Estadisiticas descriptivas
def estadisticas_descriptivas(df):
    print("Estadisticas del dataset")
    print(df.describe())    
    # Calcular medidas de tendencia central y dispersión para cada variable numérica
    print("Medidas de tendencia central y dispersión para cada variable numérica")
    measures = {}
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    ##Si los datos numericos estan en el dataframe
    if numeric_columns.size > 0:
        for column in numeric_columns:
            measures[column] = {
                'Mean': df[column].mean(),
                'Median': df[column].median(),
                'Mode': df[column].mode()[0],
                'Std Dev': df[column].std(),
                'Range': df[column].max() - df[column].min()
            }
    else:
        print("##No se encontraron columnas numéricas en el DataFrame.##")  

    # Mostrar medidas de tendencia central y dispersión
    for column, stats in measures.items():
        print(f"\n{column}:")
        for measure, value in stats.items():
            print(f"  {measure}: {value}")
    
### Verificacion de outliers
def plot_grafico_outliers(num_features,nrows, ncols, df):
    # Ajustar dinámicamente el tamaño de la figura
    figsize = (ncols * 5, nrows * 5)  # 5 unidades de ancho y alto por subplot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Aplanar para facilitar el acceso

    for i, feature in enumerate(num_features):
        sns.boxplot(x=df[feature], ax=axes[i], color="#75f8f2")
        axes[i].set_title(feature)

    # Ocultar ejes sobrantes si hay más subplots de los necesarios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

### Plot univariados de variables categoricas# Ajustar dinámicamente el tamaño de la figura
def plot_univariados_categoricos(df,ncols,nrows):
    figsize = (ncols * 5, nrows * 5)  # 5 unidades de ancho y alto por subplot
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Aplanar para facilitar el acceso

    # Plotear variables categóricas
    for i, column in enumerate(df.select_dtypes(include=['object', 'category']).columns):
        sns.countplot(data=df, x=column, ax=axes[i])
        axes[i].set_title(column)

    # Ocultar ejes sobrantes si hay más subplots de los necesarios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

# Función para plotear las variables seleccionadas en subplots
def plot_selected_variables(df, selected_columns, ncols, nrows):
    n = len(selected_columns)
    cols = ncols  # Número de columnas en el grid
    rows = nrows  # Numeor de Filas 
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()  # Asegura que sea un arreglo unidimensional
    
    for i, col in enumerate(selected_columns):
        ax = axes[i]
        if df[col].nunique() <= 10:  # Se asume categórica si tiene ≤ 10 valores únicos
            sns.countplot(data=df, x=col, palette='pastel', hue=col, ax=ax, legend=False)
            ax.set_title(f'Distribución de {col} (Categórica)')
            ax.set_ylabel('Frecuencia')
            ax.tick_params(axis='x', rotation=90)  # Rotar etiquetas
        else:
            sns.histplot(df[col], kde=True, color='skyblue', ax=ax)
            ax.set_title(f'Distribución de {col} (Numérica)')
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
    
    # Oculta subplots vacíos si las variables no llenan la cuadrícula
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


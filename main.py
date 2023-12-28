from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title='STEAM Games Queries', description='Esta es una aplicación para realizar consultas sobre todo el mundo de STEAM.')

items_games = pd.read_parquet('Archivos API\\items_games_API.parquet')
reviews_games = pd.read_parquet('Archivos API\\reviews_games.parquet')
steam_games = pd.read_parquet('Archivos API\\steam_games_API.parquet')


@app.get("/")
def read_root():
    return {"Mensaje": "Bienvenido a la aplicación de consultas de STEAM"}


@app.get("/PlayTimeGenre/{genero}", name="Año de lanzamiento con más horas jugadas por género")
def PlayTimeGenre(genero: str):
    '''
    Devuelve el año con más horas jugadas para el género de entrada

    Args: 
        genero (str): género de juego del que se quiere averiguar cuál fue el año en el que más se jugó.

    return:
        dict: diccionario donde la clave es el género y el valor es el año con más horas jugadas.
    
    '''
    # Filtramos el dataframe 'items_games' por género
    df_util = items_games[items_games['genres']== genero]
    
    # Agrupamos el dataframe anterior por año de lanzamiento, suma de minutos de juego y ordenamos en forma descendente
    df_agrupado = df_util.groupby('year_release')['playtime_forever'].sum().sort_values(ascending=False)

    # Como están ordenados de manera descendente, el valor máximo tendrá el índice [0]
    anio = df_agrupado.index[0]
    
    return {f"Año de lanzamiento con más horas jugadas para el género {genero}": int(anio)}


@app.get("/UserForGenre/{genero}", name="Usuario con más horas jugadas por género")
def UserForGenre(genero: str):
    '''
    Devuelve el usuario con más minutos jugados para el género de entrada

    Args: 
        genero (str): género de juego del que se quiere averiguar cuál fue el usuario con más horas jugadas.

    return:
        dict: diccionario donde hay dos valores:el usuario con más horas jugadas para ese género, y la cantidad total de horas que jugó cada año.
    
    '''
    # Filtramos el dataframe 'items_games' respecto al parámetro genero
    df_util = items_games[items_games['genres']== genero]

    # Agrupamos el dataframe anterior por usuario, sumamos los minutos de juego y ordenamos en forma descendente
    df_agrupado = df_util.groupby('user_id')['playtime_forever'].sum().sort_values(ascending=False)

    # Como están ordenados de manera descendente, el valor máximo tendrá el índice [0]
    user = df_agrupado.index[0]

    # Tomamos las filas del dataframe util que contengan su repsectivo usuario (user)
    df_genero_user = df_util[df_util['user_id']==user]

    # Agrupamos respecto a los años y suma de minutos de juego
    minutos_jugados = round(df_genero_user.groupby('year_release')['playtime_forever'].sum(), 3)

    # Guardamos la serie 'minutos_jugados' en una lista
    lista_minutos_jugados = [f'Año: {int(anio)}, Horas: {horas}' for anio, horas in minutos_jugados.items()]

    return {f"Usuario con más horas jugadas para género {genero}": user, "Horas jugadas": lista_minutos_jugados}


@app.get("/UsersRecommend/{anio}", name="Top 3 de juegos más recomendados por año.")
def UsersRecommend(anio: int):
    '''
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
    
    Args: 
        anio (int): año del que se quiere saber el top 3 juegos más recomendados.

    return:
        dict: diccionario con los 3 juegos más recomendados por usuarios
        ->  reviews['recommend'] = True(1)
        ->  comentarios positivos(2) o neutros(1)

    '''
    # Si el año de entrada no correasponde a algún año de los que se hizo reseña (year_posted), retorna un mensaje de error.
    if anio not in reviews_games['year_posted'].unique():
        return f"El año ingresado se encuentra fuera del rango de registros, intente nuevamente"
        
    else:
        # Filtramos el dataframe por las filas donde el año de posteo (year_posted) no es menor al año de publicación (year_release).
        df = reviews_games[reviews_games['year_posted']>=reviews_games['year_release']]
            
        # Filtramos el dataframe 'df' para el año parámetro y donde la columna 'sentiment_analysis' sea positiva (2) o neutra (1)
        df_anio = df[(df['year_posted']==anio) & (df['sentiment_analysis'].isin([1,2]))]

        # Agrupamos el dataframe 'df_anio' por el título del juego (title), sumamos las recomendaciones (recommend) para obtener los juegos más recomendados, y ordenamos de forma descendente.
        top = df_anio.groupby('title')['recommend'].sum().sort_values(ascending=False)

        # Construimos el top 3 de juegos más recomendados por los usuarios.
        top_tres = [{"Puesto 1": top.index[0]}, {"Puesto 2": top.index[1]}, {"Puesto 3": top.index[2]}]

    return top_tres


@app.get("/UsersNotRecommend/{anio}", name="Top 3 de juegos menos recomendados por año.")
def UsersNotRecommend(anio: int):
    '''
    Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado
    
    Args: 
        anio (int): año del que se quiere saber el top 3 juegos menos recomendados.

    return:
        dict: diccionario con los 3 juegos menos recomendados por usuarios
        ->  reviews['recommend'] = False(0)
        ->  comentarios negativos(reviews['sentiment_analysis']==0)

    '''
    # Si el año de entrada no correasponde a algún año de los que se hizo reseña (year_posted), retorna un mensaje de error.
    if anio not in reviews_games['year_posted'].unique():
        return f"El año ingresado se encuentra fuera del rango de registros, intente nuevamente"
            
    else:
        # Filtramos el dataframe por las filas donde el año de posteo (year_posted) no es menor al año de publicación (year_release).
        df = reviews_games[reviews_games['year_posted']>=reviews_games['year_release']]

        # Filtramos el dataframe 'df' para el año de entrada y donde las columnas 'recommend' y 'sentiment_analysis' sean negativas (0) 
        df_anio = df[(df['year_posted']==anio) & (df['recommend']==0) & (df['sentiment_analysis']==0)]

        # Agrupamos el dataframe 'df_anio' por el título del juego (title), contamos las recomendaciones (recommend) para obtener los juegos con más recomendaciones negativas, y ordenamos de forma descendente.
        top = df_anio.groupby('title')['recommend'].count().sort_values(ascending=False)

        # Construimos el top 3 de juegos más recomendados por los usuarios.
        top_tres = [{"Puesto 1": top.index[0]}, {"Puesto 2": top.index[1]}, {"Puesto 3": top.index[2]}]

    return top_tres


@app.get("/sentiment_analysis/{anio}", name="Reseñas categorizados con un análisis de sentimiento por año.")
def sentiment_analysis(anio: int):
    '''
    Según el año de lanzamiento, devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento

    Args: 
        anio (int): año del que se quiere saber la cantidad de registros de reseñas de usuarios.

    return:
        list: lista con la cantidad de registros de reseñas de usuarios en ese año. Las cantidades están separadas por tipo de reseña (negativa, positiva y neutra)

    '''
    
    # Si el año de entrada no correasponde a algún año de los que se hizo reseña (year_posted), retorna un mensaje de error.
    if anio not in reviews_games['year_posted'].unique():
        return f"El año ingresado se encuentra fuera del rango de registros, intente nuevamente"
            
    else:
        # Filtramos el dataframe con las filas cuyo año de posteo (year_posted) no es menor al año de publicación (year_release).
        df = reviews_games[reviews_games['year_posted']>=reviews_games['year_release']]

        # Filtramos el dataframe 'df' para el año de entrada.
        df_anio = df[df['year_release'] == anio]

        # Contamos las filas del dataframe 'df_anio' respecto a los valores únicos de la columna 'sentiment_analysis' (0, 1 y 2) y los agrupamos de acuerdo a su categoría.
        positive = df_anio[df_anio['sentiment_analysis']==2].shape[0]  # Número de filas de sentimientos positivos (2)
        neutral = df_anio[df_anio['sentiment_analysis']==1].shape[0]   # Número de filas de sentimientos neutros (1)
        negative = df_anio[df_anio['sentiment_analysis']==0].shape[0]  # Número de filas de sentimientos negativos (0)

    return {'Negative': negative, 'Neutral': neutral, 'Positive': positive}

# Instanciamos la clase CountVectorizer
vector = CountVectorizer(tokenizer = lambda x: x.split(', '))

# Dividimos cada cadena de descripción en palabras individuales y se creamos una matríz de conteo que representa cuántas veces aparece cada género en cada videojuego.
matriz = vector.fit_transform(steam_games['description'])

@app.get("/recomendacion_juego/{product_id}", name="Recomendación de juegos similares según ID de un producto.")
def recomendacion_juego(product_id: int):
    '''
    Se ingresa el id de producto (item_id) y retorna una lista con 5 juegos recomendados similares al ingresado (title).
    
    '''
    # Si el id de entrada no correasponde a algún id en la columna 'item_id', retorna un mensaje de error.
    if product_id not in steam_games['item_id'].values:
        return f'El ID ingresado es inexistente, intente nuevamente'
    else:
        # Buscamos el índice del id ingresado
        index = steam_games.index[steam_games['item_id']==product_id][0]

        # De la matriz, tomamos el array de descripciones donde índice es igual a 'index'
        description_index = matriz[index]

        # cosine_similarity(description_index, matriz): Calcula la similitud coseno entre la descripción de entrada (description_index) y la descripción de cada fila de la matriz.
        # Realizamos un ordenamiento descendente de los índices de la matriz de similitud coseno. Es decir, realizamos un ordenamiento de mayor a menor similitud de las descripciones.
        # Tomamos los índices del 1 al 6 ya que el índice 0 es el mismo índice de entrada.
        top_indices = np.argsort(-cosine_similarity(description_index, matriz))[0, 1:6]

        # Construimos la lista 'recomendaciones'
        recomendaciones = []
        for i in top_indices:
            recomendaciones.append(steam_games['title'][i])
        
        return recomendaciones
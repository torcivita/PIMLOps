<p align=center><img src=Im%C3%A1genes/68747470733a2f2f643331757a386c77666d796e38672e636c6f756466726f6e742e6e65742f4173736574732f6c6f676f2d68656e72792d77686974652d6c672e706e67.png><p>

# <div align="center"><u>**PROYECTO INDIVIDUAL N° 1:**</u> </div>
# <div align="center"><u>**MACHINE LEARNING OPERATIONS (MLOps)** </u> </div>

## **Descripción**
Este proyecto es una simulación de trabajo real de un Data Scientist de Steam, una plataforma multinacional de videojuegos.
El objetivo principal es disponiblizar datos útiles mediante una API y crear un modelo de Machine Learning que funcione como sistema de recomendación de videojuegos para usuarios. Para ello fue necesario pasar por todo el proceso, desde el ETL hasta la disponibilización de la API en la web.

<p align=center><img src=Im%C3%A1genes/230424_ml-model-development_infographic_2.jpg><p>

## **Proceso**
### ETL (Extracción, Transformación y Carga)
El primer paso fue descargar la [Base de Datos](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj). En la misma se encuentran tres archivos ([steam_games.json.gz](Dataset/steam_games.json.gz), [user_reviews.json.gz](Dataset/user_reviews.json.gz) y [users_items.json.gz](Dataset/users_items.json.gz)) que, a su vez, los podrán a encontrar en la carpeta [Dataset](Dataset). 
La madurez de los datos era casi nula, me encontré con datos anidados e incompletos. el proceso de transformación fue bastante largo. A este proceso, junto con parte del EDA, se lo puede encontrar en el archivo [ETL.ipynb](<Jupyter Notebooks/ETL.ipynb>) en la carpeta [Jupyter Notebooks](<Jupyter Notebooks>).

### EDA (Análisis Exploratorio de los Datos)
Dado a la enorme complejidad del purgamiento de datos, parte del analisis exploratorio se encuentra en el archivo [ETL.ipynb](<Jupyter Notebooks/ETL.ipynb>). Allí realicé la obtencion de los datos, tratamiento de faltantes, tratamiento de outliers, agrupación y exploración. Por otro lado, la visualización de algunas métricas básicas, que favorecen a la interpretación, están en el archivo [Visualizacion.ipynb](<Jupyter Notebooks/Visualizacion.ipynb>).

### Funciones de Consulta
Las funciones de consulta desarrolladas las pueden encontrar y probar en el archivo [TestAPI.ipynb](<Jupyter Notebooks/TestAPI.ipynb>) y son las siguientes:

- `def PlayTimeGenre(genero : str )` Devuelve el año con más horas jugadas para el género de entrada

    ```
    return {f"Año de lanzamiento con más horas jugadas para el género {genero}": int(anio)}
    ```

- `def UserForGenre(genero : str)` Devuelve el usuario con mas horas jugadas para el género de entrada.

    ```
    return {f"Usuario con más horas jugadas para género {genero}": user, "Horas jugadas": lista_minutos_jugados}
    ```

- `def UserRecommend(año : int)` Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.

    ```
    return [{Puesto 1: 'Juego 1'}, {Puesto 2: 'Juego 2'}, {Puesto 3: 'Juego 3'}]
    ```

- `UsersNotRecommend(anio : int)` Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año de entrada.

    ```
    return [{Puesto 1: 'Juego 1'}, {Puesto 2: 'Juego 2'}, {Puesto 3: 'Juego 3'}]
    ```

- `sentiment_analysis(anio : int)` Según el año de lanzamiento, devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento

    ```
    return {'Negative': Cantidad de Negativos (int), 'Neutral': Cantidad de Neutros (int) , 'Positive': Cantidad de Positivos (int)}
    ```

### ML (Machine Learning)
Desarrollé un sistema de recomendaciones utilizando la librería `Scikit-Learn`. Apliqué la técnica de vectorización **One-Hot-Encoding** sobre las columnas de géneros, etiquetas y especificaciones. La idea detrás de One-Hot-Encoding es convertir cada valor en una variable categórica en un vector binario. La vectorización One-Hot-Encoding crea un vector de longitud N con todos los elementos iguales a cero, excepto uno, que estará en la posición correspondiente a la categoría de la observación.

Luego, utilicé la **similitud del coseno** para calcular la similitud entre cada par de vectores de descripción y ordearlos según su similitud. 

La representación one-hot convierte las categorías en vectores binarios. Estos vectores son ortogonales entre sí, ya que solo tienen un valor no nulo en una posición específica y ceros en todas las demás. Dado que la similitud del coseno mide la similitud basada en el ángulo entre dos vectores, es especialmente útil cuando trabajamos con vectores one-hot. 

Con este proceso, en el archivo [ML.ipynb](<Jupyter Notebooks/ML.ipynb>) creé una función de recomendación de videojuegos que tiene una relación ítem-ítem, esto es se toma un item, en base a que tan similar sea ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados.

- `recomendacion_juego(product_id: int):` Se ingresa el id de producto (item_id) y retorna una lista con 5 juegos recomendados similares al ingresado (title).

    ```
    return {['Juego 1', 'Juego 2', 'Juego 3', 'Juego 4', 'Juego 5']}
    ```

### API
Para el desarrollo de la API utilicé el Framework FastAPI. A la misma la desarrollé en el archivo [main.py](main.py).
La API contiene las 5 funciones de consultas sobre videojuegos desarrolladas en el archivo [TestAPI.ipynb](<Jupyter Notebooks/TestAPI.ipynb>) y, además, contiene una sexta función que es el modelo de recomendación de videojuegos desarrollado en el archivo [ML.ipynb](<Jupyter Notebooks/ML.ipynb>).

Tuve problemas para realizar el despliegue de la API en Render, como se solicita en las consignas del trabajo, pero de todas maneras pude grabar un video de su funcionamiento haciendo un despliegue desde la terminal de mi PC con el comando *uvicorn main:app --port 8001*. 
El problema del despliegue en Render se dio por un límite en la memoria de procesamiento de la versión gratuita. Intenté solucionarlo, por consejo de mis compañeros, achicando los archivos de donde se ectraen los datos de consulta, los reduje a un 10% de su tamaño original (proceso en el archivo [Reducción Archivos.ipynb](<Jupyter Notebooks/Reducción Archivos.ipynb>)) pero ni así logré que funcione.

Link del video:

### Librerías y Frameworks
Utilicé las siguientes librerías
    - **os**, **datetime**, **gzip** y **json** para abrir los archivos del dataset, manipulación de estructuras de datos, carga de dataframes, etc.
    - **pandas** y **numpy** para la manipular y procesar datos.
    - **Matplotlib** y **Seaborn** para la visualización de datos.
    - **nltk** para el análisis de lenguaje natural.
    - **scikit-learn** para el aprendizaje automático del modelo de recomendación.
    - **FastAPI** para construir la API con Python.
    
# Guia 4

## Ejercicio 1

Entiendo que como la lista X tiene entradas <algoritmo, {hiperparametros}> entonces es un grid search más flexible. Para cada algoritmo se puede especificar los puntos exactos del grid que se quiere probar.

Un grid search más clasico definiría para el algoritmo un intervalo de busqueda para cada hiperparámetro. Por ejemplo:

```
Algoritmo: árbol de decisión

altura_max: {5, 10, 15}
medida:    {gini, entropy}

grid:
(5,  gini)
(5,  entropy)
(10, gini)
(10, entropy)
(15, gini)
(15, entropy)
```

### a

```python
GRID_SEARCH(X : List⟨Tuple⟨ALGORITMO, HYPERS⟩⟩, D : DATA, M : METRICA) : Tuple⟨ALGORITMO, HYPERS, FLOAT⟩

    # para cada tupla (algoritmo hypers) de la lista x, armo una tupla (algoritmo, hyper, cross_val)
    resultados = [(algoritmo, hyperparams, CROSS_VAL(algoritmo, hyperparams, D, M)) for algoritmo, hyperparams in X]
    
    # devuelvo el que mayor valor de metrica M dio en el cross_val
    return (max(resultados, key=lambda x: x[2]))


```

### b

Como no tengo información de los datos voy a implementar el CROSS_VAL separando en k-folds para tener una medida más robusta de performance.
Por cada fold voy a obtener un modelo y predicciones para el fold de validacion. Luego con las predicciones de todos los folds voy a calcular una metrica global con M


```python
CROSS_VAL(A : ALGORITMO, HS : HYPERS, D : DATA, M : METRICA) : FLOAT 
    # supongo D tiene tuplas de (atributos, etiqueta)

    # separo los datos en folds. Supongo que devuelve lista de listas. Cada lista tiene los datos de ese fold
    # elegi arbitrariamente 5 folds, pero podria ser parametrizado.
    folds = SEPARAR_EN_FOLDS(k=5, D)

    predicciones_per_fold = []

    # iterar i= 1..k folds
    for i in range(5):
        # separar el i-esismo fold para validacion
        val = folds[i]
        train = [x for j in range(5) if j != i for x in folds[j]]

        # obtener un modelo con los datos de entrenmaiento con el algoritmo
        # los datos ya tienen la etiqueta
        modelo = Algoritmo(train, HS).fit()       

        # guardar performance medida con metrica M sobre los datos de validacion
        predicciones_fold = modelo(val)

        predicciones_per_fold.append(predicciones_fold)

    # Las predicciones estan en una lista de lista [[pred_fold1], [pred_fold2], ...]
    # lo convierto a una lista de pred [pred1, pred2, ...]
    predicciones_aplanadas_en_una_sola_lista = [pred for ppfold in predicciones_per_fold  for pred in ppfold]
    # juntar los valores de la metrica M de cada fold y devolerlo.
    # Le paso las predicciones de todos los datos y los datos, que guardan las etiquetas reales
    return M(predicciones_aplanadas_en_una_sola_lista, D)

```

### c

Suponemos H : HYPERS es un diccionario en donde los valores también pueden ser distribuciones probabilísticas a las que se las puede muestrear con la función SAMPLE(D: DISTRIBUCIÓN) : FLOAT
(suponer que SAMPLE(CONSTANTE) = CONSTANTE).

Aca tambien vamos a usar cross_val con k-fold para obtener resultados mas robustos.

```python
RANDOM_SEARCH(X : List⟨Tuple⟨ALGORITMO, HYPERS⟩⟩, D : DATA, M : METRICA, N : INT) : Tuple⟨ALGORITMO, HYPERS, FLOAT⟩

    resultados []

    # en hyperparams tenemos un diccionario {hyperparametro: distribución}
    # armamos otro con {hyperparametro: valor} donde valor es una muestra de su distribución correspondiente
    for algoritmo, hyperparams in X
        for i in range(N)
            hs = {hyper: SAMPLE(dist) for hyper, dist in hyperparams.items()}

            resultados.append((algoritmo, hs, CROSS_VAL(algortimo, hs, D, M)))

    # devuelvo el que mayor valor de metrica M dio en el cross_val
    return (max(resultados, key=lambda x: x[2]))

```

## Ejercicio 2

### a
Verdadero. 

La medida de performance solo sobre datos de entrenamientos nos da una medida significativa solo sobre estos datos.
Hacer validación cruzada implica separar una parte de los datos para entrenar y otra para medir la performance, y así tener una medida
significativa y realista de la performance del modelo. Es decir, cómo se comportará ante datos nuevos.

### b
Falso. 

Los datos se parten en K subconjuntos. Se entrenan K modelos en donde cada uno separa uno de los folds (el k-esimo) para
la validación. El resto se usa para entrenamiento, y por lo tanto no son disjuntos.

### c
Verdadero.

Cada modelo utiliza un fold distinto como datos de validación. Podemos tener una métrica global donde primero juntamos las predicciones y luego
calculamos la métrica, o podemos calcular la métrica por fold y luego promediarlas (o juntarlas de alguna forma).

### d
Falso.

Se podria tener buena performance en entrenamiento y mala en validación. Podria ser overfitting o una mala partición de los datos.
K-fold cross validation trata de "sanar" este problema de medición de generalidad, pero igualemtente se tendría una evidencia (no garantía) que señala posible overfitting.

### e
Falso.

Entiendo que "modelos más realistas" se refiere a que mejor se acerquen a describir los datos reales. Como se está entrenando con más datos
de entrenamiento con $k \rightarrow n$ entonces se podría decir que se aproxima a un modelo más realista.

Por otro lado, los N modelos que se generan no necesariamente son independientes. Al compartir gran parte de los datos con
otras particiones, muy probablemente el algoritmo genere modelos parecidos. (También depende de los datos).

Es correcto que hay que entrenar demasiados modelos con N relativamente grande, y esto es costoso.

La herramienta Leave on out cross validation donde K=N y se usa una sola instancia como validación sirve para cuando
el conjunto de datos es pequeño y es factible corroborar la independencia de los datos (y asi sí se puede obtener
modelos independientes). 

### f
Falso.

Depende mucho del azar (de como se particionó los datos de entrenamiento y control). Hasta podría dar mejor performance en el control.

Se esperaría que los datos de entrenamiento y control provengan de una distribución parecida. Si se llega a encontrar un modelo que funciona bien
con los datos de entrenamiento, se esperaría que tambien funcione relativamente bien con los de control. 

### g
Falso.

El conjunto de control solo se usa al final de todo el proceso para tener una medida de generalización. Si se usa para
entrenar, entonces dicha medida estará sesgada a esos datos.

Por otro lado, tengo entendido que luego de elegir los valores de los hiperparámetros, se juntan los datos de entrenamiento y validación
y se hace un solo entrenamiento (sin k-fold) para obtener el modelo final.

### h
Verdadero.

Justificación en punto g

### i
Verdadero.

Los datos de control estan separados de los de entrenamiento, por lo tanto no fueron usados en el entrenamiento para ir tomando decisiones
sobre que modelos elegir.
Es razonable entonces esperar una peor performance con dichos datos.


## Ejercicio 3

### a

Hay situaciones en el que los datos tienen grupos, y se podría tener una cantidad distinta de ellos y/o con atributos con correlaciones espurias.

1. Tenemos 3 clases A, B, C que queremos predecir. Tenemos 100 datos de entrenamiento con ciertos atributos. 60 clase A, 30 clase B y 10 clase C
Si separamos 80% de los datos para entrenamiento y el resto para validación de manera aleatoria podría pasar que en validación solo tengamos datos de las clases A y B,
y todos los de C caigan en entrenamiento.
Se podría llegar a un modelo que aprendió muy bien a clasificar instancias A y B pero no C, y al no tener ninguna instancia de la clase C en validación, se obtiene un valor
de performance bueno general pero no representa que tan bien le va al modelo para la clase C.
Lo ideal es que se separe de cierta forma para tener datos de todas las clases en ambas particiones.
A esto se lo llama Stratified cross validation, y se puede hacer con k-fold tambien (Stratified k-fold cv)
2. Queremos hacer una predicción temporal de algún dato, por ejemplo el precio de un producto. Se tiene el historial de compra y ventas desde cierto rango de años (2020 a 2026).
Si partimos al azar datos de entrenamiento y validación podriamos tener que entrenamos con abril 2024, marzo 2022, mayo 2025, etc y validamos con otros periodos igual de aleatorios.
Para analisis temporal tiene más sentido ir expandiendo la "ventana" de datos de entrenamiento:
Train1: enero a abril 2020, valid1: mayo 2020
Train2: enero a mayo 2020, valid: junio 2020
...
Esto se lo llama temporal series o time series k-fold cross validation. 
3. Los datos pueden venir de distintas fuentes, cada una puede tener correlaciones espurias con la etiqueta a predecir.
Por ejemplo, datos de 3 camaras distintas. Si mezclamos aleatoriamente se podria tener en entrenamiento y validación los datos de la misma cámara. Luego
el modelo podría aprender características como el tamaño de la imagen, iluminación, resolución o demás cosas particulares a dicha cámara, en vez de algún
atributo de la imagen que sea importante al problema (datos con correlaciones espurias).
La forma de solucionarlo es separar las fuentes y usar cada fuente solo en entrenamiento o solo validación. Como los atributos espurios de una fuente no mejorarían
la performance sobre otra fuente, se obliga a aprender atributos relacionados con el problema.
A esto se lo llama leave on group out (por ejemplo camara 2 y 3 para entrenamiento y se deja camara 1 para validación)  

### b
Los datos de control sirven para calcular el desempeño del modelo encontrado sobre datos que nunca se vieron y asi medir generalidad. Si lo usamos más de una vez
y cada vez tomamos una decisión a partir de ellas, pierda la calidad de ser usado para medir generalidad. (Indirectamente se estaría overfitteando a ellos).

### c
Analizar posibles causas, plantearlas para una nueva fase de desarrollo, previamente seleccionar nuevas instancias de datos de control independientes.

Posibles causas:
- overfitting
- diferencia en distribuciones de datos de control y desarrollo
- problemas de separación de datos

## Ejercicio 4

### a

### b

### c

### d

## Ejercicio 5

### a

### b

### c

### d

### e

### f

## Ejercicio 6

## Ejercicio 7

## Ejercicio 8
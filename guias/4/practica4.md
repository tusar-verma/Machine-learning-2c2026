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

### b

### c

## Ejercicio 3

## Ejercicio 4

## Ejercicio 5

## Ejercicio 6

## Ejercicio 7

## Ejercicio 8
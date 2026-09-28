# Guia 7

## Ejercicio 1

### I

Error cuadratico: $error(y, \hat{h}(x)) = (y - \hat{h}(x))^2$

### II
### III

## Ejercicio 2

### a

Verdadero, se reduce el overfitting. Los modelos generados les costará más memorizar los datos de entrenamiento y los ruidos y deberan buscar un patrón que descrba a los datos.
Cuando hay pocos datos y algoritmos que generan modelos complejos, para distintos dataset se obtienen modelos muy distintos que cada uno se ajusta fuertemente a los datos con los que fue entrenado.

### b

Falso, si el algoritmo genera modelos que son muy simples, no importa el tamaño del dataset. Por ejemplo si el algoritmo solo genera modelos lineales y los datos siguen un patrón de una curva.

### c

Falso.

### d

Verdadero.

### e

Verdadero.

### f

Verdadero.

## Ejercicio 3

### a

Se debe comparar con C4 que obtuvo buenos resultados en entrenamiento y validación. Vamos a suponer que los datos son suficientemente buenos.

- C1 está sobre ajustando a sus datos de entrenamiento al tener una peor performance en validación
- C2 está subajustando. Como c4 demuesta que hay modelos con performance .99, obtener una performance .90 implica que no se está encontrando el patrón subyacente de los datos. No hay sobreajuste ya que en validación dio un score similar.
- C3: Sobreajuste por la diferencia de score en entrenamiento y validación y subajuste por el score bajo general del modelo.

### b

Para sobreajuste intentaría ampliar el dataset de entrenamiento (natural o artificialmente). O reducir la complejidad ajustando hiperparámetros o cantidad de atributos.

Si en cambio es subajuste, intentaría complejizar el modelo ajustando los hiperparámetros para complejizar los modelos generados, o agregando atributos al dataset. 

No haría ambas a la vez. Y ambas respuestas estan ligadas al tradeoff sesgo-varianza.

### c

C2 y C3

### d

C1 y C3

### e
Suponiendo el caso ideal y donde los datos de entrenamiento y validación que se usen no jueguen un rol importante en este analisis, varianza si implica sobreajuste y sobreajuste implica varianza.

- varianza -> sobreajuste: Con varianza alta lo que pasa es que el algoritmo tiene suficiente libertad para adaptarse a tanto la función que se quiere aprender como al ruido. Por lo que se termina sobreajustando (se tiene un error muy chico sobre los datos en los que se entrenó y a su ruido, pero al validar sobre otro dataset que tiene otro ruido, se obtiene una performance peor).
- sobreajuste -> varianza: Se obtuvo 2 valores de performance sobre 2 datasets disintos (entrenamiento y validación) y se obtuvo una brecha grande entre ambos (se sobreajustó a los de entrenamiento). Ahora uno podría pensar que justamente lo que permitió el sobreajuste es la flexibilidad del modelo. Y si se hubiese entrenado con otro dataset, se obtendría otro modelo completamente distinto. Y por lo tanto se tendría alta varianza. (No estoy seguro de estas inferencias).

### f

C1 sin altura limitada por que sobreajusta a los datos de entrenamiento y C2 de altura acotada al subajustar y tener un rendimiento similar en validación.

### g

No
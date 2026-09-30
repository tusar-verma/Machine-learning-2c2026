# Guia 9

## Ejercicio 1

### a
La normalización min-max sufre si hay valores atípicos (outliers). En este caso pudo ocurrir que hubo un dato con el precio mucho más elevado que los demás. Todos quedan en el lado cercano al 0 del intervalo y el outlier en el 1.

### b
Se tiene la estandarización o Z-score (restar a cada instancia la media muestral y dividirlo por el desvio estandar muestral). Este método también es afectado por outliers (aumenta la varianza y mueve la media). Pero si se sabe a priori que los datos siguen una distribución normal, no es mala idea usar la estandarización.


## Ejercicio 2

### a

Hubo fuga de información: se calculó la media sobre los datos de desarrollo y control. Luego la performance fue sobre-estima (error subestimado). 

### b

La imputación se hace en el dataset de entrenamiento. Se calcula la media de entrenamiento y se usa para imputar en validación. Una vez obtenida la mejor media con los datos de desarrollo (el mejor del k-fold cv), se utiliza dicha media para imputar los datos de control.

### c

Sobre la misma media que fue entrenado el modelo (la media de entrenamiento).

## Ejercicio 3

Nuevas columnas: 
- Distancia a espacio verde más cercano
- Distancia a transporte público más cercano
- Métros cuadrados cubiertos
- Antigüedad del edificio
- Expensas

Transformaciones:
- Precio: Transformar para llevarlo a una escala similar de los demás atributos numéricos y así no llevarse completamente la representación de distncia. Esto se debe hacer analizando la distribución de los datos. Si es normale, se puede estandarizar. Si es asimétrica se puede hacer una transformación logarítmica primero y luego estandarizar.
Antes de transformar, se debe imputar los valores faltantes, por ejemplo con la media o mediana.
- ubicación: Imputar faltantes, puede ser poniendo una caterogria especial "desconocido". One hot encoding de tamaño la cantidad de categorías.
- Metros cuadrados: imputar faltantes con la media o mediana y luego escalar con estandarización o z-score para que no dominen a habitaciones y baños en el calculo de distancia.
- Habitaciones: Se debe decidir entre imputar con la mediana (y agregar columna de imputación o real) o usar medida de distancia que soporte valores nulos. 
- Fecha de publicación: convertir a dias activos en mercado (31/12/2021 - fecha de publicación). Luego se puede imputar los faltantes con la mediana y escalarlos si es necesario a un rango más cercano al de habitaciones y baños.
- Baños: Se debe decidir entre imputar con la mediana (y agregar columna de imputación o real) o usar medida de distancia que soporte valores nulos.
- Distancia a espacio verde más cercano: imputar faltantes teniendo en cuenta los demás atributos que correspondan a ubicación geografica.
- Distancia a transporte público más cercano: idem anterior. Pero ambas también van a depender del dominio del problema. Además ambas deberían escalarse según el rango de valores que se maneje.
- Métros cuadrados cubiertos: igual que métros cuadrados.
- fecha de construcción: convertir a años de antiguedad (año actual - año de construcción). Imputar faltantes con la mediana y escalar si es necesario para que no dominen ante otros atributos.
- expensas: Imputar faltantes con la mediana y agregar una columna indicadora. Luego hacer una transformación similar a la de precio. La imputación se debe hacer viendo las reglas de dominio (capaz hay casas o ph que no tienen expensas).

Notas: todas las imputaciones se deben hacer teniendo en cuenta el motivo de su ausencia:
- MNAR, Missing not at random: su aucencia es legítima y tiene una justificación. Por ejemplo ciertas personas deciden no compartir su salario. Se puede imputar pero agregando una columna indicadora.
- MAR, Missing at random: Su aucencia depende de otro atributo. Por ejemplo si se tiene edad y salario, la gente menor a 16 años, en general, no tiene salario. Se puede imputar con métodos predictivos (KNN imputer o MICE)
- MCAR, Missing completly at random: aucencia dada por motivos no asociados a los datos, por ejemplo error de tipeo o perdida por cruces de bases de datos. Se puede imputar usando métodos simples como la media o mediana.

No es recomendable imputar datos MNAR y MAR ya que su ausencia también es un dato. Y sobre los atributos imputados se puede agregar una columna adicional que indique (si 1/no 0) si es un dato imputado o no (dato real).

Tener en cuenta que la decisión de si se debe imputar o no siempre depende del dominio del problema.



## Ejercicio 4

### a

- La hora tiene 24 valores que son ciclicos. Se de hacer una transformación con preoyecciones trigonométrica (por ejemplo seno).
- La fecha la podemos transformar en dia labora o no laboral. Va depender del pais (por los feriados). Queda en valores {0, 1}
- longitud y latitud no hace falta transformarlos ya que pueden ser porcesados por KNN.
- Todas se deben estandarizar para que queden en un intervalo similar y no dominen en el calculo de distancia.

### b

La supocisión de que mucha genete asisitirá al barrio chino en año nuevo nos indica que habrá muchas instancias positivas con la fecha del año nuevo.

Transformaciones:
- Con la latitud y longitud, y la ubicación del barrio chino, podemos calcular la distancia al barrio chino.
- Con la fecha podemos calcular la distancia en tiempo a año nuevo y si es fin de semana o no {1, 0}
- Estandarizamos los atributos de distancia a barrio chino, distancia en fecha a año nuevo y temperatura para que ninguno domine.

### c

Todas son numericas sin característica ciclica y se pueden usar directo en KNN. Pero falta estandarizarlos para que no haya un atributo que domine más que el otro.

Si se usa árboles de decisión, no hace falta hacer ninguna transformación.

### d

Se debe transformar los 3 al no ser numericos:
- color de ojos y pelo en one hot vector
- tamaño puede ser en categorías numericas aprovechando que es un atributo ordinal: chico < mediano < grande. Asignando 0 a chico, 1 mediano y 2 grande. Luego se puede escalar con minmax a 0..1 para que no domine en las distanias de knn.

### e


## Ejercicio 5

### 1

El unico atributo que puede tener faltantes es el salario, y al tener el atributo edad, puede ser MAR. Imputaría primero con la mediana de los salarios o con algún imputador estadistico o predictivo basado en los demás datos y agregaría la columna indicadora. 

Luego convertiría ciudad y provincia con one-hot encoding

Finalmente escalaría los resultados.

El orden final: b -> c -> a


### 2

1000 - 1 + 23 - 1 - 2 = 1023 (dropeando la ultima columna, sino 1025)

### 3

Como dicen que la distribución no es normal, usaría el z-score o minmax.

### 4

Opción b: imputar "-1". Si el corte del árbol ve que maximiza la reducción de entropía en el corte de los que tienen salario vs los que no (-1), entonces podría ser una propiedad intrinseca de los datos.

Opción d: imputar salario medio y agregar columna indicadora. Quizas la media es buena estimación. Con la columna agregamos interpretabilidad.

Descarto opción a: camufla la imputación. Luego no se puede saber si es un dato realmetne es un dato real.

Descarto opción c: idem. Lo unico que esta opción podria dar mejores predicciones del salario pero depende del dominio del problema.
# Guia 2

## Ejercicio 2.1

[notebook-1](../notebooks/notebook_01_herramientas-published.ipynb)

## Ejercicio 2.2

### a

- conjunto de datos: tweets
- métroca: cant clasificados correctamente / total a clasificar
- supervisado

### b

Simplificado a segmentar un objeto en una imagen

- conjunto de datos: set de imagenes a segmentar
- posibles salidas del modelo: 
  - mascara (1 y 0) donde los 1 marcan al objeto buscado
  - coordenadas del objeto (4 puntos (x,y) por ejemplo)
- métroca: comparación de las areas dada por el modelo y el ground truth, comparando superposición.
- supervisado

### c

- conjunto de datos: historiales de movimientos de cuenta bancaria 
- métroca: cantidad correcta clasificados como fraude / cantidad total fraude. Deseable darle más peso a la metrica para los casos falsos negativos (clasificado como NO fraude pero realmente si eran). También para falsos positivos.
- supervisado

### d

- conjunto de datos: conjunto de audios.
- El modelo agrupa (clusters) los audios que son parecidos. 
- métrica: definir alguna distancia que tenga como entrada al audios, por ejemplo usando las frecuencias de onda. Y usarlo para medir que los audios dentro de un mismo cluster tengan poca distancia y elementos entre clusters tengan mucho distancia.
- No supervisado

## Ejercicio 2.3


>  nota: no se puede predecir probabilidades. No se puede hacer un modelo que deuvelva una probabilidad, ya que cuando haya un nuevo dato (observación) la probabilidad debería cambiar pero no se puede definir que tanto deberia cambiar.

### a

- base de datos: historial de gasto de luz de la empresa.
- valores de etiquestas: valor continuo para el total de gasto de luz para el proximo seméstre. 
- tipo de problema: **regresión**

### b

- base de datos: conjunto de tweets 
- valores de etiquestas: Si o No, representado como booleano
- tipo de problema: clasificación

### c

- base de datos: datos de una persona?
- valores de etiquestas: distancia en kilómetros (float) 
- tipo de problema: regresión

### d

- base de datos: historial de gasto de luz de la empresa.
- valores de etiquestas: Si o no (bool)
- tipo de problema: clasificación

### e

- idem al anterior. No se va a predecir la probabilidad, sino si se va a gastar mas o menos de $50k

### f

- base de datos: conjunto de notas de examenes pasados
- valores de etiquestas: valores enteros $0, .., 10$
- tipo de problema: regresion

> argumento de por qué no es clasificación: clasificar erronamento una nota como 1 en vez de 10 es objetivamente peor que confundir un 8 y 9. Los valores de las clases estan relacionados. 
> El problema de dado un texto decir en que idioma está sí es de clasificación ya que los valores no tienen relación (confundir ingles con chino es igual de malo que confundir español con árabe).

### g

Idem a f. (Clasificación con relación entre clases $\rightarrow$  regresión)

### h

- base de datos: Datos de personas
- valores de etiquestas: coordenadas en un mapa.
- tipo de problema: Regresión

### i

- tipo de problema: clasificación.

### j

- valores de etiquestas: multi-etiqueta para pelota, niños, cielo, bicicleta (representados con 0, 1, 2,3 por ejemplo).
- tipo de problema: Clasificación

### d bis

No cambia. Se debe clasificar en gasto > $50k o no. No importa si se tiene etiqetas reales (dato de cuanto se fue gastando en el tiempo por ejemplo) o binarias (si o no hubo gasto > $50k en los meses pasados)

## Ejercicio 2.4

### a

- Espacio de hipotesis: funciones lineales de la forma $ax+b=y$
- parámetros para describir una hipótesis: $a$ y $b$ (pendiente y ordenada al origen)

### b

- Espacio de hipótesis: cualquier par de dos rectas paralelas al eje _y_ o al eje _x_ (o exclusivo) y distintas
- parámetros: un booleano para indicar si las rectas son verticales u horizontales y 2 valores reales correspondientes a las "posiciones" de dichas rectas.

### c

- Espacio de hipótesis: conjunto de 3 elipses 
- parámetros: los parámetros de cada elipse $a_i, b_i, h_i, k_i$ con $i\in\{1,2,3\}$  $$\frac{(x_i-h)^2}{a_i^2} + \frac{(y_i-k_i)^2}{b_i^2} = 1$$ 

## Ejercicio 2.5

[notebook-2](../notebooks/notebook_02_titanic-published.ipynb)

## Ejercicio 2.6

### a

### b

### c

## Ejercicio 2.7

### a

$4*2 = 8$ 

### b

$2^8=256$

### c

$4*2=8$ 

El 4 viene por la cardinalidad de color. Y por cada color tenemos 2 condicionales posibles.

### d

sesgo: par de tamaño y color. Las hipótesis son de la forma:

```python
if (Tamaño == chico and color = Rojo):
    A
else
    B
``` 

o viceversa (cambia la etiqueta A por B)

La cantidad de posibles pares son $4*2*2=16$ la cantidad de pares posibles multiplicado por la clasificación.

### e

Algoritmo: 
1- toma un punto aleatorio, ve su tamaño, color y etiqueta y elige la hipótesis correspondiente.

## Ejercicio 2.8

Formula sorpresa de un evento $x$ con probabilidad $p(x)$: $I(x) = -log_2(p(x))$

Formula entropía: $H(X)= - \sum P(x)log_2(P(x))$

### a

$I(X=5) = -log_2(1/6) \approx 2.58 $

### b

$H(X)= - \sum 1/6 log_2(1/6) = log_2(6) \approx 2.58$

### c

$H(x) = - (1/2 * log_2(1/2) + 5 * 1/10 * log_2(1/10)) \approx 2.16$ 

### d

Ahora hay menos sorpresa (menos incertidumbre). Ya que uno de los valores (el 1) es más probable de que se observe.

### e

$I(X=ninguno) = - log_2(inﬁnitesimal) = \inf$ 

Al tener probabilidad muy baja, si ocurriera dicho evento, se tendría una sorpresa muy grande.

## Ejercicio 2.9

### a

$P(X=A) = 5/12, P(X=B)= 7/12$

$H(X) = - (5/12 log_2(5/12) + 7/12 log_2(7/12)) \approx 0.97986$

### b

$P(X=A|x_1 < 0) = 3/5$

$P(X=B|x_1 < 0) = 2/5$

$P(X=A|x_1 >= 0) = 2/7$

$P(X=A|x_1 >= 0) = 5/7$

$H(X|x_1 < 0) = -(3/5 log_2(3/5) + 2/5 log_2(2/5)) \approx 0.97095$

$H(X|x_1 >= 0) = -(2/7 log_2(2/7) + 5/7 log_2(5/7)) \approx 0.86312$

### c

Entiendo que como se reduce la incertidumbre la probabilidad de equivocarse es menor.

### d

$H(X)= 5/12 * H(X|x_1 < 0) + 7/12 * H(X|x_1 >= 0) = 5/12 * 0.97095 + 7/12 * 0.86312 \approx 0.90804$

Esta entropía es menor, por lo tanto se reduce la incertidumbre.

### e

$H(X|x_2 < 0) = - (1/2 log_2(1/2) + 1/2 log_2(1/2)) = 1$

$H(X|x_2 >= 0) = - (1/3 log_2(1/3) + 2/3 log_2(2/3)) \approx 0.91829$

$H(X)= 6/12 * H(X|x_1 < 0) + 6/12 * H(X|x_2 >= 0) = 1/2 * 1 + 1/2 * 0.91829 \approx 0.95914$

Este corte es peor al tener una incertidumbre mayor.
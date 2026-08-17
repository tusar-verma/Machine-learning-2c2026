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
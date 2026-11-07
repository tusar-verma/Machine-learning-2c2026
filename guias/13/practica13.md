# Guia 13

## Ejercicio 1

### a

![](img/13-1-a.png)

### b

Para la función de activación necesitamos que la salida sea una probabilidad. Un valor entre 0 y 1 donde 1 representa sentimiento positivo y 0 sentimiento negativo. De esta forma también se tiene un grado de certeza en la clasificación. Por lo que la sigmoide parece adecuada.

Para la función de pérdida podemos usar binary cross-entropy que sirve para comaprar distribuciones de probabilidad.

### c

En este caso al ser un problema con una unica salida, solo la salida del tiempo T final se conecta a la función de perdida.

En general, según el problema podríamos tener una entrada a muchas salidas, muchas entradas a una salida, muchos a muchos o muchos a muchos desfasado. Para todas estos problemas se puede agregar la cantidad de entradas y salidas necesarias para que ambas coincidan. Luego fijar en 0 las entradas que no se usen y conectar a la función de perdida las salidas requeridas.

![alt text](img/uno-muchos.png)

Como las salidas intermedias $o_{\langle 1\rangle}, \dots, o_{\langle T-1\rangle}$ no se conectan a la función de pérdida (o se asume que su gradiente/ponderación de pérdida es cero). El gradiente del error en $T$ fluye hacia atrás a través de las conexiones recurrentes de los estados ocultos mediante Backpropagation Through Time (BPTT) sin necesidad de modificar el optimizador

## Ejercicio 2

### a

Uno a muchos. La entrada es una representación de la imagen en un vector / embedding y la salida es una secuencia temporal de palabras. 

### b

Muchos a uno. La entrada es una secuencia temporal de palabras y la salida es un output entre 0 y 1.

### c

Muchos a muchos desfasado. Secuencia temporal de palabras a otra secuencia temporal de palabras, no necesariamente de la misma longitud.

### d

Muchos a muchos desfasado. Secuencia temporal de ventas del ultimo año de entrada (365 dias). La salida es otra secuencia temporal con la serie de ventas diaria (longitud fija de 30 dias).

### e

Uno a muchos. La entrada es la categoría única y la salida una secuencia de sugerencias.

### f

Muchos a muchos desfasada. La corrección podría eliminar o insertar palabras.

### g

Muchos a Muchos. Por cada paso temporal de la entrada se tiene varias neuronas de salida con función de activación softmax, o sigmoide por clase si puede haber solapamiento.

### h (gemini)

Muchos a muchos desfasado. Es una tarea de síntesis de voz donde se mapea una secuencia de texto (caracteres o fonemas) a una secuencia de señales de audio (espectrogramas/muestras acústicas), donde la tasa de muestreo y duración difieren sustancialmente.

## Ejercicio 3

### a

![alt text](img/13-3-a.png)

### b

![alt text](img/13-3-b.png)

### c

![alt text](img/13-3-c.png)

### d

![alt text](img/13-3-d.png)

### e

![alt text](img/13-3-e.png)

## Ejercicio 4
## Ejercicio 5
## Ejercicio 6

## Ejercicio 10
## Ejercicio 11

### a
### b
### c
### d
### e
### f
### g
### h
### i
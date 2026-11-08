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

### a

![alt text](img/13-4-a.png)

### b

![alt text](img/13-4-b.png)

### c

```python

import numpy as np

x = [np.array([1, 2, 0]), np.array([0, 4, 2])]
h = np.array([0.0, 0.0])
U = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
W = np.array([[0.7, 0.8], [0.9, 1.0]])
V = np.array([[1.1, 1.2]])
b_h = np.array([0.1, 0.2])
b_o = np.array([0.3])

for t, x_t in enumerate(x, start=1):
  z = W @ h + U @ x_t + b_h
  h = np.tanh(z)
  o = V @ h + b_o
  print(f"Paso t={t}:")
  print(f"  h_<{t}> = {np.round(h, 5)}")
  print(f"  o_<{t}> = {np.round(o[0], 5)}")


"""
Paso t=1:
  h_<1> = [0.53705 0.92167]
  o_<1> = 1.99676
Paso t=2:
  h_<2> = [0.98931 0.99987]
  o_<2> = 2.58808
"""
```

## Ejercicio 5

### a

![alt text](img/13-5-a.png)

### b

![alt text](img/13-5-b.png)

### c

![alt text](img/13-5-c.png)

### d

Se da un "camino" a cada nodo $h_t$ para que el gradiente viaje directamente sin tener que pasar por toda la cadena. Así las primeras no sufren el gradient vanish.

#### Gemini

### 1. ¿Cómo cambia la arquitectura y la pérdida?

En el planteo original de muchos a uno, la predicción se realiza exclusivamente con el estado final:


$$\hat{y} = \sigma(V h_{\langle T\rangle}) \quad \text{[cite: 55]}$$

En el ejercicio 13.5d, se reemplaza $h_{\langle T\rangle}$ por el **promedio temporal de todos los estados ocultos** $\bar{h}$:


$$\bar{h} = \frac{1}{T} \sum_{t=1}^{T} h_{\langle t\rangle} \quad \text{[cite: 56]}$$

$$\hat{y} = \sigma(V \bar{h}) \quad \text{[cite: 55, 56]}$$

$$J = L(\hat{y}, y) \quad \text{[cite: 55, 56]}$$

---

### 2. ¿Cómo cambia el cálculo del gradiente del costo?

#### Para la matriz de salida $V$

* **Antes:** $\frac{\partial J}{\partial V} = \frac{\partial J}{\partial \hat{y}} \sigma'(V h_{\langle T\rangle}) h_{\langle T\rangle}^\top$.


* **Ahora:** Como la salida se proyecta a partir de $\bar{h}$:



$$\frac{\partial J}{\partial V} = \frac{\partial J}{\partial \hat{y}} \sigma'(V \bar{h}) \bar{h}^\top = \delta_{\text{salida}} \cdot \bar{h}^\top \quad \text{[cite: 55, 56]}$$



---

#### Para los estados ocultos $h_{\langle t\rangle}$

En el caso original, solo el último estado $h_{\langle T\rangle}$ recibía gradiente directo de la pérdida ($L$ solo dependía de $h_{\langle T\rangle}$). Para los pasos anteriores ($t < T$), el gradiente debía viajar forzosamente por la cadena recurrente hacia el pasado.

Ahora, cada estado oculto individual $h_{\langle t\rangle}$ interviene en $J$ a través de **dos caminos distintos**:

1. **Camino directo (a través de $\bar{h}$):**

$$\frac{\partial J}{\partial \bar{h}} = V^\top \delta_{\text{salida}} \implies \frac{\partial J}{\partial h_{\langle t\rangle}}\Bigg\vert{}_{\text{directo}} = \frac{\partial J}{\partial \bar{h}} \frac{\partial \bar{h}}{\partial h_{\langle t\rangle}} = \frac{1}{T} \frac{\partial J}{\partial \bar{h}}$$


2. **Camino recurrente (a través del estado siguiente $h_{\langle t+1\rangle}$):**

$$\frac{\partial J}{\partial h_{\langle t\rangle}}\Bigg\vert{}_{\text{recurrente}} = \frac{\partial J}{\partial h_{\langle t+1\rangle}} \frac{\partial h_{\langle t+1\rangle}}{\partial h_{\langle t\rangle}}$$



Por la regla de la cadena multivariable, el gradiente total acumulado respecto a cualquier estado intermedio $h_{\langle t\rangle}$ es:


$$\frac{\partial J}{\partial h_{\langle t\rangle}} = \underbrace{\frac{1}{T} \frac{\partial J}{\partial \bar{h}}}_{\text{inyección directa sin importar el tiempo } t} + \underbrace{\frac{\partial J}{\partial h_{\langle t+1\rangle}} \operatorname{diag}\big(1 - h_{\langle t+1\rangle}^2\big) W}_{\text{propagación recurrente hacia atrás}} \quad \text{[cite: 5, 56]}$$

---

#### Para la matriz de pesos recurrente $W$

El gradiente respecto a $W$ sigue sumando los aportes de cada paso:


$$\frac{\partial J}{\partial W} = \sum_{t=1}^{T} \frac{\partial J}{\partial z_{\langle t\rangle}} h_{\langle t-1\rangle}^\top \quad \text{[cite: 5, 21]}$$


donde $\frac{\partial J}{\partial z_{\langle t\rangle}} = \operatorname{diag}\big(1 - h_{\langle t\rangle}^2\big) \frac{\partial J}{\partial h_{\langle t\rangle}}$.

La diferencia clave está en el primer paso ($t=1$):

* **Antes:** $\frac{\partial J}{\partial h_{\langle 1\rangle}} = \frac{\partial J}{\partial h_{\langle T\rangle}} \prod_{k=2}^{T} \operatorname{diag}\big(1 - h_{\langle k\rangle}^2\big) W \approx \vec{0}$ (desvanecido por $T-1$ multiplicaciones).


* **Ahora:**

$$\frac{\partial J}{\partial h_{\langle 1\rangle}} = \frac{1}{T} \frac{\partial J}{\partial \bar{h}} + \text{(términos recurrentes atenuados)}$$



---

### 3. Ventajas y justificación ("¿Por qué?")

* **Mitiga el *vanishing gradient* para las primeras palabras:** El término directo $\frac{1}{T} \frac{\partial J}{\partial \bar{h}}$ no depende de la distancia temporal ni pasa por ninguna multiplicación sucesiva por $W$ o por derivadas de la $\tanh$. El gradiente del error llega de forma directa e intacta a $h_{\langle 1\rangle}$, garantizando que la red pueda ajustar $W$ y $U$ para aprender de las palabras del inicio del texto.


* **Actúa como una "autopista de gradiente" (similar a una conexión residual):** En lugar de forzar a que toda la señal de error viaje secuencialmente a través de $T$ pasos hacia atrás, el promedio distribuye el gradiente en paralelo sobre todos los instantes temporales en un solo salto.


* **Mayor robustez en reseñas largas:** El modelo deja de depender exclusivamente de que el vector $h_{\langle T\rangle}$ tenga que retener y comprimir toda la información de la frase sin olvidar el pasado, permitiendo que palabras clave ubicadas al principio (como un "no" o "nunca") tengan un impacto directo y no disipado en la clasificación.

### e

Se cambia la salida (output):

- Enfoque de Clasificación Multiclase: $C = 5$ neuronas en la capa de salida con función de activación Softmax y pérdida de Entropía Cruzada Categórica.   
- Enfoque de Regresión: $C = 1$ neurona con activación lineal (o sigmoidea reescalada al intervalo $[1, 5]$) optimizada con Error Cuadrático Medio (MSE).

## Ejercicio 6

### a

$$h_{\langle t\rangle}^{(k)} = g_1\big(W_1 h_{\langle t-1\rangle}^{(k)} + W_4 o_{\langle t-1\rangle}^{(k)} + W_3 x_{\langle t\rangle}^{(k)}\big)$$

$$o_{\langle t\rangle}^{(k)} = g_2(W_2 h_{\langle t\rangle}^{(k)})$$

### b

Frente a realimentar solo la salida ($W_1 = 0$):
- Ventajas: El estado oculto $h$ es un espacio latente de alta dimensionalidad no restringido por las clases de salida. Permite retener representaciones complejas y dinámicas internas ricas que un vector de salida colapsado no puede encapsular. 
(Se restringe la "memoria" al espacio de salida que en general es reducido).

Frente a realimentar solo el estado oculto ($W_4 = 0$):
- Ventajas: Le brinda al modelo conocimiento directo y explícito del token o decisión previamente discretizada/emitida.  
- Desventajas: Introduce mayor cantidad de parámetros, expone al modelo a exposure bias si predice incorrectamente en tiempo de inferencia, y dificulta la paralelización del cómputo recurrente. 
(A coste de más paramétros, pones a disposición del aprendizaje del tiempo t+1 la decisión tomada del tiempo t)


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
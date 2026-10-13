# Guia 10

## Ejercicio 1

1. Falso: promedio de las instancias de las hojas como valor predicho.
2. Verdadero
3. Verdadero
4. 

## Ejercicio 2

### a

Si la regresion fuera perfecta, se espera que los puntos sigan una diagonal

### b

En orden de las imagenes: regresion lineal, knn y decision tree

- Como decision tree tiene altura maxima 3, solo puede asignar 8 valores posibles.
- Regresión lineal se espera que los puntos sigan la forma de una linea
- knn con 1 solo vecino, asigna como predicción el valor exacto del vecino más cercano del conjunto de entrenamiento. Por eso las predicciones están estrictamente acotadas en $[0, 5]$

### c

No se puede obtener info de los atributos solo viendo los gráficos.

## Ejercicio 3

Ambos graficos corresponden a solo 2 atributos del dataset y la intensidad del color del punto corresponde al precio. 
Se podría graficar en $R^3$ con ejes latitud, longitud y precio. Sobre este espacio, la regresión lineal se vería como un plano inclinado. (más alto para los puntos de la superficie que corresponde a precios altos). Si luego proyectamos este plano a los ejes originales, obtendremos rectas paralelas (asignandole color por altura) con degradado (colores oscuros, valores altos).

## Ejercicio 4

1. Cambia el valor de los pesos. Las predicciones no cambian por que se ajustan los pesos justamente por esa normalización.
Por otro lado, la normalización hace que gradient descent converga más rápido. Las curvas de nivel dejan de ser curvas alargadas y pasan a ser circulares: $\hat{y} = w_0 x_0 + w_1 x_1$. Si $x_1$ tiene escala de los 1000 y $x_2$ del orden 1, si variamos 0.1 el peso, en $x_1$ el valor cambia mucho. En cambio, si lo hacemos en $x_2$, cambia poco. 
Ahora al normalizar, ambos atributos se encuentran en los mismos rangos y no tenemos dicho fenómeno.
2. Ridge penaliza los pesos con valores altos. Si los atributos no están normalizados y se tiene un atributo $x_1$ en el orden de los 10000 y otro $x_2$ en el orden de los 10 y ambos tienen la misma importancia en la predicción del valor, se esperaría que el peso asignado a $x_1$ sea chico y el de $x_2$ sea grande. La regularización Ridge entonces penalizará a $x_2$ solo por su rango de valores. Esto no es deseable.
Ahora si usamos regularización Ridge con los datos normalizados penalizaremos valores grandes de pesos independientemente del rango de valores.
Queremos regularizar para restringir el tamaño de los pesos para evitar el sobreajuste y reducir sensibilidad por ruido (si los pesos son altos, los errores de entrenamiento son 0, pero instancias parecidas a las de entrenamiento dan valores muy variados: predicción de precio por tamaño en $m^2$, $w_1=20000$ con $100m^2$ da $2M, con $101m^2$ da $2020000. Con pesos grandes, se tiene resultados inestables).
3. Al aplicar normalización y luego entrenar, cambian los pesos pero no las predicciones, por lo que el MSE final no debería cambiar.
$$X_{\text{norm}} = X A + B$$
$$\hat{y} = X w + w_0$$
$$\hat{y}_{\text{norm}} = X_{\text{norm}} w^* + w_0^* = (X A + B) w^* + w_0^* = X (A w^*) + (B w^* + w_0^*)$$
4. No afecta, se podría usar el mismo modelo y sus predicciones escalarlos por 1000 (en este ejemplo). Pero esto si se conoce el factor de escalado del target
$$\text{MSE}_{\text{nuevo}} = \frac{1}{n} \sum_{i=1}^n (c \cdot \hat{y}^{(i)} - c \cdot y^{(i)})^2 = c^2 \cdot \text{MSE}_{\text{original}}$$
Como multiplicar por una constante positiva $c^2 > 0$ no cambia la ubicación del punto mínimo (es decir, el argumento que minimiza $c^2 \cdot f(w)$ sigue siendo el mismo óptimo reescalado), las decisiones relativas que toma el modelo no se ven afectadas. Cada punto conserva exactamente el mismo peso e importancia relativa en la optimización. 

La normalización no cambia el MSE sin regularización ya que el el algoritmo compensa este cambio de escala. Ahora si usamos regularización y comparamos los MSE (sin el termino de regularizacion) normalizado contra no normalizado, si cambia el MSE por que justamente la regularización cambia los pesos, y por lo tanto las predicciones

## Ejercicio 5

### 1
```python
def descenso_gradiente(g: [float] -> float, gd: [float] -> [float], z_init: [float], alpha: float, num_iter: int, tol: float):
    """
    Descenso de gradiente para minimizar g.
    Args:
    g: La función a optimizar.
    dg: El gradiente de la función.
    z_init: Valor inicial.
    alpha: El “learning rate”.
    num_iterations: Máx iteraciones.
    tol: Tolerancia para la convergencia.
    """
    z = z_init
    for i in range(num_iter):
        gradiente = gd(z)
        z_new = z - alpha * gradiente
        if (abs(g(z_new) - g(z)) < tol):
            break
        
        z = z_new

    return z
```

### 2
Para regresión lineal se usa para conseguir el vector de pesos que minimice el MSE.
$$\text{MSE}_{X,y}(w) = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{h}_w(x^{(i)}) - y^{(i)}\right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left(w_0 + \sum_{j=1}^p w_j x_j^{(i)} - y^{(i)}\right)^2$$

### 3
Se calculan las derivadas parciales con respecto a cada peso de la función MSE.
$$\frac{\partial}{\partial w_j} \text{MSE}(w) = \frac{\partial}{\partial w_j} \left( \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2 \right)$$
$$\nabla_w \text{MSE}_{X,y}(w) = \frac{2}{n} \sum_{i=1}^n (\hat{h}_w(x^{(i)}) - y^{(i)}) x^{(i)} = \frac{2}{n} X^T (Xw - y)$$

### 4

```python
def grad_desc_mini_batch(X_train, y_train, alpha, z_init, g, gd):
    z = z_init
    while (no se cumpla criterio de detención):
        # epoch
        while (haya datos de entrenamiento para hacer batch):
            X_batch, y_batch = Muestrear_batch_sin_reposicion(X_train, y_train)
            grad_batch = gd(X_batch, y_batch, z)

            z = z - alpha * grad_batch
    
    return z

```

Para regresión lineal g es el MSE y gd es el gradiente con respecto al batch particular que se tiene en la iteración. Notar en el punto 3 que la fórmula del gradiente del MSE depende del conjunto de datos que se use. Es decir, tanto el MSE y su gradiente dependen de los datos que se usen para calcularlo. Y en cada batch entonces se tiene una superficie distinta en el que se calcula el proximo movimiento de z. 
Se espera que los batchs sean parecidos a los datos completos en distribución y acerquen al mínimo real. Además como es un subconjunto, se puede tener la ventaja de resolver problemas de divergencia o saltearse minimos locales.

Al cambiar el lote en cada iteración, la función de pérdida y el gradiente estimado cambian ligeramente de forma. Ese "ruido estocástico" introduce oscilaciones que ayudan a escapar de mínimos locales o puntos de ensilladura en superficies no convexas.   Eficiencia computacional: Permite entrenar con conjuntos de datos masivos que no caben enteros en memoria RAM. 

## Ejercicio 6

### A

En ambos los pesos $w_1, \dots, w_p$ se settean en 0. Luego veamos que $w_0$ toma el valor de la media de las predicciones: de la formula del MSE tenemos que queremos minimizar
$$f(w_0) = \frac{1}{n} \sum_{i=1}^n (w_0 - y^{(i)})^2$$ 

Derivamos respecto a $w_0$ e igualamos a 0:Aplicando la regla de la cadena:

$$\frac{df}{dw_0} = \frac{2}{n} \sum_{i=1}^n (w_0 - y^{(i)}) = 0$$

Despejamos $w_0$:

Distribuimos la sumatoria:

$$\sum_{i=1}^n w_0 - \sum_{i=1}^n y^{(i)} = 0$$

Como $w_0$ es una constante sumada $n$ veces, $\sum_{i=1}^n w_0 = n \cdot w_0$:

$$n \cdot w_0 = \sum_{i=1}^n y^{(i)} \implies w_0 = \frac{1}{n} \sum_{i=1}^n y^{(i)} = \bar{y}$$

### B

Como en la regularización no e toma en cuenta el valor de $w_0$ pudimos hacer el analisis del punto a. Ahora en este punto nos dicen que también lo  penalizamos, por lo que cuando $\lambda$ tienda a infinito, el algoritmo setteará $w_0$ también en 0 para minimizar la función de costo. El modelo entocnes siempre predecirá 0.  

### C

Recordemos que minimizar la función de costo regularizada es equivalente a resolver un problema de optimización con restricciones:

* **Problema 1 (Sin penalizar $w_0$):**

$$\min_{w} \text{SSE}(w) \quad \text{sujeto a} \quad \sum_{j=1}^p \vert{}w_j\vert{}^q \le s$$

*(donde $q=1$ para Lasso y $q=2$ para Ridge, y $s$ es una constante asociada a $\lambda$)*.

Llamemos al conjunto de soluciones permitidas:

$$S_1 = \left\{ (w_0, w_1, \dots, w_p) \in \mathbb{R}^{p+1} : \sum_{j=1}^p \vert{}w_j\vert{}^q \le s \right\}$$

*(aquí $w_0$ puede tomar cualquier valor real libremente)*.

* **Problema 2 (Penalizando también $w_0$):**

$$\min_{w} \text{SSE}(w) \quad \text{sujeto a} \quad \vert{}w_0\vert{}^q + \sum_{j=1}^p \vert{}w_j\vert{}^q \le s$$

Llamemos al nuevo conjunto de soluciones permitidas:

$$S_2 = \left\{ (w_0, w_1, \dots, w_p) \in \mathbb{R}^{p+1} : \vert{}w_0\vert{}^q + \sum_{j=1}^p \vert{}w_j\vert{}^q \le s \right\}$$

* **Relación entre ambos conjuntos:**
Como $\vert{}w_0\vert{}^q \ge 0$, cualquier vector que cumpla la condición de $S_2$ automáticamente cumple la de $S_1$:

$$\sum_{j=1}^p \vert{}w_j\vert{}^q \le \vert{}w_0\vert{}^q + \sum_{j=1}^p \vert{}w_j\vert{}^q \le s \implies S_2 \subseteq S_1$$

* **Conclusión **
Si minimizamos la misma función objetivo $\text{SSE}(w)$ en un conjunto $S_1$ y en un subconjunto $S_2 \subseteq S_1$, el mínimo sobre el subconjunto siempre es mayor o igual:

$$\min_{w \in S_2} \text{SSE}(w) \ge \min_{w \in S_1} \text{SSE}(w)$$

Por lo tanto, la suma de errores al cuadrado penalizando la constante es necesariamente **mayor o igual** que sin penalizarla.

## Ejercicio 7



## Ejercicio 8
## Ejercicio 9
## Ejercicio 10
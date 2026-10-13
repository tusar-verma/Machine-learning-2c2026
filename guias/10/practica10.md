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
## Ejercicio 6
## Ejercicio 7
## Ejercicio 8
## Ejercicio 9
## Ejercicio 10
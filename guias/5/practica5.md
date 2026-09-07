# Guia 5

## Ejercicio 1

### I

Dado un problema de clasificación con $k$ clases, clasifica una instancia $x$ viendo cual es la probabilidad de que dicha instancia pertenezca a una clase. La clasificación corresponde a la clase con mayor probabilidad: $c = argmax_{c \in K} P(Y=c | X=x)$

### II

En la practica no se conoce las distribuciones de $P(Y=c | X=x)$.

### III

$P(Y=verde) = 0.2$
$P(Y=azul) = 1 - P(Y=verde) = 0.8$

$f_{verde}(x) = X|Y = Verde ∼ N (0, 2), f_{azul}(x) = X|Y = Azul ∼ N (3, 1)$

$P(Y=Verde | X=2) = \frac{P(Y=verde) f_{verde}(2) }{ P(Y=verde) f_{azul}(2) * P(Y=azul) f_{verde}(2) }$

$P(Y=azul | X=2) = \frac{P(Y=azul) f_{azul}(2) }{ P(Y=verde) f_{azul}(2) * P(Y=azul) f_{verde}(2) }$

Y debemos quedarnos con el maximo. Como ambos denominadores son iguales, basta con ver el mayor numerador:

$c = argmax_{c\in \{verde, azul\}}(P(Y=verde) f_{verde}(2), P(Y=azul) f_{azul}(2)) \approx argmax(0.2 * 0.12098, 0.8 * 0.2419) = max(0.0242, 0.1935) = azul$

[Calculadora función densidad de normal](https://www.statskingdom.com/normal-distribution-calculator.html)

### IV

Una clasificación en este contexto:

$\hat{y}(x_1, x_2) = argmax_c (P(Y=c)* f_c(x_1, x_2))$

La frontera de decisión es un punto en el espacio donde hay 2 o más clases que "empatan": $P(Y=c_1)* f_{c_1}(x_1, x_2) = P(Y=c_2)* f_{c_2}(x_1, x_2)$ para $c_1, c_2$ clases distintas.

Podriamos decir que la frontera entre dos clases esta dada por $g_{c_1, c_2}(x_1, x_2) = 0$:

$g_{c_1, c_2}(x_1, x_2) = P(Y=c_1)* f_{c_1}(x_1, x_2) - P(Y=c_2)* f_{c_2}(x_1, x_2) = 0$

## Ejercicio 3

### a

Si, no veo por que no. Se tiene solo 2 atributos. Los problemas de KNN surgen con muchos atributos o con atributos en unidades de medidas distintos

### b

Por lo mencionado en el punto a, se debe estandarizar los datos de ambos atributos (restar la media y dividir por el desvio estandar). Así se "desplaza" los puntos a una normal (0,1) y la distancia no sufre por problemas de diferencias entre las unidades de medidas diferentes entre atributos.

## Ejercicio 4

### a

Alcanza por si sola. 

$P(Y=0 | X = x^{(t)}) = 1 - P(Y=1 | X = x^{(t)})$

Y estas 2 probabilidades son justamente las que necesitamos para el clasificador.

### b

No alcanza. No es util solo tener esta información. En la formula del argmax justamente lo ignoramos ya que no participa (no depende de la clase).

### c

No es suficiente información. Nos falta la prior.

### d

Idem a c, pero aca tenemos un poco más de info (ambas likelihoods) pero nos faltan las priors.

### e

Con $P(Y=0) = 1 - P(Y=1)$ tengo ambas priors, pero me falta uno de los likelihoods para poder armar el clasificador.

### f

Faltan los likelihoods.

## Ejercicio 5

Hecho en clase

## Ejercicio 6 

SVM con un kernel lineal fallaría por que no hay separación lineal posible.

Con algun kernel polinomico se puede resolver. (Aprovechando la ecuación de un ciruclo $x_1^2 + x_2^2 = r$)

## Ejercicio 7

El primer sesgo corresponde a la independencia condicional de los atributos (dada la clase $Y=c$, los atributos son independientes), dando en el paso (5) la productoria del likelihood.

El segundo sesgo, en el paso 7, es que las distribuciones de X | Y = c (cada atributo condicionado a una clase $c$) sigue una distribución normal con cierta media y desvio estandar.

## Ejercicio 8

### 1

### 2

### 3

### 4

### 5
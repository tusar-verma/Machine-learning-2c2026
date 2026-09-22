# Guia 8

## Ejercicio 1


**Bagging** (Bootstrap Aggregating) es un algoritmo de ensamble diseñado para reducir la varianza de estimadores inestables (como árboles de decisión no podados) sin perjudicar sustancialmente su sesgo.

**Funcionamiento:**

1. **Bootstrap:** Dado un dataset original $D$ de tamaño $N$, se generan $B$ subconjuntos de entrenamiento independientes $D_1, D_2, \dots, D_B$. Cada $D_b$ se construye muestreando $N$ instancias de $D$ de forma aleatoria **con reemplazo**.
2. **Entrenamiento en paralelo:** Se entrena un modelo base $h_b$ sobre cada muestra $D_b$ sin modificar el algoritmo base.
3. **Agregación:**
* **Regresión:** Se promedian las predicciones continuas:

$$\hat{h}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B h_b(x)$$


* **Clasificación:** Se realiza una votación mayoritaria (*hard voting*) o se promedian las probabilidades de clase estimadas (*soft voting*):

$$\hat{h}_{\text{bag}}(x) = \operatorname{argmax}_{c} \sum_{b=1}^B \mathbb{I}(h_b(x) = c)$$

## Ejercicio 2


## Ejercicio 3

Como se dijo en el punto 1, bagging para el problema de clasificación toma la votación entre los modelos entrenamos. Sea B la cantidad de modelos entrenados, $x_1, x_2, x_3, x_4$ tal que $[x_1/B, x_2/B, x_3/B, x_4/B] = [0.75, 0.50, 0.25, 0.75, 1.0]$. 

Reescribiendo las probabilidades en fracciones: $[3/4, 2/4, 1/4, 3/4, 4/4]$ 

El minimo B es 4, y realmente se podría tener cualquier número múltiplo de 4. 


## Ejercicio 4

### a

Verdadero.

### b

Verdadero, es bagging + acotación de atributos en cada iteracion de corte (en cada nodo).

### c

Falso. Se utiliza todos los atributos, pero en cada nodo se define un subconjunto para buscar mejor corte

### d

Falso. m es el hiperparámetro para configurar el tamaño del subconjunto de atributos que se considera en cada nodo.

### e

Falso. No es el mismo atributo en todo el arbol, sino que en cada nodo se considera un solo atributo al azar.

### f
### g


### h

Falso. La justificación de random forest es que en bagging al usar siempre todos los atributos, es muy probable que se generen los mismos árboles ya ue la importancia de los atributos es la misma. Por lo tanto la correlación entre los modelos será alta y no se reducirá tanto la varianza.
Random forest al seleccionar atributos al azar para cada nodo aumenta los posibles árboles que puede generar no deterministicamente. Disminuye la correlación y por lo tanto la varianza más que bagging.

### i
### j

## Ejercicio 5
## Ejercicio 6
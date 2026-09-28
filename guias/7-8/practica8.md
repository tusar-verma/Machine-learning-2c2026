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

El algoritmo consiste en:
1. Extraer de manera aleatoria con reposición un conjunto de datos de los datos de entrenamiento (bootstrap).
2. Entrenar un árbol sobre dicho bootstrap sin poda. En cada noodo del árbol se elige aleatoriamente un subconjunto $$
F << M$ de las $M$ features. Luego se elije el mejor corte en dicho nodo sobre ese subconjunto.
3. Se repite hasta obtener N árboles (el bosque completo).
4. Para realizar una predicción:
   1. Si es un problema de clasificación, se genera una votación entre todos los árboles
   2. Si es un problema de regresión, se pide la predicción a cada árbol y se agrega el resultado de alguna manera (promedio por ejemplo).

A diferencia de bagging, cada árbol se entrena sobre un subconjunto $F$ de atributos elegidos de manera aleatoria. Esto con el fin de reducir el "parecido" de los árboles generados. En bagging, si se tiene un atributo que es el más importante, todos los árboles elegirán dicho atributo para hacer el corte. Reduciendo las posibilidades de atributos a elegir, garantizamos que cada árbol sea distinto.

El hiperparámetro $F$ es fundamental. Si es muy chico, entonces es poco probable que en cada árbol se asigne los features realmente importantes para el problema, y los árboles obtenidos darán predicciones malas. En cambio, si $F$ es muy grande estaremos más cerca de la situación del algoritmo de bagging, donde los árboles serán muy parecidos al ser muy probable elegir las features importantes.

Una analogía buena es pensar en una votación. Si se tiene a varias personas que saben mucho del tema, es muy probable que la decisión votada sea muy buena. En cambio si se tiene varias personas que saben poco del tema, votarán cada uno una cosa distinta y la predicción será mala. Tener muchos votantes que no están correlacionados permitirá "corregir" errores: si se tiene 100 votantes donde 5 votaron mal, se compensará con las votaciones de los demás votantes.
Y maedida que aumentamos la cantidad de votantes, se estabiliza el resultado final (baja la variabilidad).

Por todo esto es deseable tener un valor $F$ que sea suficientemente bajo para que los árboles esten poco correlacionados y suficientemente alto para que realmente puedan aprender las features importantes. Y si no lo hacen, que la cantidad de árboles N sea lo suficientemente grande para que la votación lleve a la respuesta correcta.

Como los árboles se entrenan usando un bootstrap de los datos de entrenamiento, se tiene un subconjunto de ellos que no se utiliza para entrenar llamado out-of-bag (OOB). Que se estima es el $36.8\%$ de los datos, y podemos utilizarlos como conjunto de validación del modelo.

> P(estar en bootstrap) = 1 - P(no estar en bootsrap)
> P(no estar en bootsrap) = P(no ser elegido para bootstrap)$^n$
> P(no ser elegido para bootstrap) = 1 - P(ser elegido para bootstrap)
> P(ser elegido para bootstrap) = $\frac{1}{n}$ siendo n el tamaño de los datos de entrenamiento y la elección equiprobable. 
> Luego P(estar en bootstrap) = $1 - (1 - \frac{1}{n})^n = 1 - 1/e \approx 0.632$ cuando $n \rightarrow \infty$

Breiman propone medir la importancia de una feature viendo que tanto se modifica el error cuando se cambia los valores de dicha feature: Si se quiere calcular la importancia de la feature $m$
1. Para cada árbol se toma el OOB
2. Se permutan los valores de $m$ del OOB
3. Se registra el incremento porcentual del error respecto de las variables sin la modificación.
4. el promedio del incremento entre todos los árboles indica el nivel de importancia de la feature $m$.

En sklearn la importancia de una feature mide la disminución promedio de la impureza. 
Para cada árbol:
   1. calculo la reducción de impureza dada por cada feature X
      1. Recorro el arbol viendo que nodos usaron la feature X para el corte.
      2. Sobre ellos calculo la impureza antes del corte (ponderada por la cantidad de instancias en el nodo) $I_n$, y la cantidad de impureza en cada nodo hijo ponderada por la cantidad de instancias que se envió a cada hijo $I_1, I_2$
      3. La disminución de la impureza en el nodo por el atributo X es $I_n - (I_1 + I_2)$
      4. Sumo todos los decrementos de impuerza en todos los nodos que participó X. Lo divido por la suma de todas disminuciones de impuerzas de todos los atributos para normalizarlo a un valor entre 0 y 1.
   2. Promedio la reducción de impuerza de la feature X de cada árbol. Obteniendo así la del bosque.

Notar entonces que la importancia que usa la reducción de impureza (estrategia de sklearn) siempre es sobre los datos de entrenamiento. En cambio, el propuesto por Breiman son sobre los datos de validación.
Podemos decir entonces que la primera sobreestima la importancia de las features. Más aún (sobreestima más fuertemente) si la feature es de valor continuo o categórica con k categorías. Este tipo de atributos tiene un espacio de corte más grande, y por lo tanto una probabildad más grande de ser elegida.



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
Falso. MDI hace dicha suma pero como es un promedio, se tiene que dividir sobre la cantidad de árboles. Siendo B la cantidad de bootstraps generados entonces se tiene B árboles entrenados (uno para cada bootstrap).

### g
Verdadero. Justificación en f.

### h

Falso. La justificación de random forest es que en bagging al usar siempre todos los atributos, es muy probable que se generen los mismos árboles ya ue la importancia de los atributos es la misma. Por lo tanto la correlación entre los modelos será alta y no se reducirá tanto la varianza.
Random forest al seleccionar atributos al azar para cada nodo aumenta los posibles árboles que puede generar no deterministicamente. Disminuye la correlación y por lo tanto la varianza más que bagging.

### i
Falso.

### j
Verdadero.  

Se puede estiamr el error de generalización con los datos OOB. Cada dato del training puede no haber participado en varios bootstraps. Podemos tomar dichos árboles que no fueron entrenados con ese dato y calcular un error de estimación:
1. Identificación de árboles OOB: Para cada observación $(x_i, y_i)$ del conjunto de entrenamiento original, se buscan únicamente los árboles que no incluyeron a esa observación en su muestra de entrenamiento (aproximadamente un tercio de los árboles del bosque, $\approx 36.8\%$).
2. Votación: Solo ese subconjunto de árboles evalúa la observación $x_i$ y emite su voto (o su valor predicho en regresión).   
3. Predicción agregada OOB: Se determina la clase ganadora por mayoría simple (o el promedio en regresión) entre los votos de ese subconjunto.  
4.  Cálculo del error OOB: Se compara la predicción con su etiqueta real $y_i$, y se calcula el promedio de error de todo el dataset de entrenamiento. Esto es, el error Out-Of-Bag.
   
Como en las votaciones de cada instancia estamos usando menos árboles que el ensamble completo (aproximadamente un tercio de los árboles) entonces se tiende a sobre-estimar el error de generalización. (Justificado sobre la construcción del ensamble basado en baja correlación y alta cantidad de árboles, produce buenas predicciones).


## Ejercicio 5
## Ejercicio 6
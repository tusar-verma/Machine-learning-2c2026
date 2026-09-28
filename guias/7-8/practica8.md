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

Ada-boast es un meta-algoritmo que entrena estimadores iterativamente. Cada nuevo modelo se especializa en los errores cometidos por los modelos anteriores. Para esto, al finalizar una iteración y calcular la performance, se re-calculan los pesos que se le asigna a los datos de entrenamiento dandole más peso a aquellas en las que se falló la predicción.
Al finalizar, la predicción global se construye mediante una combinación lineal ponderada de los votos de todos los estimadores en función de su precisión individual.

Es un meta-algoritmo por que el estimador puede ser cualquier algoritmo (LDA, SVM, árboles, etc).

### a

Los weak learners son estimadores muy sesgados. No tienen capacidad para aprender los patrones de muchos datos. La idea es acumular varios de estos, donde cada uno aprende los errores del otro y así en el ensamble, las predicciones se fortalecen (en la votación, se pondera asignandole mayor peso a aquellos learners que mejor les fue).
El fundamento teórico de Boosting (demostrado por Schapire y Freund) radica en que una combinación ponderada de clasificadores débiles puede converger a un clasificador fuerte (strong learner) con un error de generalización arbitrariamente bajo.

### b

En pocas palabras: inicialmente se asigna peso uniforme a cada instancia. Se entrena un modelo, se calcula el error obtenido y se recalculan los pesos en base a ese error. Si la instancia se clasificó correctamente, disminuye su peso. Si se clasificó incorrectamente aumenta su peso. Además se le asigna un peso (o importancia) al modelo obtenido en base a dicho error. Si es alto, la importancia del modelo se acerca a 0. Una vez recalculado los pesos, se vuelve a entrenar.

Gemini:

Dado un conjunto de entrenamiento con $N$ observaciones $(x_1, y_1), \dots, (x_N, y_N)$ con etiquetas $y_i \in \{-1, +1\}$:
- Inicialización: Se asigna un peso uniforme a todas las instancias en la iteración $t = 1$:
$$w_1(i) = \frac{1}{N} \quad \forall i \in \{1, \dots, N\}$$

- Cálculo del error ponderado: Con los pesos de la ronda $t$, se entrena el modelo base $h_t(x)$ y se mide su tasa de error ponderada $\epsilon_t$:

$$\epsilon_t = \sum_{i: y_i \ne h_t(x_i)} w_t(i)$$

- Cálculo del peso del modelo ($\alpha_t$): Se determina la importancia o confianza que tendrá el clasificador $h_t$ en la decisión final:

$$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$

Si $\epsilon_t$ es bajo, $\alpha_t$ es un valor positivo grande; si $\epsilon_t \to 0.5$, $\alpha_t \to 0$.

- Actualización de pesos de las instancias: Los pesos para la siguiente iteración $t+1$ se recalculan aplicando un factor multiplicativo según el acierto o desacierto:

$$w_{t+1}(i) = \frac{w_t(i) \exp\left( -\alpha_t \, y_i \, h_t(x_i) \right)}{Z_t}$$

donde $Z_t$ es la constante de normalización $\sum_{i=1}^N w_t(i) \exp(-\alpha_t y_i h_t(x_i))$ que garantiza $\sum_{i=1}^N w_{t+1}(i) = 1$:
    - Instancia bien clasificada ($y_i = h_t(x_i)$): El exponente es $-\alpha_t$, por lo que su peso disminuye. 
    - Instancia mal clasificada ($y_i \ne h_t(x_i)$): El exponente es $+\alpha_t$, por lo que su peso aumenta, obligando al siguiente clasificador a priorizar su resolución correcta

### c

En pocas palabras: los criterios de corte se corresponden a detección de sobreajuste sobre un conjunto de control; si la condición de weak learner se rompe (los modelos no aprender nada nuevo (cometen los mismos errores y los pesos no se ajustan) o son peores que el azar); alcanzado un máximo T que se setea como hiperparámetro (para evitar ajuste a ruido). 

Gemini:

El número total de iteraciones/modelos $T$ se determina mediante los siguientes criterios prácticos y analíticos:

- Parada por convergencia anticipada (Early Stopping): Se monitorea el error sobre un conjunto de validación independiente (o validación cruzada). Si el error de validación deja de descender durante un número determinado de rondas y comienza a incrementarse, se detiene el entrenamiento para evitar el sobreajuste.
- Criterio de corte algorítmico interno:
  - Fallo del clasificador base: Si en alguna ronda un modelo base obtiene un error $\epsilon_t \ge 0.5$, el clasificador ya no cumple la condición de weak learner y el algoritmo se interrumpe de inmediato. (si $\epsilon_t > 0.5$ el modelo se está equivocando más de lo que está acertando; y si $\epsilon_t = 0.5$ La fracción dentro del logaritmo es $\frac{0.5}{0.5} = 1$. Como $\ln(1) = 0$, resulta $\alpha_t = 0$. El modelo recibe un peso de voto nulo en la decisión final; no aporta ninguna información y el algoritmo se estanca).
  - Ajuste perfecto: Si un clasificador alcanza $\epsilon_t = 0$, su peso $\alpha_t$ tiende a infinito y el ciclo se detiene por haber resuelto el conjunto de entrenamiento. (AdaBoost necesita identificar qué instancias se equivocaron para subirles el peso y enfocarse en ellas en la siguiente ronda. Si no hubo ninguna equivocación, no hay errores residuales que corregir ni pesos que reasignar. El ensamble ya resolvió el entrenamiento al 100%, por lo que la iteración concluye de inmediato)
- Control de sobreajuste ante ruido: Aunque en problemas sin ruido AdaBoost amplía los márgenes de separación sin sobreajustar fácilmente, en presencia de ruido o datos atípicos (outliers), un $T$ excesivo concentra pesos desproporcionados en etiquetas espurias y degrada fuertemente el rendimiento. En tales escenarios, $T$ debe regularse como un hiperparámetro acotado. 
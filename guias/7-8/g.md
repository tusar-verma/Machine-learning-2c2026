## 7. Sesgo y Varianza

### Ejercicio 7.1

#### I) Métrica de error en regresión

Para la descomposición clásica de sesgo y varianza se utiliza la función de pérdida cuadrática o **error cuadrático**:


$$error(y, \hat{h}(x)) = (y - \hat{h}(x))^2$$

---

#### II) Demostración de la descomposición

Supongamos el modelo generador estándar de regresión:


$$y = f(x) + \epsilon$$

Donde:

* $f(x) = \mathbb{E}[y \mid x]$ es la función determinista real que describe la relación entre las variables.
* $\epsilon$ es el ruido irreducible con $\mathbb{E}[\epsilon] = 0$ y varianza constante $\operatorname{Var}(\epsilon) = \mathbb{E}[\epsilon^2] = \sigma^2$.
* El ruido $\epsilon$ del punto de evaluación es estadísticamente independiente del conjunto de entrenamiento $D_n$.
* $\hat{h}_{D_n}(x)$ es el estimador entrenado sobre una muestra aleatoria $D_n$. Su valor esperado sobre todas las posibles muestras $D_n$ de tamaño $n$ se denota como:

$$\bar{h}(x) = \mathbb{E}_{D_n}[\hat{h}_{D_n}(x)]$$



Para un punto fijo $x$, el error cuadrático esperado promediando sobre los conjuntos de entrenamiento $D_n$ y el ruido $\epsilon$ es:


$$\mathbb{E}_{D_n, \epsilon} \left[ (y - \hat{h}_{D_n}(x))^2 \right]$$

Sumamos y restamos $\bar{h}(x)$ dentro del término cuadrático:


$$y - \hat{h}_{D_n}(x) = (y - \bar{h}(x)) + (\bar{h}(x) - \hat{h}_{D_n}(x))$$

Elevando al cuadrado:


$$(y - \hat{h}_{D_n}(x))^2 = (y - \bar{h}(x))^2 + (\bar{h}(x) - \hat{h}_{D_n}(x))^2 + 2(y - \bar{h}(x))(\bar{h}(x) - \hat{h}_{D_n}(x))$$

Aplicamos la esperanza $\mathbb{E}_{D_n}[\cdot]$:

* **Término cruzado:**

$$\mathbb{E}_{D_n} \left[ 2(y - \bar{h}(x))(\bar{h}(x) - \hat{h}_{D_n}(x)) \right] = 2(y - \bar{h}(x)) \left( \bar{h}(x) - \mathbb{E}_{D_n}[\hat{h}_{D_n}(x)] \right)$$



Dado que $\mathbb{E}_{D_n}[\hat{h}_{D_n}(x)] = \bar{h}(x)$, este término es exactamente 0.
* **Segundo término:**

$$\mathbb{E}_{D_n} \left[ (\hat{h}_{D_n}(x) - \bar{h}(x))^2 \right] = \operatorname{Var}_{D_n}(\hat{h}_{D_n}(x))$$



Nos queda:


$$\mathbb{E}_{D_n} \left[ (y - \hat{h}_{D_n}(x))^2 \right] = (y - \bar{h}(x))^2 + \operatorname{Var}_{D_n}(\hat{h}_{D_n}(x))$$

Ahora reemplazamos $y = f(x) + \epsilon$ en el primer término:


$$y - \bar{h}(x) = (f(x) - \bar{h}(x)) + \epsilon$$

Elevando al cuadrado:


$$(y - \bar{h}(x))^2 = (f(x) - \bar{h}(x))^2 + \epsilon^2 + 2\epsilon(f(x) - \bar{h}(x))$$

Tomamos la esperanza respecto al ruido $\epsilon$:

* $(f(x) - \bar{h}(x))^2$ no depende de $\epsilon$, por lo que permanece idéntico. Por definición:

$$(f(x) - \bar{h}(x))^2 = \left( \mathbb{E}_{D_n}[\hat{h}_{D_n}(x)] - f(x) \right)^2 = \operatorname{Sesgo}(\hat{h}_{D_n}(x))^2$$


* $\mathbb{E}_\epsilon[\epsilon^2] = \operatorname{Var}(\epsilon) = \sigma^2$.
* $\mathbb{E}_\epsilon[2\epsilon(f(x) - \bar{h}(x))] = 2(f(x) - \bar{h}(x))\mathbb{E}_\epsilon[\epsilon] = 0$ (porque $\mathbb{E}[\epsilon] = 0$).

Reuniendo todos los términos:


$$\mathbb{E}_{D_n, \epsilon} \left[ (y - \hat{h}_{D_n}(x))^2 \right] = \operatorname{Sesgo}(\hat{h}_{D_n}(x))^2 + \operatorname{Var}_{D_n}(\hat{h}_{D_n}(x)) + \operatorname{Var}(\epsilon)$$

---

#### III) Error lineal: $error(y, pred) = y - pred$

**No se obtiene la misma descomposición.**

Si tomamos la esperanza del error lineal:


$$\mathbb{E}_{D_n, \epsilon} [y - \hat{h}_{D_n}(x)] = \mathbb{E}_\epsilon [f(x) + \epsilon] - \mathbb{E}_{D_n} [\hat{h}_{D_n}(x)] = f(x) + 0 - \bar{h}(x) = -\operatorname{Sesgo}(\hat{h}_{D_n}(x))$$

**Problemas encontrados:**

1. **Desaparición de los momentos de segundo orden:** La varianza del modelo $\operatorname{Var}(\hat{h})$ y la varianza del ruido $\operatorname{Var}(\epsilon)$ son medidas de dispersión cuadrática. Debido a la linealidad del operador esperanza, las fluctuaciones aleatorias alrededor de la media se cancelan mutuamente ($\mathbb{E}[\hat{h}(x) - \bar{h}(x)] = 0$), haciendo imposible aislar la varianza.
2. **Cancelación de errores de distinto signo:** Un estimador que comete un error de $+100$ en la mitad de los casos y $-100$ en la otra mitad tendría un error esperado de $0$, ocultando por completo la imprecisión del modelo.

---

### Ejercicio 7.2

* **(a) Verdadero.** Al incrementar el número de instancias de entrenamiento $N$, la muestra representa con mayor fidelidad a la distribución poblacional y se reduce la sensibilidad del estimador frente al ruido de una muestra particular, disminuyendo la varianza del modelo.


* **(b) Falso.** El sesgo está determinado por la rigidez y las restricciones del espacio de hipótesis $\mathcal{H}$ (por ejemplo, ajustar una recta a datos que tienen curvatura). Incluso con infinitos datos ($N \to \infty$), un modelo con capacidad insuficiente mantendrá su error sistemático intacto.


* **(c) Falso.** Un modelo de alta complejidad posee gran flexibilidad y grados de libertad para adaptarse a geometrías complejas, por lo que su sesgo es típicamente **bajo**.


* **(d) Verdadero.** Una alta complejidad permite al modelo ajustarse a pequeñas perturbaciones y al ruido estocástico específico del conjunto de entrenamiento, lo que provoca que el modelo varíe notablemente si se entrena con un conjunto distinto (alta varianza).


* **(e) Verdadero.** El subajuste (*underfitting*) ocurre cuando el modelo carece de la capacidad necesaria para capturar el patrón subyacente de los datos, lo que genera errores sistemáticos altos tanto en entrenamiento como en validación (alto sesgo).


* **(f) Verdadero.** El sobreajuste (*overfitting*) sucede cuando el modelo memoriza particularidades y ruido del conjunto de entrenamiento, lo que genera una gran brecha entre el error de entrenamiento y el de generalización debido a la inestabilidad de sus predicciones ante nuevos datos (alta varianza).



---

### Ejercicio 7.3

#### (a) Sobreajuste y subajuste

Analizando el rendimiento de la Tabla 2:

* **Sobreajuste:**
* **C1:** Accuracy train $= 0.99$, validación $= 0.89$. Presenta una brecha de 0.10 entre entrenamiento y validación.


* **C3:** Accuracy train $= 0.90$, validación $= 0.75$. Presenta la brecha más grande ($0.15$), con una degradación severa en validación.




* **Subajuste:**
* **C2:** Accuracy train $= 0.90$, validación $= 0.89$. Casi no hay brecha de generalización ($0.01$), pero su rendimiento en entrenamiento está 9 puntos por debajo del límite que la tarea permite alcanzar (como muestra C4 con 0.99 en train).


* **C3:** También presenta subajuste en entrenamiento ($0.90$ vs $0.99$ de C4), combinado con un fuerte sobreajuste hacia validación.



---

#### (b) Acciones correctivas

* **Ante subajuste (alto sesgo):**
* Incrementar la complejidad del modelo (por ejemplo, permitir mayor profundidad en árboles o agregar neuronas y capas).
* Agregar nuevos atributos predictivos o crear términos de interacción y características polinómicas.
* Reducir la regularización (disminuir penalizaciones $L_1/L_2$ o flexibilizar criterios de parada).


* **Ante sobreajuste (alta varianza):**
* Recolectar más datos de entrenamiento o aplicar aumento de datos (*data augmentation*).
* Aplicar o incrementar técnicas de regularización (poda de árboles, regularización $L_1/L_2$, *dropout*).
* Reducir la dimensionalidad o aplicar selección de atributos.
* Utilizar ensambles reductores de varianza como Bagging o Random Forest.



---

#### (c) Configuraciones con alto sesgo

**C2 y C3**.
El sesgo se evalúa observando el error de entrenamiento en relación con el nivel óptimo alcanzable para la tarea. Dado que la configuración C4 demuestra que es posible alcanzar un accuracy de $0.99$ en entrenamiento y $0.98$ en validación, un accuracy de entrenamiento de sólo $0.90$ (error del 10%) en C2 y C3 evidencia un sesgo significativamente más alto.

---

#### (d) Configuraciones con alta varianza

**C3 y C1**.
La varianza se manifiesta en la discrepancia (*gap*) entre el rendimiento en entrenamiento y validación:

* C3 tiene una brecha de $0.90 - 0.75 = 0.15$ (la más alta).


* C1 tiene una brecha de $0.99 - 0.89 = 0.10$.



---

#### (e) Relación entre alta varianza y sobreajuste

* **¿Sobreajuste implica alta varianza? Sí.** El sobreajuste se define empíricamente por una pérdida de rendimiento significativa en datos de prueba en comparación con los de entrenamiento, lo cual ocurre precisamente porque el estimador varió en exceso para capturar las particularidades de la muestra de entrenamiento.
* **¿Alta varianza implica sobreajuste? Sí.** En aprendizaje supervisado, un modelo con alta varianza produce hipótesis marcadamente distintas ante distintas realizaciones del conjunto de datos, memorizando ruido muestral y exhibiendo una brecha de generalización desfavorable en validación respecto a entrenamiento.

---

#### (f) Características de C1 y C2 si fueran árboles de decisión

* **C1 (Train 0.99, Val 0.89):** Sería un árbol **muy profundo y no podado**. Sus hiperparámetros tendrían valores como `max_depth = None`, `min_samples_split = 2` y `min_samples_leaf = 1`. Particiona el espacio hasta aislar casi cada ejemplo, produciendo muchas hojas y fronteras de decisión irregulares.


* **C2 (Train 0.90, Val 0.89):** Sería un árbol **fuertemente restringido o podado**. Tendría un `max_depth` bajo, o valores elevados de `min_samples_leaf` o regularización por complejidad de costo (`ccp_alpha`). Esto restringe su flexibilidad, impidiéndole memorizar el ruido pero limitando su capacidad para aprender patrones más sutiles.



---

#### (g) Escenario con alto ruido de fondo (límite humano $\approx$ 90%)

**No, la configuración C2 no tendría sesgo alto bajo estas condiciones.**
Si el nivel de ruido es tal que ni siquiera los humanos pueden superar el $90\%$ de acierto, el **error de Bayes** (ruido irreducible $\sigma^2$) ronda el $10\%$. En ese escenario:

* El máximo accuracy teórico alcanzable es $\approx 0.90$.
* C2 logra $0.90$ en entrenamiento y $0.89$ en validación.
Por lo tanto, C2 estaría capturando prácticamente todo el patrón determinista existente sin ajustar el ruido irreducible. El error del 10% provendría del ruido irreducible y no de un sesgo del algoritmo.



---

## 8. Ensambles

### Ejercicio 8.1: Algoritmo Bagging

**Bagging** (*Bootstrap Aggregating*, Breiman 1996) es un meta-algoritmo de ensamble diseñado para reducir la varianza de estimadores inestables (como árboles de decisión no podados) sin perjudicar sustancialmente su sesgo.

**Funcionamiento:**

1. **Bootstrap:** Dado un dataset original $D$ de tamaño $N$, se generan $B$ subconjuntos de entrenamiento independientes $D_1, D_2, \dots, D_B$. Cada $D_b$ se construye muestreando $N$ instancias de $D$ de forma aleatoria **con reemplazo**.
2. **Entrenamiento en paralelo:** Se entrena un modelo base $h_b$ sobre cada muestra $D_b$ sin modificar el algoritmo base.
3. **Agregación:**
* **Regresión:** Se promedian las predicciones continuas:

$$\hat{h}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B h_b(x)$$


* **Clasificación:** Se realiza una votación mayoritaria (*hard voting*) o se promedian las probabilidades de clase estimadas (*soft voting*):

$$\hat{h}_{\text{bag}}(x) = \operatorname{argmax}_{c} \sum_{b=1}^B \mathbb{I}(h_b(x) = c)$$





---

### Ejercicio 8.2: Random Forest (Breiman, 2001)

#### Principal diferencia con Bagging

En Bagging estándar con árboles, cada árbol evalúa los $p$ atributos disponibles en cada nodo para encontrar la división óptima. Si existen unos pocos predictores dominantes, la mayoría de los árboles los elegirán en los primeros niveles, generando árboles altamente correlacionados entre sí.

**Random Forest** introduce aleatoriedad a nivel de atributos (*random feature subspace selection*): en **cada partición de cada nodo**, se selecciona de forma aleatoria un subconjunto de tamaño $m \le p$ atributos (por defecto $m \approx \sqrt{p}$ en clasificación y $m \approx p/3$ en regresión). La mejor división se busca únicamente entre esos $m$ atributos candidatos. Esto **descorrelaciona los árboles**, permitiendo una reducción mucho más drástica de la varianza del ensamble.

---

#### Estimación de error "Out-Of-Bag" (OOB)

Al muestrear $N$ elementos con reemplazo de un total de $N$, la probabilidad de que una instancia dada **no** sea elegida en una muestra bootstrap particular es:


$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368$$

Aproximadamente el $36.8\%$ de los datos quedan fuera de cada árbol; estas instancias se denominan **Out-Of-Bag (OOB)** para ese árbol específico.

**Procedimiento de estimación:**

1. Para cada instancia $x_i$ del conjunto original, se identifican únicamente los árboles en cuyo subconjunto bootstrap $x_i$ no participó.
2. Se agrega la predicción de $x_i$ utilizando exclusivamente ese subconjunto de árboles.
3. Se compara esta predicción agregada con la etiqueta verdadera $y_i$ para calcular la métrica de error global sobre todo el dataset.

El error OOB es un estimador no sesgado del error de generalización que prescinde de un conjunto de validación separado o de validación cruzada.

---

#### Importancia de features: Breiman vs. scikit-learn

* Propuesta original de Breiman (Permutation Feature Importance basada en OOB):


1. Para cada árbol $b$, se evalúa su desempeño (accuracy o error cuadrático) sobre su conjunto OOB.
2. Se permutan al azar los valores del atributo $j$ en dicho conjunto OOB (rompiendo la asociación entre el atributo $j$ y la variable objetivo $y$, preservando la distribución marginal).
3. Se vuelve a calcular el desempeño sobre los datos OOB permutados.
4. La diferencia entre el desempeño original y el degradado promediada a lo largo de todos los árboles (y normalizada por su desviación estándar) define la importancia del atributo $j$.


* Implementación por defecto en scikit-learn (`feature_importances_`):


* Utiliza la importancia por reducción de impureza (**MDI**, *Mean Decrease in Impurity* o impureza de Gini).


* Suma la disminución total ponderada de impureza que genera cada atributo en cada nodo en el que fue seleccionado para realizar una partición, calculada sobre los datos de **entrenamiento** (in-bag) y promediada sobre todos los árboles del bosque.
* **Diferencia clave:** MDI se calcula en entrenamiento y tiende a sobrestimar severamente la importancia de atributos continuos o variables categóricas con alta cardinalidad (muchas categorías), ya que estos atributos ofrecen más oportunidades numéricas para reducir la impureza en la muestra de entrenamiento. El método de permutación OOB propuesto por Breiman no presenta este sesgo sistemático.



---

### Ejercicio 8.3: Cantidad de submodelos en ensamble Bagging

Un clasificador Bagging con decisiones duras calcula la probabilidad estimada de la clase positiva contando la fracción de submodelos que votaron por dicha clase:


$$p(x) = \frac{k}{B}, \quad \text{con } k \in \{0, 1, 2, \dots, B\}$$

Donde $B$ es el número total de submodelos y $k$ es el número de votos positivos emitidos. Por lo tanto, cualquier probabilidad devuelta por el modelo debe ser un múltiplo entero de $\frac{1}{B}$.

Las probabilidades observadas son $[0.75, 0.50, 0.25, 0.75, 1.0]$. Expresadas como fracciones irreducibles:


$$\frac{3}{4}, \quad \frac{1}{2} = \frac{2}{4}, \quad \frac{1}{4}, \quad \frac{3}{4}, \quad 1 = \frac{4}{4}$$

El mínimo común denominador de estas fracciones es $4$. Por lo tanto, la cantidad de submodelos $B$ debe ser un **múltiplo de 4** ($B \in \{4, 8, 12, 16, \dots\}$). Asumiendo el principio de parsimonia (el ensamble mínimo capaz de generar estos valores), se utilizaron **$B = 4$ submodelos**.

---

### Ejercicio 8.4: Verdadero o Falso

* **(a) Verdadero.** Por definición del remuestreo bootstrap, cada subconjunto se obtiene extrayendo $N$ observaciones con reemplazo a partir de un conjunto original de tamaño $N$.


* **(b) Verdadero.** Random Forest hereda el esquema de remuestreo bootstrap de Bagging para las filas: cada árbol se entrena sobre una muestra bootstrap de tamaño $N$.


* **(c) Falso.** La restricción de atributos se aplica **nodo por nodo** y no a nivel de árbol completo. En cada nodo se sortea un subconjunto aleatorio de atributos candidatos; a lo largo de sus distintas ramificaciones, el árbol puede utilizar cualquiera de los atributos del dataset.


* **(d) Falso.** Tomar $m = 1$ implica que en cada división de nodo se evalúa un único atributo seleccionado al azar. Si esa división genera una partición válida con ganancia positiva, el nodo se divide. En los nodos hijos resultantes se vuelve a sortear un atributo, por lo que el árbol puede crecer en múltiples niveles de profundidad.


* **(e) Falso.** Como el sorteo del atributo se realiza de forma independiente en cada nodo, el árbol puede seleccionar el atributo $X_1$ en la raíz, el atributo $X_4$ en un nodo del segundo nivel, el atributo $X_2$ en el tercer nivel, etc., involucrando múltiples atributos a lo largo de su estructura.


* **(f) Verdadero.** Corresponde a la definición no normalizada de importancia basada en impureza (MDI), donde se acumula la ganancia o reducción de impureza total atribuible a un predictor a lo largo de todas las particiones del bosque.


* **(g) Verdadero.** Corresponde a la definición estándar de MDI promediada por árbol (dividiendo por el total de estimadores $B$), lo cual hace que la métrica no escale artificialmente al añadir más estimadores.


* **(h) Falso.** La varianza del promedio de $B$ estimadores con varianza $\sigma^2$ y correlación par a par $\rho$ es:

$$\operatorname{Var} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$



Al forzar que cada partición elija entre un subconjunto aleatorio de atributos, Random Forest **reduce la correlación $\rho$ entre árboles**, logrando una **mayor** reducción de varianza que Bagging ($\operatorname{Var}_{RF} < \operatorname{Var}_{\text{Bagging}}$).


* **(i) Falso.** El error de generalización estimado mediante Out-Of-Bag (OOB) es un estimador no sesgado del error fuera de muestra. No subestima sistemáticamente el error real.


* **(j) Falso.** Para un número adecuado de árboles $B$, el estimador OOB converge asintóticamente al error de validación cruzada y al error de generalización de forma no sesgada. No sobreestima de manera sistemática el error real.



---

### Ejercicio 8.5: Meta-algoritmo AdaBoost

AdaBoost (*Adaptive Boosting*, Freund & Schapire 1997) es un algoritmo de ensamble secuencial donde cada nuevo modelo se entrena corrigiendo los errores cometidos por los modelos anteriores. Las instancias difíciles o mal clasificadas reciben progresivamente mayor ponderación, y el ensamble final combina las predicciones mediante una suma ponderada por el rendimiento de cada modelo.

#### (a) Concepto de "weak learners"

Un **aprendiz débil** (*weak learner*) es un algoritmo de aprendizaje que sólo requiere tener un rendimiento ligeramente superior al azar (en clasificación binaria, una tasa de error ponderado $\epsilon < 0.5$). Frecuentemente se utilizan árboles de decisión de un solo nivel de división (*decision stumps*), caracterizados por un alto sesgo y una varianza muy baja.

---

#### (b) Cálculo y actualización de pesos

1. **Inicialización:** Cada instancia recibe un peso uniforme:

$$w_1(i) = \frac{1}{N}, \quad \forall i = 1, \dots, N$$


2. **En cada iteración $t = 1, \dots, T$:**
* Se entrena el estimador débil $h_t$ y se calcula su error ponderado:

$$\epsilon_t = \sum_{i=1}^N w_t(i) \mathbb{I}(y_i \ne h_t(x_i))$$


* Se calcula el peso o importancia del modelo en el ensamble:

$$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$


* Se actualizan los pesos de cada instancia para la siguiente iteración ($y_i \in \{-1, +1\}$):

$$w_{t+1}(i) = \frac{w_t(i) \exp\left(-\alpha_t y_i h_t(x_i)\right)}{Z_t}$$



Donde $Z_t$ es el factor de normalización $\sum_{i=1}^N w_t(i) \exp(-\alpha_t y_i h_t(x_i))$ que garantiza que $\sum_i w_{t+1}(i) = 1$. Si una instancia se clasificó incorrectamente ($y_i h_t(x_i) = -1$), su peso se multiplica por $e^{\alpha_t} > 1$ (aumenta); si se clasificó bien, se multiplica por $e^{-\alpha_t} < 1$ (disminuye).



---

#### (c) Criterio para determinar la cantidad de modelos

1. **Parada temprana por validación (*Early Stopping*):** Se evalúa una métrica de desempeño sobre un conjunto de validación independiente en cada iteración y se interrumpe el entrenamiento si no hay mejora tras una cierta cantidad de rondas.
2. **Criterios de parada analítica interna:**
* Si $\epsilon_t \ge 0.5$: el modelo débil no supera al azar y se detiene el ensamble para no incorporar ruido.
* Si $\epsilon_t = 0$: el clasificador actual logra separación perfecta sobre la muestra ponderada ($\alpha_t \to \infty$).


3. **Maximización del margen funcional:** En ausencia de ruido, AdaBoost continúa refinando el margen de separación de las instancias incluso después de que el error de entrenamiento cae a cero, por lo que $T$ suele fijarse como un hiperparámetro calibrado mediante búsqueda en grilla (*Grid Search*).

---

### Ejercicio 8.6: (Opcional) AdaBoost vs. GradientBoosting y XGBoost

#### Diferencia entre AdaBoost y GradientBoosting

* **Mecanismo de optimización:**
* **AdaBoost** ajusta la función de pérdida exponencial $L(y, f(x)) = \exp(-y f(x))$ reponderando las instancias de entrenamiento en cada paso.
* **Gradient Boosting** es un marco general de descenso por gradiente en el espacio de funciones (*Functional Gradient Descent*). Admite cualquier función de pérdida diferenciable (como desvío binomial / *log-loss* para clasificación, o error cuadrático medio y pérdida de Huber para regresión).


* **Objetivo de los modelos base:**
* En AdaBoost, cada modelo se entrena sobre el dataset original con los pesos muestrales actualizados.
* En Gradient Boosting, cada nuevo árbol se entrena para predecir los **pseudorresiduos**, que corresponden al gradiente negativo de la función de pérdida evaluado en las predicciones del ensamble previo:

$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$





---

#### Implementación y mejoras clave en XGBoost

XGBoost (*Extreme Gradient Boosting*, Chen & Guestrin 2016) optimiza computacional y estadísticamente el algoritmo de Gradient Boosting a través de los siguientes pilares:

1. **Aproximación de Taylor de segundo orden:** Mientras que Gradient Boosting tradicional utiliza únicamente el gradiente de primer orden ($g_i$), XGBoost incorpora el gradiente de segundo orden (Hessiano, $h_i$):

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$



Esto permite derivar de forma analítica el peso óptimo de cada hoja y evaluar la ganancia de cada división de manera exacta.
2. **Término formal de regularización ($\Omega$):** Incorpora penalizaciones explícitas contra el sobreajuste directamente en la función objetivo:

$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$



donde $T$ es la cantidad de hojas y $w_j$ los pesos de las hojas (penalización $L_2$).
3. **Manejo automático de valores faltantes (*Sparsity-aware Split Finding*):** Asigna una dirección por defecto para los valores ausentes en cada nodo, aprendida a partir de cuál rama optimiza mejor la ganancia durante el entrenamiento.
4. **Algoritmo de cuantiles ponderados (*Weighted Quantile Sketch*):** Para datasets masivos donde buscar todos los puntos de corte posibles es inviable, XGBoost divide el dominio de los atributos en cuantiles aproximados basados en los pesos hessianos.
5. **Optimización a nivel de hardware:** Utiliza estructuras de datos en columnas preordenadas almacenadas en bloques de memoria (*Column Block*), permitiendo la búsqueda de divisiones en paralelo y el procesamiento fuera de la memoria principal (*out-of-core computing*).
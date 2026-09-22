# Guia 6

## Ejercicio 1

### a

$MAE^{esc} = \frac{1}{n}\sum \vert{}(a\cdot y^{(i)}+b) - (a\cdot \hat{y}^{(i)}+b)\vert{} = \frac{1}{n}\sum a\cdot \vert{}y^{(i)} - \hat{y}^{(i)}\vert{} = a \cdot MAE$. 

El MAE se escala linealmente por el factor $a$

$MSE^{esc} = \frac{1}{n}\sum ((a\cdot y^{(i)}+b) - (a\cdot \hat{y}^{(i)}+b))^{2} = \frac{1}{n}\sum a^{2}\cdot (y^{(i)} - \hat{y}^{(i)})^{2} = a^{2}\cdot MSE$

El MSE se escala cuadráticamente por $a^{2}$

### c

$R^2$ Vale 1 cuando el modelo predice correctamente todas las instancias. El numerador se hace 0 y $R^2 = 1 - 0$

Vale 0 cuando el numerador es igual al denominador. Es decir, cuando las predicciones tienen el mismo error que predecir siempre la media de las etiquetas.

### d

Vale $< 0$ cuando el numerador es más grande que el denominador. Es decir, cuando el error de las predicciones es peor que el modelo que prediciría siempre la media de las etiquetas.

### e

Las escalas son las mismas que en el punto a y aparecen en el numerador y denominador. Se cancelan si son distintos de 0.

### f
La formula se puede ver como:

$$R^{2} = 1 - \frac{MSE}{Var(y)}$$

como tanto el numerador y el denominador estan en la misma unidad (por ejemplo, $dolares^2$) entonces se cancelan y queda un valor adimencional. Por lo que se puede usar para comparar con otros problemas que usen la misma medida.

La desventaja es que es una medida muy atada a la varianza de los datos. Por ejemplo, si se quiere comparar 2 modelos entrenados con datasets distintos, en donde ambos dieron el mismo MSE pero el primero tiene una varianza menor que la segunda, entonces el de mayor varianza (mayor denominador) hará que el término $\frac{MSE}{Var(y)}$ sea más chico y que $R^2$ se acerque a más a 1.
Se reportaría que el segundo modelo es mejor, pero realmente se cometió la misma cantidad de errores (no necesariamente los mismos errores).

### g

#### i

Si existen etiquetas $y^{(i)}=0$ o $y^{(i)}\approx 0$, el denominador en la sumatoria de MAPE se anula o tiende a cero, causando que el error tienda a infinito (división por cero) e invalidando la métrica.

### ii

Si $y=100$:
- Subestimar prediciendo $h(x)=0$: $\frac{\vert{}100-0\vert{}}{100} = 100\%$.   
- Sobreestimar prediciendo $h(x)=200$: $\frac{\vert{}100-200\vert{}}{100} = 100\%$.   
- Sobreestimar prediciendo $h(x)=300$: $\frac{\vert{}100-300\vert{}}{100} = 200\%$.   
 
Sesgo: sobreestimar produce penalizaciones porcentuales que crecen al infinito. Como el objetivo es minimizar el error, entonces siempre se preferirá modelos que subestimen.
En este caso como los valores predichos son siempre positivos (ventas o consumo eléctrico), el término $\frac{|Y-t|}{|Y|}$ para valores $0 \leq t \leq Y$ siempre será menor a 1 (es decir para los casos donde t es una subestimación), dando errores menor o iguales al $100%$.


## Ejercicio 2

### a

Se denomina clase positiva a aquel fenómeno que se quiere que un modelo aprenda a distinguir o siguiendo alguna convención según el problema. En el caso de spam vs no-spam, la clase positiva es spam. En el caso de imagenes perros vs gatos, podria ser cualquiera de las dos.
Siempre hay que tener en cuenta que hay métricas que son sensibles al desbalance de clases, y por lo tanto cambian el valor reportado según cual sea la clase positiva.

### b

- Verdadero positivo: tanto la etiqueta real como lo predicho por el modelo es positivo
- Verdadero negativo: idem pero negativo
- Falso negativo: el medole predijo negativo y la verdadera etiqueta era postiiva.
- Falso positivo: el modelo predijo positivo y la verdadera etiqueta era negativa.

### d

Depende del problema en el que se esté trabajando.

En casos medicos donde se quiere clasificar paciente enfermo o no enfermo, con etiqueta positiva enfermo, un falso positivo podría implicar nuevos tests para el paciente. Y un falso negativo podría darle de alta al paciente erroneamente, y dependiendo de la enfermedad puede ser fatal.

En el caso de spam vs no spam (etiqueta positiva spam) un falso positivo es menos deseable, ya que filtra un mail que no era spam como spam.


## Ejercicio 3

### a

- accuracy: promedio de aciertos correctos en todas las clases (para el caso de clasificación)
- precision: de las observaciones etiquetadas como positivas por el modelo, cuantas realmente son positivas.
- recall: de todas las observaciones positivas, cuantas el modelo etiqueta correctamente como positivas.

### c

El accuracy en problemas con desbalance de clases podría reportar valores que ocultan el mal desempeño sobre alguna de las clases.
Por ejemplo en un dataset de 100 instancias donde 10 son negativas y 90 positivas, un modelo que predice siempre positiva tiene un accuracy de 90%. Se podría interpretar como un modelo que clasifica muy bien, pero el error en la clase negativa es del 100% y esto queda oculto en el promedio.

### d

$$
\begin{align*}
F_{\beta} =& (1+\beta^{2}) \frac{(\frac{TP}{TP+FP}) (\frac{TP}{TP+FN})}{\beta^{2} (\frac{TP}{TP+FP}) + (\frac{TP}{TP+FN})} \\

F_{\beta} =& (1+\beta^{2}) \frac{(\frac{TP}{TP+FP}) (\frac{TP}{TP+FN})}{\beta^{2} (\frac{TP}{TP+FP}) + (\frac{TP}{TP+FN})} \frac{\frac{(TP+FP)(TP+FN)}{TP}}{\frac{(TP+FP)(TP+FN)}{TP}} \\

F_{\beta} =& (1+\beta^{2}) \frac{TP}{\beta^{2} (TP+FN) + (TP+FP)} \\

F_{\beta} =& (1+\beta^{2}) \frac{TP}{\beta^{2}TP + \beta^{2}FN + TP + FP} \\

F_{\beta} =&  \frac{(1+\beta^{2})TP}{(1 + \beta^{2})TP + \beta^{2}FN + FP} \\
\end{align*} 
$$

### e

#### i

No

#### ii

Para este caso la clase postiiva es "fraude" y la negativa "legitimo". 

Como se aumentan los TN la métrica $F_\beta$ no se ve alterada.

El problema de desbalance de clases em accuracy se nota sobre la clase minoritaria. Si usamos las métricas de precision y recall sobre la clase minoritaria justamente para poder distinguir modelos que clasifiquen bien la clase positiva (la minoritaria), entonces tiene sentido. Va a depender del problema y que tanta importancia se le dá a la clase negativa. Para este caso, si interesaría más poder clasificar correctamente casos fraudulentos entonces suena razonable que la métrica quede inalterada por el incremento en la capacidad de descartar negativos.

#### iii

Se esta comparando el mismo clasificador entre 2 poblaciones con distinta proporcion de instancias negativas. El objetivo es ver cómo el clasificador se desempeña en cada uno. Si ambos desempeñan igual en la clase positiva, pero uno es claramente mejor en la clase negativa, la métrica $F_1$ no lo reflejará.

### f

Lo que se quiere demostrar es: a medida que aumentamos el umbral, el recall disminuye.

$recall = \frac{TP}{TP + FN}$. El denominador ($TP + FN$) es la suma total de instancias reales positivas en $\mathcal{D}$ y por lo tanto es constante.
Si elevamos el umbral ($\mu_{1} > \mu_{2}$), las condiciones para ser clasificado como positivo se vuelven más estrictas. Toda instancia que supera el umbral $\mu_{1}$ matemáticamente tuvo que superar el umbral $\mu_{2}$, pero puede haber instancias que superaban $\mu_{2}$ y ya no alcanzan $\mu_{1}$. Por ende, el número de verdaderos positivos (TP) solo puede disminuir o mantenerse: $TP(\mu_{1}) \le TP(\mu_{2})$. Al ser el denominador constante, $recall_{h,\mathcal{D}}(\mu_{1})\le recall_{h,\mathcal{D}}(\mu_{2})$. 

### g

Supongamos 3 instancias ordenadas por predicción: 
- Instancia A Positiva (0.9)
- Instancia B Negativa (0.8)
- Instancia C Positiva (0.7).

Si ponemos el umbral $\mu = 0.85$, clasifica A como positiva. TP=1, FP=0. $precision = \frac{1}{1} = 1.0$.
Si bajamos el umbral a $\mu = 0.75$, clasifica A y B como positivas. TP=1, FP=1. $precision = \frac{1}{2} = 0.5$.
Si bajamos el umbral a $\mu = 0.65$, clasifica A, B y C. TP=2, FP=1. $precision = \frac{2}{3} \approx 0.67$.

En este escenario, para $\mu_1 = 0.75$ y $\mu_2 = 0.65$ ($\mu_1 > \mu_2$), tenemos que $precision(\mu_1) = 0.5 < precision(\mu_2) = 0.67$. Por lo tanto, no decrece de forma monótona.   


## Ejercicio 5

### a

Clase positiva: tumor maligno. Clase negativa: tumor benigno

9 instancias positivas
23 instancias negativas

- $\mu_1$: TP=9, TN=8, FP=15, FN=0
- $\mu_2$: TP=9, TN=9, FP=14, FN=0
- $\mu_3$: TP=8, TN=17, FP=6, FN=1
- $\mu_4$: TP=5, TN=19, FP=4, FN=4

### b

Para este caso en particular que queremos maximizazr la cantidad de instancias con positivas encontradas y correctamente clasificadas por el modelo (recall = 1), el mejor es $\mu_2$ al tener menos error en la clase negativa que $\mu_1$

### c

$\mu_{1}$: $P = \frac{9}{24} = 0.375$, $R = \frac{9}{9} = 1.0$ $\rightarrow (1.0, 0.375)$
$\mu_{2}$: $P = \frac{9}{23} \approx 0.391$, $R = \frac{9}{9} = 1.0$ $\rightarrow (1.0, 0.391)$
$\mu_{3}$: $P = \frac{8}{14} \approx 0.571$, $R = \frac{8}{9} \approx 0.889$ $\rightarrow (0.889, 0.571)$
$\mu_{4}$: $P = \frac{5}{9} \approx 0.556$, $R = \frac{5}{9} \approx 0.556$ $\rightarrow (0.556, 0.556)$

Pseudocodigo:

```python
CURVA-ROC(LABELS: List(bool), SCORES: List(float)): List(Tuple(Umbral, FPR, TPR))
    Ordenar instancias por SCORES de mayor a menor.
    P = Cantidad de casos con LABELS == True
    N = Cantidad de casos con LABELS == False
    TP = 0
    FP = 0
    RESULTADOS = []
    FD    
    i = 0
    Mientras i < longitud(LABELS):
        Score_Actual = SCORES[i]
        
        // Procesar TODAS las instancias que comparten este mismo score
        Mientras i < longitud(LABELS) y SCORES[i] == Score_Actual:
            Si LABELS[i] == True:
                TP = TP + 1
            Sino:
                FP = FP + 1
            i = i + 1
            
        // Se registra el punto de corte en la curva ROC solo después 
        // de procesar todo el bloque de score idéntico
        Añadir (Score_Actual, FP/N, TP/P) a RESULTADOS
        
    Retornar RESULTADOS
```


## Ejercicio 6

### a

Verdadero. Es una métrica que se calcula entre todos los umbrales y se puede intepretar como que tan bien un modelo separa una clase positiva de una negativa 

### b

Verdadero. Se calcula con los valores de la matriz de confusión.

### c

Falso. Ordenamientos iguales sin scores repetidos hacen que ambos tengan los mismos valores de TP, FP, TN, FN, y por lo tanto sus curvas sean iguales. (Los umbrales podrian estar en otros valores pero separan las instancias de igual forma).


## Ejercicio 7

- Precision disminuye si aumenta los falsos positivos. Es más adecuado si se desea penalizar falsos positivos
- $F_\beta$ permite ponerle peso a precision o recall. Precision disminuye con falsos positivos y recall con falsos negativos. Este es el más adecuado ya que permite configurar una preferencia sobre que tipo de error se prefiere.
- AUCROC es una métrica de que tan robusto es el modelo (que tanta confianza tiene al separar positivos y negativos). No veo como se usaría para parametrizar la penalización de uno de los errores.

## Ejercicio 8

### a

Al disminuir el umbral los TP suben en la misma proporción que baja los FN. Además podrían subirse los FP.

Si se nota que el $F_1$ disminuye, entonces se obtuvo muchos más FP. Si se nota que $F_1$ aumenta, entonces se tiene más TP. Es decir, el modelo B está clasificando mejor las instancias positivas.
También se pude ver como que sube el recall y la precision no se ve afectada (se esta aumentando las instancias correctamente clasificadas como positivas también). Por ejemplo si un modelo asigno una prob de 0.45 a instancias postivias, con umbral 0.5 se tendrian muchos FN. Al bajar el umbral a 0.4, aumenta los TP y el recall.

### b

No se tienen info suficiente para decir nada sobre la curva completa. Pero nos da unos indicios de que el modelo B separa peor instancias positivas de las negativas al estar asignandole un numero bajo de probabilidad a dichas instancias.

## Ejercicio 9

### a
Verdadero, se ve en las fórmulas (no tienen en cuenta TN).

### b
Falso, si TP = 0 se indefine (podemos asignare precision 0) (Vedadero si TP > 0)

### c
Falso. Idem.

### d
Falso. Una matriz de confusión por cada umbral. En general se reporta el de $\mu_{0.5}$

### e
Verdadero

### f
Falso. Podemos armar uno por instancia en el caso extremo donde a cada uno le asigna una probabilidad distinta y al ordenarlos se tiene instancias intercaladas por clase.

### g
Falso. Por ejemplo se tiene un umbral $\mu_1$ donde se clasifican 10 instancias como positivas y son solo 2 TP. Si ordenamos las instancias, a las 8 TN se le asigna valores de probabilidad más altos. Al mover el umbral hacia arriba $\mu_2 > \mu_1$ se clasifica alguna (o ambas) de las instancias que eran realmente positivas como negativas, bajando la precision.

### h
Verdadero, demostrado en 6.3.f. El denominador de recall es constante y aumentar el umbral disminuye el numerador.

### i
Falso, usa el recall que es el TPR y el FPR.


## Ejercicio 10

### a

### b

### c

## Ejercicio 11

### a

### b

### c

## Ejercicio 12
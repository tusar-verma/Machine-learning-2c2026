# Guia 12

## Ejercicio 1

### 1

Verdadero

### 2

Verdadero

### 3

Verdadero.

Para el caso de una única red, la matriz de pesos $W$ tendrá $p$ columnas (1 por atributo) y $q$ filas (1 por target). Las predicciones de cada target corresponde a:

$$\hat{y}_k = \sum_{j=0}^{p} w_{jk} x_j = w_k^T x \quad \text{para cada salida } k \in \{1, \dots, q\}$$

Cada fila de pesos corresponde a un único target. Y como no hay capas intermedias, los pesos entre targets no se relacionan.
Luego es equivalente a haber entrenado $q$ regresores lineales simples.

### 4

Falso. Siguiendo la idea anterior, la capa oculta relaciona los pesos. Luego al entrenar, se busca optimizar el error conjunto de las q salidas a la vez. Por el contrario, $q$ redes independientes especializan sus capas ocultas únicamente en su salida particular, aprendiendo pesos internos distintos.

## Ejercicio 2

La idea es que al tener activaciones lineales, todas las operaciones matriciales de todas las capas menos la ultima se puede reducir a una sola matriz. Finalmente, con la última activación sigmoidea, se tiene la regresión logística.

No tiene sentido usar muchas capas.

<details open>
<summary>Desarrollo por gemini</summary>
Toda función lineal general $g_l(z)$ se expresa como un escalamiento y un desplazamiento (o transformación afín coordenada a coordenada):

$$g_l(z) = c_l z + d_l \quad (c_l, d_l \in \mathbb{R}, \ c_l \neq 0)$$

O en su versión multivariable más general, $g_l(z) = C^{[l]} z + d^{[l]}$, donde $C^{[l]}$ es una matriz de transformación lineal.

#### Demostración general paso a paso

Consideremos una red de $L$ capas donde cada capa $l \in \{1, \dots, L-1\}$ tiene una transformación afín en sus pesos más una función de activación lineal $g_l$:

1. **Capa 1:**
* Preactivación: $z^{[1]} = W^{[1]} x + b^{[1]}$

* Activación lineal:

$$a^{[1]} = g_1(z^{[1]}) = C^{[1]} (W^{[1]} x + b^{[1]}) + d^{[1]} = (C^{[1]} W^{[1]}) x + (C^{[1]} b^{[1]} + d^{[1]})$$

* Definiendo $\tilde{W}^{[1]} = C^{[1]} W^{[1]}$ y $\tilde{b}^{[1]} = C^{[1]} b^{[1]} + d^{[1]}$, la activación es afín:

$$a^{[1]} = \tilde{W}^{[1]} x + \tilde{b}^{[1]}$$

2. **Capa $l$ (inducción hasta $L-1$):**
* Supongamos que $a^{[l-1]} = \tilde{W}^{[l-1]} x + \tilde{b}^{[l-1]}$.

* La capa $l$ calcula:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} = W^{[l]} (\tilde{W}^{[l-1]} x + \tilde{b}^{[l-1]}) + b^{[l]} = (W^{[l]} \tilde{W}^{[l-1]}) x + (W^{[l]} \tilde{b}^{[l-1]} + b^{[l]})$$

* Al aplicar la activación lineal $g_l(z^{[l]}) = C^{[l]} z^{[l]} + d^{[l]}$:

$$a^{[l]} = C^{[l]} \left[ (W^{[l]} \tilde{W}^{[l-1]}) x + (W^{[l]} \tilde{b}^{[l-1]} + b^{[l]}) \right] + d^{[l]}$$

$$a^{[l]} = (C^{[l]} W^{[l]} \tilde{W}^{[l-1]}) x + (C^{[l]} W^{[l]} \tilde{b}^{[l-1]} + C^{[l]} b^{[l]} + d^{[l]})$$

* Esto vuelve a ser de la forma afín:

$$a^{[l]} = \tilde{W}^{[l]} x + \tilde{b}^{[l]}$$

3. **Colapso hasta la penúltima capa ($L-1$):**
Toda la secuencia de capas intermedias colapsa en una **única matriz equivalente** $W_{\text{equiv}}$ y un **único vector de sesgo** $b_{\text{equiv}}$:

$$a^{[L-1]} = W_{\text{equiv}} x + b_{\text{equiv}}$$

donde:

$$W_{\text{equiv}} = \prod_{k=1}^{L-1} (C^{[k]} W^{[k]})$$

4. **Última capa ($L$):**
Posee una sola neurona de salida con pesos $w^{[L]}$ y activación sigmoidea $\sigma$:

$$z^{[L]} = w^{[L]} a^{[L-1]} + b^{[L]} = w^{[L]} (W_{\text{equiv}} x + b_{\text{equiv}}) + b^{[L]} = (w^{[L]} W_{\text{equiv}}) x + (w^{[L]} b_{\text{equiv}} + b^{[L]})$$

$$\hat{y} = \sigma(z^{[L]}) = \sigma(w^* x + b^*)$$

donde $w^* = w^{[L]} W_{\text{equiv}}$ y $b^* = w^{[L]} b_{\text{equiv}} + b^{[L]}$.

#### Conclusión

Cualquier combinación de capas lineales intermedias colapsa algebraicamente en una única transformación matricial afín. Al aplicar la sigmoide al final, el modelo completo resultante es indistinguible de una **regresión logística**. Por ello, usar múltiples capas lineales intermedias no aporta capacidad representacional adicional respecto a entrenar un único clasificador logístico directo sobre las entradas.
</details>

## Ejercicio 3
## Ejercicio 4
## Ejercicio 5
## Ejercicio 6
## Ejercicio 7
## Ejercicio 8
## Ejercicio 9
## Ejercicio 10
## Ejercicio 11
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

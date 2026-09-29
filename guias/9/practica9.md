# Guia 9

## Ejercicio 1

### a
La normalización min-max sufre si hay valores atípicos (outliers). En este caso pudo ocurrir que hubo un dato con el precio mucho más elevado que los demás. Todos quedan en el lado cercano al 0 del intervalo y el outlier en el 1.

### b
Se tiene la estandarización o Z-score (restar a cada instancia la media muestral y dividirlo por el desvio estandar muestral). Este método también es afectado por outliers (aumenta la varianza y mueve la media). Pero si se sabe a priori que los datos siguen una distribución normal, no es mala idea usar la estandarización.


## Ejercicio 2

### a

Hubo fuga de información: se calculó la media sobre los datos de desarrollo y control. Luego la performance fue sobre-estima (error subestimado). 

### b

La imputación se hace en el dataset de entrenamiento. Se calcula la media de entrenamiento y se usa para imputar en validación. Una vez obtenida la mejor media con los datos de desarrollo (el mejor del k-fold cv), se utiliza dicha media para imputar los datos de control.

### c

Sobre la misma media que fue entrenado el modelo (la media de entrenamiento).

## Ejercicio 3



## Ejercicio 4
### a
### b
### c
### d
### e

## Ejercicio 5
### 1
### 2
### 3
### 4
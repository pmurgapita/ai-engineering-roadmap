# Lesson 013 — Linear Regression: The First Machine Learning Model

## Objetivo

Comprender qué es la regresión lineal, por qué es uno de los modelos fundamentales de Machine Learning y cómo se relaciona con todo lo que ya hemos implementado desde cero.

Al terminar esta lección deberías entender:

* qué significa regresión,
* qué significa lineal,
* cómo se interpreta un modelo lineal,
* qué representan los pesos y el sesgo,
* qué limitaciones tiene la regresión lineal,
* por qué sigue siendo importante en IA moderna.

---

## 1. Punto de partida

En lecciones anteriores entrenamos una neurona con esta forma:

```text
y_pred = xw + b
```

Esa estructura tiene un nombre:

```text
regresión lineal
```

La regresión lineal es uno de los modelos más simples y fundamentales del Machine Learning.

Se usa para predecir valores numéricos continuos.

Ejemplos:

```text
precio de una vivienda
temperatura de mañana
ventas del próximo mes
consumo eléctrico
salario estimado
```

---

## 2. Qué significa regresión

En Machine Learning supervisado distinguimos dos tipos de problemas muy importantes:

### Regresión

La regresión predice valores numéricos.

Ejemplos:

```text
precio = 280000 €
temperatura = 23.5 ºC
ventas = 1520 unidades
```

### Clasificación

La clasificación predice categorías.

Ejemplos:

```text
spam / no spam
perro / gato
fraude / no fraude
aprobado / suspendido
```

La regresión lineal pertenece al primer grupo.

---

## 3. Qué significa lineal

Un modelo lineal asume que la salida puede aproximarse mediante una combinación lineal de las entradas.

Para una sola feature:

```text
y_pred = wx + b
```

Donde:

* `x` es la entrada,
* `w` es el peso,
* `b` es el sesgo,
* `y_pred` es la predicción.

Geométricamente, esto representa una recta.

```text
y
↑
|          /
|        /
|      /
|____/____________→ x
```

---

## 4. Ejemplo con viviendas

Imagina que solo usamos una feature:

```text
metros cuadrados
```

Queremos predecir:

```text
precio
```

El modelo podría ser:

```text
precio_predicho = metros_cuadrados × w + b
```

Por ejemplo:

```text
w = 3000
b = 50000
```

Para una vivienda de 90 metros cuadrados:

```text
precio = 90 × 3000 + 50000
precio = 270000 + 50000
precio = 320000
```

Interpretación:

* cada metro cuadrado añade aproximadamente 3000 €,
* el sesgo representa un precio base aproximado.

---

## 5. Regresión lineal con varias features

En la vida real normalmente usamos más de una feature.

Para viviendas podríamos usar:

```text
metros cuadrados
habitaciones
baños
distancia al centro
antigüedad
```

El modelo sería:

```text
y_pred = x1w1 + x2w2 + x3w3 + ... + b
```

Ejemplo:

```text
precio =
metros × w1
+ habitaciones × w2
+ baños × w3
+ distancia × w4
+ b
```

En forma compacta:

```text
y_pred = x · w + b
```

Esto conecta directamente con el producto escalar.

---

## 6. Interpretación de pesos

Cada peso indica cómo influye una feature en la predicción.

Ejemplo:

```text
precio = metros × 2500 + habitaciones × 12000 + distancia × (-7000) + 40000
```

Interpretación:

* `2500`: cada metro adicional suma aproximadamente 2500 €.
* `12000`: cada habitación adicional suma aproximadamente 12000 €.
* `-7000`: cada unidad adicional de distancia resta aproximadamente 7000 €.
* `40000`: precio base o sesgo del modelo.

Los pesos pueden ser:

```text
positivos → aumentan la predicción
negativos → reducen la predicción
cercanos a cero → poca influencia
```

---

## 7. Cuidado con la escala de las features

No siempre podemos comparar pesos directamente.

Ejemplo:

```text
metros = 80
habitaciones = 3
distancia = 5
```

Los metros suelen tener valores mucho más grandes que el número de habitaciones.

Por eso un peso más grande o pequeño no siempre significa más o menos importancia de forma directa.

Más adelante estudiaremos normalización y escalado de features.

Esto será importante para entrenar modelos de forma estable.

---

## 8. Ejemplo resuelto

Modelo:

```text
precio = metros × 2500 + habitaciones × 12000 + distancia × (-7000) + 40000
```

Datos:

```text
metros = 80
habitaciones = 3
distancia = 5
```

Cálculo:

```text
precio = 80 × 2500 + 3 × 12000 + 5 × (-7000) + 40000

precio = 200000 + 36000 - 35000 + 40000

precio = 241000
```

Predicción:

```text
precio = 241000 €
```

---

## 9. Qué aprende la regresión lineal

La regresión lineal aprende los pesos y el sesgo que minimizan la pérdida.

Es decir, intenta encontrar:

```text
w y b
```

para que:

```text
predicciones ≈ valores reales
```

Flujo:

```text
datos
 ↓
predicción lineal
 ↓
pérdida
 ↓
gradientes
 ↓
actualización de pesos y sesgo
 ↓
mejor modelo
```

Esto es exactamente lo que ya implementamos en sesiones anteriores.

---

## 10. Función de pérdida habitual

En regresión se usa mucho el MSE:

```text
Mean Squared Error
```

La fórmula conceptual es:

```text
MSE = media de (y_real - y_pred)²
```

Si el modelo predice mal, el MSE sube.

Si predice bien, el MSE baja.

---

## 11. Limitaciones de la regresión lineal

La regresión lineal puede quedarse corta cuando el problema no se comporta de forma aproximadamente lineal.

Puede fallar cuando:

* hay relaciones no lineales,
* las variables interactúan entre sí,
* hay patrones complejos,
* faltan features importantes,
* los datos tienen mucho ruido,
* existen excepciones o cambios bruscos.

Ejemplo:

```text
El precio de una vivienda no depende solo de sumar efectos independientes.
Puede depender de combinaciones complejas: barrio + luz + planta + estado + mercado.
```

Para resolver problemas más complejos necesitaremos:

* mejores features,
* transformaciones,
* modelos no lineales,
* redes neuronales.

---

## 12. Por qué sigue siendo importante

Aunque la regresión lineal es simple, enseña los fundamentos de muchos modelos:

```text
datos
features
pesos
sesgo
predicción
pérdida
optimización
generalización
```

Además, muchas redes neuronales están construidas a partir de capas lineales.

Una capa lineal en Deep Learning hace una versión general de:

```text
salida = entrada × pesos + sesgos
```

Después se añaden funciones de activación para introducir no linealidad.

---

## 13. Relación con IA moderna

Los modelos modernos como redes neuronales y Transformers usan muchísimas operaciones lineales internamente.

Aunque un Transformer sea mucho más complejo, dentro aparecen constantemente operaciones del tipo:

```text
x · w + b
```

Por eso entender regresión lineal no es estudiar algo antiguo.

Es estudiar uno de los ladrillos básicos de la IA moderna.

---

## 14. Idea fundamental

**La regresión lineal aprende pesos y sesgo para aproximar una relación numérica entre entradas y salidas.**

---

## 15. Conceptos clave

* Regresión
* Clasificación
* Modelo lineal
* Feature
* Peso
* Sesgo
* Predicción
* MSE
* Optimización
* Normalización
* Escala de features

---

## 16. Preguntas de repaso

1. ¿Qué es la regresión lineal?
2. ¿Cuál es la diferencia entre regresión y clasificación?
3. ¿Qué significa que un modelo sea lineal?
4. ¿Qué representa un peso?
5. ¿Qué representa el sesgo?
6. ¿Por qué no siempre podemos comparar pesos directamente?
7. ¿Qué limitaciones tiene la regresión lineal?
8. ¿Por qué sigue siendo importante para entender Deep Learning?

---

## 17. Errores comunes

### Error 1: pensar que regresión significa “volver hacia atrás”

En Machine Learning, regresión significa predecir un valor numérico continuo.

---

### Error 2: comparar pesos sin mirar la escala

Un peso grande no siempre significa que la feature sea más importante si las escalas son diferentes.

---

### Error 3: pensar que lineal significa “simple e inútil”

Los modelos lineales son simples, pero enseñan conceptos fundamentales y pueden funcionar muy bien en algunos problemas.

---

### Error 4: olvidar que la regresión lineal solo aprende relaciones lineales

Si el problema tiene patrones complejos, la regresión lineal puede quedarse corta.

---

## 18. Pregunta del ingeniero

Si tuviera que predecir un valor numérico, ¿qué problema resuelve la regresión lineal?

Respuesta esperada:

La regresión lineal permite aprender una relación aproximada entre características de entrada y una salida numérica ajustando pesos y sesgo para minimizar la diferencia entre predicciones y valores reales.

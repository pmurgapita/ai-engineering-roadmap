# Lesson 018 — Vector Math: Magnitude, Direction and Dimension

## Objetivo

Entender los vectores con más rigor matemático y conectarlos con Machine Learning, embeddings, modelos lineales y gradientes.

Al terminar esta lección deberías entender:

* qué es un vector,
* qué significa la dimensión de un vector,
* qué es la magnitud o norma,
* qué es la dirección,
* por qué el orden de las posiciones importa,
* cómo se usan vectores en features, pesos, embeddings y gradientes.

---

## 1. Por qué los vectores importan en IA

En IA usamos vectores constantemente.

Ejemplos:

```text
perro = [0.9, 0.8]
gato = [0.8, 0.7]
casa = [80, 3, 5]
weights = [2.5, 12, -7]
```

Los vectores aparecen en:

```text
features
pesos
embeddings
gradientes
representaciones internas
activaciones neuronales
```

Por eso entender vectores es fundamental.

---

## 2. Qué es un vector

Un vector es una lista ordenada de números.

Puede representar:

* una posición,
* una dirección,
* una colección de características,
* una representación aprendida por un modelo.

Ejemplo:

```text
[80, 3, 5]
```

Puede representar una casa:

```text
80 metros
3 habitaciones
5 km de distancia
```

Otro ejemplo:

```text
[0.9, 0.8]
```

Puede representar un objeto en un espacio vectorial.

---

## 3. Dimensión

La dimensión de un vector es la cantidad de números que contiene.

Ejemplos:

```text
[5]                 → dimensión 1
[2, 7]              → dimensión 2
[1, 0, 3]           → dimensión 3
[4, 8, 2, 9, 1]     → dimensión 5
```

En IA moderna podemos tener vectores de muchas dimensiones.

Por ejemplo, un embedding de texto puede tener:

```text
384 dimensiones
768 dimensiones
1536 dimensiones
3072 dimensiones
```

Aunque no podamos visualizar espacios de tantas dimensiones, matemáticamente siguen funcionando con las mismas ideas básicas.

---

## 4. Vector como punto

En dos dimensiones:

```text
v = [3, 4]
```

podemos verlo como un punto:

```text
x = 3
y = 4
```

Visualmente:

```text
y
↑
|
|        • [3,4]
|
|
|________________→ x
```

En tres dimensiones:

```text
v = [3, 4, 2]
```

sería un punto con:

```text
x = 3
y = 4
z = 2
```

En muchas dimensiones no podemos dibujarlo fácilmente, pero la idea matemática sigue siendo válida.

---

## 5. Vector como dirección

También podemos ver un vector como una flecha desde el origen.

Ejemplo:

```text
v = [3, 4]
```

Es una flecha desde:

```text
[0, 0]
```

hasta:

```text
[3, 4]
```

Visualmente:

```text
y
↑
|
|        • [3,4]
|       /
|      /
|     /
|____/____________→ x
  [0,0]
```

Esta interpretación es importante porque algunas medidas, como la similitud coseno, se fijan mucho en la dirección.

---

## 6. Magnitud o norma

La magnitud de un vector es su longitud.

Para:

```text
v = [3, 4]
```

la magnitud es:

```text
||v|| = sqrt(3² + 4²)
```

```text
||v|| = sqrt(9 + 16)
```

```text
||v|| = sqrt(25)
```

```text
||v|| = 5
```

Esto viene del teorema de Pitágoras.

---

## 7. Fórmula general de la norma

Para un vector:

```text
v = [v1, v2, v3, ..., vn]
```

su norma es:

```text
||v|| = sqrt(v1² + v2² + v3² + ... + vn²)
```

Ejemplo:

```text
v = [1, 2, 2]
```

```text
||v|| = sqrt(1² + 2² + 2²)
```

```text
||v|| = sqrt(1 + 4 + 4)
```

```text
||v|| = sqrt(9)
```

```text
||v|| = 3
```

---

## 8. Ejemplo manual

Norma de:

```text
[6, 8]
```

Cálculo:

```text
||[6, 8]|| = sqrt(6² + 8²)
           = sqrt(36 + 64)
           = sqrt(100)
           = 10
```

---

## 9. Magnitud y dirección

La magnitud indica la longitud del vector.

La dirección indica hacia dónde apunta.

Ejemplo:

```text
a = [1, 1]
b = [10, 10]
```

Ambos tienen la misma dirección, pero distinta magnitud.

```text
b = 10 × a
```

Esto conecta con la similitud coseno:

```text
coseno → se fija en la dirección
distancia euclídea → se fija en la separación
```

Por eso dos vectores pueden tener la misma dirección pero estar lejos en distancia euclídea.

---

## 10. Operaciones básicas con vectores

### Suma

```text
[1, 2] + [3, 4] = [4, 6]
```

### Resta

```text
[5, 7] - [2, 3] = [3, 4]
```

### Multiplicación por escalar

```text
3 × [2, 4] = [6, 12]
```

### Norma

```text
||[3, 4]|| = 5
```

### Producto escalar

```text
[1, 2] · [3, 4] = 1×3 + 2×4 = 11
```

---

## 11. Producto escalar

El producto escalar multiplica componente a componente y luego suma.

Ejemplo:

```text
[2, 3, 4] · [5, 1, -2]
```

Cálculo:

```text
2×5 + 3×1 + 4×(-2)
= 10 + 3 - 8
= 5
```

Resultado:

```text
5
```

---

## 12. Vectores y features

En Machine Learning, un vector suele representar un ejemplo.

Ejemplo:

```text
coche = [kilómetros, año, potencia, puertas]
```

```text
coche = [120000, 2018, 150, 5]
```

Cada posición del vector tiene un significado:

```text
índice 0 → kilómetros
índice 1 → año
índice 2 → potencia
índice 3 → puertas
```

Por eso el orden importa.

Esto:

```text
[120000, 2018, 150, 5]
```

no significa lo mismo que esto:

```text
[5, 150, 2018, 120000]
```

Aunque contengan los mismos números.

---

## 13. Vectores y pesos

Un modelo lineal usa dos vectores:

```text
x = vector de features
w = vector de pesos
```

Ejemplo:

```text
x = [90, 3, 4]
w = [2.5, 12, -7]
```

El producto escalar:

```text
x · w
```

calcula:

```text
90 × 2.5 + 3 × 12 + 4 × (-7)
```

```text
225 + 36 - 28 = 233
```

Luego añadimos el sesgo:

```text
y_pred = x · w + b
```

Si:

```text
b = 40
```

entonces:

```text
y_pred = 273
```

---

## 14. Vectores y embeddings

Un embedding también es un vector.

Pero en lugar de representar features claras como:

```text
metros
habitaciones
distancia
```

representa características aprendidas por el modelo.

Ejemplo conceptual:

```text
perro = [0.82, -0.12, 0.44, ..., 0.09]
gato  = [0.79, -0.10, 0.40, ..., 0.11]
mesa  = [-0.30, 0.91, -0.22, ..., 0.04]
```

No siempre sabemos interpretar cada dimensión de forma humana directa.

Pero el espacio vectorial conserva relaciones útiles.

Por eso:

```text
perro cerca de gato
perro lejos de mesa
```

---

## 15. Vectores y gradientes

El gradiente también puede ser un vector.

Si un modelo tiene varios pesos:

```text
w = [w1, w2, w3]
```

entonces podemos tener un gradiente:

```text
gradient = [g1, g2, g3]
```

Cada componente indica cómo cambiar un peso.

```text
w1 se actualiza usando g1
w2 se actualiza usando g2
w3 se actualiza usando g3
```

Actualización:

```text
w = w - learning_rate × gradient
```

Con vectores:

```text
[w1, w2, w3] = [w1, w2, w3] - lr × [g1, g2, g3]
```

Esto es lo mismo que ya implementamos en código, pero escrito de forma matemática.

---

## 16. Código

Archivo recomendado:

```text
code/fundamentals/vector_math.py
```

Código:

```python
import math


def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    result = []

    for i in range(len(a)):
        result.append(a[i] + b[i])

    return result


def vector_subtract(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    result = []

    for i in range(len(a)):
        result.append(a[i] - b[i])

    return result


def scalar_multiply(scalar, vector):
    result = []

    for value in vector:
        result.append(scalar * value)

    return result


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def vector_norm(vector):
    total = 0

    for value in vector:
        total += value ** 2

    return math.sqrt(total)


a = [3, 4]
b = [1, 2]

print("a + b:", vector_add(a, b))
print("a - b:", vector_subtract(a, b))
print("3 * a:", scalar_multiply(3, a))
print("a · b:", dot_product(a, b))
print("||a||:", vector_norm(a))


features = [90, 3, 4]
weights = [2.5, 12, -7]
bias = 40

prediction = dot_product(features, weights) + bias

print("\nPrediction:", prediction)
```

---

## 17. Salida observada

```text
a + b: [4, 6]
a - b: [2, 2]
3 * a: [9, 12]
a · b: 11
||a||: 5.0

Prediction: 273.0
```

---

## 18. Similitud coseno

La similitud coseno mide cuánto se parecen dos vectores en dirección.

Fórmula:

```text
cosine = dot_product(a, b) / (vector_norm(a) × vector_norm(b))
```

Código:

```python
def cosine_similarity(a, b):
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))
```

Ejemplo:

```python
x = [1, 1]
y = [10, 10]
z = [-1, 1]

print(cosine_similarity(x, y))
print(cosine_similarity(x, z))
```

Interpretación:

```text
x e y tienen la misma dirección → similitud coseno ≈ 1
x y z son perpendiculares → similitud coseno = 0
```

Si aparece un resultado como:

```text
0.9999999999999998
```

eso es prácticamente `1`.

La diferencia mínima se debe a precisión decimal del ordenador.

---

## 19. Idea fundamental

**Un vector es una representación numérica con dimensión, magnitud y dirección, y es una de las estructuras básicas de la IA.**

---

## 20. Conceptos clave

* Vector
* Dimensión
* Magnitud
* Norma
* Dirección
* Origen
* Producto escalar
* Multiplicación por escalar
* Feature vector
* Weight vector
* Embedding
* Gradient vector
* Similitud coseno

---

## 21. Preguntas de repaso

1. ¿Qué es un vector?
2. ¿Qué significa la dimensión de un vector?
3. ¿Qué es la magnitud de un vector?
4. ¿Qué diferencia hay entre magnitud y dirección?
5. ¿Por qué el orden de las posiciones importa en un vector de features?
6. ¿Qué relación hay entre un vector de features y un vector de pesos?
7. ¿Por qué un embedding también es un vector?
8. ¿Por qué el gradiente puede verse como un vector?

---

## 22. Errores comunes

### Error 1: pensar que un vector es solo una lista sin estructura

Un vector tiene orden, dimensión, magnitud y dirección.

---

### Error 2: cambiar el orden de las features

Si cambiamos el orden de las features, los pesos se aplican a datos equivocados.

---

### Error 3: confundir producto escalar con producto vectorial

En Machine Learning usamos muchísimo el producto escalar:

```text
x · w = x1w1 + x2w2 + ... + xnwn
```

---

### Error 4: pensar que vectores con distinta magnitud no pueden ser similares

Dos vectores pueden tener distinta magnitud y aun así apuntar en la misma dirección.

Ejemplo:

```text
[1, 1]
[10, 10]
```

---

## 23. Pregunta del ingeniero

Si tuviera que representar un objeto del mundo real para que un modelo pueda usarlo, ¿qué problema resuelve un vector?

Respuesta esperada:

Un vector permite convertir un objeto o ejemplo en una representación numérica ordenada, donde cada dimensión contiene una feature o una característica aprendida. Esto permite al modelo calcular predicciones, similitudes, pérdidas y actualizaciones matemáticas.

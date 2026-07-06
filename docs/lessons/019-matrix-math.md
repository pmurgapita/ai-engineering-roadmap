# Lesson 019 — Matrix Math: Many Vectors Together

## Objetivo

Comprender qué es una matriz, cómo representa muchos vectores juntos y por qué las matrices son fundamentales para datasets, batches, modelos lineales y redes neuronales.

Al terminar esta lección deberías entender:

* qué es una matriz,
* qué significa el shape de una matriz,
* cómo una matriz puede representar un dataset,
* cómo funciona el producto matriz-vector,
* por qué las dimensiones deben ser compatibles,
* por qué las matrices permiten calcular muchas predicciones a la vez.

---

## 1. De vectores a matrices

En la lección anterior vimos que un vector puede representar un ejemplo.

Ejemplo:

```text
casa = [80, 3, 5]
```

Donde cada posición representa una feature:

```text
80 metros
3 habitaciones
5 km de distancia
```

Una matriz nos permite juntar muchos vectores.

Ejemplo:

```text
[
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2]
]
```

Cada fila puede representar una vivienda.

Cada columna puede representar una feature.

---

## 2. Qué es una matriz

Una matriz es una tabla de números organizada en filas y columnas.

Ejemplo:

```text
[
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2]
]
```

Esta matriz tiene:

```text
3 filas
3 columnas
```

Las filas son:

```text
[50, 1, 8]
[80, 3, 5]
[120, 4, 2]
```

Las columnas representan:

```text
columna 1 → metros
columna 2 → habitaciones
columna 3 → distancia
```

---

## 3. Matriz como dataset

En Machine Learning solemos llamar `X` a la matriz de entradas.

Ejemplo:

```text
X = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2],
  [60, 2, 7],
  [100, 3, 3]
]
```

Y llamamos `y` al vector de respuestas reales:

```text
y = [139, 241, 376, 167, 319]
```

Esto significa:

```text
[50, 1, 8]   → 139
[80, 3, 5]   → 241
[120, 4, 2]  → 376
[60, 2, 7]   → 167
[100, 3, 3]  → 319
```

---

## 4. Shape de una matriz

El shape describe el tamaño de una matriz.

Se suele escribir como:

```text
(filas, columnas)
```

Ejemplo:

```text
X = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2],
  [60, 2, 7],
  [100, 3, 3]
]
```

Tiene:

```text
5 filas
3 columnas
```

Por tanto:

```text
shape = (5, 3)
```

Interpretación:

```text
5 ejemplos
3 features por ejemplo
```

Esto es fundamental en IA.

Muchos errores en NumPy, PyTorch o TensorFlow son errores de shape.

---

## 5. Filas y columnas

En una matriz de datos típica:

```text
filas    → ejemplos
columnas → features
```

Ejemplo:

```text
X shape = (100, 4)
```

Significa:

```text
100 ejemplos
4 features por ejemplo
```

Una matriz de datos suele tener esta forma:

```text
n_examples × n_features
```

---

## 6. Matriz y vector de pesos

Antes hacíamos una predicción con un solo ejemplo:

```text
x = [90, 3, 4]
w = [2.5, 12, -7]
```

La predicción era:

```text
y_pred = x · w + b
```

Ahora tenemos muchos ejemplos:

```text
X = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2]
]
```

Y el mismo vector de pesos:

```text
w = [2.5, 12, -7]
```

Podemos hacer una predicción para cada fila.

---

## 7. Producto matriz-vector

Multiplicar una matriz `X` por un vector `w` significa hacer un producto escalar entre cada fila de `X` y el vector `w`.

```text
X · w
```

Ejemplo:

```text
X = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2]
]

w = [2.5, 12, -7]
```

Primera fila:

```text
[50, 1, 8] · [2.5, 12, -7]
= 50×2.5 + 1×12 + 8×(-7)
= 125 + 12 - 56
= 81
```

Segunda fila:

```text
[80, 3, 5] · [2.5, 12, -7]
= 80×2.5 + 3×12 + 5×(-7)
= 200 + 36 - 35
= 201
```

Tercera fila:

```text
[120, 4, 2] · [2.5, 12, -7]
= 120×2.5 + 4×12 + 2×(-7)
= 300 + 48 - 14
= 334
```

Resultado:

```text
X · w = [81, 201, 334]
```

Si añadimos sesgo:

```text
b = 40
```

entonces:

```text
y_pred = X · w + b
```

Resultado:

```text
[121, 241, 374]
```

---

## 8. Ejemplo completo con cinco viviendas

Dataset:

```text
X = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2],
  [60, 2, 7],
  [100, 3, 3]
]
```

Pesos:

```text
w = [2.5, 12, -7]
```

Sesgo:

```text
b = 40
```

Productos sin sesgo:

```text
Raw predictions = [81, 201, 334, 125, 265]
```

Predicciones finales:

```text
Predictions = [121, 241, 374, 165, 305]
```

---

## 9. Comparación con los valores reales

Valores reales:

```text
real_prices = [139, 241, 376, 167, 319]
```

Predicciones:

```text
predictions = [121, 241, 374, 165, 305]
```

Comparación:

```text
[50, 1, 8]   → real 139 | predicción 121
[80, 3, 5]   → real 241 | predicción 241
[120, 4, 2]  → real 376 | predicción 374
[60, 2, 7]   → real 167 | predicción 165
[100, 3, 3]  → real 319 | predicción 305
```

No es perfecto, pero sigue bastante bien la tendencia.

---

## 10. MSE sobre varias predicciones

Podemos calcular la pérdida sobre todas las predicciones usando MSE:

```text
MSE = media de (real - predicción)²
```

Para:

```text
real_prices = [139, 241, 376, 167, 319]
predictions = [121, 241, 374, 165, 305]
```

Errores:

```text
139 - 121 = 18
241 - 241 = 0
376 - 374 = 2
167 - 165 = 2
319 - 305 = 14
```

Errores cuadrados:

```text
18² = 324
0²  = 0
2²  = 4
2²  = 4
14² = 196
```

Suma:

```text
324 + 0 + 4 + 4 + 196 = 528
```

Media:

```text
528 / 5 = 105.6
```

Por tanto:

```text
MSE = 105.6
```

---

## 11. Código del laboratorio

Archivo recomendado:

```text
code/fundamentals/matrix_math.py
```

Código:

```python
def matrix_shape(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    return rows, columns


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def matrix_vector_multiply(matrix, vector):
    result = []

    for row in matrix:
        value = dot_product(row, vector)
        result.append(value)

    return result


def add_bias(predictions, bias):
    result = []

    for prediction in predictions:
        result.append(prediction + bias)

    return result


def mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud")

    total = 0

    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        total += error ** 2

    return total / len(y_true)


X = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

weights = [2.5, 12, -7]
bias = 40

real_prices = [139, 241, 376, 167, 319]

print("Shape of X:", matrix_shape(X))

raw_predictions = matrix_vector_multiply(X, weights)
predictions = add_bias(raw_predictions, bias)

print("Raw predictions:", raw_predictions)
print("Predictions:", predictions)
print("MSE:", mean_squared_error(real_prices, predictions))
```

---

## 12. Salida observada

```text
Shape of X: (5, 3)
Raw predictions: [81.0, 201.0, 334.0, 125.0, 265.0]
Predictions: [121.0, 241.0, 374.0, 165.0, 305.0]
MSE: 105.6
```

---

## 13. Por qué las matrices importan

Antes calculábamos una predicción cada vez:

```text
un ejemplo → una predicción
```

Con matrices podemos calcular muchas predicciones juntas:

```text
muchos ejemplos → muchas predicciones
```

Esto es la base del entrenamiento eficiente.

Los modelos modernos procesan grupos de ejemplos llamados batches.

---

## 14. Qué es un batch

Un batch es un grupo de ejemplos procesados juntos.

Ejemplo:

```text
batch = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2]
]
```

Shape:

```text
(3, 3)
```

Significa:

```text
3 ejemplos
3 features
```

En Deep Learning se usan batches constantemente.

Ejemplo:

```text
batch_size = 32
```

Puede significar:

```text
32 ejemplos procesados a la vez
```

---

## 15. Matrices en

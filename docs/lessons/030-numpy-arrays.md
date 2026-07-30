# Lesson 030 — Introduction to NumPy: Arrays, Dimensions and Shapes

## Objetivo

Comprender qué es un array de NumPy, cómo se diferencia de una lista de Python y cómo permite realizar operaciones vectorizadas con vectores, matrices y tensores.

Al terminar esta lección deberías entender:

* qué es un `ndarray`,
* cómo crear arrays,
* qué significan `shape`, `ndim`, `size` y `dtype`,
* qué diferencia hay entre un vector 1D y una matriz 2D,
* qué diferencia hay entre `*` y `@`,
* cómo realizar varias predicciones en una sola operación.

---

## 1. Importar NumPy

La convención habitual es:

```python
import numpy as np
```

Esto permite utilizar:

```python
np.array(...)
np.zeros(...)
np.mean(...)
```

La estructura principal de NumPy es el array multidimensional.

---

## 2. Crear un array

Lista de Python:

```python
python_vector = [1, 2, 3]
```

Array de NumPy:

```python
numpy_vector = np.array(
    [1, 2, 3],
    dtype=float,
)
```

Aunque visualmente se parecen, su comportamiento es diferente.

---

## 3. Multiplicación

Con una lista:

```python
python_vector * 2
```

obtenemos:

```text
[1, 2, 3, 1, 2, 3]
```

La lista se repite.

Con un array:

```python
numpy_vector * 2
```

obtenemos:

```text
[2. 4. 6.]
```

Cada elemento se multiplica por dos.

---

## 4. Suma

Con listas:

```python
[1, 2, 3] + [10, 20, 30]
```

obtenemos una concatenación:

```text
[1, 2, 3, 10, 20, 30]
```

Con arrays:

```python
np.array([1, 2, 3]) + np.array([10, 20, 30])
```

obtenemos una suma elemento a elemento:

```text
[11, 22, 33]
```

---

## 5. Atributos principales

### `shape`

Indica el tamaño en cada dimensión.

```python
array.shape
```

### `ndim`

Indica el número de dimensiones.

```python
array.ndim
```

### `size`

Indica el número total de elementos.

```python
array.size
```

### `dtype`

Indica el tipo de los valores almacenados.

```python
array.dtype
```

---

## 6. Vector

```python
vector = np.array([1, 2, 3])
```

Tiene:

```text
shape = (3,)
ndim = 1
size = 3
```

Es un array unidimensional con tres elementos.

No tiene explícitamente filas y columnas.

---

## 7. Vector frente a matriz

Vector:

```python
vector = np.array([1, 2, 3])
```

```text
shape = (3,)
ndim = 1
```

Matriz con una fila:

```python
row_matrix = np.array([
    [1, 2, 3]
])
```

```text
shape = (1, 3)
ndim = 2
```

Matriz con una columna:

```python
column_matrix = np.array([
    [1],
    [2],
    [3],
])
```

```text
shape = (3, 1)
ndim = 2
```

Resumen:

```text
(3,)   → vector con 3 elementos
(1, 3) → matriz con 1 fila y 3 columnas
(3, 1) → matriz con 3 filas y 1 columna
```

Un vector 1D no distingue entre orientación de fila y columna.

Podemos transformarlo mediante:

```python
vector.reshape(1, 3)
```

o:

```python
vector.reshape(3, 1)
```

---

## 8. Matriz

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
```

Tiene:

```text
shape = (2, 3)
ndim = 2
size = 6
```

En un dataset podría significar:

```text
2 ejemplos
3 features por ejemplo
```

En una capa neuronal también podría significar:

```text
2 neuronas
3 pesos por neurona
```

El contexto determina el significado.

---

## 9. Tensor 3D

```python
tensor = np.array([
    [
        [1, 2],
        [3, 4],
    ],
    [
        [5, 6],
        [7, 8],
    ],
])
```

Tiene:

```text
shape = (2, 2, 2)
ndim = 3
size = 8
```

NumPy permite utilizar la misma estructura general para vectores, matrices y tensores.

---

## 10. Multiplicación elemento a elemento

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

result = a * b
```

Resultado:

```text
[10, 40, 90]
```

Cada posición se multiplica con la posición correspondiente.

---

## 11. Producto escalar con `@`

```python
result = a @ b
```

Resultado:

```text
140
```

Cálculo:

```text
1×10 + 2×20 + 3×30 = 140
```

Resumen:

```text
* → multiplicación elemento a elemento
@ → producto escalar o matricial
```

---

## 12. Varias predicciones a la vez

Dataset:

```python
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
```

Pesos:

```python
weights = np.array([0.5, 1.0, -1.0])
```

Shapes:

```text
X shape = (2, 3)
weights shape = (3,)
```

Calculamos:

```python
raw_predictions = X @ weights
```

NumPy realiza un producto escalar por cada fila:

```text
fila 1 → una predicción
fila 2 → una predicción
```

Resultado:

```text
[-0.5, 1.0]
```

Shape:

```text
(2,)
```

---

## 13. Añadir el bias

```python
bias = 2.0

predictions = X @ weights + bias
```

Resultado:

```text
[1.5, 3.0]
```

El escalar `bias` se suma a todas las predicciones.

---

## 14. Vectorización

Antes:

```python
predictions = []

for example in X:
    prediction = dot_product(example, weights)
    prediction += bias
    predictions.append(prediction)
```

Con NumPy:

```python
predictions = X @ weights + bias
```

Las matemáticas son las mismas.

NumPy permite expresar y ejecutar la operación sobre el conjunto completo sin escribir manualmente cada bucle.

---

## 15. Ejemplo de viviendas

```python
houses = np.array([
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
], dtype=float)

weights = np.array(
    [2.5, 12, -7],
    dtype=float,
)

bias = 40

predictions = houses @ weights + bias
```

Resultado:

```text
[121. 241. 374. 165. 305.]
```

Shapes:

```text
houses shape = (5, 3)
weights shape = (3,)
predictions shape = (5,)
```

Cada fila representa una vivienda y cada valor de salida representa su predicción.

---

## 16. Código completo

Archivo:

```text
code/numpy/030_numpy_arrays.py
```

```python
import numpy as np


python_vector = [1, 2, 3]
python_other_vector = [10, 20, 30]

numpy_vector = np.array(
    [1, 2, 3],
    dtype=float,
)

numpy_other_vector = np.array(
    [10, 20, 30],
    dtype=float,
)

print("Python list * 2:")
print(python_vector * 2)

print("\nNumPy array * 2:")
print(numpy_vector * 2)

print("\nPython list + list:")
print(python_vector + python_other_vector)

print("\nNumPy array + array:")
print(numpy_vector + numpy_other_vector)

matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    dtype=float,
)

tensor_3d = np.array(
    [
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ],
    dtype=float,
)

print("\nVector:")
print(numpy_vector)
print("Shape:", numpy_vector.shape)
print("Dimensions:", numpy_vector.ndim)
print("Size:", numpy_vector.size)
print("Data type:", numpy_vector.dtype)

print("\nMatrix:")
print(matrix)
print("Shape:", matrix.shape)
print("Dimensions:", matrix.ndim)
print("Size:", matrix.size)
print("Data type:", matrix.dtype)

print("\nTensor 3D:")
print(tensor_3d)
print("Shape:", tensor_3d.shape)
print("Dimensions:", tensor_3d.ndim)
print("Size:", tensor_3d.size)
print("Data type:", tensor_3d.dtype)

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print("\nElement-wise multiplication:")
print(a * b)

print("\nDot product with @:")
print(a @ b)

X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

weights = np.array([0.5, 1.0, -1.0])
bias = 2.0

raw_predictions = X @ weights
predictions = raw_predictions + bias

print("\nX:")
print(X)
print("X shape:", X.shape)

print("\nWeights:")
print(weights)
print("Weights shape:", weights.shape)

print("\nRaw predictions:")
print(raw_predictions)
print("Raw predictions shape:", raw_predictions.shape)

print("\nPredictions with bias:")
print(predictions)
```

---

## 17. Idea fundamental

**Un array de NumPy representa datos numéricos multidimensionales y permite operar sobre vectores, matrices y tensores sin escribir manualmente todos los bucles.**

---

## 18. Conceptos clave

* NumPy
* `ndarray`
* Array
* Shape
* Dimensión
* Size
* Dtype
* Vector 1D
* Matriz 2D
* Tensor
* Multiplicación elemento a elemento
* Producto escalar
* Vectorización
* Bias

---

## 19. Errores comunes

### Confundir `(3,)` con `(1, 3)`

```text
(3,)   → una dimensión
(1, 3) → dos dimensiones
```

### Utilizar `*` cuando queremos multiplicación matricial

```text
* → elemento a elemento
@ → producto escalar o matricial
```

### Interpretar el shape sin contexto

```text
(2, 3)
```

puede representar ejemplos y features o neuronas y pesos. El significado depende del uso.

### Pensar que NumPy cambia las matemáticas

NumPy implementa las mismas operaciones que ya hemos estudiado, pero de manera vectorizada.

---

## 20. Pregunta del ingeniero

¿Qué problema resuelve NumPy en nuestros modelos?

NumPy permite representar datos y parámetros como arrays multidimensionales y ejecutar operaciones sobre ejemplos, features y neuronas completas sin programar manualmente cada producto y cada bucle.

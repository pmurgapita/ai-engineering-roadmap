# Lesson 020 — Matrix-Matrix Multiplication: From One Output to Many Outputs

## Objetivo

Comprender cómo una matriz de entradas multiplicada por una matriz de pesos permite producir varias salidas para muchos ejemplos a la vez.

Al terminar esta lección deberías entender:

* la diferencia entre matriz-vector y matriz-matriz,
* por qué usamos una matriz de pesos `W`,
* cómo una capa lineal produce varias salidas,
* cómo interpretar los shapes,
* cómo contar parámetros en una capa lineal.

---

## 1. Punto de partida

En la lección anterior usamos:

```text
X · w
```

Donde:

```text
X = matriz de entradas
w = vector de pesos
```

Esto producía una salida por cada ejemplo.

Ejemplo:

```text
X shape = (5, 3)
w shape = (3)
X · w shape = (5)
```

Interpretación:

```text
5 ejemplos
1 predicción por ejemplo
```

---

## 2. De una salida a muchas salidas

A veces queremos que cada ejemplo produzca varias salidas.

Ejemplo con viviendas:

```text
[80, 3, 5] → [precio, demanda]
```

O incluso:

```text
[80, 3, 5] → [precio, demanda, impuestos]
```

Para producir varias salidas necesitamos varios conjuntos de pesos.

---

## 3. De vector de pesos a matriz de pesos

Antes teníamos un vector de pesos:

```text
w = [2.5, 12, -7]
```

Eso produce una salida.

Para producir dos salidas, podemos usar dos vectores de pesos:

```text
w_price = [2.5, 12, -7]
w_demand = [0.3, 5, -2]
```

Podemos juntarlos en una matriz de pesos:

```text
W = [
  [2.5, 0.3],
  [12, 5],
  [-7, -2]
]
```

Shape:

```text
W shape = (3, 2)
```

Interpretación:

```text
3 features de entrada
2 salidas
```

---

## 4. Shape principal

Si tenemos:

```text
X shape = (5, 3)
W shape = (3, 2)
```

Entonces:

```text
X · W shape = (5, 2)
```

Porque:

```text
(5, 3) · (3, 2) = (5, 2)
```

Interpretación:

```text
5 ejemplos
2 salidas por ejemplo
```

La dimensión interna debe coincidir:

```text
features de X = filas de W
```

---

## 5. Cálculo manual con un ejemplo

Tomamos una vivienda:

```text
x = [80, 3, 5]
```

Y la matriz de pesos:

```text
W = [
  [2.5, 0.3],
  [12, 5],
  [-7, -2]
]
```

Para calcular la primera salida, usamos la primera columna de `W`:

```text
[2.5, 12, -7]
```

```text
precio_raw = 80×2.5 + 3×12 + 5×(-7)
precio_raw = 200 + 36 - 35
precio_raw = 201
```

Para calcular la segunda salida, usamos la segunda columna de `W`:

```text
[0.3, 5, -2]
```

```text
demanda_raw = 80×0.3 + 3×5 + 5×(-2)
demanda_raw = 24 + 15 - 10
demanda_raw = 29
```

Resultado:

```text
x · W = [201, 29]
```

Una entrada, dos salidas.

---

## 6. Añadir sesgos

Si tenemos dos salidas, también tenemos dos sesgos:

```text
b = [40, 10]
```

Entonces:

```text
y_pred = x · W + b
```

```text
[201, 29] + [40, 10] = [241, 39]
```

Resultado final:

```text
[precio_predicho, demanda_predicha] = [241, 39]
```

---

## 7. Aplicado a muchos ejemplos

Usamos toda la matriz `X`:

```text
X = [
  [50, 1, 8],
  [80, 3, 5],
  [120, 4, 2],
  [60, 2, 7],
  [100, 3, 3]
]
```

Shape:

```text
X shape = (5, 3)
```

Pesos:

```text
W = [
  [2.5, 0.3],
  [12, 5],
  [-7, -2]
]
```

Shape:

```text
W shape = (3, 2)
```

Sesgos:

```text
b = [40, 10]
```

Resultado:

```text
Y_pred = X · W + b
```

Shape:

```text
Y_pred shape = (5, 2)
```

Interpretación:

```text
5 ejemplos
2 predicciones por ejemplo
```

---

## 8. Cómo pensar en las columnas de `W`

Cada columna de `W` produce una salida.

```text
W = [
  [2.5, 0.3],
  [12, 5],
  [-7, -2]
]
```

Primera columna:

```text
[2.5, 12, -7]
```

Produce la salida 1.

Segunda columna:

```text
[0.3, 5, -2]
```

Produce la salida 2.

Por tanto:

```text
número de columnas de W = número de salidas
```

---

## 9. Ejemplo manual

Datos:

```text
x = [2, 1]

W = [
  [3, 5],
  [4, -1]
]

b = [10, 100]
```

Primera salida:

```text
2×3 + 1×4 = 6 + 4 = 10
```

Con bias:

```text
10 + 10 = 20
```

Segunda salida:

```text
2×5 + 1×(-1) = 10 - 1 = 9
```

Con bias:

```text
9 + 100 = 109
```

Resultado:

```text
x · W + b = [20, 109]
```

---

## 10. Conexión con capas neuronales

Una capa lineal hace:

```text
Y = XW + b
```

Donde:

```text
X = batch de entradas
W = matriz de pesos
b = vector de sesgos
Y = salidas
```

En PyTorch, algo como:

```python
nn.Linear(in_features=3, out_features=2)
```

significa conceptualmente:

```text
entrada con 3 features
salida con 2 valores
```

Eso implica:

```text
W shape conceptual = (3, 2)
b shape = (2)
```

Algunas librerías pueden almacenar internamente los pesos transpuestos, pero la idea conceptual es esta:

```text
entrada × pesos + sesgos
```

---

## 11. De una neurona a una capa

Una neurona lineal:

```text
una entrada vectorial → una salida
```

Matemáticamente:

```text
y = x · w + b
```

Una capa con varias neuronas:

```text
una entrada vectorial → varias salidas
```

Matemáticamente:

```text
y = x · W + b
```

Cada columna de `W` puede verse como una neurona lineal dentro de la misma capa.

Si la capa tiene 2 salidas, podemos verla como 2 neuronas.

Si tiene 128 salidas, podemos verla como 128 neuronas.

---

## 12. Código

Archivo recomendado:

```text
code/fundamentals/matrix_matrix_multiply.py
```

Código:

```python
def matrix_shape(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    return rows, columns


def get_column(matrix, column_index):
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def matrix_multiply(A, B):
    a_rows, a_cols = matrix_shape(A)
    b_rows, b_cols = matrix_shape(B)

    if a_cols != b_rows:
        raise ValueError("No se pueden multiplicar: dimensiones incompatibles")

    result = []

    for row in A:
        result_row = []

        for column_index in range(b_cols):
            column = get_column(B, column_index)
            value = dot_product(row, column)
            result_row.append(value)

        result.append(result_row)

    return result


def add_bias(matrix, bias):
    result = []

    for row in matrix:
        result_row = []

        for i in range(len(row)):
            result_row.append(row[i] + bias[i])

        result.append(result_row)

    return result


def count_parameters(W, b):
    rows, columns = matrix_shape(W)

    number_of_weights = rows * columns
    number_of_biases = len(b)
    total_parameters = number_of_weights + number_of_biases

    return number_of_weights, number_of_biases, total_parameters


X = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

W = [
    [2.5, 0.3],
    [12, 5],
    [-7, -2],
]

b = [40, 10]

raw_outputs = matrix_multiply(X, W)
outputs = add_bias(raw_outputs, b)

print("Shape of X:", matrix_shape(X))
print("Shape of W:", matrix_shape(W))

print("Raw outputs:")
for row in raw_outputs:
    print(row)

print("\nOutputs:")
for row in outputs:
    print(row)

weights_count, bias_count, total_count = count_parameters(W, b)

print("\nWeights:", weights_count)
print("Biases:", bias_count)
print("Total parameters:", total_count)
```

---

## 13. Salida observada

```text
Shape of X: (5, 3)
Shape of W: (3, 2)

Raw outputs:
[81.0, 4.0]
[201.0, 29.0]
[334.0, 52.0]
[125.0, 14.0]
[265.0, 39.0]

Outputs:
[121.0, 14.0]
[241.0, 39.0]
[374.0, 62.0]
[165.0, 24.0]
[305.0, 49.0]

Weights: 6
Biases: 2
Total parameters: 8
```

---

## 14. Interpretación de la salida

Cada fila de `outputs` corresponde a un ejemplo.

Ejemplo:

```text
[121.0, 14.0]
```

puede interpretarse como:

```text
precio estimado = 121
demanda estimada = 14
```

La segunda fila:

```text
[241.0, 39.0]
```

corresponde a la segunda vivienda.

Así que:

```text
X shape = (5, 3)
W shape = (3, 2)
outputs shape = (5, 2)
```

---

## 15. Conteo de parámetros

En una capa lineal, los parámetros son:

```text
pesos + sesgos
```

Si:

```text
W shape = (3, 2)
b shape = (2)
```

entonces:

```text
pesos = 3 × 2 = 6
sesgos = 2
total = 8
```

En modelos grandes ocurre lo mismo, pero a gran escala.

Cuando decimos que un modelo tiene millones o billones de parámetros, hablamos principalmente de pesos y sesgos repartidos por muchas capas.

---

## 16. Idea fundamental

**Multiplicar una matriz de entradas por una matriz de pesos permite producir varias salidas para muchos ejemplos a la vez.**

---

## 17. Conceptos clave

* Matriz de entradas
* Matriz de pesos
* Vector de sesgos
* Multiplicación matriz-matriz
* Shape
* In features
* Out features
* Capa lineal
* Neurona lineal
* Parámetro
* Batch

---

## 18. Preguntas de repaso

1. ¿Qué diferencia hay entre `X · w` y `X · W`?
2. ¿Por qué `W` tiene varias columnas?
3. ¿Qué significa `out_features`?
4. ¿Qué shape tiene conceptualmente `W` si `in_features = 10` y `out_features = 5`?
5. ¿Por qué una capa lineal puede verse como varias neuronas juntas?
6. ¿Cómo se calcula el número de parámetros de una capa lineal?
7. ¿Qué representa el vector `b`?

---

## 19. Errores comunes

### Error 1: pensar que varias salidas significan varias capas

Varias salidas pueden pertenecer a una misma capa.

Una capa con 5 salidas puede verse como 5 neuronas lineales juntas.

---

### Error 2: olvidar que las dimensiones internas deben coincidir

Para multiplicar:

```text
A shape = (m, n)
B shape = (n, p)
```

El resultado es:

```text
(m, p)
```

Las dimensiones internas deben coincidir.

---

### Error 3: confundir filas y columnas de `W`

En esta convención conceptual:

```text
filas de W = features de entrada
columnas de W = salidas
```

---

### Error 4: olvidar los sesgos

Si hay 2 salidas, normalmente hay 2 sesgos.

Si hay 128 salidas, normalmente hay 128 sesgos.

---

## 20. Pregunta del ingeniero

Si quiero que un modelo produzca varias predicciones por cada entrada, ¿qué problema resuelve usar una matriz de pesos?

Respuesta esperada:

Una matriz de pesos permite usar varios conjuntos de pesos a la vez. Cada columna produce una salida distinta, de modo que una entrada vectorial puede transformarse en un vector de salidas dentro de una misma capa lineal.

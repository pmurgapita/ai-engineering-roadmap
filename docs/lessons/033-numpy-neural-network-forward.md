# Lesson 033 — Neural Network Forward Pass with NumPy

## Objetivo

Construir una red neuronal multicapa utilizando operaciones matriciales de NumPy y procesar un batch completo sin bucles manuales.

---

## 1. Convención de datos

Representamos un batch mediante:

```text id="f3a83f"
X shape = (batch, features)
```

Por ejemplo:

```text id="un0evk"
X shape = (5,3)
```

significa:

```text id="clfg0u"
5 ejemplos
3 features por ejemplo
```

---

## 2. Convención de pesos

Con NumPy utilizamos:

```text id="q5g6fa"
W shape = (inputs, neurons)
```

Esto permite escribir:

```python id="cd39th"
X @ W
```

Si:

```text id="xnidwu"
X = (batch, inputs)
W = (inputs, neurons)
```

entonces:

```text id="r1ekaa"
X @ W = (batch, neurons)
```

---

## 3. Primera capa

Tenemos:

```text id="rm5qef"
X = (5,3)
W1 = (3,4)
b1 = (4,)
```

La operación es:

```python id="59suj6"
Z1 = X @ W1 + b1
```

Producto matricial:

```text id="etsl8i"
(5,3) @ (3,4)
→
(5,4)
```

Broadcasting:

```text id="4hr90f"
(5,4) + (4,)
→
(5,4)
```

Por tanto:

```text id="nt7d6o"
Z1 shape = (5,4)
```

Interpretación:

```text id="mz3pe9"
5 ejemplos
4 salidas lineales por ejemplo
```

---

## 4. Interpretar `Z[i,j]`

En:

```text id="i3bzoy"
Z shape = (batch, neurons)
```

un elemento:

```python id="8zfxhi"
Z[i, j]
```

representa:

```text id="ynimjs"
salida de la neurona j
para el ejemplo i
```

Por ejemplo:

```python id="2y3a43"
Z1[2,3]
```

es la salida de la cuarta neurona oculta para el tercer ejemplo.

En nuestro caso:

```text id="ly56yd"
Z1[2,3] = 2.2
```

---

## 5. ReLU vectorizada

Definimos:

```python id="rnd282"
def relu(x):
    return np.maximum(0, x)
```

Esta función opera elemento a elemento sobre todo el array.

Por tanto:

```python id="5gofyo"
A1 = relu(Z1)
```

mantiene el mismo shape:

```text id="qj83ri"
Z1 = (5,4)
A1 = (5,4)
```

ReLU cambia valores, no la estructura dimensional.

---

## 6. Segunda capa

Tenemos:

```text id="qwm9wg"
A1 = (5,4)
W2 = (4,2)
b2 = (2,)
```

Calculamos:

```python id="rs6v4d"
Z2 = A1 @ W2 + b2
```

Producto:

```text id="izp2hj"
(5,4) @ (4,2)
→
(5,2)
```

Después:

```text id="q36qrn"
(5,2) + (2,)
→
(5,2)
```

Interpretación:

```text id="q0v1ig"
5 ejemplos
2 outputs por ejemplo
```

---

## 7. Forward pass completo

La red completa puede escribirse como:

```python id="3m31n9"
Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
```

O utilizando una función:

```python id="9i2hcj"
def dense_forward(inputs, weights, biases):
    return inputs @ weights + biases
```

Entonces:

```python id="7vgw39"
Z1 = dense_forward(X, W1, b1)
A1 = relu(Z1)

Z2 = dense_forward(A1, W2, b2)
```

---

## 8. Qué hace una capa dense

Una capa dense conecta cada input con cada neurona.

Si tenemos:

```text id="zowdik"
3 inputs
4 neuronas
```

necesitamos:

```text id="s9gkln"
3×4 = 12 pesos
```

y:

```text id="6m1cga"
4 biases
```

Total:

```text id="mlialg"
16 parámetros
```

---

## 9. Conteo de parámetros

Primera capa:

```text id="pw5ihl"
W1 = (3,4)
→ 12 pesos

b1 = (4,)
→ 4 biases

total = 16
```

Segunda capa:

```text id="fz1qxp"
W2 = (4,2)
→ 8 pesos

b2 = (2,)
→ 2 biases

total = 10
```

Red completa:

```text id="8e2r9u"
16 + 10 = 26 parámetros
```

---

## 10. Batch y parámetros

El tamaño del batch puede cambiar:

```text id="fvd5yi"
X = (5,3)
```

a:

```text id="aq4o82"
X = (500,3)
```

sin cambiar:

```text id="jy45zh"
W1 = (3,4)
```

Porque el shape de los pesos depende de:

```text id="7h4qda"
número de features
número de neuronas
```

no del número de ejemplos.

Los mismos parámetros se aplican a todos los ejemplos.

---

## 11. Broadcasting de los biases

Para:

```text id="sflmh6"
Z1 = (5,4)
b1 = (4,)
```

NumPy aplica:

```text id="j8suoc"
[b1,b2,b3,b4]
```

a todas las filas.

Por eso podemos escribir:

```python id="b8hws0"
Z1 = X @ W1 + b1
```

sin un bucle por ejemplo.

---

## 12. Salida correcta del ejemplo

Con:

```python id="8q1sze"
Z1 = X @ W1 + b1
```

obtenemos:

```text id="aoytv6"
[[ 0.5   2.2   2.1   0.4 ]
 [ 1.   -0.65  2.05  1.4 ]
 [ 1.7  -0.6   3.1   2.2 ]
 [ 0.45  1.5   1.2   0.45]
 [ 0.8  -0.3   4.8   1.1 ]]
```

Después de ReLU:

```text id="zatb5r"
[[0.5  2.2  2.1  0.4 ]
 [1.   0.   2.05 1.4 ]
 [1.7  0.   3.1  2.2 ]
 [0.45 1.5  1.2  0.45]
 [0.8  0.   4.8  1.1 ]]
```

Y la capa de salida produce:

```text id="fph3mw"
[[ 1.11   2.09 ]
 [ 0.03   0.95 ]
 [ 0.07   1.39 ]
 [ 1.085  1.04 ]
 [-1.84   3.91 ]]
```

Shape:

```text id="yfym57"
(5,2)
```

---

## 13. Añadir una tercera capa

Si:

```text id="5nn11c"
Z2 = (5,2)
```

y queremos tres neuronas:

```text id="yh6cmf"
W3 = (2,3)
b3 = (3,)
```

entonces:

```python id="ipb0hc"
Z3 = Z2 @ W3 + b3
```

produce:

```text id="ofug4c"
(5,2) @ (2,3)
→
(5,3)
```

La nueva capa añade:

```text id="d2c52v"
2×3 = 6 pesos
3 biases
```

Total:

```text id="yirfvy"
9 parámetros
```

La red pasa de:

```text id="5pxsoi"
26 → 35 parámetros
```

---

## 14. Vectorización

Antes necesitábamos conceptualmente:

```text id="sognnm"
por cada ejemplo:
    por cada neurona:
        por cada entrada:
            multiplicar
            sumar
```

Con NumPy:

```python id="f6muxt"
Z = X @ W + b
```

La multiplicación matricial procesa todos los ejemplos y neuronas y broadcasting aplica los biases.

---

## 15. Idea fundamental

**Una capa neuronal vectorizada procesa un batch completo mediante `X @ W + b`, donde `X` tiene shape `(batch, inputs)`, `W` tiene shape `(inputs, neurons)` y el resultado tiene shape `(batch, neurons)`.**

---

## 16. Conceptos clave

* Dense layer
* Batch
* Features
* Neuronas
* Matriz de pesos
* Vector de biases
* Forward pass
* Vectorización
* Multiplicación matricial
* Broadcasting
* Parameter sharing
* Número de parámetros

---

## 17. Pregunta del ingeniero

¿Por qué una capa de cuatro neuronas puede procesar cinco o quinientos ejemplos sin cambiar sus parámetros?

Porque los pesos representan la transformación aprendida entre las features de entrada y las neuronas, no los ejemplos individuales. El mismo conjunto de parámetros se aplica a todas las filas del batch.

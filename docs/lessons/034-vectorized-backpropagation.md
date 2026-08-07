# Lesson 034 — Vectorized Backpropagation with NumPy

## Objetivo

Implementar el ciclo completo de entrenamiento de una red neuronal utilizando operaciones vectorizadas de NumPy.

Al finalizar esta lección deberías entender cómo calcular:

* forward pass para un batch,
* MSE vectorizado,
* gradientes de la capa de salida,
* gradientes de pesos y biases,
* propagación hacia capas anteriores,
* backward de ReLU,
* actualización vectorizada de todos los parámetros.

## 1. Arquitectura

Trabajamos con:

```text
3 features
↓
4 neuronas ocultas
↓
ReLU
↓
2 neuronas de salida
```

para un batch de cinco ejemplos.

Shapes:

```text
X  = (5,3)

W1 = (3,4)
b1 = (4,)

Z1 = (5,4)
A1 = (5,4)

W2 = (4,2)
b2 = (2,)

Z2 = (5,2)
Y  = (5,2)
```

## 2. Forward pass

```python
Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
```

El mismo conjunto de parámetros se aplica a todos los ejemplos del batch.

## 3. MSE vectorizado

```python
loss = np.mean(
    (Z2 - Y) ** 2
)
```

Si `Z2` contiene diez valores, la media divide la suma de errores cuadráticos entre diez.

Por eso:

```python
dZ2 = (
    2
    * (Z2 - Y)
    / Z2.size
)
```

Cada elemento de `Z2` tiene asociado un gradiente.

Por tanto:

```text
Z2.shape == dZ2.shape
```

## 4. Gradientes de la segunda capa

Los pesos tienen:

```text
W2 = (4,2)
```

Calculamos:

```python
dW2 = A1.T @ dZ2
```

Shapes:

```text
A1   = (5,4)
A1.T = (4,5)

dZ2 = (5,2)

(4,5) @ (5,2)
→
(4,2)
```

Así:

```text
dW2.shape == W2.shape
```

Conceptualmente, esta multiplicación acumula las contribuciones de todos los ejemplos para cada peso.

## 5. Gradiente de los biases

```python
db2 = np.sum(
    dZ2,
    axis=0,
)
```

Cada bias es compartido por todos los ejemplos.

Por eso sumamos las contribuciones del batch.

```text
dZ2 = (5,2)

axis=0
↓
db2 = (2,)
```

## 6. Propagación hacia la capa oculta

```python
dA1 = dZ2 @ W2.T
```

Shapes:

```text
dZ2  = (5,2)
W2.T = (2,4)

(5,2) @ (2,4)
→
(5,4)
```

Por tanto:

```text
dA1.shape == A1.shape
```

La transpuesta permite recorrer hacia atrás las conexiones utilizadas durante el forward pass.

## 7. Backward de ReLU

```python
def relu_derivative(x):
    return (x > 0).astype(float)
```

Aplicamos:

```python
dZ1 = (
    dA1
    * relu_derivative(Z1)
)
```

Aquí utilizamos `*`, porque queremos una multiplicación elemento a elemento.

```text
gradiente
×
derivada local de ReLU
```

Si una neurona tenía `Z ≤ 0`, la derivada de ReLU vale cero y bloquea ese gradiente.

## 8. Gradientes de W1

```python
dW1 = X.T @ dZ1
```

Shapes:

```text
X   = (5,3)
X.T = (3,5)

dZ1 = (5,4)

(3,5) @ (5,4)
→
(3,4)
```

Por tanto:

```text
dW1.shape == W1.shape
```

El patrón general es:

```text
dW =
entrada_de_la_capa.T
@
gradiente_de_la_capa
```

## 9. Gradientes de b1

```python
db1 = np.sum(
    dZ1,
    axis=0,
)
```

Partimos de:

```text
(5,4)
```

y eliminamos el eje del batch:

```text
(5,4)
→
(4,)
```

Así:

```text
db1.shape == b1.shape
```

## 10. Regla fundamental de shapes

Cada parámetro necesita exactamente un gradiente.

Por tanto:

```text
W.shape == dW.shape

b.shape == db.shape
```

Esto permite actualizar:

```python
W -= learning_rate * dW
b -= learning_rate * db
```

elemento a elemento.

## 11. Backpropagation completo

```python
dZ2 = (
    2
    * (Z2 - Y)
    / Z2.size
)

dW2 = A1.T @ dZ2
db2 = np.sum(dZ2, axis=0)

dA1 = dZ2 @ W2.T

dZ1 = (
    dA1
    * relu_derivative(Z1)
)

dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0)
```

Estas pocas operaciones sustituyen los bucles manuales por ejemplos, neuronas y pesos.

## 12. Actualización

Después de calcular todos los gradientes:

```python
W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2
```

Primero se completa todo el backward pass y después se modifican los parámetros.

## 13. Entrenamiento durante muchas épocas

El ciclo completo es:

```python
for epoch in range(100):
    # Forward
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2

    loss = mse(Z2, Y)

    # Backward
    dZ2 = 2 * (Z2 - Y) / Z2.size

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T

    dZ1 = (
        dA1
        * relu_derivative(Z1)
    )

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
```

Con nuestros datos, la pérdida evoluciona aproximadamente:

```text
epoch 0  → 0.00831250
epoch 10 → 0.00189602
epoch 20 → 0.00113190
epoch 30 → 0.00076561
epoch 40 → 0.00056615
epoch 50 → 0.00045120
epoch 60 → 0.00038127
epoch 70 → 0.00033610
epoch 80 → 0.00030498
epoch 90 → 0.00028212
```

La pérdida continúa descendiendo.

## 14. Por qué recalculamos los gradientes

Después de cada update cambian los parámetros:

```text
W1, b1, W2, b2
```

Entonces cambia el forward pass:

```text
Z1, A1, Z2
```

y cambia la pérdida.

Por tanto también cambian todos los gradientes.

El ciclo correcto es:

```text
parámetros actuales
→ forward
→ loss
→ backward
→ gradientes actuales
→ update
→ nuevos parámetros
→ repetir
```

No debemos reutilizar gradientes antiguos.

## 15. Gradientes cerca del objetivo

Cuando:

```text
Z2 → Y
```

entonces:

```text
Z2-Y → 0
```

y por tanto:

```text
dZ2 → 0
```

Los gradientes propagados hacia capas anteriores también tienden a hacerse más pequeños.

Las actualizaciones se vuelven progresivamente menores.

## 16. Forward y backward matricial

Forward:

```text
X @ W1 → Z1

A1 @ W2 → Z2
```

Backward para propagar:

```text
dZ2 @ W2.T → dA1
```

Backward para pesos:

```text
A1.T @ dZ2 → dW2

X.T @ dZ1 → dW1
```

Patrones generales:

```text
propagar hacia atrás:
dA_previous = dZ @ W.T

gradiente de pesos:
dW = A_previous.T @ dZ
```

## 17. Qué aporta NumPy

Las matemáticas son exactamente las mismas que en nuestra implementación manual.

NumPy permite calcular simultáneamente:

```text
todos los ejemplos
todas las neuronas
todos los pesos
todos los gradientes
```

mediante operaciones vectorizadas.

## 18. Idea fundamental

**El backpropagation vectorizado utiliza transpuestas, multiplicaciones matriciales, operaciones elemento a elemento y reducciones por ejes para calcular simultáneamente los gradientes de todos los parámetros de una red.**

## 19. Conexión con PyTorch

En PyTorch podremos escribir:

```python
prediction = model(X)
loss = loss_function(prediction, Y)

loss.backward()
```

`loss.backward()` calculará automáticamente los gradientes que nosotros hemos implementado manualmente:

```text
dZ2
dW2
db2
dA1
dZ1
dW1
db1
```

La automatización no cambia las matemáticas. Automatiza el backpropagation que ya comprendemos.

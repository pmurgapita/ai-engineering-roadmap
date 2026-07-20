# Lesson 027 — From Prediction to Error: First Step of the Backward Pass

## Objetivo

Comprender cómo empieza el backward pass y cómo se calculan los gradientes de los pesos y el bias de la capa de salida.

Al terminar esta lección deberías entender:

* por qué necesitamos una función de pérdida,
* qué significa derivar la pérdida respecto a la predicción,
* cómo se calcula el gradiente de un peso,
* por qué una activación igual a cero produce un gradiente cero,
* qué es la regla de la cadena,
* cómo se actualiza la capa de salida,
* por qué los gradientes se hacen menores al acercarnos al objetivo.

---

## 1. Red utilizada

La red tiene esta arquitectura:

```text
2 entradas
↓
2 neuronas ocultas
↓
ReLU
↓
1 neurona de salida
↓
predicción
```

Para:

```text
inputs = [2, 3]
```

la capa oculta produce:

```text
hidden_outputs = [0, 5]
```

La capa de salida utiliza:

```text
output_weights = [2.0, -0.5]
output_bias = 1.0
```

La predicción es:

```text
prediction = 0×2.0 + 5×(-0.5) + 1.0
prediction = -1.5
```

---

## 2. Valor real y pérdida

Necesitamos comparar la predicción con el valor correcto:

```text
y_true = 2.0
```

Utilizamos error cuadrático:

```text
L = (y_pred - y_true)²
```

Sustituyendo:

```text
L = (-1.5 - 2.0)²
L = (-3.5)²
L = 12.25
```

La pérdida nos permite medir lo mala que ha sido la predicción.

---

## 3. Forward pass y backward pass

El forward pass calcula:

```text
entrada → predicción
```

El backward pass intenta responder:

```text
¿cómo afecta cada parámetro a la pérdida?
```

El recorrido conceptual es:

```text
pérdida
↑
predicción
↑
pesos y bias de salida
↑
activaciones ocultas
↑
parámetros ocultos
```

En esta sesión solo propagamos el gradiente hasta los parámetros de la capa de salida.

---

## 4. Gradiente de la pérdida respecto a la predicción

La pérdida es:

```text
L = (y_pred - y_true)²
```

Su derivada respecto a la predicción es:

```text
∂L/∂y_pred = 2(y_pred - y_true)
```

Sustituimos:

```text
∂L/∂y_pred = 2(-1.5 - 2.0)
∂L/∂y_pred = -7
```

Interpretación:

```text
si aumentamos la predicción, la pérdida disminuye
```

Esto tiene sentido porque la predicción `-1.5` está por debajo del objetivo `2.0`.

---

## 5. Capa de salida

La salida se calcula mediante:

```text
y_pred = a1w1 + a2w2 + b
```

Con:

```text
a1 = 0
a2 = 5
w1 = 2.0
w2 = -0.5
b = 1.0
```

Por tanto:

```text
y_pred = 0×2.0 + 5×(-0.5) + 1.0
```

---

## 6. Gradiente del primer peso

La predicción depende del primer peso así:

```text
a1w1 = 0×w1
```

Por tanto:

```text
∂y_pred/∂w1 = a1 = 0
```

Aplicamos la regla de la cadena:

```text
∂L/∂w1
=
∂L/∂y_pred × ∂y_pred/∂w1
```

```text
∂L/∂w1
=
-7 × 0
=
0
```

El peso no recibe actualización para este ejemplo porque la activación que lo atraviesa es cero.

No es el peso el que vale cero. Es su entrada.

---

## 7. Gradiente del segundo peso

La segunda activación vale:

```text
a2 = 5
```

Por tanto:

```text
∂y_pred/∂w2 = a2 = 5
```

Aplicamos la regla de la cadena:

```text
∂L/∂w2
=
∂L/∂y_pred × ∂y_pred/∂w2
```

```text
∂L/∂w2
=
-7 × 5
=
-35
```

Interpretación:

```text
-7  → efecto de la predicción sobre la pérdida
5   → efecto del peso sobre la predicción
-35 → efecto total del peso sobre la pérdida
```

---

## 8. Gradiente del bias

La predicción depende del bias así:

```text
y_pred = ... + b
```

Por tanto:

```text
∂y_pred/∂b = 1
```

Aplicamos la regla de la cadena:

```text
∂L/∂b
=
∂L/∂y_pred × ∂y_pred/∂b
```

```text
∂L/∂b
=
-7 × 1
=
-7
```

---

## 9. Regla de la cadena

Un parámetro puede no afectar directamente a la pérdida.

Por ejemplo:

```text
w2
↓
prediction
↓
loss
```

Para descubrir cómo afecta `w2` a la pérdida, combinamos:

```text
cómo afecta w2 a la predicción
```

con:

```text
cómo afecta la predicción a la pérdida
```

Matemáticamente:

```text
∂L/∂w2
=
∂L/∂y_pred
×
∂y_pred/∂w2
```

La regla de la cadena multiplica las sensibilidades locales de cada tramo del camino.

En una red profunda, el backpropagation aplica esta regla repetidamente desde la pérdida hasta las primeras capas.

---

## 10. Actualización de parámetros

Usamos:

```text
learning_rate = 0.01
```

La regla es:

```text
parameter =
parameter - learning_rate × gradient
```

Primer peso:

```text
w1 = 2.0 - 0.01×0
w1 = 2.0
```

Segundo peso:

```text
w2 = -0.5 - 0.01×(-35)
w2 = -0.15
```

Bias:

```text
b = 1.0 - 0.01×(-7)
b = 1.07
```

Nuevos parámetros:

```text
output_weights = [2.0, -0.15]
output_bias = 1.07
```

---

## 11. Nuevo forward pass

La capa oculta continúa produciendo:

```text
[0, 5]
```

La nueva predicción es:

```text
prediction =
0×2.0 + 5×(-0.15) + 1.07
```

```text
prediction = 0.32
```

La predicción ha cambiado:

```text
-1.5 → 0.32
```

Se ha acercado al objetivo:

```text
2.0
```

---

## 12. Nueva pérdida

```text
new_loss = (0.32 - 2.0)²
new_loss = 2.8224
```

La pérdida baja:

```text
12.25 → 2.8224
```

Esto indica que la actualización se realizó en la dirección correcta.

---

## 13. Entrenamiento durante diez épocas

Si repetimos el proceso:

```text
forward pass
→ pérdida
→ gradientes
→ actualización
```

obtenemos aproximadamente:

| Época | Predicción |   Pérdida |     `w1` |      `w2` |     Bias |
| ----: | ---------: | --------: | -------: | --------: | -------: |
|     0 |  -1.500000 | 12.250000 | 2.000000 | -0.500000 | 1.000000 |
|     1 |   0.320000 |  2.822400 | 2.000000 | -0.150000 | 1.070000 |
|     2 |   1.193600 |  0.650281 | 2.000000 |  0.018000 | 1.103600 |
|     3 |   1.612928 |  0.149825 | 2.000000 |  0.098640 | 1.119728 |
|     4 |   1.814205 |  0.034520 | 2.000000 |  0.137347 | 1.127469 |
|     5 |   1.910819 |  0.007953 | 2.000000 |  0.155927 | 1.131185 |
|     6 |   1.957193 |  0.001832 | 2.000000 |  0.164845 | 1.132969 |
|     7 |   1.979453 |  0.000422 | 2.000000 |  0.169126 | 1.133825 |
|     8 |   1.990137 |  0.000097 | 2.000000 |  0.171180 | 1.134236 |
|     9 |   1.995266 |  0.000022 | 2.000000 |  0.172167 | 1.134433 |

La predicción se acerca a `2` y la pérdida se aproxima a cero.

---

## 14. Por qué el primer peso no cambia

Durante todo este ejemplo:

```text
a1 = 0
```

Por tanto:

```text
∂L/∂w1
=
∂L/∂y_pred × a1
=
cualquier valor × 0
=
0
```

El primer peso no recibe información de aprendizaje con esta entrada.

Con otra entrada que activara la primera neurona, su gradiente podría dejar de ser cero.

---

## 15. Por qué los cambios se hacen más pequeños

La derivada de la pérdida es:

```text
∂L/∂y_pred = 2(y_pred-y_true)
```

Cuando la predicción se acerca al valor real:

```text
y_pred-y_true → 0
```

Entonces:

```text
∂L/∂y_pred → 0
```

Como los demás gradientes dependen de este valor, también disminuyen.

Por eso las actualizaciones son cada vez menores al acercarnos al mínimo.

---

## 16. Código completo

```python
hidden_outputs = [0, 5.0]

output_weights = [
    [2.0, -0.5],
]

output_biases = [1.0]

y_true = 2.0
learning_rate = 0.01

for epoch in range(10):
    prediction = layer_forward(
        hidden_outputs,
        output_weights,
        output_biases,
    )[0]

    loss = squared_error(y_true, prediction)

    loss_prediction_gradient = squared_error_derivative(
        y_true,
        prediction,
    )

    weight_gradients, bias_gradient = output_layer_gradients(
        hidden_outputs,
        loss_prediction_gradient,
    )

    print(f"Epoch: {epoch}")
    print(f"Prediction: {prediction:.6f}")
    print(f"Loss: {loss:.6f}")
    print(f"Output weights: {output_weights}")
    print(f"Output bias: {output_biases[0]:.6f}")
    print()

    for i in range(len(output_weights[0])):
        output_weights[0][i] -= (
            learning_rate * weight_gradients[i]
        )

    output_biases[0] -= learning_rate * bias_gradient
```

---

## 17. Idea fundamental

**El backward pass comienza calculando cómo cambia la pérdida respecto a la predicción y utiliza la regla de la cadena para descubrir cómo afecta cada parámetro al error final.**

---

## 18. Conceptos clave

* Forward pass
* Backward pass
* Función de pérdida
* Gradiente de salida
* Gradiente de peso
* Gradiente de bias
* Regla de la cadena
* Sensibilidad local
* Actualización
* Época
* Activación cero

---

## 19. Errores comunes

### Confundir un peso cero con una entrada cero

El primer peso no recibe gradiente porque su entrada es cero, no porque el propio peso valga cero.

### Pensar que todos los pesos cambian siempre

Un peso puede recibir gradiente cero para un ejemplo concreto.

### Olvidar la regla de la cadena

Para conocer el efecto de un parámetro lejano sobre la pérdida, hay que combinar todos los efectos intermedios.

### No recalcular los gradientes

Después de cada actualización cambian la predicción, la pérdida y los gradientes.

---

## 20. Pregunta del ingeniero

¿Cómo descubre una red cuánto debe cambiar un peso de la capa de salida?

Calcula cómo afecta el peso a la predicción y cómo afecta la predicción a la pérdida. Después multiplica ambos efectos mediante la regla de la cadena. Ese resultado es el gradiente del peso y permite actualizarlo en la dirección que reduce la pérdida.

# Lesson 028 — Full Backpropagation: Updating the Hidden Layer

## Objetivo

Comprender cómo el gradiente atraviesa la capa de salida y la función ReLU hasta llegar a los pesos y biases de una capa oculta.

Al terminar esta lección deberías entender:

* cómo propagar el gradiente hacia las activaciones ocultas,
* qué hace la derivada de ReLU,
* cómo calcular los gradientes de los pesos ocultos,
* cómo se aplica la regla de la cadena a una red completa,
* por qué todos los gradientes deben calcularse antes de actualizar,
* cómo aprenden coordinadamente las distintas capas.

---

## 1. Arquitectura

La red utilizada es:

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

Datos:

```text
inputs = [2, 3]
y_true = 2.0
```

El forward pass inicial produce:

```text
z_hidden = [-1.5, 5.0]
a_hidden = [0, 5.0]
prediction = -1.5
loss = 12.25
```

---

## 2. Camino del backward pass

El forward pass sigue:

```text
inputs
→ z_hidden
→ ReLU
→ a_hidden
→ prediction
→ loss
```

Backpropagation recorre ese camino en sentido inverso:

```text
loss
→ prediction
→ a_hidden
→ z_hidden
→ hidden weights and biases
```

En cada tramo utilizamos una derivada local.

---

## 3. Gradiente respecto a la predicción

La pérdida es:

```text
L = (y_pred-y_true)²
```

Por tanto:

```text
∂L/∂y_pred = 2(y_pred-y_true)
```

Para:

```text
y_pred = -1.5
y_true = 2.0
```

obtenemos:

```text
∂L/∂y_pred = -7
```

El signo negativo indica que aumentar la predicción reduciría la pérdida.

---

## 4. Gradientes de la capa de salida

La predicción es:

```text
y_pred = a₁w₁ + a₂w₂ + b
```

Con:

```text
a_hidden = [0, 5]
output_weights = [2.0, -0.5]
```

Los gradientes son:

```text
∂L/∂w₁ = -7 × 0 = 0
∂L/∂w₂ = -7 × 5 = -35
∂L/∂b = -7
```

Resultado:

```text
output_weight_gradients = [0, -35]
output_bias_gradient = -7
```

---

## 5. Gradientes de las activaciones ocultas

Para continuar hacia atrás necesitamos calcular:

```text
∂L/∂a₁
∂L/∂a₂
```

La regla de la cadena establece:

```text
∂L/∂aᵢ
=
∂L/∂y_pred
×
∂y_pred/∂aᵢ
```

Como:

```text
∂y_pred/∂aᵢ = output_weightᵢ
```

obtenemos:

```text
∂L/∂a₁ = -7 × 2.0 = -14
∂L/∂a₂ = -7 × (-0.5) = 3.5
```

Resultado:

```text
hidden_activation_gradients = [-14, 3.5]
```

`∂L/∂a₂ = 3.5` significa que aumentar ligeramente `a₂` aumentaría la pérdida. En ese momento, para reducirla, conviene disminuir `a₂`.

---

## 6. Derivada de ReLU

Durante el forward pass:

```text
ReLU(z) = max(0,z)
```

Durante el backward pass utilizamos:

```text
ReLU'(z) = 0  si z ≤ 0
ReLU'(z) = 1  si z > 0
```

La derivada actúa como una compuerta:

```text
z ≤ 0 → bloquea el gradiente
z > 0 → deja pasar el gradiente
```

En nuestros ejercicios tomamos:

```text
ReLU'(0) = 0
```

---

## 7. Gradientes respecto a `z_hidden`

Sabemos que:

```text
aᵢ = ReLU(zᵢ)
```

Por tanto:

```text
∂L/∂zᵢ
=
∂L/∂aᵢ
×
∂aᵢ/∂zᵢ
```

Primera neurona:

```text
z₁ = -1.5
ReLU'(z₁) = 0
```

```text
∂L/∂z₁ = -14 × 0 = 0
```

Segunda neurona:

```text
z₂ = 5
ReLU'(z₂) = 1
```

```text
∂L/∂z₂ = 3.5 × 1 = 3.5
```

Resultado:

```text
hidden_z_gradients = [0, 3.5]
```

---

## 8. Gradientes de los parámetros ocultos

Cada neurona calcula:

```text
z = x₁w₁ + x₂w₂ + b
```

Por tanto:

```text
∂z/∂wᵢ = xᵢ
∂z/∂b = 1
```

Y:

```text
∂L/∂wᵢ
=
∂L/∂z
×
xᵢ
```

### Primera neurona

Como:

```text
∂L/∂z₁ = 0
```

obtenemos:

```text
weight_gradients₁ = 0 × [2,3] = [0,0]
bias_gradient₁ = 0
```

### Segunda neurona

Como:

```text
∂L/∂z₂ = 3.5
```

obtenemos:

```text
weight_gradients₂ = 3.5 × [2,3]
weight_gradients₂ = [7,10.5]
```

Y:

```text
bias_gradient₂ = 3.5
```

Gradientes completos:

```text
hidden_weight_gradients = [
    [0, 0],
    [7, 10.5],
]

hidden_bias_gradients = [0, 3.5]
```

---

## 9. Regla de la cadena completa

Para un peso de la segunda neurona:

```text
peso
→ z₂
→ ReLU
→ a₂
→ prediction
→ loss
```

La derivada total es:

```text
∂L/∂w
=
∂L/∂y_pred
×
∂y_pred/∂a₂
×
∂a₂/∂z₂
×
∂z₂/∂w
```

Para el primer peso:

```text
∂L/∂w₂₁
=
(-7)
×
(-0.5)
×
1
×
2
=
7
```

Para el segundo:

```text
∂L/∂w₂₂
=
(-7)
×
(-0.5)
×
1
×
3
=
10.5
```

Backpropagation aplica esta misma idea repetidamente por toda la red.

---

## 10. Orden correcto de entrenamiento

Todos los gradientes deben calcularse con el mismo estado de la red.

El orden correcto es:

```text
1. Forward pass
2. Calcular pérdida
3. Calcular todos los gradientes
4. Actualizar todos los parámetros
5. Repetir
```

No debemos actualizar la capa de salida antes de calcular los gradientes ocultos. Eso mezclaría parámetros nuevos con activaciones calculadas usando parámetros antiguos.

---

## 11. Primera actualización

Con:

```text
learning_rate = 0.01
```

los parámetros de salida cambian:

```text
[2.0, -0.5] → [2.0, -0.15]
1.0 → 1.07
```

La primera neurona oculta no cambia porque sus gradientes son cero.

La segunda cambia:

```text
[1.5, 1.0] → [1.43, 0.895]
-1.0 → -1.035
```

El nuevo forward pass produce:

```text
hidden activations = [0, 4.51]
prediction = 0.3935
loss ≈ 2.580842
```

La pérdida baja:

```text
12.25 → 2.580842
```

---

## 12. Entrenamiento durante diez épocas

Repitiendo el ciclo completo obtenemos:

| Época | Predicción |   Pérdida | Segunda activación | Segundo peso de salida |
| ----: | ---------: | --------: | -----------------: | ---------------------: |
|     0 |  −1.500000 | 12.250000 |           5.000000 |              −0.500000 |
|     1 |   0.393500 |  2.580842 |           4.510000 |              −0.150000 |
|     2 |   1.079501 |  0.847318 |           4.442527 |              −0.005094 |
|     3 |   1.461151 |  0.290359 |           4.441214 |               0.076693 |
|     4 |   1.685938 |  0.098635 |           4.452785 |               0.124556 |
|     5 |   1.818430 |  0.032968 |           4.463739 |               0.152525 |
|     6 |   1.895725 |  0.010873 |           4.471493 |               0.168735 |
|     7 |   1.940386 |  0.003554 |           4.476419 |               0.178060 |
|     8 |   1.966015 |  0.001155 |           4.479392 |               0.183397 |
|     9 |   1.980658 |  0.000374 |           4.481137 |               0.186442 |

La predicción se acerca a `2` y la pérdida se acerca a cero.

---

## 13. Aprendizaje coordinado

La segunda activación primero disminuye:

```text
5 → 4.51 → 4.4425
```

Al principio, su peso de salida es negativo. Reducir la activación ayuda a subir la predicción.

Después, el peso de salida pasa a ser positivo:

```text
-0.5 → -0.15 → -0.005 → 0.0767
```

Entonces aumentar la activación empieza a ayudar a subir la predicción, por lo que su gradiente cambia de signo y la activación aumenta ligeramente.

Las distintas capas no aprenden de manera aislada. Sus parámetros se ajustan coordinadamente.

---

## 14. Primera neurona apagada

La primera neurona mantiene:

```text
z₁ = -1.5
a₁ = 0
```

Como su derivada de ReLU es cero, no recibe gradiente y sus parámetros no cambian.

Esto ocurre para esta entrada concreta. Otra entrada podría activar la neurona.

---

## 15. Código del entrenamiento

```python
for epoch in range(10):
    # Forward pass
    z_hidden, a_hidden = hidden_layer_forward(
        inputs,
        hidden_weights,
        hidden_biases,
    )

    prediction = output_layer_forward(
        a_hidden,
        output_weights,
        output_bias,
    )

    loss = squared_error(y_true, prediction)

    # Backward: output
    loss_prediction_gradient = (
        squared_error_derivative(y_true, prediction)
    )

    output_weight_gradients = [
        loss_prediction_gradient * activation
        for activation in a_hidden
    ]

    output_bias_gradient = loss_prediction_gradient

    # Backward: hidden
    hidden_activation_gradients = [
        loss_prediction_gradient * weight
        for weight in output_weights
    ]

    hidden_z_gradients = [
        hidden_activation_gradients[i]
        * relu_derivative(z_hidden[i])
        for i in range(len(z_hidden))
    ]

    hidden_weight_gradients = [
        [
            hidden_z_gradients[neuron_index] * input_value
            for input_value in inputs
        ]
        for neuron_index in range(len(hidden_z_gradients))
    ]

    hidden_bias_gradients = hidden_z_gradients.copy()

    # Actualizar salida
    for i in range(len(output_weights)):
        output_weights[i] -= (
            learning_rate * output_weight_gradients[i]
        )

    output_bias -= learning_rate * output_bias_gradient

    # Actualizar capa oculta
    for neuron_index in range(len(hidden_weights)):
        for weight_index in range(
            len(hidden_weights[neuron_index])
        ):
            hidden_weights[neuron_index][weight_index] -= (
                learning_rate
                * hidden_weight_gradients[neuron_index][weight_index]
            )

        hidden_biases[neuron_index] -= (
            learning_rate
            * hidden_bias_gradients[neuron_index]
        )
```

---

## 16. Idea fundamental

**Backpropagation utiliza la regla de la cadena para llevar el gradiente desde la pérdida hasta todos los parámetros de la red.**

---

## 17. Conceptos clave

* Backpropagation
* Regla de la cadena
* Gradiente de activación
* Gradiente de `z`
* Derivada de ReLU
* Gradiente de peso
* Gradiente de bias
* Compuerta de gradiente
* Forward pass
* Actualización coordinada

---

## 18. Errores comunes

### Invertir la dirección de una derivada

```text
∂L/∂a
```

significa cómo cambia `L` cuando cambia `a`, no cómo cambia `a` cuando cambia `L`.

### Confundir ReLU con su derivada

ReLU transforma activaciones durante el forward pass.

Su derivada controla el paso del gradiente durante el backward pass.

### Actualizar parámetros demasiado pronto

Primero deben calcularse todos los gradientes. Después se actualizan todos los parámetros.

### Pensar que una neurona apagada nunca podrá activarse

Puede estar apagada para un ejemplo y activa para otro.

---

## 19. Pregunta del ingeniero

¿Cómo descubre una red neuronal cómo debe cambiar un peso de una capa oculta?

Sigue el camino que conecta ese peso con la pérdida y multiplica las derivadas locales de cada transformación: salida lineal, activación, capas posteriores, predicción y pérdida. Ese producto es el gradiente del peso.

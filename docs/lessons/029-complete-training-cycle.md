# Lesson 029 — Consolidation: The Complete Training Cycle

## Objetivo

Consolidar el ciclo completo de entrenamiento de una red neuronal antes de pasar a NumPy y a operaciones vectorizadas.

Al terminar esta lección deberías entender:

* las cuatro fases principales del entrenamiento,
* qué valores se calculan durante el forward pass,
* por qué algunos valores deben guardarse,
* cómo se desarrolla el backward pass,
* por qué todos los gradientes deben calcularse antes de actualizar,
* qué hace una función `train_step`,
* cómo influyen la mutabilidad y la inmutabilidad de Python.

---

## 1. Arquitectura utilizada

La red tiene:

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

```python
inputs = [2, 3]
y_true = 2.0
```

Parámetros:

```python
hidden_weights = [
    [0.5, -1.0],
    [1.5, 1.0],
]

hidden_biases = [0.5, -1.0]

output_weights = [2.0, -0.5]
output_bias = 1.0
```

---

## 2. Las cuatro fases

Entrenar una red consiste en repetir:

```text
1. Forward pass
2. Cálculo de pérdida
3. Backward pass
4. Actualización de parámetros
```

De forma resumida:

```text
entrada
→ predicción
→ pérdida
→ gradientes
→ parámetros nuevos
```

---

## 3. Forward pass

El forward pass calcula la predicción usando los parámetros actuales.

Primera neurona:

```text
z₁ = 2×0.5 + 3×(-1.0) + 0.5
z₁ = -1.5

a₁ = ReLU(-1.5)
a₁ = 0
```

Segunda neurona:

```text
z₂ = 2×1.5 + 3×1.0 - 1.0
z₂ = 5

a₂ = ReLU(5)
a₂ = 5
```

Resultados ocultos:

```text
z_hidden = [-1.5, 5]
a_hidden = [0, 5]
```

Predicción:

```text
y_pred = 0×2.0 + 5×(-0.5) + 1.0
y_pred = -1.5
```

---

## 4. Función de pérdida

Usamos error cuadrático:

```text
L = (y_pred-y_true)²
```

Sustituyendo:

```text
L = (-1.5-2.0)²
L = 12.25
```

La pérdida transforma el error del modelo en un valor que podemos minimizar.

---

## 5. Backward pass

El backward pass responde:

```text
¿Cómo afecta cada parámetro a la pérdida?
```

Empieza en la pérdida y recorre hacia atrás el grafo de cálculo:

```text
loss
→ prediction
→ output layer
→ hidden activations
→ ReLU
→ hidden layer
```

---

## 6. Gradiente respecto a la predicción

```text
∂L/∂y_pred = 2(y_pred-y_true)
```

Para los valores actuales:

```text
∂L/∂y_pred = 2(-1.5-2)
∂L/∂y_pred = -7
```

El signo negativo indica que aumentar la predicción reduciría la pérdida.

---

## 7. Gradientes de la capa de salida

```text
∂L/∂wᵢ
=
∂L/∂y_pred × aᵢ
```

Por tanto:

```text
output_weight_gradients = [
    -7×0,
    -7×5,
]
```

```text
output_weight_gradients = [0, -35]
```

Para el bias:

```text
output_bias_gradient = -7
```

---

## 8. Gradientes de las activaciones ocultas

```text
∂L/∂aᵢ
=
∂L/∂y_pred × output_weightᵢ
```

Entonces:

```text
∂L/∂a₁ = -7×2.0 = -14
∂L/∂a₂ = -7×(-0.5) = 3.5
```

Resultado:

```text
hidden_activation_gradients = [-14, 3.5]
```

---

## 9. Gradientes a través de ReLU

Sabemos que:

```text
a = ReLU(z)
```

Y:

```text
ReLU'(z) = 0 si z ≤ 0
ReLU'(z) = 1 si z > 0
```

Por tanto:

```text
∂L/∂z
=
∂L/∂a × ReLU'(z)
```

Primera neurona:

```text
z₁ = -1.5
∂L/∂z₁ = -14×0 = 0
```

Segunda neurona:

```text
z₂ = 5
∂L/∂z₂ = 3.5×1 = 3.5
```

Resultado:

```text
hidden_z_gradients = [0, 3.5]
```

---

## 10. Gradientes de la capa oculta

Cada neurona calcula:

```text
z = x₁w₁+x₂w₂+b
```

Por tanto:

```text
gradiente de un peso
=
gradiente de z × entrada correspondiente
```

Primera neurona:

```text
weight_gradients₁ = 0×[2,3] = [0,0]
bias_gradient₁ = 0
```

Segunda neurona:

```text
weight_gradients₂ = 3.5×[2,3]
weight_gradients₂ = [7,10.5]

bias_gradient₂ = 3.5
```

---

## 11. Actualización

Regla:

```text
parameter =
parameter - learning_rate × gradient
```

Usando:

```text
learning_rate = 0.01
```

actualizamos todos los pesos y biases después de terminar el backward pass completo.

---

## 12. Por qué guardar valores del forward pass

Necesitamos guardar:

```text
inputs
z_hidden
a_hidden
prediction
```

porque el backward pass utiliza esos valores.

Ejemplos:

```text
gradiente de peso de salida
=
gradiente de predicción × a_hidden
```

```text
derivada de ReLU
depende de z_hidden
```

```text
gradiente de peso oculto
=
gradiente de z × inputs
```

Estos valores intermedios pueden verse como una cache temporal.

---

## 13. Grafo de cálculo

La red puede representarse como:

```text
inputs
→ operaciones lineales
→ z_hidden
→ ReLU
→ a_hidden
→ operación lineal
→ prediction
→ loss
```

El forward pass recorre el grafo hacia delante.

El backward pass recorre el grafo hacia atrás aplicando la regla de la cadena.

---

## 14. Orden correcto

El orden correcto es:

```text
1. Forward pass completo
2. Cálculo de pérdida
3. Backward pass completo
4. Actualización completa
```

Todos los gradientes deben pertenecer al mismo estado de la red.

Actualizar una capa antes de terminar el backward pass mezclaría parámetros nuevos con activaciones antiguas.

---

## 15. Qué es `train_step`

`train_step` es una función que realiza un paso completo de entrenamiento:

```text
forward
→ loss
→ backward
→ update
```

Recibe:

```text
un ejemplo
la respuesta real
los parámetros actuales
el learning rate
```

Y produce:

```text
una predicción
una pérdida
un modelo ligeramente actualizado
```

Al llamarla repetidamente, la red aprende de forma progresiva.

---

## 16. Mutabilidad de las listas

Las listas de Python son mutables.

```python
weights[0] = new_value
```

modifica directamente la lista original.

Por eso una función puede cambiar:

```text
hidden_weights
hidden_biases
output_weights
```

sin devolver necesariamente nuevas listas.

---

## 17. Inmutabilidad de los números

Los números de Python son inmutables.

Cuando hacemos:

```python
output_bias -= learning_rate * gradient
```

dentro de una función, la variable local pasa a apuntar a un nuevo número.

La variable exterior no se modifica automáticamente.

Por eso debemos devolver:

```python
return output_bias
```

y reasignar:

```python
output_bias = returned_output_bias
```

---

## 18. Entrenamiento durante diez épocas

La red evoluciona así:

```text
Epoch 00 | prediction = -1.500000 | loss = 12.250000
Epoch 01 | prediction = 0.393500 | loss = 2.580842
Epoch 02 | prediction = 1.079501 | loss = 0.847318
Epoch 03 | prediction = 1.461151 | loss = 0.290359
Epoch 04 | prediction = 1.685938 | loss = 0.098635
Epoch 05 | prediction = 1.818430 | loss = 0.032968
Epoch 06 | prediction = 1.895725 | loss = 0.010873
Epoch 07 | prediction = 1.940386 | loss = 0.003554
Epoch 08 | prediction = 1.966015 | loss = 0.001155
Epoch 09 | prediction = 1.980658 | loss = 0.000374
```

La predicción se aproxima al objetivo `2.0` y la pérdida se aproxima a cero.

---

## 19. Entrenamiento con un solo ejemplo

Ajustar correctamente:

```text
[2,3] → 2
```

no demuestra que la red generalice.

Podría haber aprendido únicamente ese caso.

Para aprender patrones necesitaremos:

```text
muchos ejemplos
batches
pérdida media
train/test split
evaluación en datos nuevos
```

---

## 20. Pseudocódigo completo

```text
definir funciones auxiliares

inicializar pesos y biases

durante cada época:

    hacer forward pass

    calcular pérdida

    hacer backward pass:
        gradiente de predicción
        gradientes de salida
        gradientes de activaciones ocultas
        derivada de ReLU
        gradientes ocultos

    actualizar todos los parámetros

    mostrar predicción y pérdida
```

---

## 21. Idea fundamental

**Entrenar una red consiste en repetir forward pass, pérdida, backward pass y actualización de parámetros.**

---

## 22. Conceptos clave

* Forward pass
* Pérdida
* Backward pass
* Actualización
* Regla de la cadena
* Grafo de cálculo
* Cache
* Train step
* Mutabilidad
* Inmutabilidad
* Generalización

---

## 23. Pregunta del ingeniero

¿Qué problema resuelve organizar el entrenamiento como un ciclo repetible?

Permite separar claramente la predicción, la medición del error, el cálculo de gradientes y la actualización de parámetros. Al repetir este ciclo, el modelo puede mejorar progresivamente sus predicciones.

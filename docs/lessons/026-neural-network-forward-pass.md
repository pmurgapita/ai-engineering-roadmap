# Lesson 026 — Building a Neural Network From Scratch: Forward Pass

## Objetivo

Comprender cómo una red neuronal transforma una entrada en una predicción mediante capas lineales y funciones de activación.

Al terminar esta lección deberías entender:

* qué es un forward pass,
* cómo funciona una neurona dentro de una capa,
* por qué cada neurona tiene sus propios pesos,
* qué diferencia hay entre una salida lineal y una activación,
* qué representa una capa oculta,
* cómo calcular manualmente una predicción,
* cómo interpretar los shapes de los parámetros.

---

## 1. Arquitectura de la red

Construimos una red con:

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

La red recibe un vector con dos features:

```text
x = [x1, x2]
```

La capa oculta transforma estas dos features en dos características nuevas.

Después, la capa de salida combina esas características para producir una predicción.

---

## 2. Qué es el forward pass

El forward pass es el proceso de calcular la salida de una red neuronal utilizando una entrada y los parámetros actuales.

```text
entrada
↓
capa lineal
↓
activación
↓
capa lineal
↓
predicción
```

Durante el forward pass no se modifican los parámetros.

Solo se calcula la predicción.

---

## 3. Entrada utilizada

Usamos:

```text
inputs = [2, 3]
```

Shape:

```text
(2,)
```

Esto significa que cada ejemplo tiene dos features.

---

## 4. Capa oculta

La capa oculta tiene dos neuronas.

Cada neurona recibe las dos entradas y tiene:

```text
2 pesos
1 bias
```

Parámetros:

```python
hidden_weights = [
    [0.5, -1.0],
    [1.5, 1.0],
]

hidden_biases = [0.5, -1.0]
```

Primera fila:

```text
[0.5, -1.0]
```

son los pesos de la primera neurona.

Segunda fila:

```text
[1.5, 1.0]
```

son los pesos de la segunda neurona.

---

## 5. Primera neurona oculta

La operación lineal es:

```text
z1 = inputs · weights1 + bias1
```

Sustituyendo:

```text
z1 = [2, 3] · [0.5, -1.0] + 0.5
```

Calculamos:

```text
z1 = 2×0.5 + 3×(-1.0) + 0.5
z1 = 1 - 3 + 0.5
z1 = -1.5
```

Aplicamos ReLU:

```text
a1 = ReLU(-1.5)
a1 = 0
```

La neurona está apagada para esta entrada.

Esto no significa que esté siempre apagada. Una entrada diferente podría producir un valor positivo.

---

## 6. Segunda neurona oculta

La operación lineal es:

```text
z2 = inputs · weights2 + bias2
```

Sustituyendo:

```text
z2 = [2, 3] · [1.5, 1.0] - 1.0
```

Calculamos:

```text
z2 = 2×1.5 + 3×1.0 - 1.0
z2 = 3 + 3 - 1
z2 = 5
```

Aplicamos ReLU:

```text
a2 = ReLU(5)
a2 = 5
```

La segunda neurona está activa.

---

## 7. Salida de la capa oculta

Antes de aplicar ReLU:

```text
z_hidden = [-1.5, 5]
```

Después de aplicar ReLU:

```text
a_hidden = [0, 5]
```

La diferencia es:

```text
z_hidden → salida de la transformación lineal
a_hidden → salida después de la activación
```

La siguiente capa recibe `a_hidden`.

---

## 8. Nueva representación

La entrada original era:

```text
[2, 3]
```

La capa oculta produce:

```text
[0, 5]
```

Esto es una nueva representación de la entrada.

Los valores ya no representan directamente las features originales. Representan respuestas de neuronas a distintas combinaciones de features.

Podemos pensar en cada neurona como un detector:

```text
neurona 1 → detecta una combinación
neurona 2 → detecta otra combinación
```

Deep Learning aprende nuevas features mediante este tipo de transformaciones.

---

## 9. Capa de salida

La capa de salida recibe:

```text
a_hidden = [0, 5]
```

Sus parámetros son:

```python
output_weights = [
    [2.0, -0.5]
]

output_biases = [1.0]
```

La única neurona de salida tiene:

```text
weights = [2.0, -0.5]
bias = 1.0
```

Calculamos:

```text
prediction = 0×2.0 + 5×(-0.5) + 1.0
prediction = 0 - 2.5 + 1
prediction = -1.5
```

La salida de la red es:

```text
[-1.5]
```

---

## 10. Forward pass completo

El proceso completo es:

```text
inputs = [2, 3]

↓ capa oculta lineal

z_hidden = [-1.5, 5]

↓ ReLU

a_hidden = [0, 5]

↓ capa de salida

prediction = [-1.5]
```

Matemáticamente:

```text
z_hidden = linear_hidden(x)
a_hidden = ReLU(z_hidden)
prediction = linear_output(a_hidden)
```

---

## 11. Convención de shapes utilizada en el código

En nuestros scripts desde cero, cada fila de la matriz de pesos corresponde a una neurona.

La convención es:

```text
weights shape = (número de neuronas, número de entradas)
```

Para la capa oculta:

```text
hidden_weights shape = (2, 2)
```

Interpretación:

```text
2 neuronas
2 entradas por neurona
```

Para la capa de salida:

```text
output_weights shape = (1, 2)
```

Interpretación:

```text
1 neurona
2 entradas procedentes de la capa oculta
```

Shapes completos:

```text
inputs shape = (2,)
hidden_weights shape = (2, 2)
hidden_biases shape = (2,)
hidden_outputs shape = (2,)
output_weights shape = (1, 2)
output_biases shape = (1,)
prediction shape = (1,)
```

---

## 12. Otra convención habitual

En notación matemática también es frecuente representar los pesos como:

```text
W shape = (número de entradas, número de salidas)
```

Con esa convención:

```text
x · W
```

produce directamente las salidas.

Por ejemplo:

```text
x shape = (2,)
W_output shape = (2, 1)
prediction shape = (1,)
```

Ambas convenciones son válidas.

La diferencia es que las matrices están transpuestas.

Lo importante es no mezclar ambas convenciones dentro del mismo cálculo.

---

## 13. Por qué cada neurona tiene pesos diferentes

Todas las neuronas de una capa reciben la misma entrada.

Pero cada una tiene pesos y bias diferentes.

Por eso pueden aprender patrones distintos:

```text
neurona 1 → combinación A de features
neurona 2 → combinación B de features
```

Si todas tuvieran exactamente los mismos parámetros, producirían la misma salida y no aportarían detectores diferentes.

---

## 14. Código

Archivo:

```text
code/fundamentals/neural_network_forward.py
```

Código:

```python
def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def relu(x):
    return max(0, x)


def neuron_forward(inputs, weights, bias, activation=None):
    z = dot_product(inputs, weights) + bias

    if activation is None:
        return z

    return activation(z)


def layer_forward(inputs, weights, biases, activation=None):
    if len(weights) != len(biases):
        raise ValueError("Debe haber un bias por neurona")

    outputs = []

    for i in range(len(weights)):
        output = neuron_forward(
            inputs,
            weights[i],
            biases[i],
            activation,
        )

        outputs.append(output)

    return outputs


inputs = [2, 3]

hidden_weights = [
    [0.5, -1.0],
    [1.5, 1.0],
]

hidden_biases = [0.5, -1.0]

output_weights = [
    [2.0, -0.5],
]

output_biases = [1.0]

hidden_outputs = layer_forward(
    inputs,
    hidden_weights,
    hidden_biases,
    relu,
)

prediction = layer_forward(
    hidden_outputs,
    output_weights,
    output_biases,
)

print("Inputs:", inputs)
print("Hidden outputs:", hidden_outputs)
print("Prediction:", prediction)
```

---

## 15. Salida observada

```text
Inputs: [2, 3]
Hidden outputs: [0, 5.0]
Prediction: [-1.5]
```

---

## 16. Segundo ejemplo

Cambiamos la entrada:

```text
inputs = [-2, 1]
```

Primera neurona:

```text
z1 = (-2)×0.5 + 1×(-1.0) + 0.5
z1 = -1.5

a1 = ReLU(-1.5)
a1 = 0
```

Segunda neurona:

```text
z2 = (-2)×1.5 + 1×1.0 - 1.0
z2 = -3

a2 = ReLU(-3)
a2 = 0
```

Capa oculta:

```text
a_hidden = [0, 0]
```

Capa de salida:

```text
prediction = 0×2.0 + 0×(-0.5) + 1.0
prediction = 1.0
```

Resultado:

```text
Prediction: [1.0]
```

Cuando todas las neuronas ocultas se apagan, la salida puede depender únicamente del bias de la última capa.

---

## 17. Forward pass frente a entrenamiento

El forward pass calcula una predicción:

```text
entrada
→ capas
→ predicción
```

El entrenamiento incluye más pasos:

```text
forward pass
→ calcular pérdida
→ calcular gradientes
→ actualizar parámetros
→ repetir
```

El forward pass forma parte del entrenamiento, pero también se usa durante la inferencia, cuando un modelo entrenado realiza predicciones sin modificar sus parámetros.

---

## 18. Idea fundamental

**Una red neuronal transforma una entrada mediante capas lineales y activaciones. El forward pass calcula la predicción usando los parámetros actuales.**

---

## 19. Conceptos clave

* Red neuronal
* Forward pass
* Capa oculta
* Capa de salida
* Neurona
* Peso
* Bias
* Salida lineal
* Activación
* ReLU
* Representación interna
* Feature aprendida
* Inferencia
* Shape

---

## 20. Errores comunes

### Error 1: pensar que una neurona con salida cero estará siempre apagada

La salida depende de la entrada.

Una neurona puede estar apagada para un ejemplo y activa para otro.

### Error 2: confundir `z` y `a`

```text
z → salida antes de la activación
a → salida después de la activación
```

### Error 3: mezclar convenciones de matrices

Puede usarse:

```text
(neuronas, entradas)
```

o:

```text
(entradas, salidas)
```

Pero hay que mantener la misma convención durante todo el cálculo.

### Error 4: pensar que el forward pass entrena la red

El forward pass solo calcula la predicción.

Para entrenar también necesitamos pérdida, gradientes y actualización de parámetros.

---

## 21. Pregunta del ingeniero

¿Qué problema resuelve una capa oculta?

Una capa oculta transforma las features originales en nuevas características aprendidas. Sus neuronas pueden responder a diferentes combinaciones de entrada, permitiendo que las capas posteriores trabajen con representaciones más útiles para producir la predicción.

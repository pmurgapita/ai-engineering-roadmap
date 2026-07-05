# Lesson 014 — Lab: Linear Regression With Multiple Features

## Objetivo

Implementar una regresión lineal con varias features desde cero y comprender cómo un modelo aprende un peso por cada característica de entrada.

Al terminar esta lección deberías entender:

* cómo representar una entrada con varias features,
* por qué necesitamos un vector de pesos,
* cómo se calcula `y_pred = x · w + b`,
* cómo se actualizan varios pesos durante el entrenamiento,
* por qué la escala de los datos afecta al aprendizaje.

---

## 1. Punto de partida

Hasta ahora hemos usado modelos con una sola entrada:

```text
y_pred = xw + b
```

En esta lección pasamos a varias entradas:

```text
y_pred = x1w1 + x2w2 + x3w3 + b
```

En forma compacta:

```text
y_pred = x · w + b
```

Esto es mucho más parecido a los modelos usados en problemas reales.

---

## 2. Problema del laboratorio

Queremos predecir precios de viviendas usando tres features:

```text
metros cuadrados
habitaciones
distancia al centro
```

Cada vivienda se representa como un vector:

```text
[metros, habitaciones, distancia]
```

Ejemplo:

```text
[80, 3, 5]
```

El modelo tendrá un peso para cada feature:

```text
[w_metros, w_habitaciones, w_distancia]
```

---

## 3. Modelo

La predicción se calcula así:

```text
precio_predicho =
metros × w1
+ habitaciones × w2
+ distancia × w3
+ b
```

Ejemplo:

```text
x = [80, 3, 5]
w = [2500, 12000, -7000]
b = 40000
```

Cálculo:

```text
predicción =
80 × 2500
+ 3 × 12000
+ 5 × (-7000)
+ 40000
```

```text
predicción =
200000 + 36000 - 35000 + 40000
```

```text
predicción = 241000
```

---

## 4. Producto escalar

La parte:

```text
x1w1 + x2w2 + x3w3
```

es un producto escalar.

Por eso podemos escribir:

```text
y_pred = x · w + b
```

Donde:

* `x` es el vector de features,
* `w` es el vector de pesos,
* `b` es el sesgo,
* `y_pred` es la predicción.

Expandido:

```text
y_pred = x1w1 + x2w2 + x3w3 + b
```

---

## 5. Datos del laboratorio

Usamos un dataset artificial pequeño:

```python
houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

prices = [
    139,
    241,
    376,
    167,
    319,
]
```

Los precios están expresados en miles de euros.

Por ejemplo:

```text
139 → 139000 €
241 → 241000 €
376 → 376000 €
```

---

## 6. Regla original

Los datos siguen aproximadamente esta regla:

```text
precio_en_miles =
metros × 2.5
+ habitaciones × 12
+ distancia × (-7)
+ 40
```

Por tanto, los valores esperados serían aproximadamente:

```text
w ≈ [2.5, 12, -7]
b ≈ 40
```

El modelo no recibe esta regla directamente.

Solo recibe ejemplos.

---

## 7. Por qué escalamos los precios

Si usáramos precios en euros:

```text
139000, 241000, 376000
```

los errores serían muy grandes.

Eso podría producir gradientes enormes y volver el entrenamiento inestable.

Al trabajar en miles de euros:

```text
139, 241, 376
```

los números son más manejables.

Esto es un ejemplo de escalado de datos.

---

## 8. Código

Archivo recomendado:

```text
code/fundamentals/multiple_features_regression.py
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


def predict(features, weights, bias):
    return dot_product(features, weights) + bias


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(features, y_true, y_pred):
    error = y_true - y_pred

    weight_gradients = []

    for feature in features:
        weight_gradients.append(-2 * feature * error)

    bias_gradient = -2 * error

    return weight_gradients, bias_gradient


houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

prices = [
    139,
    241,
    376,
    167,
    319,
]

weights = [0.0, 0.0, 0.0]
bias = 0.0

learning_rate = 0.00001
epochs = 1000

for epoch in range(epochs):
    total_loss = 0

    total_weight_gradients = [0.0, 0.0, 0.0]
    total_bias_gradient = 0

    for features, price in zip(houses, prices):
        prediction = predict(features, weights, bias)

        loss = squared_error(price, prediction)
        weight_gradients, bias_gradient = gradients(features, price, prediction)

        total_loss += loss

        for i in range(len(weights)):
            total_weight_gradients[i] += weight_gradients[i]

        total_bias_gradient += bias_gradient

    mean_loss = total_loss / len(houses)

    mean_weight_gradients = [
        gradient / len(houses)
        for gradient in total_weight_gradients
    ]

    mean_bias_gradient = total_bias_gradient / len(houses)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:04d} | "
            f"weights = {[round(w, 4) for w in weights]} | "
            f"bias = {bias:.4f} | "
            f"loss = {mean_loss:.4f}"
        )

    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * mean_weight_gradients[i]

    bias = bias - learning_rate * mean_bias_gradient

print("Final weights:", weights)
print("Final bias:", bias)
print("Final loss:", mean_loss)
```

---

## 9. Resultado observado

Salida aproximada:

```text
Epoch 0000 | weights = [0.0, 0.0, 0.0] | bias = 0.0000 | loss = 69685.6000
Epoch 0100 | weights = [3.0629, 0.1005, 0.0606] | bias = 0.0272 | loss = 150.8940
Epoch 0200 | weights = [3.0669, 0.1023, -0.0227] | bias = 0.0206 | loss = 143.8935
Epoch 0300 | weights = [3.0708, 0.104, -0.1038] | bias = 0.0142 | loss = 137.2606
Epoch 0400 | weights = [3.0746, 0.1055, -0.1827] | bias = 0.0079 | loss = 130.9761
Epoch 0500 | weights = [3.0783, 0.107, -0.2595] | bias = 0.0018 | loss = 125.0217
Epoch 0600 | weights = [3.0819, 0.1085, -0.3343] | bias = -0.0041 | loss = 119.3799
Epoch 0700 | weights = [3.0854, 0.1098, -0.4071] | bias = -0.0099 | loss = 114.0344
Epoch 0800 | weights = [3.0888, 0.111, -0.478] | bias = -0.0155 | loss = 108.9697
Epoch 0900 | weights = [3.0921, 0.1122, -0.5469] | bias = -0.0210 | loss = 104.1709
Final weights: [3.0953599202016187, 0.11326125638399409, -0.614072884987337]
Final bias: -0.026259413300628694
Final loss: 99.66831730854753
```

---

## 10. Interpretación del resultado

La pérdida baja mucho al principio:

```text
69685.6 → 150.894
```

Después sigue bajando, pero más lentamente:

```text
150.894 → 99.668
```

Esto significa que el modelo está aprendiendo, aunque no llegue exactamente a los pesos esperados.

---

## 11. Por qué los pesos no coinciden con la regla original

La regla original era aproximadamente:

```text
w ≈ [2.5, 12, -7]
b ≈ 40
```

Pero el modelo aprendió algo parecido a:

```text
w ≈ [3.095, 0.113, -0.614]
b ≈ -0.026
```

Esto ocurre por varias razones:

1. Tenemos pocos datos.
2. Las features tienen escalas diferentes.
3. El learning rate es pequeño.
4. Entrenamos un número limitado de épocas.
5. Algunos parámetros pueden compensarse entre sí.

La razón más importante aquí es la escala de las features.

---

## 12. El problema de la escala

Las features tienen rangos muy distintos:

```text
metros:       50 - 120
habitaciones: 1 - 4
distancia:    2 - 8
```

El modelo puede reducir mucho la pérdida usando principalmente la feature de metros, porque sus valores son mucho mayores.

Por eso el peso de metros cambia más rápidamente.

Esto nos lleva a una idea clave:

```text
Para entrenar modelos de forma estable, muchas veces necesitamos escalar o normalizar las features.
```

Lo estudiaremos en profundidad más adelante.

---

## 13. Predicción con el modelo aprendido

Podemos usar los pesos aprendidos para predecir una vivienda nueva.

Ejemplo:

```python
def predict_price(meters, rooms, distance):
    features = [meters, rooms, distance]
    return predict(features, weights, bias)


print(predict_price(90, 3, 4))
```

Con el modelo aprendido, puede salir algo cercano a:

```text
276.44
```

La regla original daría:

```text
precio = 90 × 2.5 + 3 × 12 + 4 × (-7) + 40

precio = 225 + 36 - 28 + 40

precio = 273
```

El resultado aprendido es bastante cercano.

Esto muestra que un modelo puede hacer predicciones razonables aunque sus pesos no coincidan exactamente con los valores teóricos.

---

## 14. Qué significa realmente `y_pred = x · w + b`

La expresión:

```text
y_pred = x · w + b
```

significa que el modelo:

1. toma un vector de entrada,
2. multiplica cada feature por su peso correspondiente,
3. suma todos esos productos,
4. añade el sesgo,
5. produce una predicción.

Expandido:

```text
y_pred = x1w1 + x2w2 + x3w3 + b
```

---

## 15. Relación con modelos grandes

Este laboratorio usa solo tres pesos y un sesgo.

Pero el mecanismo es el mismo en modelos más grandes:

```text
muchas features
muchos pesos
muchos sesgos
muchas predicciones
muchos gradientes
```

Una red neuronal no cambia la idea fundamental.

La escala.

---

## 16. Idea fundamental

**Cuando una entrada tiene varias features, el modelo aprende un peso por cada feature y los ajusta todos para reducir la pérdida.**

---

## 17. Conceptos clave

* Regresión lineal múltiple
* Feature
* Vector de entrada
* Vector de pesos
* Producto escalar
* Sesgo
* Escalado de datos
* Gradiente por parámetro
* Predicción
* Pérdida

---

## 18. Preguntas de repaso

1. ¿Por qué necesitamos un vector de pesos?
2. ¿Qué significa `y_pred = x · w + b`?
3. ¿Qué ocurre con la pérdida durante el entrenamiento?
4. ¿Por qué los pesos aprendidos pueden no coincidir con los valores teóricos?
5. ¿Por qué la escala de las features afecta al entrenamiento?
6. ¿Qué diferencia hay entre regresión lineal simple y regresión lineal con varias features?

---

## 19. Errores comunes

### Error 1: confundir producto escalar con producto vectorial

En este modelo usamos producto escalar:

```text
x · w = x1w1 + x2w2 + x3w3
```

No producto vectorial.

---

### Error 2: esperar que los pesos aprendidos coincidan exactamente con los teóricos

Con pocos datos, escalas distintas y entrenamiento limitado, los pesos pueden no coincidir exactamente.

---

### Error 3: pensar que si los pesos son raros el modelo no sirve

Lo importante es evaluar las predicciones y la pérdida, no solo mirar los pesos.

---

### Error 4: ignorar la escala de las features

Si una feature tiene valores mucho mayores que otra, puede dominar el entrenamiento.

---

## 20. Pregunta del ingeniero

Si tuviera que entrenar un modelo con muchas características, ¿qué problema resuelve usar un vector de pesos?

Respuesta esperada:

Un vector de pesos permite asignar una influencia distinta a cada feature de entrada. Así el modelo puede combinar varias señales para producir una predicción y ajustar cada peso según cómo contribuya al error.

# Lesson 022 — Functions: The Model as a Transformation

## Objetivo

Comprender que un modelo de Machine Learning puede verse como una función parametrizada que transforma entradas en salidas.

Al terminar esta lección deberías entender:

* qué es una función,
* qué significa que una función tenga parámetros,
* qué diferencia hay entre función real y función aprendida,
* cómo se relacionan funciones con modelos,
* qué es composición de funciones,
* por qué una red neuronal puede verse como una composición de funciones,
* por qué la función de pérdida también es una función.

---

## 1. Qué es una función

Una función toma una entrada y produce una salida.

```text
entrada → función → salida
```

Matemáticamente:

```text
y = f(x)
```

Ejemplo:

```text
f(x) = 2x
```

Si:

```text
x = 3
```

entonces:

```text
f(3) = 2 × 3 = 6
```

Una función puede verse como una regla o transformación.

---

## 2. Función como máquina

Podemos pensar en una función como una máquina:

```text
         ┌─────────┐
x ─────▶ │   f     │ ─────▶ y
         └─────────┘
```

Ejemplo:

```text
         ┌────────────┐
3 ─────▶ │ multiplica │ ─────▶ 6
         │   por 2    │
         └────────────┘
```

La máquina siempre aplica una regla.

---

## 3. Funciones con parámetros

En Machine Learning normalmente no conocemos la función correcta.

Queremos aprenderla.

Por eso usamos funciones con parámetros.

Ejemplo:

```text
f(x) = wx + b
```

Donde:

```text
w = peso
b = sesgo
```

Si:

```text
w = 2
b = 1
```

entonces:

```text
f(x) = 2x + 1
```

Y:

```text
f(3) = 2×3 + 1 = 7
```

La estructura de la función es la misma, pero su comportamiento cambia según los parámetros.

---

## 4. Qué significa función parametrizada

Una función parametrizada es una función cuya regla depende de valores ajustables.

Ejemplo:

```text
f(x) = wx + b
```

Aquí `w` y `b` son parámetros.

Durante el entrenamiento, el modelo cambia esos parámetros para mejorar sus predicciones.

---

## 5. Aprender una función

Cuando entrenamos un modelo, intentamos encontrar buenos valores para sus parámetros.

Ejemplo:

```text
f(x) = wx + b
```

Al principio:

```text
w = 0
b = 0
```

El modelo predice mal.

Después del entrenamiento:

```text
w ≈ 2
b ≈ 1
```

El modelo predice mejor.

Podemos decir que el modelo ha aprendido una función aproximada.

---

## 6. Función real vs función aprendida

Supón que la regla real es:

```text
y = 2x + 1
```

Pero el modelo aprende:

```text
y_pred = 2.07x + 0.79
```

No es perfecta, pero se parece.

En Machine Learning muchas veces no aprendemos la función exacta.

Aprendemos una aproximación útil.

```text
función real      → regla ideal o desconocida
función aprendida → aproximación obtenida desde datos
```

En problemas reales puede haber ruido, excepciones o factores ocultos, así que la función real perfecta puede no ser accesible.

---

## 7. Dominio y codominio

Dos palabras matemáticas útiles:

### Dominio

El dominio es el conjunto de entradas posibles.

Ejemplo:

```text
x = metros cuadrados
```

Dominio posible:

```text
casas entre 20 y 300 m²
```

### Codominio

El codominio es el tipo de salidas posibles.

Ejemplo:

```text
precio estimado
```

Codominio posible:

```text
números reales positivos
```

En Machine Learning podemos pensarlo así:

```text
dominio → qué tipo de datos acepta el modelo
codominio → qué tipo de salida produce
```

---

## 8. Funciones de una variable

Una función de una variable recibe un solo valor.

Ejemplo:

```text
f(x) = 3x + 2
```

Si:

```text
x = 4
```

entonces:

```text
f(4) = 3×4 + 2 = 14
```

Esto se parece a una neurona lineal simple:

```text
y_pred = xw + b
```

---

## 9. Funciones de varias variables

Una función también puede recibir varias entradas.

Ejemplo:

```text
f(x1, x2, x3) = 2.5x1 + 12x2 - 7x3 + 40
```

Donde:

```text
x1 = metros
x2 = habitaciones
x3 = distancia
```

Si:

```text
x1 = 90
x2 = 3
x3 = 4
```

entonces:

```text
f(90, 3, 4)
= 2.5×90 + 12×3 - 7×4 + 40
= 225 + 36 - 28 + 40
= 273
```

Esto es lo mismo que:

```text
y_pred = x · w + b
```

---

## 10. Funciones vectoriales

Una función puede recibir un vector como entrada.

```text
f(x) = x · w + b
```

Donde:

```text
x = [90, 3, 4]
w = [2.5, 12, -7]
b = 40
```

Entonces:

```text
f(x) = 273
```

Entrada:

```text
vector
```

Salida:

```text
escalar
```

Esto es típico en regresión:

```text
muchas features → un número
```

---

## 11. Funciones que devuelven vectores

También podemos tener funciones que devuelven varias salidas.

Ejemplo:

```text
f(x) = xW + b
```

Si:

```text
x shape = (3,)
W shape = (3, 2)
b shape = (2,)
```

Entonces:

```text
f(x) shape = (2,)
```

Entrada:

```text
vector
```

Salida:

```text
vector
```

Ejemplo:

```text
entrada vivienda → [precio, demanda]
```

---

## 12. Funciones sobre batches

Cuando usamos matrices:

```text
X shape = (5, 3)
```

tenemos 5 ejemplos, cada uno con 3 features.

Podemos aplicar una función a todos:

```text
Y = XW + b
```

Si:

```text
W shape = (3, 2)
b shape = (2,)
```

entonces:

```text
Y shape = (5, 2)
```

Interpretación:

```text
5 ejemplos → 2 salidas por ejemplo
```

Esto es una función aplicada a un batch.

---

## 13. Composición de funciones

Podemos encadenar funciones.

Ejemplo:

```text
h(x) = g(f(x))
```

Esto significa:

```text
primero aplicamos f
después aplicamos g
```

Visualmente:

```text
x → f → salida intermedia → g → salida final
```

La salida de una función se convierte en la entrada de otra.

---

## 14. Redes neuronales como composición de funciones

Una red neuronal puede verse como una función grande formada por muchas funciones pequeñas.

Ejemplo:

```text
entrada
→ capa lineal
→ activación
→ capa lineal
→ activación
→ salida
```

Matemáticamente:

```text
modelo(x) = f3(f2(f1(x)))
```

Durante el entrenamiento, ajustamos los parámetros de esas funciones.

Ejemplo:

```text
W1, b1, W2, b2...
```

Una red neuronal no es magia.

Es una composición de transformaciones parametrizadas.

---

## 15. Función de pérdida

La pérdida también es una función.

```text
loss = L(y_true, y_pred)
```

Ejemplo:

```text
L(y_true, y_pred) = (y_true - y_pred)²
```

Esta función toma:

```text
respuesta real
predicción
```

y produce:

```text
un número que mide el error
```

---

## 16. MSE como función de pérdida

Cuando tenemos varias predicciones, podemos usar MSE:

```text
MSE = media de los errores cuadrados
```

Ejemplo:

```text
real_values = [139, 241, 376]
predicted_values = [121, 241, 374]
```

Errores:

```text
139 - 121 = 18
241 - 241 = 0
376 - 374 = 2
```

Errores cuadrados:

```text
18² = 324
0² = 0
2² = 4
```

Media:

```text
(324 + 0 + 4) / 3 = 109.33333333333333
```

Resultado:

```text
MSE = 109.33333333333333
```

El MSE resume varios errores individuales en un solo número.

---

## 17. Entrenamiento como optimización de funciones

Podemos ver el entrenamiento así:

```text
modelo: f(x; parámetros)
pérdida: L(y_true, f(x; parámetros))
objetivo: encontrar parámetros que minimicen L
```

En palabras simples:

```text
queremos ajustar los parámetros del modelo para que la función de pérdida sea lo más baja posible
```

---

## 18. Código

Archivo recomendado:

```text
code/fundamentals/functions_as_models.py
```

Código:

```python
def linear_function(x, w, b):
    return x * w + b


def house_price_function(features, weights, bias):
    total = 0

    for i in range(len(features)):
        total += features[i] * weights[i]

    return total + bias


def squared_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_squared_error(y_true_values, y_pred_values):
    if len(y_true_values) != len(y_pred_values):
        raise ValueError("Las listas deben tener la misma longitud")

    total_error = 0

    for i in range(len(y_true_values)):
        error = y_true_values[i] - y_pred_values[i]
        squared_error = error ** 2
        total_error += squared_error

    return total_error / len(y_true_values)


def apply_function_to_batch(batch, weights, bias):
    predictions = []

    for features in batch:
        prediction = house_price_function(features, weights, bias)
        predictions.append(prediction)

    return predictions


print("Simple function:")
print(linear_function(3, 2, 1))

features = [90, 3, 4]
weights = [2.5, 12, -7]
bias = 40

prediction = house_price_function(features, weights, bias)

print("\nHouse prediction:")
print(prediction)

loss = squared_loss(273, prediction)

print("\nLoss:")
print(loss)

batch = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
]

batch_predictions = apply_function_to_batch(batch, weights, bias)

print("\nBatch predictions:")
print(batch_predictions)

real_values = [139, 241, 376]
predicted_values = [121, 241, 374]

mse = mean_squared_error(real_values, predicted_values)

print("\nMSE:")
print(mse)
```

---

## 19. Salida observada

```text
Simple function:
7

House prediction:
273.0

Loss:
0.0

Batch predictions:
[121.0, 241.0, 374.0]

MSE:
109.33333333333333
```

---

## 20. Ejemplos resueltos

### Ejemplo 1

```text
f(x) = 4x - 3
x = 5
```

Cálculo:

```text
f(5) = 4×5 - 3
f(5) = 20 - 3
f(5) = 17
```

### Ejemplo 2

```text
f(x1, x2, x3) = 2x1 - x2 + 3x3
```

Con:

```text
x1 = 4
x2 = 5
x3 = 2
```

Cálculo:

```text
f(4, 5, 2) = 2×4 - 5 + 3×2
f(4, 5, 2) = 8 - 5 + 6
f(4, 5, 2) = 9
```

---

## 21. Idea fundamental

**Un modelo de Machine Learning puede entenderse como una función parametrizada que transforma entradas en salidas.**

---

## 22. Conceptos clave

* Función
* Entrada
* Salida
* Parámetro
* Función parametrizada
* Función real
* Función aprendida
* Dominio
* Codominio
* Composición de funciones
* Modelo
* Función de pérdida
* Optimización

---

## 23. Preguntas de repaso

1. ¿Qué es una función?
2. ¿Qué significa que una función sea parametrizada?
3. ¿Qué diferencia hay entre una función real y una función aprendida?
4. ¿Por qué un modelo puede verse como una función?
5. ¿Qué significa componer funciones?
6. ¿Por qué una red neuronal puede verse como composición de funciones?
7. ¿Por qué la función de pérdida también es una función?
8. ¿Qué significa entrenar un modelo desde el punto de vista de funciones?

---

## 24. Errores comunes

### Error 1: pensar que una función solo puede recibir un número

Una función puede recibir un escalar, un vector, una matriz o un tensor.

---

### Error 2: pensar que una función solo puede devolver un número

Una función puede devolver un escalar, un vector, una matriz o un tensor.

---

### Error 3: confundir función real con función aprendida

La función aprendida es una aproximación obtenida desde datos.

No tiene por qué ser perfecta.

---

### Error 4: olvidar que la pérdida también es una función

La pérdida transforma una predicción y una respuesta real en un número que mide error.

---

## 25. Pregunta del ingeniero

Si tengo datos de entrada y quiero producir predicciones, ¿qué problema resuelve pensar en el modelo como una función?

Respuesta esperada:

Pensar en el modelo como una función permite entender que el modelo transforma entradas en salidas mediante una regla parametrizada. Entrenar consiste en ajustar esos parámetros para que la transformación produzca predicciones cada vez más cercanas a la realidad.

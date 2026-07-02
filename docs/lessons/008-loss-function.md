# Lesson 008 — How Does an AI Know It Is Wrong?

## Objetivo

Comprender qué es una función de pérdida, por qué es necesaria para entrenar modelos de Machine Learning y cómo permite medir si una red neuronal está aprendiendo.

---

## 1. Punto de partida

En la lección anterior vimos que una neurona artificial básica calcula:

```text
salida = entrada · pesos + sesgo
```

O matemáticamente:

```text
y = x · w + b
```

Pero si una neurona produce una salida, necesitamos saber si esa salida es buena o mala.

Ahí aparece la función de pérdida.

---

## 2. Error

El error es la diferencia entre el valor real y la predicción del modelo.

```text
error = valor real - predicción
```

Ejemplo:

```text
valor real = 360000
predicción = 340000

error = 360000 - 340000
error = 20000
```

El modelo se ha quedado corto por 20.000 euros.

---

## 3. Función de pérdida

Una función de pérdida mide cuánto se ha equivocado un modelo.

No solo queremos saber que hay error.

Queremos convertir ese error en un número que podamos minimizar.

```text
predicción
    ↓
comparación con valor real
    ↓
pérdida
```

La función de pérdida responde a esta pregunta:

```text
¿Cuánto de mala ha sido esta predicción?
```

---

## 4. Por qué no basta con sumar errores

Supón estos casos:

```text
Caso A:
real = 100
predicción = 90
error = 10

Caso B:
real = 100
predicción = 110
error = -10
```

Si sumamos los errores:

```text
10 + (-10) = 0
```

Parecería que el modelo no se ha equivocado.

Pero eso es falso.

Se ha equivocado dos veces.

Por eso necesitamos transformar los errores para que no se cancelen.

Dos formas comunes son:

```text
valor absoluto del error
```

y

```text
error cuadrático
```

---

## 5. Error cuadrático

El error cuadrático eleva el error al cuadrado.

```text
loss = (real - predicción)²
```

Ejemplo:

```text
real = 100
predicción = 90

loss = (100 - 90)²
loss = 10²
loss = 100
```

Si el error es negativo:

```text
real = 100
predicción = 110

loss = (100 - 110)²
loss = (-10)²
loss = 100
```

Al elevar al cuadrado, la pérdida siempre es positiva.

---

## 6. Por qué elevar al cuadrado

Elevar el error al cuadrado tiene varias ventajas:

1. Elimina signos negativos.
2. Penaliza más los errores grandes.
3. Es matemáticamente cómodo para derivar.

Ejemplo:

```text
error = 2  → loss = 4
error = 10 → loss = 100
error = 50 → loss = 2500
```

Un error grande genera una pérdida mucho mayor.

Esto hace que el modelo tenga más presión para corregir fallos grandes.

---

## 7. Mean Squared Error

Cuando tenemos muchos ejemplos, no miramos un único error.

Calculamos la media de todos los errores cuadrados.

Esto se llama Mean Squared Error, o MSE.

```text
MSE = media de (real - predicción)²
```

Ejemplo:

```text
reales        = [100, 200, 300]
predicciones  = [90, 220, 310]
```

Errores:

```text
100 - 90  = 10
200 - 220 = -20
300 - 310 = -10
```

Errores cuadrados:

```text
10²   = 100
(-20)² = 400
(-10)² = 100
```

Media:

```text
MSE = (100 + 400 + 100) / 3
MSE = 600 / 3
MSE = 200
```

---

## 8. Ejemplo resuelto

Datos:

```text
reales        = [10, 20, 30]
predicciones  = [12, 18, 33]
```

Errores:

```text
10 - 12 = -2
20 - 18 = 2
30 - 33 = -3
```

Errores cuadrados:

```text
(-2)² = 4
2²    = 4
(-3)² = 9
```

MSE:

```text
MSE = (4 + 4 + 9) / 3
MSE = 17 / 3
MSE = 5.666...
```

---

## 9. Relación con el aprendizaje

Antes dijimos:

```text
Aprender consiste en ajustar pesos y sesgos para cometer menos errores.
```

Ahora podemos decirlo mejor:

```text
Aprender consiste en ajustar pesos y sesgos para minimizar la función de pérdida.
```

Flujo básico:

```text
pesos actuales
      ↓
predicción
      ↓
comparación con valor real
      ↓
pérdida
      ↓
ajuste de pesos
      ↓
menor pérdida
```

Sin función de pérdida, el modelo no tiene forma de saber si está mejorando o empeorando.

---

## 10. La pérdida como brújula

Podemos imaginar la pérdida como la altura de una montaña.

```text
pérdida alta → mal modelo
pérdida baja → mejor modelo
```

Entrenar un modelo consiste en bajar poco a poco hacia una zona de menor pérdida.

Más adelante estudiaremos el descenso del gradiente, que es el método que permite decidir hacia dónde mover los pesos para reducir esa pérdida.

---

## 11. Código en Python

Archivo recomendado:

```text
code/fundamentals/loss_function.py
```

Código:

```python
def squared_error(y_true, y_pred):
    error = y_true - y_pred
    return error ** 2


def mean_squared_error(y_true_values, y_pred_values):
    if len(y_true_values) != len(y_pred_values):
        raise ValueError("Las listas deben tener la misma longitud")

    total_loss = 0

    for i in range(len(y_true_values)):
        total_loss += squared_error(y_true_values[i], y_pred_values[i])

    return total_loss / len(y_true_values)


real_price = 360000
predicted_price = 340000

print("Error cuadrático:", squared_error(real_price, predicted_price))


real_values = [100, 200, 300]
predictions = [90, 220, 310]

print("MSE:", mean_squared_error(real_values, predictions))
```

Salida esperada:

```text
Error cuadrático: 400000000
MSE: 200.0
```

---

## 12. Reto opcional: MAE

Otra función de pérdida común es el Mean Absolute Error, o MAE.

En lugar de elevar el error al cuadrado, usamos el valor absoluto.

```text
MAE = media de |real - predicción|
```

Ejemplo:

```text
reales        = [10, 20, 30]
predicciones  = [12, 18, 33]
```

Errores absolutos:

```text
|10 - 12| = 2
|20 - 18| = 2
|30 - 33| = 3
```

MAE:

```text
MAE = (2 + 2 + 3) / 3
MAE = 7 / 3
MAE = 2.333...
```

Implementación:

```python
def mean_absolute_error(y_true_values, y_pred_values):
    if len(y_true_values) != len(y_pred_values):
        raise ValueError("Las listas deben tener la misma longitud")

    total_error = 0

    for i in range(len(y_true_values)):
        error = abs(y_true_values[i] - y_pred_values[i])
        total_error += error

    return total_error / len(y_true_values)
```

---

## 13. Interpretación de la pérdida

Una pérdida alta o baja siempre debe interpretarse en el contexto del problema.

Por ejemplo:

```text
MSE = 200
```

puede ser muy malo si estamos prediciendo notas de 0 a 10.

Pero puede ser aceptable si estamos prediciendo precios de viviendas de cientos de miles de euros.

La pérdida no se interpreta en abstracto.

Se interpreta según la escala del problema.

---

## 14. Idea fundamental

**Una función de pérdida mide cuánto se equivoca un modelo y permite saber si está aprendiendo.**

---

## 15. Conceptos clave

* Error
* Predicción
* Valor real
* Función de pérdida
* Loss function
* Error cuadrático
* Mean Squared Error
* Mean Absolute Error
* Minimización
* Aprendizaje

---

## 16. Preguntas de repaso

1. ¿Qué es el error de una predicción?
2. ¿Qué mide una función de pérdida?
3. ¿Por qué no basta con sumar errores positivos y negativos?
4. ¿Qué es el error cuadrático?
5. ¿Qué es el MSE?
6. ¿Qué significa minimizar una función de pérdida?
7. ¿Por qué una red neuronal necesita una función de pérdida para aprender?

---

## 17. Errores comunes

### Error 1: confundir error con pérdida

El error es la diferencia entre valor real y predicción.

La pérdida es una medida de lo mala que ha sido esa diferencia.

---

### Error 2: sumar errores directamente

Los errores positivos y negativos pueden cancelarse.

Por eso usamos funciones como MSE o MAE.

---

### Error 3: interpretar la pérdida sin contexto

El mismo valor de pérdida puede ser bueno o malo dependiendo del problema y de la escala de los datos.

---

### Error 4: pensar que el modelo aprende solo por calcular la pérdida

Calcular la pérdida solo mide el error.

Para aprender, el modelo necesita usar esa pérdida para ajustar pesos y sesgos.

Ese ajuste lo estudiaremos con descenso del gradiente.

---

## 18. Pregunta del ingeniero

Si tuviera que entrenar un modelo, ¿qué problema resuelve la función de pérdida?

Respuesta esperada:

La función de pérdida proporciona una medida numérica del error del modelo. Sin esa medida, no podríamos saber si las predicciones son buenas o malas ni ajustar los pesos para mejorar.

# Lesson 012 — Learning Weight and Bias

## Objetivo

Comprender cómo una neurona artificial puede aprender no solo un peso, sino también un sesgo.

Al terminar esta lección deberías entender:

* por qué necesitamos el sesgo,
* cómo se aprende una relación del tipo `y = wx + b`,
* cómo se actualizan peso y sesgo durante el entrenamiento,
* por qué una neurona con peso y sesgo es más flexible que una neurona con solo peso.

---

## 1. Punto de partida

En la lección anterior entrenamos una neurona simple para aprender:

```text
y = 2x
```

El modelo era:

```text
predicción = x × w
```

Solo aprendía un parámetro:

```text
w
```

En esta lección damos un paso más y entrenamos una neurona completa:

```text
predicción = x × w + b
```

Donde:

* `w` es el peso,
* `b` es el sesgo,
* ambos se ajustan durante el entrenamiento.

---

## 2. Problema

Queremos que la neurona aprenda esta relación:

```text
y = 2x + 1
```

Ejemplos:

```text
x = 1 → y = 3
x = 2 → y = 5
x = 3 → y = 7
x = 4 → y = 9
```

El modelo no recibe la fórmula.

Solo recibe ejemplos.

Debe aprender los valores adecuados de `w` y `b`.

---

## 3. Datos de entrenamiento

Usamos:

```python
xs = [1, 2, 3, 4]
ys = [3, 5, 7, 9]
```

Cada par representa:

```text
entrada → salida correcta
```

```text
1 → 3
2 → 5
3 → 7
4 → 9
```

---

## 4. Modelo

El modelo será:

```text
y_pred = xw + b
```

En Python:

```python
def predict(x, w, b):
    return x * w + b
```

Esto es una neurona lineal completa.

---

## 5. Por qué necesitamos el sesgo

Si usamos un modelo sin sesgo:

```text
y = wx
```

entonces cuando `x = 0`, siempre ocurre:

```text
y = 0
```

Eso significa que la recta siempre pasa por el origen.

Pero si queremos aprender:

```text
y = 2x + 1
```

entonces cuando `x = 0`:

```text
y = 1
```

La recta no pasa por el origen.

El sesgo permite desplazar la recta hacia arriba o hacia abajo.

```text
sin sesgo: y = wx
con sesgo: y = wx + b
```

---

## 6. Intuición geométrica

Una relación lineal puede escribirse como:

```text
y = wx + b
```

Donde:

* `w` controla la inclinación de la recta,
* `b` controla el punto donde corta el eje vertical.

Para:

```text
y = 2x + 1
```

tenemos:

```text
w = 2
b = 1
```

---

## 7. Función de pérdida

Seguimos usando error cuadrático:

```text
loss = (y_true - y_pred)²
```

Y para todos los ejemplos usamos MSE:

```text
MSE = media de los errores cuadrados
```

La pérdida mide cuánto se equivoca el modelo en sus predicciones.

---

## 8. Gradientes

Ahora tenemos dos parámetros:

```text
w
b
```

Por tanto necesitamos dos gradientes:

```text
gradient_w
gradient_b
```

Para un ejemplo:

```text
y_pred = xw + b
loss = (y_true - y_pred)²
```

Los gradientes son:

```text
gradient_w = -2 × x × (y_true - y_pred)
gradient_b = -2 × (y_true - y_pred)
```

La idea importante es:

```text
gradient_w indica cómo ajustar el peso
gradient_b indica cómo ajustar el sesgo
```

---

## 9. Código

Archivo recomendado:

```text
code/fundamentals/learning_weight_and_bias.py
```

Código:

```python
def predict(x, w, b):
    return x * w + b


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(x, y_true, y_pred):
    error = y_true - y_pred

    gradient_w = -2 * x * error
    gradient_b = -2 * error

    return gradient_w, gradient_b


xs = [1, 2, 3, 4]
ys = [3, 5, 7, 9]

w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    total_gradient_w = 0
    total_gradient_b = 0

    for x, y_true in zip(xs, ys):
        y_pred = predict(x, w, b)

        loss = squared_error(y_true, y_pred)
        gradient_w, gradient_b = gradients(x, y_true, y_pred)

        total_loss += loss
        total_gradient_w += gradient_w
        total_gradient_b += gradient_b

    mean_loss = total_loss / len(xs)
    mean_gradient_w = total_gradient_w / len(xs)
    mean_gradient_b = total_gradient_b / len(xs)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"w = {w:.4f} | "
            f"b = {b:.4f} | "
            f"loss = {mean_loss:.4f} | "
            f"grad_w = {mean_gradient_w:.4f} | "
            f"grad_b = {mean_gradient_b:.4f}"
        )

    w = w - learning_rate * mean_gradient_w
    b = b - learning_rate * mean_gradient_b

final_loss = mean_loss

print(f"Final weight: {w:.4f}")
print(f"Final bias: {b:.4f}")
print(f"Final loss: {final_loss:.4f}")
```

---

## 10. Resultado observado

Salida aproximada:

```text
Epoch 000 | w = 0.0000 | b = 0.0000 | loss = 41.0000 | grad_w = -35.0000 | grad_b = -12.0000
Epoch 010 | w = 1.7572 | b = 0.6071 | loss = 1.0733 | grad_w = -5.6061 | grad_b = -1.9997
Epoch 020 | w = 2.0375 | b = 0.7116 | loss = 0.0396 | grad_w = -0.8788 | grad_b = -0.3891
Epoch 030 | w = 2.0804 | b = 0.7351 | loss = 0.0122 | grad_w = -0.1191 | grad_b = -0.1280
Epoch 040 | w = 2.0851 | b = 0.7453 | loss = 0.0108 | grad_w = 0.0024 | grad_b = -0.0841
Epoch 050 | w = 2.0837 | b = 0.7532 | loss = 0.0102 | grad_w = 0.0213 | grad_b = -0.0751
Epoch 060 | w = 2.0814 | b = 0.7606 | loss = 0.0096 | grad_w = 0.0238 | grad_b = -0.0719
Epoch 070 | w = 2.0790 | b = 0.7677 | loss = 0.0090 | grad_w = 0.0236 | grad_b = -0.0696
Epoch 080 | w = 2.0767 | b = 0.7745 | loss = 0.0085 | grad_w = 0.0229 | grad_b = -0.0675
Epoch 090 | w = 2.0744 | b = 0.7812 | loss = 0.0080 | grad_w = 0.0223 | grad_b = -0.0655
Final weight: 2.0722
Final bias: 0.7876
Final loss: 0.0076
```

El modelo aprendido es aproximadamente:

```text
y ≈ 2.0722x + 0.7876
```

La relación real era:

```text
y = 2x + 1
```

No es perfecto todavía, pero la pérdida ya es baja.

---

## 11. Interpretación

Durante el entrenamiento:

```text
w se acerca a 2
b se acerca a 1
loss baja hacia 0
```

Al principio los cambios son grandes porque la pérdida es alta.

Después los cambios se vuelven más pequeños porque el modelo está más cerca de una buena solución.

Cuando hay más de un parámetro, los ajustes pueden compensarse entre sí.

Por eso `w` puede pasar un poco de `2` mientras `b` todavía está por debajo de `1`.

---

## 12. Diferencia con aprender solo `w`

### Modelo sin sesgo

```text
y_pred = xw
```

Solo puede aprender relaciones que pasan por el origen.

Ejemplo:

```text
y = 2x
```

### Modelo con sesgo

```text
y_pred = xw + b
```

Puede aprender relaciones desplazadas.

Ejemplo:

```text
y = 2x + 1
```

Esto hace que el modelo sea más flexible.

---

## 13. Reto: aprender `y = 3x - 2`

Si usamos:

```python
xs = [1, 2, 3, 4]
ys = [1, 4, 7, 10]
```

la relación real es:

```text
y = 3x - 2
```

Por tanto, el modelo debería aprender aproximadamente:

```text
w ≈ 3
b ≈ -2
```

El proceso es el mismo:

```text
datos → predicción → pérdida → gradientes → actualización de w y b
```

---

## 14. Relación con PyTorch

Más adelante, cuando usemos PyTorch, veremos capas como:

```python
torch.nn.Linear(...)
```

Por dentro, una capa lineal hace una versión más general de:

```text
entrada × pesos + sesgos
```

Esta lección es la versión mínima de esa idea.

---

## 15. Idea fundamental

**El sesgo permite que una neurona aprenda relaciones que no pasan por cero.**

---

## 16. Conceptos clave

* Peso
* Sesgo
* Parámetro
* Predicción
* Error
* Función de pérdida
* Gradiente respecto a `w`
* Gradiente respecto a `b`
* Regresión lineal
* Capa lineal

---

## 17. Preguntas de repaso

1. ¿Qué representa `w`?
2. ¿Qué representa `b`?
3. ¿Por qué una relación como `y = 2x + 1` necesita sesgo?
4. ¿Qué diferencia hay entre `y = wx` y `y = wx + b`?
5. ¿Por qué `w` y `b` se ajustan juntos durante el entrenamiento?
6. ¿Qué ocurre con la pérdida si el entrenamiento funciona correctamente?

---

## 18. Errores comunes

### Error 1: pensar que el sesgo es opcional en todos los casos

A veces se puede entrenar sin sesgo, pero muchos problemas necesitan un desplazamiento para ajustarse bien.

---

### Error 2: esperar que `w` y `b` lleguen exactamente al valor correcto en pocas épocas

El entrenamiento es gradual. Dependiendo del learning rate y del número de épocas, puede tardar más o menos.

---

### Error 3: pensar que un learning rate más alto siempre es mejor

Subir el learning rate puede acelerar el entrenamiento, pero también puede hacerlo inestable.

---

### Error 4: interpretar `w` y `b` por separado sin mirar la predicción completa

Lo importante no es solo que `w` o `b` sean exactos.

Lo importante es que la combinación:

```text
xw + b
```

produzca buenas predicciones.

---

## 19. Pregunta del ingeniero

Si tuviera que construir un modelo más flexible, ¿qué problema resuelve añadir un sesgo?

Respuesta esperada:

Añadir un sesgo permite que el modelo aprenda relaciones que no pasan por el origen. Esto hace posible desplazar la salida hacia arriba o hacia abajo y reducir la pérdida en problemas donde el peso por sí solo no basta.

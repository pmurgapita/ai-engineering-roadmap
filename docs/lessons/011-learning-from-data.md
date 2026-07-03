# Lesson 011 — A Neuron That Learns From Data

## Objetivo

Construir una neurona simple que aprende una relación a partir de datos usando predicción, pérdida, gradiente y actualización de pesos.

Al terminar esta lección deberías entender:

* qué significa aprender desde ejemplos,
* cómo un peso se ajusta usando datos,
* por qué el modelo puede descubrir una relación como `y = 2x`,
* qué diferencia hay entre optimizar una función conocida y aprender desde datos.

---

## 1. Punto de partida

En la lección anterior implementamos descenso del gradiente sobre una función artificial:

```text
loss = (w - 3)²
```

Sabíamos que el mínimo estaba en:

```text
w = 3
```

En esta lección damos un paso más importante: el modelo aprenderá desde datos.

---

## 2. Problema

Queremos que una neurona aprenda esta relación:

```text
y = 2x
```

Ejemplos:

```text
1 → 2
2 → 4
3 → 6
4 → 8
```

El modelo no sabe que la regla es `y = 2x`.

Solo recibe ejemplos.

Debe ajustar su peso para descubrir la relación.

---

## 3. Modelo

Usaremos una neurona sin sesgo para simplificar:

```text
predicción = x × w
```

Donde:

* `x` es la entrada,
* `w` es el peso,
* `predicción` es la salida del modelo.

Si el peso correcto es `2`, entonces:

```text
predicción = x × 2
```

y el modelo habrá aprendido la relación.

---

## 4. Datos de entrenamiento

Usamos:

```python
xs = [1, 2, 3, 4]
ys = [2, 4, 6, 8]
```

Cada par indica:

```text
entrada → salida correcta
```

Ejemplo:

```text
1 → 2
2 → 4
3 → 6
4 → 8
```

---

## 5. Predicción inicial

Si empezamos con:

```python
w = 0
```

entonces:

```text
x = 1 → predicción = 1 × 0 = 0
x = 2 → predicción = 2 × 0 = 0
x = 3 → predicción = 3 × 0 = 0
x = 4 → predicción = 4 × 0 = 0
```

Todas las predicciones son malas.

La pérdida será alta.

---

## 6. Función de pérdida

Para cada ejemplo usamos error cuadrático:

```text
loss = (y_true - y_pred)²
```

Para todos los ejemplos usamos la media de los errores cuadrados:

```text
MSE = media de (y_true - y_pred)²
```

La pérdida nos dice cuánto se equivoca el modelo sobre el conjunto de entrenamiento.

---

## 7. Gradiente

Nuestro modelo es:

```text
y_pred = x × w
```

La pérdida para un ejemplo es:

```text
loss = (y_true - y_pred)²
```

Sustituyendo:

```text
loss = (y_true - xw)²
```

El gradiente respecto al peso `w` es:

```text
gradient = -2 × x × (y_true - y_pred)
```

La idea importante no es memorizar la fórmula.

La idea importante es que el gradiente indica cómo cambiar `w` para reducir el error.

---

## 8. Código

Archivo recomendado:

```text
code/fundamentals/learning_neuron.py
```

Código:

```python
def predict(x, w):
    return x * w


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradient(x, y_true, y_pred):
    return -2 * x * (y_true - y_pred)


xs = [1, 2, 3, 4]
ys = [2, 4, 6, 8]

w = 0.0
learning_rate = 0.01
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    total_gradient = 0

    for x, y_true in zip(xs, ys):
        y_pred = predict(x, w)

        loss = squared_error(y_true, y_pred)
        grad = gradient(x, y_true, y_pred)

        total_loss += loss
        total_gradient += grad

    mean_loss = total_loss / len(xs)
    mean_gradient = total_gradient / len(xs)

    print(
        f"Epoch {epoch:02d} | "
        f"w = {w:.4f} | "
        f"loss = {mean_loss:.4f} | "
        f"gradient = {mean_gradient:.4f}"
    )

    w = w - learning_rate * mean_gradient

print(f"Final weight: {w:.4f}")
```

---

## 9. Resultado observado

Salida inicial:

```text
Epoch 00 | w = 0.0000 | loss = 30.0000 | gradient = -30.0000
Epoch 01 | w = 0.3000 | loss = 21.6750 | gradient = -25.5000
Epoch 02 | w = 0.5550 | loss = 15.6602 | gradient = -21.6750
Epoch 03 | w = 0.7717 | loss = 11.3145 | gradient = -18.4237
Epoch 04 | w = 0.9560 | loss = 8.1747 | gradient = -15.6602
Epoch 05 | w = 1.1126 | loss = 5.9062 | gradient = -13.3112
Epoch 06 | w = 1.2457 | loss = 4.2673 | gradient = -11.3145
Epoch 07 | w = 1.3588 | loss = 3.0831 | gradient = -9.6173
Epoch 08 | w = 1.4550 | loss = 2.2275 | gradient = -8.1747
```

Interpretación:

```text
w aumenta hacia 2
loss disminuye hacia 0
gradient se acerca a 0
```

Esto significa que el modelo está aprendiendo.

---

## 10. Primera actualización paso a paso

Al inicio:

```text
w = 0
mean_gradient = -30
learning_rate = 0.01
```

Actualización:

```text
w = w - learning_rate × gradient
w = 0 - 0.01 × (-30)
w = 0 + 0.3
w = 0.3
```

El peso aumenta porque el gradiente era negativo.

El modelo se mueve hacia el valor correcto.

---

## 11. Por qué el peso se acerca a 2

Los datos siguen la relación:

```text
y = 2x
```

El modelo tiene la forma:

```text
predicción = x × w
```

Para que el modelo produzca buenas predicciones, `w` debe acercarse a `2`.

Cuando `w` se acerca a `2`, la pérdida baja.

Por eso el entrenamiento empuja el peso hacia ese valor.

---

## 12. Diferencia con la lección anterior

### Lección 010

Optimizábamos directamente una función artificial:

```text
loss = (w - 3)²
```

### Lección 011

El modelo aprende una relación desde ejemplos:

```text
xs = [1, 2, 3, 4]
ys = [2, 4, 6, 8]
```

Esto es más parecido al Machine Learning real.

No decimos al modelo cuál es la regla.

Le damos ejemplos y dejamos que ajuste sus parámetros.

---

## 13. Reto: aprender `y = 3x`

Si cambiamos los datos:

```python
xs = [1, 2, 3, 4]
ys = [3, 6, 9, 12]
```

la relación es:

```text
y = 3x
```

El modelo debería aprender:

```text
w ≈ 3
```

El proceso es el mismo:

```text
datos → predicción → pérdida → gradiente → actualización
```

Solo cambia el valor al que converge el peso.

---

## 14. Por qué esto ya es Machine Learning

Aunque este ejemplo es muy pequeño, contiene los ingredientes fundamentales del aprendizaje automático:

```text
datos de entrenamiento
modelo
predicción
función de pérdida
gradiente
actualización de parámetros
```

Más adelante cambiaremos:

```text
un peso → millones de pesos
una neurona → muchas neuronas
datos simples → imágenes, texto, audio
```

Pero el mecanismo central seguirá siendo el mismo.

---

## 15. Idea fundamental

**Entrenar consiste en usar datos para ajustar parámetros hasta que las predicciones se parezcan a las respuestas correctas.**

---

## 16. Conceptos clave

* Datos de entrenamiento
* Entrada
* Salida correcta
* Predicción
* Peso
* Pérdida
* Gradiente
* Learning rate
* Actualización de parámetros
* Machine Learning supervisado

---

## 17. Preguntas de repaso

1. ¿Qué relación intentaba aprender el modelo?
2. ¿Qué representa el peso `w`?
3. ¿Por qué el peso se acerca a `2`?
4. ¿Qué ocurre con la pérdida durante el entrenamiento?
5. ¿Qué diferencia hay entre optimizar `loss = (w - 3)²` y aprender desde datos?
6. ¿Por qué este ejemplo ya puede considerarse Machine Learning?

---

## 18. Errores comunes

### Error 1: pensar que el learning rate va cambiando

En este laboratorio, el learning rate se mantiene constante.

Lo que cambia es el peso `w`.

---

### Error 2: pensar que el modelo sabe que la respuesta es `2`

El modelo no sabe la regla.

Solo ajusta el peso usando los datos, la pérdida y el gradiente.

---

### Error 3: confundir datos con regla

Los datos muestran ejemplos de la relación, pero no expresan directamente la regla.

El modelo debe descubrir una aproximación mediante entrenamiento.

---

### Error 4: pensar que este ejemplo es demasiado simple para ser Machine Learning

Aunque sea pequeño, contiene los elementos esenciales del aprendizaje supervisado.

---

## 19. Pregunta del ingeniero

Si tuviera que construir un modelo que aprende desde ejemplos, ¿qué problema resuelve el entrenamiento sobre datos?

Respuesta esperada:

El entrenamiento sobre datos permite que el modelo ajuste sus parámetros a partir de ejemplos reales, en lugar de recibir reglas programadas manualmente. Esto le permite aprender relaciones entre entradas y salidas y hacer predicciones sobre nuevos casos.

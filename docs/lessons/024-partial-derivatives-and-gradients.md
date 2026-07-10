# Lesson 024 — Partial Derivatives and Gradients

## Objetivo

Comprender qué son las derivadas parciales y el gradiente, y cómo se usan para ajustar muchos parámetros de un modelo al mismo tiempo.

Al terminar esta lección deberías entender:

* qué es una derivada parcial,
* qué es el gradiente,
* por qué el gradiente es un vector,
* cómo se calcula el gradiente en una neurona lineal,
* cómo se actualizan varios pesos y sesgos,
* por qué los gradientes deben recalcularse en cada paso de entrenamiento.

---

## 1. Punto de partida

En la sesión anterior vimos:

```text id="wfpbs2"
derivada → cambio respecto a una variable
```

Ahora necesitamos extender esa idea.

Un modelo real no suele tener un solo parámetro.

Puede tener muchos:

```text id="b4yrl4"
w1, w2, w3, b
```

Por ejemplo:

```text id="hh4jm5"
y_pred = x1w1 + x2w2 + x3w3 + b
```

Entonces la pérdida depende de muchos valores:

```text id="cfghyy"
loss = L(w1, w2, w3, b)
```

Ahora queremos saber:

```text id="xbihqe"
si cambio solo w1, ¿qué pasa con la pérdida?
si cambio solo w2, ¿qué pasa con la pérdida?
si cambio solo b, ¿qué pasa con la pérdida?
```

Para responder usamos derivadas parciales.

---

## 2. Derivada parcial

Una derivada parcial mide cómo cambia una función respecto a una variable, manteniendo las demás constantes.

Ejemplo:

```text id="9letk7"
f(x, y) = 3x + 2y
```

Si queremos saber cómo cambia la función al cambiar `x`, mantenemos `y` constante.

La derivada parcial respecto a `x` es:

```text id="v1ttq2"
∂f/∂x = 3
```

Porque por cada unidad que aumenta `x`, `f` aumenta 3.

La derivada parcial respecto a `y` es:

```text id="bd14zf"
∂f/∂y = 2
```

Porque por cada unidad que aumenta `y`, `f` aumenta 2.

Idea clave:

```text id="72ej8m"
cambio una variable y dejo las demás quietas
```

---

## 3. Notación

La derivada normal usa:

```text id="kjz0jj"
d
```

Ejemplo:

```text id="2fqo7g"
df/dx
```

La derivada parcial usa:

```text id="1xcu7z"
∂
```

Ejemplo:

```text id="brgr7s"
∂f/∂x
```

Se lee:

```text id="bbefxs"
derivada parcial de f respecto a x
```

No hace falta obsesionarse con el símbolo. Lo importante es la idea.

---

## 4. Ejemplo intuitivo

Función:

```text id="6p7cqa"
f(x, y) = x² + y²
```

Derivada parcial respecto a `x`:

```text id="oqn520"
∂f/∂x = 2x
```

Derivada parcial respecto a `y`:

```text id="50datv"
∂f/∂y = 2y
```

Si estamos en el punto:

```text id="q9nvj3"
x = 3
y = 4
```

Entonces:

```text id="c6rtsj"
∂f/∂x = 2×3 = 6
∂f/∂y = 2×4 = 8
```

Esto significa:

```text id="kz4uik"
si muevo x un poco, f cambia con sensibilidad 6
si muevo y un poco, f cambia con sensibilidad 8
```

---

## 5. Gradiente

El gradiente junta todas las derivadas parciales en un vector.

Para:

```text id="7nn8gl"
f(x, y) = x² + y²
```

el gradiente es:

```text id="3rku92"
∇f = [∂f/∂x, ∂f/∂y]
```

Como:

```text id="tgplao"
∂f/∂x = 2x
∂f/∂y = 2y
```

entonces:

```text id="mvrixi"
∇f = [2x, 2y]
```

En el punto:

```text id="6selnt"
x = 3
y = 4
```

el gradiente es:

```text id="nllzx1"
∇f = [6, 8]
```

---

## 6. Qué significa el gradiente

El gradiente apunta hacia la dirección de mayor subida de la función.

Si la función es una pérdida:

```text id="me117u"
loss = L(w1, w2)
```

entonces:

```text id="am4ef4"
gradiente = dirección donde la pérdida sube más rápido
```

Como queremos minimizar la pérdida, vamos en dirección contraria:

```text id="aw4jxa"
parámetros = parámetros - learning_rate × gradiente
```

---

## 7. Por qué el gradiente es un vector

Si el modelo tiene varios parámetros:

```text id="0a00sr"
w1, w2, w3
```

necesitamos saber cómo cambia la pérdida respecto a cada uno:

```text id="3l47bg"
∂L/∂w1
∂L/∂w2
∂L/∂w3
```

El gradiente los junta:

```text id="yr52yq"
∇L = [
  ∂L/∂w1,
  ∂L/∂w2,
  ∂L/∂w3
]
```

Por eso el gradiente es un vector.

Cada componente responde:

```text id="qe75n5"
si cambio este parámetro un poco, ¿cómo cambia la pérdida?
```

---

## 8. Actualización de varios pesos

Supón un modelo con pesos:

```text id="t73gay"
w = [w1, w2, w3]
```

Y una pérdida:

```text id="ft9afl"
loss = L(w1, w2, w3)
```

El gradiente será:

```text id="j2lij1"
∇L = [
  ∂L/∂w1,
  ∂L/∂w2,
  ∂L/∂w3
]
```

Luego actualizamos:

```text id="o772db"
w1 = w1 - lr × ∂L/∂w1
w2 = w2 - lr × ∂L/∂w2
w3 = w3 - lr × ∂L/∂w3
```

Esto es lo que ocurre en modelos con muchas features.

---

## 9. Derivadas parciales en una neurona lineal

Modelo:

```text id="8k8mat"
y_pred = x1w1 + x2w2 + b
```

Pérdida:

```text id="tb214w"
loss = (y_true - y_pred)²
```

Queremos saber cómo cambiar:

```text id="t0e45l"
w1
w2
b
```

Los gradientes son:

```text id="mji4ha"
∂loss/∂w1 = -2 × x1 × (y_true - y_pred)
∂loss/∂w2 = -2 × x2 × (y_true - y_pred)
∂loss/∂b  = -2 × (y_true - y_pred)
```

Esto es lo que ya habíamos implementado antes.

Ahora le estamos poniendo nombre formal.

---

## 10. Ejemplo numérico

Datos:

```text id="d9d9mk"
x = [2, 3]
w = [1, 1]
b = 0
y_true = 10
```

Modelo:

```text id="kpwqud"
y_pred = 2×1 + 3×1 + 0 = 5
```

Error:

```text id="deqvx9"
error = y_true - y_pred = 10 - 5 = 5
```

Pérdida:

```text id="jn0au8"
loss = 5² = 25
```

Gradientes:

```text id="q8r5ul"
grad_w1 = -2 × 2 × 5 = -20
grad_w2 = -2 × 3 × 5 = -30
grad_b  = -2 × 5 = -10
```

Gradiente total:

```text id="4hi1dx"
gradient = [-20, -30, -10]
```

Como son negativos, al restarlos los parámetros aumentan.

Con:

```text id="lagm6s"
learning_rate = 0.01
```

Actualización:

```text id="3yzl80"
w1 = 1 - 0.01 × (-20) = 1.2
w2 = 1 - 0.01 × (-30) = 1.3
b  = 0 - 0.01 × (-10) = 0.1
```

La predicción sube, que es lo que necesitamos porque antes era demasiado baja.

---

## 11. Código base

Archivo recomendado:

```text id="sgd9q5"
code/fundamentals/partial_derivatives.py
```

Código:

```python id="z41spx"
def predict(features, weights, bias):
    total = 0

    for i in range(len(features)):
        total += features[i] * weights[i]

    return total + bias


def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(features, y_true, y_pred):
    error = y_true - y_pred

    weight_gradients = []

    for feature in features:
        gradient = -2 * feature * error
        weight_gradients.append(gradient)

    bias_gradient = -2 * error

    return weight_gradients, bias_gradient


features = [2, 3]
weights = [1, 1]
bias = 0
y_true = 10

learning_rate = 0.01

y_pred = predict(features, weights, bias)
loss = squared_error(y_true, y_pred)

weight_gradients, bias_gradient = gradients(features, y_true, y_pred)

print("Initial prediction:", y_pred)
print("Initial loss:", loss)
print("Weight gradients:", weight_gradients)
print("Bias gradient:", bias_gradient)

for i in range(len(weights)):
    weights[i] = weights[i] - learning_rate * weight_gradients[i]

bias = bias - learning_rate * bias_gradient

new_prediction = predict(features, weights, bias)
new_loss = squared_error(y_true, new_prediction)

print("\nUpdated weights:", weights)
print("Updated bias:", bias)
print("New prediction:", new_prediction)
print("New loss:", new_loss)
```

---

## 12. Salida observada

```text id="lmv0ia"
Initial prediction: 5
Initial loss: 25
Weight gradients: [-20, -30]
Bias gradient: -10

Updated weights: [1.2, 1.3]
Updated bias: 0.1
New prediction: 6.4
New loss: 12.959999999999997
```

---

## 13. Interpretación

Al principio:

```text id="wsykc9"
predicción = 5
real = 10
```

El modelo predice demasiado bajo.

El error es positivo:

```text id="glwljt"
y_true - y_pred = 5
```

Los gradientes salen negativos:

```text id="c6bsmx"
[-20, -30, -10]
```

Al aplicar:

```text id="1bzjny"
parámetro = parámetro - lr × gradiente
```

restar un número negativo aumenta el parámetro.

Por eso:

```text id="rr8ohs"
w1 sube de 1 a 1.2
w2 sube de 1 a 1.3
b sube de 0 a 0.1
```

La nueva predicción es:

```text id="x57m3m"
6.4
```

Sigue lejos de `10`, pero está más cerca que `5`.

La pérdida baja:

```text id="zzvo5i"
25 → 12.96
```

Eso significa que el paso de aprendizaje fue en buena dirección.

---

## 14. Entrenamiento durante varias épocas

Para entrenar durante varias épocas, hay que repetir:

```text id="s5rt8n"
predicción
pérdida
gradientes
actualización
```

Código:

```python id="mpwut3"
features = [2, 3]
weights = [1, 1]
bias = 0
y_true = 10

learning_rate = 0.01

for epoch in range(10):
    y_pred = predict(features, weights, bias)
    loss = squared_error(y_true, y_pred)

    weight_gradients, bias_gradient = gradients(features, y_true, y_pred)

    print("\nEpoch:", epoch)
    print("Prediction:", y_pred)
    print("Loss:", loss)
    print("Weight gradients:", weight_gradients)
    print("Bias gradient:", bias_gradient)
    print("Weights:", weights)
    print("Bias:", bias)

    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * weight_gradients[i]

    bias = bias - learning_rate * bias_gradient
```

---

## 15. Por qué hay que recalcular gradientes

Los gradientes dependen de la predicción actual.

Y la predicción depende de los pesos y del bias actuales.

Por eso, después de actualizar parámetros, cambia el punto donde estamos.

Entonces también debe cambiar el gradiente.

```text id="olom15"
gradiente antiguo → dirección antigua
gradiente nuevo → dirección adaptada a la posición actual
```

Si usamos siempre el mismo gradiente antiguo, el modelo puede pasarse de largo y la pérdida puede dispararse.

---

## 16. Error común: actualizar con gradientes antiguos

Código incorrecto:

```python id="7cqrq7"
for epoch in range(10):
    y_pred = predict(features, weights, bias)
    loss = squared_error(y_true, y_pred)

    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * weight_gradients[i]

    bias = bias - learning_rate * bias_gradient
```

El problema es que `weight_gradients` y `bias_gradient` no se recalculan dentro del bucle.

Eso hace que el modelo siga moviéndose en una dirección antigua, aunque ya haya pasado el punto correcto.

Resultado típico:

```text id="3bn950"
la predicción se acerca al principio
luego se pasa
la pérdida baja al principio
luego sube mucho
```

---

## 17. Patrón correcto de entrenamiento

El patrón correcto es:

```text id="iybopa"
for epoch:
    calcular predicción
    calcular pérdida
    calcular gradientes actuales
    actualizar parámetros
```

En pseudocódigo:

```text id="0lg7vo"
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_function(y_true, y_pred)
    gradients = compute_gradients(loss, parameters)
    parameters = parameters - learning_rate * gradients
```

Este patrón aparecerá una y otra vez en Machine Learning y Deep Learning.

---

## 18. Idea fundamental

**El gradiente reúne las derivadas parciales y nos dice cómo ajustar cada parámetro para reducir la pérdida.**

---

## 19. Conceptos clave

* Derivada parcial
* Gradiente
* Vector de gradientes
* Parámetro
* Peso
* Bias
* Pérdida
* Actualización
* Learning rate
* Época
* Recalcular gradientes
* Entrenamiento

---

## 20. Preguntas de repaso

1. ¿Qué mide una derivada parcial?
2. ¿Qué es el gradiente?
3. ¿Por qué el gradiente es un vector?
4. ¿Por qué un gradiente negativo puede hacer que un peso suba?
5. ¿Por qué hay que recalcular gradientes en cada época?
6. ¿Qué puede pasar si usamos gradientes antiguos?
7. ¿Por qué la pérdida baja después de una buena actualización?

---

## 21. Errores comunes

### Error 1: pensar que una derivada parcial cambia todas las variables

No.

Una derivada parcial mide el cambio respecto a una variable, manteniendo las demás constantes.

---

### Error 2: pensar que el gradiente es un solo número

No siempre.

Si hay varios parámetros, el gradiente es un vector.

---

### Error 3: olvidar que el gradiente apunta hacia la subida

El gradiente indica hacia dónde sube más rápido la pérdida.

Para minimizar, vamos en dirección contraria.

---

### Error 4: no recalcular gradientes

Los gradientes deben calcularse con los parámetros actuales.

Si usamos gradientes antiguos, el entrenamiento puede ir en una dirección incorrecta.

---

## 22. Pregunta del ingeniero

Si un modelo tiene muchos pesos y sesgos, ¿qué problema resuelve el gradiente?

Respuesta esperada:

El gradiente permite saber cómo afecta cada parámetro a la pérdida. Al reunir todas las derivadas parciales en un vector, nos da una dirección de actualización para ajustar todos los pesos y sesgos de forma coordinada y reducir la pérdida.

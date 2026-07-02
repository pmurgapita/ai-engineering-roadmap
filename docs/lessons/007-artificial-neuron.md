# Lesson 007 — From Dot Product to the First Artificial Neuron

## Objetivo

Comprender cómo una operación matemática sencilla, el producto escalar, puede convertirse en la base de una neurona artificial.

Al terminar esta lección, deberías entender:

* qué son los pesos,
* qué es el sesgo,
* cómo una neurona combina entradas para producir una salida,
* por qué ajustar pesos puede entenderse como aprender.

---

## 1. Punto de partida

En lecciones anteriores estudiamos vectores, similitud y producto escalar.

El producto escalar parecía una operación matemática simple:

```text
[1, 2, 3] · [4, 5, 6]

= 1×4 + 2×5 + 3×6

= 32
```

Pero esta operación es una de las bases de las redes neuronales.

Una neurona artificial básica puede expresarse así:

```text
salida = entrada · pesos + sesgo
```

O matemáticamente:

```text
y = x · w + b
```

Donde:

* `x` es el vector de entrada,
* `w` es el vector de pesos,
* `b` es el sesgo,
* `y` es la salida.

---

## 2. Ejemplo: predicción del precio de una vivienda

Imagina que queremos predecir el precio de una vivienda usando tres características:

```text
[metros_cuadrados, habitaciones, distancia_al_centro]
```

Una vivienda concreta podría representarse así:

```text
x = [90, 3, 4]
```

Ahora imaginemos que el modelo tiene estos pesos:

```text
w = [3000, 20000, -10000]
```

Interpretación:

```text
metros_cuadrados      → peso positivo grande
habitaciones          → peso positivo
distancia_al_centro   → peso negativo
```

Esto significa:

* más metros cuadrados aumentan la predicción,
* más habitaciones aumentan la predicción,
* más distancia al centro reduce la predicción.

Calculamos:

```text
90×3000 + 3×20000 + 4×(-10000)

= 270000 + 60000 - 40000

= 290000
```

Si añadimos un sesgo:

```text
b = 50000
```

Entonces:

```text
salida = 290000 + 50000

salida = 340000
```

La neurona predice:

```text
precio ≈ 340000 €
```

---

## 3. Qué representa un peso

Un peso indica cuánta importancia tiene una feature en la salida.

```text
feature importante → peso grande
feature poco importante → peso pequeño
feature que reduce la salida → peso negativo
```

Ejemplo:

```text
x = [90, 3, 4]
w = [3000, 20000, -10000]
```

Cada entrada se multiplica por su peso correspondiente:

```text
90 × 3000
3 × 20000
4 × -10000
```

Una red neuronal aprende ajustando estos pesos.

No aprende reglas escritas manualmente.

Aprende números que modifican cómo se combinan las entradas.

---

## 4. Qué representa el sesgo

El sesgo permite desplazar la salida del modelo.

Ejemplo simple:

```text
y = 2x
```

Si `x = 0`, entonces `y = 0`.

Pero quizá necesitamos:

```text
y = 2x + 5
```

Ahora, aunque `x = 0`, la salida es `5`.

Ese `+5` es el sesgo.

En una neurona:

```text
y = x · w + b
```

El sesgo permite que la neurona tenga más flexibilidad y no dependa únicamente de las entradas.

---

## 5. Aprender como ajustar pesos y sesgos

Antes definimos aprender como modificar el comportamiento a partir de experiencias pasadas para obtener mejores resultados en el futuro.

En una red neuronal, aprender significa:

```text
ajustar pesos y sesgos para cometer menos errores
```

Flujo básico:

```text
entrada
  ↓
predicción
  ↓
error
  ↓
ajuste de pesos y sesgos
  ↓
mejor predicción
```

Cuando el modelo se equivoca, los pesos se modifican ligeramente.

Después vuelve a probar.

Con suficientes ejemplos, el modelo puede aprender qué features son más importantes.

---

## 6. Una neurona como detector

Una forma útil de pensar en una neurona es verla como un detector de combinaciones de features.

Ejemplo: detección de spam.

Features:

```text
[
  número_de_links,
  palabras_en_mayúsculas,
  contiene_gratis
]
```

Pesos inventados:

```text
[30, 10, 50]
```

Interpretación:

* muchos links son sospechosos,
* las mayúsculas son algo sospechosas,
* la palabra “gratis” es una señal muy fuerte de spam.

Si un email activa muchas de estas señales, la salida de la neurona será alta.

```text
x · w + b = score alto → probable spam
```

Si activa pocas señales, la salida será baja.

```text
x · w + b = score bajo → probablemente no spam
```

---

## 7. Código básico en Python

Archivo recomendado:

```text
code/fundamentals/artificial_neuron.py
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


def neuron_output(inputs, weights, bias):
    return dot_product(inputs, weights) + bias


house = [90, 3, 4]
weights = [3000, 20000, -10000]
bias = 50000

prediction = neuron_output(house, weights, bias)

print("Predicción:", prediction)
```

Resultado esperado:

```text
Predicción: 340000
```

---

## 8. Ejemplo con varias viviendas

Podemos probar varias entradas con los mismos pesos y el mismo sesgo:

```python
houses = [
    [50, 1, 8],
    [90, 3, 4],
    [120, 4, 2],
]

weights = [3000, 20000, -10000]
bias = 50000

for house in houses:
    prediction = neuron_output(house, weights, bias)
    print("Vivienda:", house, "Predicción:", prediction)
```

Esto representa una idea importante:

```text
mismo modelo + diferentes entradas → diferentes predicciones
```

---

## 9. Limitación de esta neurona

La neurona que hemos visto es lineal:

```text
y = x · w + b
```

Esto significa que combina las entradas de forma proporcional.

Pero muchos problemas reales no son lineales.

Para resolver problemas más complejos necesitaremos añadir una función de activación:

```text
x · w + b → función de activación → salida final
```

Estudiaremos esto más adelante.

---

## 10. Idea fundamental

**Una neurona artificial combina entradas mediante pesos, suma un sesgo y produce una salida.**

---

## 11. Conceptos clave

* Producto escalar
* Entrada
* Peso
* Sesgo
* Salida
* Neurona artificial
* Predicción
* Parámetro
* Aprendizaje
* Ajuste de pesos

---

## 12. Preguntas de repaso

1. ¿Qué representa un peso en una neurona artificial?
2. ¿Qué representa el sesgo?
3. ¿Por qué una neurona artificial usa producto escalar?
4. ¿Qué significa `y = x · w + b`?
5. ¿Por qué ajustar pesos puede considerarse aprender?
6. ¿Qué limitación tiene una neurona lineal?

---

## 13. Errores comunes

### Error 1: pensar que los pesos son reglas escritas a mano

Los pesos no son reglas como `if/else`.

Son números que el modelo ajusta durante el entrenamiento.

---

### Error 2: pensar que el sesgo es un error

El sesgo no es algo malo.

Es un parámetro que da más flexibilidad al modelo.

---

### Error 3: pensar que una neurona ya es inteligencia compleja

Una neurona artificial básica es una operación muy sencilla.

La potencia aparece cuando combinamos muchas neuronas, capas y funciones de activación.

---

### Error 4: olvidar que las features deben ser numéricas

La neurona no procesa conceptos directamente.

Procesa números.

Por eso necesitamos representar los datos adecuadamente antes de usarlos.

---

## 14. Pregunta del ingeniero

Si tuviera que construir un modelo que haga predicciones, ¿qué problema resuelve una neurona artificial?

Respuesta esperada:

Una neurona artificial permite combinar varias características de entrada, asignarles distinta importancia mediante pesos, añadir un sesgo y producir una salida numérica. Es una forma básica de transformar datos de entrada en una predicción.

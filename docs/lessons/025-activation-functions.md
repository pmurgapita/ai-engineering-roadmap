# Lesson 025 — Activation Functions: Why Neural Networks Need Non-Linearity

## Objetivo

Comprender por qué una red neuronal necesita funciones de activación y cómo permiten aprender relaciones mucho más complejas que una simple función lineal.

Al terminar esta lección deberías entender:

* por qué varias capas lineales siguen siendo una función lineal,
* qué es una función de activación,
* qué hacen ReLU y Sigmoid,
* por qué la no linealidad es imprescindible en Deep Learning,
* cómo encaja una activación dentro de una red neuronal.

---

# 1. El problema de las funciones lineales

Una función lineal tiene la forma:

```text
y = wx + b
```

Puede modelar relaciones simples, por ejemplo:

* metros cuadrados → precio aproximado
* horas trabajadas → salario

Pero muchos problemas reales no son lineales:

* reconocer un gato en una imagen,
* entender una frase,
* detectar spam,
* traducir un idioma,
* conducir un coche.

Una sola recta no puede representar relaciones tan complejas.

---

# 2. Varias capas lineales no solucionan el problema

Supongamos:

```text
f(x) = 2x + 1
g(x) = 3x - 4
```

Componemos ambas funciones:

```text
g(f(x))
= 3(2x + 1) - 4
= 6x - 1
```

El resultado sigue siendo una función lineal.

Por tanto:

```text
Linear
↓

Linear
↓

Linear
```

es matemáticamente equivalente a:

```text
Linear
```

Añadir más capas lineales no aumenta la capacidad del modelo.

---

# 3. Función de activación

Una función de activación transforma la salida de una capa lineal antes de enviarla a la siguiente.

Proceso:

```text
Entrada
↓

z = xW + b

↓

a = activation(z)

↓

Siguiente capa
```

Donde:

* `z` es la salida lineal.
* `a` es la salida después de aplicar la activación.

La activación introduce no linealidad.

---

# 4. ReLU

ReLU significa:

```text
Rectified Linear Unit
```

Su definición es:

```text
ReLU(x) = max(0, x)
```

Comportamiento:

```text
ReLU(-3) = 0
ReLU(-1) = 0
ReLU(0)  = 0
ReLU(2)  = 2
ReLU(5)  = 5
```

Características:

* elimina valores negativos,
* mantiene los positivos,
* es muy rápida de calcular,
* es la activación más utilizada en redes neuronales modernas.

---

# 5. ¿Por qué ReLU es no lineal?

Porque no puede describirse mediante una única recta.

Tiene dos comportamientos distintos:

```text
x < 0  → 0

x ≥ 0  → x
```

Ese cambio de comportamiento introduce la no linealidad necesaria para aprender funciones complejas.

---

# 6. Sigmoid

Sigmoid transforma cualquier número en un valor comprendido entre 0 y 1.

Su fórmula es:

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

No es necesario memorizarla.

Lo importante es su comportamiento:

```text
muy negativo → cerca de 0

0 → 0.5

muy positivo → cerca de 1
```

Por ello suele utilizarse cuando queremos representar probabilidades.

---

# 7. ReLU vs Sigmoid

| Activación | Salida   | Uso principal           |
| ---------- | -------- | ----------------------- |
| ReLU       | `[0, ∞)` | Capas ocultas           |
| Sigmoid    | `(0,1)`  | Probabilidades binarias |

---

# 8. Una red neuronal sencilla

Sin activación:

```text
Entrada

↓

Linear

↓

Linear

↓

Salida
```

Equivale a una sola transformación lineal.

Con activación:

```text
Entrada

↓

Linear

↓

ReLU

↓

Linear

↓

Salida
```

Ahora el modelo ya puede aprender relaciones mucho más complejas.

---

# 9. Interpretación de z y a

Normalmente se usa la siguiente notación:

```text
z = salida lineal

a = salida activada
```

Es decir:

```text
z = xW + b

a = ReLU(z)
```

La siguiente capa recibe `a`, no `z`.

---

# 10. Ejemplo práctico

Para:

```text
z = -5
```

obtenemos:

```text
ReLU(z) = 0
```

Mientras que para:

```text
z = 7
```

obtenemos:

```text
ReLU(z) = 7
```

La activación modifica la información antes de pasarla a la siguiente capa.

---

# 11. ¿Por qué necesitamos activaciones?

Porque sin ellas:

* una red profunda seguiría siendo una función lineal,
* no podría aprender relaciones complejas,
* muchas capas no aportarían ninguna ventaja.

Las activaciones son las responsables de que una red neuronal pueda resolver problemas reales.

---

# 12. Código utilizado

Archivo:

```text
code/fundamentals/activation_functions.py
```

Funciones implementadas:

```python
import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def linear(x, w, b):
    return x * w + b
```

Se probaron distintos valores de entrada para comparar el comportamiento de:

* ReLU,
* Sigmoid,
* una capa lineal,
* una capa lineal seguida de ReLU.

---

# 13. Conceptos clave

* Función lineal
* No linealidad
* Función de activación
* ReLU
* Sigmoid
* Salida lineal (`z`)
* Salida activada (`a`)
* Composición de funciones
* Red neuronal

---

# 14. Errores comunes

### Error 1

Pensar que muchas capas lineales hacen una red más potente.

No.

Siguen siendo equivalentes a una única función lineal.

---

### Error 2

Pensar que ReLU cambia todos los valores.

No.

Solo modifica los negativos.

Los positivos permanecen iguales.

---

### Error 3

Pensar que Sigmoid calcula una probabilidad exacta.

No.

Produce un número entre 0 y 1 que normalmente interpretamos como una probabilidad.

---

# 15. Idea fundamental

**Las funciones de activación introducen no linealidad, permitiendo que una red neuronal aprenda relaciones complejas que serían imposibles usando únicamente capas lineales.**

---

# 16. Pregunta del ingeniero

Si una red neuronal ya tiene muchas capas lineales, ¿por qué necesita añadir funciones de activación?

**Respuesta esperada:**

Porque la composición de funciones lineales sigue siendo una función lineal. Las activaciones introducen la no linealidad necesaria para que la red pueda aprender patrones y relaciones complejas presentes en los datos reales.

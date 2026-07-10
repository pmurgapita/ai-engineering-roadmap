# Lesson 023 — Slope and Derivative: How a Function Changes

## Objetivo

Comprender qué son la pendiente y la derivada, y cómo se conectan con el descenso del gradiente en Machine Learning.

Al terminar esta lección deberías entender:

* qué mide la pendiente,
* qué es una derivada,
* qué diferencia hay entre pendiente media e instantánea,
* cómo interpretar el signo de una derivada,
* por qué la derivada ayuda a reducir la pérdida,
* qué relación hay entre derivada y gradiente.

---

## 1. La pregunta central

Una función transforma entradas en salidas:

```text
y = f(x)
```

Ahora queremos preguntar:

```text
si cambio x un poco, ¿cuánto cambia y?
```

Esa pregunta es la base de la derivada.

---

## 2. Pendiente

La pendiente mide cuánto cambia `y` cuando cambia `x`.

Fórmula:

```text
pendiente = cambio en y / cambio en x
```

O:

```text
pendiente = Δy / Δx
```

Donde:

```text
Δ = cambio
```

Ejemplo:

```text
punto A = (1, 2)
punto B = (3, 6)
```

Entonces:

```text
Δy = 6 - 2 = 4
Δx = 3 - 1 = 2
```

Pendiente:

```text
4 / 2 = 2
```

---

## 3. Pendiente positiva, negativa y cero

### Pendiente positiva

Si `x` sube, `y` sube.

Ejemplo:

```text
f(x) = 2x
```

Pendiente:

```text
2
```

---

### Pendiente negativa

Si `x` sube, `y` baja.

Ejemplo:

```text
f(x) = -3x
```

Pendiente:

```text
-3
```

---

### Pendiente cero

Si `x` cambia, `y` no cambia.

Ejemplo:

```text
f(x) = 5
```

Pendiente:

```text
0
```

---

## 4. Pendiente en una recta

Para una función lineal:

```text
f(x) = wx + b
```

la pendiente es:

```text
w
```

Ejemplo:

```text
f(x) = 3x + 10
```

La pendiente es:

```text
3
```

El sesgo `b` desplaza la recta, pero no cambia su inclinación.

Ejemplo:

```text
f(x) = 3x + 10
f(x) = 3x - 50
```

Ambas tienen pendiente:

```text
3
```

---

## 5. Derivada

La derivada es la pendiente instantánea de una función en un punto.

Para funciones lineales, la pendiente es igual en todas partes.

Para funciones curvas, la pendiente cambia según el punto.

Ejemplo:

```text
f(x) = x²
```

Valores:

```text
x = 1 → y = 1
x = 2 → y = 4
x = 3 → y = 9
```

La función cada vez sube más rápido.

La pendiente no es constante.

---

## 6. Pendiente media vs pendiente instantánea

Entre dos puntos podemos calcular una pendiente media:

```text
pendiente media = Δy / Δx
```

Ejemplo:

```text
f(x) = x²
```

Entre `x = 1` y `x = 3`:

```text
f(1) = 1
f(3) = 9
```

Entonces:

```text
Δy = 9 - 1 = 8
Δx = 3 - 1 = 2
```

Pendiente media:

```text
8 / 2 = 4
```

Pero la pendiente instantánea en `x = 3` es otra cosa.

La derivada de:

```text
f(x) = x²
```

es:

```text
f'(x) = 2x
```

Entonces:

```text
f'(3) = 2 × 3 = 6
```

La pendiente instantánea en `x = 3` es `6`.

---

## 7. Notación de derivada

Si tenemos:

```text
f(x)
```

su derivada puede escribirse como:

```text
f'(x)
```

Se lee:

```text
f prima de x
```

Ejemplo:

```text
f(x) = x²
f'(x) = 2x
```

Otro ejemplo:

```text
f(x) = 3x + 2
f'(x) = 3
```

---

## 8. Derivada como sensibilidad

La derivada mide sensibilidad.

Pregunta:

```text
si cambio x un poco, ¿cuánto cambia f(x)?
```

Si la derivada es grande:

```text
pequeño cambio en x → gran cambio en y
```

Si la derivada es pequeña:

```text
pequeño cambio en x → poco cambio en y
```

Si la derivada es negativa:

```text
al aumentar x, y baja
```

---

## 9. Relación con la pérdida

En Machine Learning nos importa mucho esta pregunta:

```text
si cambio un peso un poco, ¿qué pasa con la pérdida?
```

Ejemplo:

```text
loss = L(w)
```

La derivada nos dice:

```text
si aumento w, ¿sube o baja la pérdida?
```

Si:

```text
L'(w) > 0
```

entonces aumentar `w` hace subir la pérdida.

Para reducirla, bajamos `w`.

Si:

```text
L'(w) < 0
```

entonces aumentar `w` hace bajar la pérdida.

Para reducirla, subimos `w`.

Esto explica la regla:

```text
w = w - learning_rate × gradient
```

---

## 10. Ejemplo con pérdida simple

Usamos esta función:

```text
loss(w) = (w - 3)²
```

Su derivada es:

```text
loss'(w) = 2(w - 3)
```

Si:

```text
w = 0
```

entonces:

```text
loss'(0) = 2(0 - 3) = -6
```

Esto significa:

```text
cuando w = 0, la pendiente de la función de pérdida es -6
```

No significa que la función pase por `-6`.

De hecho:

```text
loss(0) = (0 - 3)² = 9
```

La función pasa por:

```text
w = 0, loss = 9
```

Pero su pendiente en ese punto es:

```text
-6
```

Como la derivada es negativa, si aumentamos `w`, la pérdida baja.

---

## 11. Descenso del gradiente

La regla de actualización es:

```text
w = w - learning_rate × gradient
```

Si:

```text
w = 0
learning_rate = 0.1
gradient = -6
```

entonces:

```text
w = 0 - 0.1 × (-6)
w = 0 + 0.6
w = 0.6
```

El peso aumenta, que es justo lo que necesitamos para acercarnos al mínimo en `w = 3`.

---

## 12. Otro ejemplo de actualización

Si:

```text
w = 5
learning_rate = 0.1
```

Derivada:

```text
loss'(w) = 2(w - 3)
```

Entonces:

```text
loss'(5) = 2(5 - 3) = 4
```

Actualización:

```text
new_w = 5 - 0.1 × 4
new_w = 5 - 0.4
new_w = 4.6
```

Como `4.6` está más cerca de `3` que `5`, la actualización va en la dirección correcta.

---

## 13. Por qué restamos el gradiente

El gradiente apunta hacia donde la pérdida sube más rápido.

Como queremos reducir la pérdida, nos movemos en la dirección contraria.

Por eso:

```text
nuevo_peso = peso_actual - learning_rate × gradiente
```

Restar el gradiente significa bajar la pérdida.

---

## 14. Derivada y gradiente

La derivada se usa para una función con una variable.

El gradiente generaliza esta idea a varias variables.

Ejemplo con una variable:

```text
loss = L(w)
```

Usamos derivada:

```text
dL/dw
```

Ejemplo con varias variables:

```text
loss = L(w1, w2, w3)
```

Usamos gradiente:

```text
∇L = [dL/dw1, dL/dw2, dL/dw3]
```

Es decir:

```text
gradiente = vector de derivadas parciales
```

Idea clave:

```text
derivada → cambio respecto a una variable
gradiente → cambios respecto a muchas variables
```

---

## 15. Código

Archivo recomendado:

```text
code/fundamentals/derivatives.py
```

Código:

```python
def f_linear(x):
    return 3 * x + 2


def derivative_linear(x):
    return 3


def f_square(x):
    return x ** 2


def derivative_square(x):
    return 2 * x


def loss_function(w):
    return (w - 3) ** 2


def derivative_loss(w):
    return 2 * (w - 3)


print("Linear function:")
for x in [0, 1, 2, 3]:
    print(f"x = {x} | f(x) = {f_linear(x)} | f'(x) = {derivative_linear(x)}")

print("\nSquare function:")
for x in [0, 1, 2, 3]:
    print(f"x = {x} | f(x) = {f_square(x)} | f'(x) = {derivative_square(x)}")

print("\nLoss function:")
for w in [0, 1, 2, 3, 4, 5]:
    print(f"w = {w} | loss = {loss_function(w)} | loss'(w) = {derivative_loss(w)}")


print("\nGradient descent step:")
w = 5
learning_rate = 0.1

gradient = derivative_loss(w)
new_w = w - learning_rate * gradient

print("w:", w)
print("gradient:", gradient)
print("new_w:", new_w)
```

---

## 16. Salida observada

```text
Linear function:
x = 0 | f(x) = 2 | f'(x) = 3
x = 1 | f(x) = 5 | f'(x) = 3
x = 2 | f(x) = 8 | f'(x) = 3
x = 3 | f(x) = 11 | f'(x) = 3

Square function:
x = 0 | f(x) = 0 | f'(x) = 0
x = 1 | f(x) = 1 | f'(x) = 2
x = 2 | f(x) = 4 | f'(x) = 4
x = 3 | f(x) = 9 | f'(x) = 6

Loss function:
w = 0 | loss = 9 | loss'(w) = -6
w = 1 | loss = 4 | loss'(w) = -4
w = 2 | loss = 1 | loss'(w) = -2
w = 3 | loss = 0 | loss'(w) = 0
w = 4 | loss = 1 | loss'(w) = 2
w = 5 | loss = 4 | loss'(w) = 4

Gradient descent step:
w: 5
gradient: 4
new_w: 4.6
```

---

## 17. Interpretación de la salida

### Función lineal

```text
f(x) = 3x + 2
```

Su derivada siempre es:

```text
3
```

La pendiente no cambia.

---

### Función cuadrática

```text
f(x) = x²
```

Su derivada es:

```text
2x
```

Por eso:

```text
x = 0 → pendiente 0
x = 1 → pendiente 2
x = 2 → pendiente 4
x = 3 → pendiente 6
```

La pendiente aumenta.

---

### Función de pérdida

```text
loss(w) = (w - 3)²
```

La pérdida es mínima en:

```text
w = 3
```

Ahí:

```text
loss = 0
loss'(w) = 0
```

A la izquierda de `3`:

```text
derivada negativa
```

El descenso del gradiente aumenta `w`.

A la derecha de `3`:

```text
derivada positiva
```

El descenso del gradiente reduce `w`.

---

## 18. Idea fundamental

**La derivada mide cómo cambia una función cuando cambia su entrada, y en Machine Learning nos dice cómo ajustar parámetros para reducir la pérdida.**

---

## 19. Conceptos clave

* Pendiente
* Cambio
* Delta
* Derivada
* Pendiente instantánea
* Sensibilidad
* Función de pérdida
* Descenso del gradiente
* Gradiente
* Learning rate
* Mínimo

---

## 20. Preguntas de repaso

1. ¿Qué mide la pendiente?
2. ¿Qué es una derivada?
3. ¿Qué diferencia hay entre pendiente media y pendiente instantánea?
4. ¿Cuál es la derivada de una función lineal?
5. ¿Cuál es la derivada de `x²`?
6. ¿Qué significa que una derivada sea negativa?
7. ¿Por qué restamos el gradiente?
8. ¿Qué diferencia hay entre derivada y gradiente?

---

## 21. Errores comunes

### Error 1: pensar que derivada significa valor de la función

La derivada no es el valor de la función.

La derivada es la pendiente de la función en un punto.

---

### Error 2: pensar que `loss'(0) = -6` significa que la pérdida vale `-6`

No.

```text
loss(0) = 9
loss'(0) = -6
```

La pérdida vale `9`, pero la pendiente en ese punto vale `-6`.

---

### Error 3: olvidar que el gradiente apunta hacia subida

El gradiente apunta hacia donde la pérdida sube.

Por eso, para minimizar, vamos en dirección contraria.

---

### Error 4: confundir derivada y gradiente

La derivada mide cambio respecto a una variable.

El gradiente es un vector de derivadas para varias variables.

---

## 22. Pregunta del ingeniero

Si quiero reducir la pérdida ajustando un parámetro, ¿qué problema resuelve la derivada?

Respuesta esperada:

La derivada indica cómo cambia la pérdida cuando cambio un parámetro. Si la derivada es positiva, aumentar el parámetro aumenta la pérdida, así que conviene reducirlo. Si es negativa, aumentar el parámetro reduce la pérdida, así que conviene aumentarlo.

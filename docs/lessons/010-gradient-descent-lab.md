# Lesson 010 — Lab: Gradient Descent From Scratch

## Objetivo

Implementar descenso del gradiente desde cero en Python para entender cómo un modelo ajusta un peso poco a poco con el objetivo de reducir una función de pérdida.

Al terminar esta lección deberías entender:

* cómo cambia un peso durante el entrenamiento,
* cómo se relacionan pérdida, gradiente y actualización,
* qué ocurre cuando el learning rate es pequeño,
* qué ocurre cuando el learning rate es demasiado grande,
* qué significa que un entrenamiento converja o diverja.

---

## 1. Punto de partida

En la lección anterior estudiamos la intuición del descenso del gradiente.

La idea principal era:

```text
nuevo_peso = peso_actual - learning_rate × gradiente
```

El objetivo es ajustar los pesos para reducir la función de pérdida.

En este laboratorio vamos a verlo con código.

---

## 2. Problema del laboratorio

Usaremos esta función de pérdida:

```text
loss = (w - 3)²
```

Queremos encontrar el valor de `w` que minimiza la pérdida.

Sabemos que el mínimo está en:

```text
w = 3
```

porque:

```text
loss = (3 - 3)² = 0
```

Pero imaginamos que el modelo no lo sabe. Solo puede ir ajustando `w` usando el gradiente.

---

## 3. Función de pérdida

```python
def loss_function(w):
    return (w - 3) ** 2
```

Ejemplos:

```text
w = 0 → loss = 9
w = 1 → loss = 4
w = 2 → loss = 1
w = 3 → loss = 0
w = 4 → loss = 1
```

La pérdida baja a medida que `w` se acerca a `3`.

---

## 4. Gradiente

Para esta función:

```text
loss = (w - 3)²
```

el gradiente es:

```text
gradient = 2 × (w - 3)
```

Intuición:

```text
si w está por debajo de 3 → gradiente negativo
si w está por encima de 3 → gradiente positivo
si w = 3 → gradiente 0
```

El gradiente indica hacia dónde aumenta la pérdida.

Como queremos reducirla, nos movemos en dirección contraria.

---

## 5. Código base

Archivo recomendado:

```text
code/fundamentals/gradient_descent.py
```

Código:

```python
def loss_function(w):
    return (w - 3) ** 2


def gradient(w):
    return 2 * (w - 3)


w = 0
learning_rate = 0.1
epochs = 20

for epoch in range(epochs):
    current_loss = loss_function(w)
    current_gradient = gradient(w)

    print(
        f"Epoch {epoch:02d} | "
        f"w = {w:.4f} | "
        f"loss = {current_loss:.4f} | "
        f"gradient = {current_gradient:.4f}"
    )

    w = w - learning_rate * current_gradient
```

---

## 6. Ejecución

Dependiendo del sistema, se puede ejecutar con:

```bash
python code/fundamentals/gradient_descent.py
```

o:

```bash
python3 code/fundamentals/gradient_descent.py
```

o en Windows:

```bash
py code/fundamentals/gradient_descent.py
```

---

## 7. Resultado con learning rate 0.1

Salida inicial:

```text
Epoch 00 | w = 0.0000 | loss = 9.0000 | gradient = -6.0000
Epoch 01 | w = 0.6000 | loss = 5.7600 | gradient = -4.8000
Epoch 02 | w = 1.0800 | loss = 3.6864 | gradient = -3.8400
Epoch 03 | w = 1.4640 | loss = 2.3593 | gradient = -3.0720
Epoch 04 | w = 1.7712 | loss = 1.5099 | gradient = -2.4576
Epoch 05 | w = 2.0170 | loss = 0.9664 | gradient = -1.9661
```

Interpretación:

```text
w se acerca a 3
loss baja hacia 0
gradient se acerca a 0
```

Esto es aprendizaje en su forma más simple.

---

## 8. Primera actualización paso a paso

Al inicio:

```text
w = 0
```

Gradiente:

```text
gradient = 2 × (0 - 3)
gradient = -6
```

Actualización:

```text
w = w - learning_rate × gradient
w = 0 - 0.1 × (-6)
w = 0 + 0.6
w = 0.6
```

Como el gradiente era negativo, restarlo hace que `w` aumente.

El peso se mueve hacia el mínimo.

---

## 9. Learning rate pequeño

Con:

```python
learning_rate = 0.01
```

el modelo avanza lentamente.

```text
learning_rate pequeño → pasos pequeños → aprendizaje lento
```

Esto puede ser estable, pero puede tardar demasiado en llegar a una buena solución.

---

## 10. Learning rate alto pero estable

Con:

```python
learning_rate = 0.9
```

el modelo puede oscilar alrededor del mínimo.

Pasa de un lado a otro de `w = 3`, pero todavía puede acercarse progresivamente.

```text
learning_rate alto → oscilaciones
```

Si no es demasiado alto, aún puede converger.

---

## 11. Learning rate demasiado alto

Con:

```python
learning_rate = 1.1
```

la pérdida puede crecer cada vez más.

El modelo se aleja del mínimo.

```text
learning_rate demasiado alto → divergencia
```

Esto significa que el entrenamiento falla.

---

## 12. Convergencia y divergencia

### Convergencia

Un entrenamiento converge cuando los pesos se acercan a una solución buena y la pérdida disminuye.

```text
loss baja → modelo mejora
```

### Divergencia

Un entrenamiento diverge cuando los pesos se alejan de una solución buena y la pérdida aumenta.

```text
loss sube → modelo empeora
```

---

## 13. Early stopping

Podemos parar el entrenamiento cuando la pérdida ya es suficientemente pequeña.

Ejemplo:

```python
if current_loss < 0.0001:
    break
```

Esto se llama early stopping.

En modelos reales se usa para evitar entrenar más de lo necesario.

---

## 14. Código mejorado

```python
def loss_function(w):
    return (w - 3) ** 2


def gradient(w):
    return 2 * (w - 3)


w = 0
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    current_loss = loss_function(w)
    current_gradient = gradient(w)

    print(
        f"Epoch {epoch:02d} | "
        f"w = {w:.4f} | "
        f"loss = {current_loss:.4f} | "
        f"gradient = {current_gradient:.4f}"
    )

    if current_loss < 0.0001:
        print("Early stopping: loss is small enough.")
        break

    w = w - learning_rate * current_gradient

final_loss = loss_function(w)

print(f"Final weight: {w:.4f}")
print(f"Final loss: {final_loss:.4f}")
```

---

## 15. Idea fundamental

**El descenso del gradiente convierte el error en pequeños cambios de pesos que reducen la pérdida.**

---

## 16. Conceptos clave

* Descenso del gradiente
* Peso
* Pérdida
* Gradiente
* Learning rate
* Epoch
* Convergencia
* Divergencia
* Oscilación
* Early stopping

---

## 17. Preguntas de repaso

1. ¿Qué ocurre con `w` cuando el modelo aprende correctamente?
2. ¿Qué ocurre con la pérdida durante el entrenamiento?
3. ¿Por qué el gradiente se acerca a cero cuando llegamos al mínimo?
4. ¿Qué pasa si el learning rate es demasiado pequeño?
5. ¿Qué pasa si el learning rate es demasiado grande?
6. ¿Qué significa que un entrenamiento diverja?
7. ¿Qué es early stopping?

---

## 18. Errores comunes

### Error 1: pensar que el modelo “sabe” que el mínimo está en 3

En este ejemplo nosotros sabemos que el mínimo está en `w = 3`, pero el modelo no lo sabe.

Solo ajusta `w` usando la pérdida y el gradiente.

---

### Error 2: pensar que un learning rate grande siempre aprende más rápido

Un learning rate grande puede hacer que el entrenamiento sea inestable o que diverja.

---

### Error 3: imprimir el resultado final antes de la última actualización

Si imprimimos el resultado final dentro del bucle antes de actualizar `w`, podemos estar mostrando el valor anterior a la última actualización.

Es mejor calcular la pérdida final después del bucle.

---

### Error 4: creer que el gradiente siempre será grande

Cuando el peso se acerca al mínimo, el gradiente suele hacerse más pequeño.

Eso significa que el modelo ya no necesita hacer cambios grandes.

---

## 19. Pregunta del ingeniero

Si tuviera que entrenar un modelo desde cero, ¿qué problema resuelve implementar descenso del gradiente?

Respuesta esperada:

El descenso del gradiente permite transformar una medida de error en ajustes concretos de los pesos. Sin este mecanismo, podríamos saber que el modelo se equivoca, pero no tendríamos una forma sistemática de mejorar sus parámetros.

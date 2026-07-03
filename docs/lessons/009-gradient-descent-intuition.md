# Lesson 009 — How Does an AI Adjust Its Weights?

## Objetivo

Comprender la intuición del descenso del gradiente y cómo permite ajustar pesos y sesgos para reducir la función de pérdida.

Al terminar esta lección deberías entender:

* qué problema resuelve el descenso del gradiente,
* qué representa el gradiente,
* por qué nos movemos en dirección contraria al gradiente,
* qué papel tiene el learning rate,
* cómo se actualiza un peso.

---

## 1. Punto de partida

En la lección anterior vimos que una función de pérdida mide cuánto se equivoca un modelo.

Pero medir el error no basta.

Una vez sabemos que el modelo se ha equivocado, necesitamos responder:

```text
¿Qué cambio hago en los pesos para reducir la pérdida?
```

Ahí aparece el descenso del gradiente.

---

## 2. La función de pérdida como montaña

Podemos imaginar la función de pérdida como una montaña.

Cada punto de la montaña representa una combinación distinta de pesos.

La altura representa la pérdida.

```text
pérdida alta → modelo malo
pérdida baja → modelo mejor
```

Entrenar consiste en moverse poco a poco hacia zonas donde la pérdida sea menor.

```text
        pérdida alta
             ▲
            / \
           /   \
          /     \
         /       \____ pérdida baja
```

---

## 3. Qué es el gradiente

El gradiente indica hacia dónde aumenta más rápido la función de pérdida.

```text
gradiente = dirección de subida más rápida
```

Pero durante el entrenamiento no queremos aumentar la pérdida.

Queremos reducirla.

Por eso nos movemos en la dirección contraria al gradiente.

```text
gradiente → subida
-gradiente → bajada
```

---

## 4. Descenso del gradiente

El descenso del gradiente es el método que ajusta pesos poco a poco para reducir la pérdida.

La regla básica de actualización es:

```text
nuevo_peso = peso_actual - learning_rate × gradiente
```

Donde:

* `peso_actual` es el valor actual del peso,
* `gradiente` indica cómo cambia la pérdida respecto a ese peso,
* `learning_rate` controla el tamaño del paso.

---

## 5. Learning rate

El `learning_rate`, o tasa de aprendizaje, indica cuánto se mueve el modelo en cada actualización.

### Learning rate demasiado pequeño

Si el learning rate es demasiado pequeño, el modelo avanza muy lentamente.

```text
pasos pequeños → aprendizaje lento
```

Puede tardar demasiado en llegar a una buena solución.

### Learning rate demasiado grande

Si el learning rate es demasiado grande, el modelo puede pasarse de largo.

```text
pasos enormes → saltos inestables
```

Puede oscilar alrededor del mínimo o incluso alejarse de él.

---

## 6. Ejemplo simple

Supongamos:

```text
peso_actual = 10
gradiente = 2
learning_rate = 0.1
```

Usamos la fórmula:

```text
nuevo_peso = peso_actual - learning_rate × gradiente
```

Sustituimos:

```text
nuevo_peso = 10 - 0.1 × 2
nuevo_peso = 10 - 0.2
nuevo_peso = 9.8
```

El peso se ha movido ligeramente hacia abajo.

---

## 7. Por qué funciona

La función de pérdida nos dice si el modelo está fallando.

El gradiente nos dice cómo afecta cada peso a esa pérdida.

El learning rate decide cuánto modificamos el peso.

Juntos forman el mecanismo básico de aprendizaje:

```text
predicción
    ↓
pérdida
    ↓
gradiente
    ↓
actualización de pesos
    ↓
nueva predicción
```

---

## 8. Relación con redes neuronales

Una red neuronal tiene muchos pesos y sesgos.

Durante el entrenamiento, el modelo calcula cómo cambiar cada uno para reducir la pérdida.

Aunque una red moderna puede tener millones o miles de millones de parámetros, la idea base sigue siendo la misma:

```text
ajustar parámetros para minimizar la pérdida
```

---

## 9. Ejemplo intuitivo con viviendas

Imagina que un modelo predice precios de viviendas demasiado bajos.

Puede que el peso asociado a `metros_cuadrados` sea demasiado pequeño.

El gradiente ayuda a descubrir cómo cambiar ese peso.

Si aumentar el peso reduce la pérdida, el entrenamiento tenderá a aumentarlo.

Si disminuirlo reduce la pérdida, tenderá a disminuirlo.

---

## 10. Definición mejorada de aprendizaje

Ahora podemos expresar mejor qué significa aprender en una red neuronal:

```text
Aprender = ajustar pesos y sesgos para minimizar la función de pérdida usando el gradiente.
```

Esta es una de las ideas centrales del Deep Learning.

---

## 11. Idea fundamental

**El descenso del gradiente es el método que ajusta pesos poco a poco para reducir la pérdida.**

---

## 12. Conceptos clave

* Función de pérdida
* Gradiente
* Descenso del gradiente
* Learning rate
* Actualización de pesos
* Minimización
* Convergencia
* Oscilación
* Parámetros
* Entrenamiento

---

## 13. Preguntas de repaso

1. ¿Qué problema resuelve el descenso del gradiente?
2. ¿Qué representa el gradiente?
3. ¿Por qué nos movemos en dirección contraria al gradiente?
4. ¿Qué papel tiene el learning rate?
5. ¿Qué puede pasar si el learning rate es demasiado grande?
6. ¿Qué puede pasar si el learning rate es demasiado pequeño?
7. ¿Cómo se actualiza un peso usando descenso del gradiente?

---

## 14. Errores comunes

### Error 1: pensar que la pérdida ajusta pesos automáticamente

La pérdida solo mide el error.

El gradiente indica cómo ajustar los pesos para reducir ese error.

---

### Error 2: pensar que el gradiente apunta hacia el mínimo

El gradiente apunta hacia donde la pérdida aumenta más rápido.

Para minimizar, usamos la dirección contraria.

---

### Error 3: pensar que un learning rate grande siempre es mejor

Un learning rate grande puede parecer rápido, pero puede causar inestabilidad y evitar que el modelo converja.

---

### Error 4: pensar que un learning rate muy pequeño siempre es seguro

Un learning rate muy pequeño puede hacer que el entrenamiento sea demasiado lento.

---

## 15. Pregunta del ingeniero

Si tuviera que entrenar una red neuronal, ¿qué problema resuelve el descenso del gradiente?

Respuesta esperada:

El descenso del gradiente permite decidir cómo modificar los pesos y sesgos para reducir la función de pérdida. Sin él, podríamos medir el error, pero no sabríamos cómo cambiar el modelo para mejorar.

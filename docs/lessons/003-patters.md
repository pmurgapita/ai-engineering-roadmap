# Lesson 003 — From Experience to Patterns

## Objetivo

Comprender qué es un patrón, por qué los modelos de IA intentan descubrir patrones en los datos y cómo esto se relaciona con la idea de aprender una función `f(x) = y`.

---

## 1. Punto de partida

En la lección anterior vimos que aprender no significa memorizar.

Aprender significa modificar el comportamiento a partir de experiencias pasadas para obtener mejores resultados en situaciones futuras.

Pero esto nos lleva a una nueva pregunta:

**¿Qué aprende realmente una IA?**

La respuesta inicial es:

Una IA aprende patrones.

---

## 2. ¿Qué es un patrón?

Un patrón es una regularidad que aparece en los datos y que permite hacer predicciones o generalizar a casos nuevos.

Por ejemplo, para reconocer una bicicleta, no basta con memorizar imágenes concretas. El modelo necesita detectar características que suelen aparecer en muchas bicicletas:

* dos ruedas,
* manillar,
* sillín,
* pedales,
* cuadro,
* estructura general.

Estas características forman un patrón.

---

## 3. Patrón y generalización

Un patrón es útil porque permite generalizar.

Si un modelo solo memoriza imágenes concretas de bicicletas, fallará cuando vea una bicicleta nueva.

Si aprende el patrón general de una bicicleta, podrá reconocer bicicletas que nunca ha visto antes.

Ejemplo:

```text
Imagen nueva
     ↓
Patrones reconocidos
     ↓
Predicción: bicicleta
```

---

## 4. Programa tradicional vs Machine Learning

En un programa tradicional, el programador escribe las reglas.

Ejemplo:

```text
Si tiene dos ruedas
y tiene manillar
y tiene sillín
entonces es una bicicleta
```

El problema es que el mundo real tiene demasiadas variaciones.

¿Qué ocurre si la bicicleta está borrosa?

¿Qué ocurre si aparece de lado?

¿Qué ocurre si está parcialmente tapada?

¿Qué ocurre si es una bicicleta infantil?

¿Qué ocurre si tiene ruedines?

Programar manualmente todas las reglas sería inviable.

---

## 5. El cambio de paradigma

En Machine Learning no escribimos todas las reglas.

En lugar de eso, damos ejemplos al modelo.

```text
Datos + respuestas correctas
          ↓
        Modelo
          ↓
   Aprende patrones
```

No programamos directamente la solución.

Programamos un sistema capaz de aprender una solución a partir de datos.

---

## 6. La idea matemática: `f(x) = y`

Muchos problemas de IA pueden entenderse como una función.

Una función transforma una entrada en una salida.

```text
f(x) = y
```

Donde:

* `x` es la entrada,
* `f` es el modelo,
* `y` es la salida.

Ejemplos:

```text
f(foto) = "bicicleta"
f(email) = "spam"
f(texto) = "respuesta"
f(audio) = "transcripción"
```

Entrenar una IA consiste en ajustar esa función para que produzca buenas salidas.

---

## 7. El modelo no ve el mundo como nosotros

Una IA no ve una bicicleta como una persona.

Una imagen para una IA es una colección de números.

Una imagen en blanco y negro puede representarse como una matriz:

```text
[
  [0, 0, 0, 255],
  [0, 120, 200, 255],
  [0, 80, 180, 255]
]
```

Cada número representa la intensidad de un píxel.

Una imagen a color suele tener tres canales:

```text
R, G, B
```

Rojo, verde y azul.

Por ejemplo:

```text
224 × 224 × 3
```

Eso representa una imagen de 224 píxeles de alto, 224 píxeles de ancho y 3 canales de color.

El modelo no recibe “una bicicleta”.

Recibe números.

---

## 8. De números a conceptos

Una de las grandes preguntas de la IA es:

**¿Cómo puede una máquina pasar de números a conceptos?**

La respuesta corta es:

Detectando patrones estadísticos.

El camino completo nos llevará a estudiar:

* vectores,
* matrices,
* funciones,
* pérdida,
* gradientes,
* redes neuronales,
* embeddings,
* atención,
* Transformers.

Pero el hilo conductor será siempre:

```text
números → patrones → predicción
```

---

## 9. Features

Una feature es una característica de los datos que puede ayudar al modelo a tomar una decisión.

Ejemplos para detectar spam:

* longitud del correo,
* palabras en mayúsculas,
* palabras como “gratis” o “urgente”,
* enlaces sospechosos,
* faltas de ortografía,
* remitente desconocido,
* archivos adjuntos sospechosos.

Cada feature por separado puede no ser suficiente.

Pero varias features combinadas pueden formar un patrón.

```text
MAYÚSCULAS + GRATIS + ENLACE SOSPECHOSO + URGENCIA
                         ↓
                    probable spam
```

---

## 10. Machine Learning clásico vs Deep Learning

En Machine Learning clásico, muchas features son diseñadas manualmente por humanos.

Ejemplo:

Para predecir el precio de una casa, podríamos definir features como:

* metros cuadrados,
* número de habitaciones,
* ubicación,
* año de construcción.

En Deep Learning, el modelo puede aprender muchas features automáticamente.

Esto es muy importante porque algunos patrones son demasiado complejos para diseñarlos a mano.

Por ejemplo, en imágenes, el modelo puede aprender progresivamente:

```text
bordes → formas → partes de objetos → objetos completos
```

Esta capacidad se llama representation learning.

---

## 11. Idea fundamental

**Una IA no aprende objetos directamente; aprende patrones numéricos que se relacionan con esos objetos.**

Cuando un modelo reconoce una bicicleta, no está pensando “bicicleta” como una persona.

Está detectando una estructura numérica que, según su entrenamiento, suele corresponder a bicicletas.

---

## 12. Conceptos clave

* Patrón
* Generalización
* Función
* Entrada
* Salida
* Feature
* Machine Learning
* Deep Learning
* Representation learning
* Predicción

---

## 13. Preguntas de repaso

1. ¿Qué es un patrón?
2. ¿Por qué una IA necesita patrones para generalizar?
3. ¿Por qué no es práctico programar manualmente todas las reglas?
4. ¿Qué significa `f(x) = y` en IA?
5. ¿Qué es una feature?
6. ¿Cuál es la diferencia entre diseñar features manualmente y aprenderlas automáticamente?

---

## 14. Errores comunes

### Error 1: pensar que la IA ve conceptos directamente

Una IA no ve “bicicletas”, “perros” o “gatos”.

Procesa números.

---

### Error 2: pensar que más datos siempre significa mejor aprendizaje

La cantidad importa, pero la variedad y la calidad de los datos son igual o más importantes.

---

### Error 3: confundir memorizar con aprender

Un modelo que memoriza puede funcionar bien con ejemplos conocidos, pero fallará con datos nuevos.

---

## 15. Pregunta del ingeniero

Si tuviera que construir un sistema capaz de reconocer objetos, ¿qué problema intenta resolver el aprendizaje de patrones?

Respuesta esperada:

El aprendizaje de patrones permite que el sistema detecte regularidades en los datos y use esas regularidades para tomar decisiones sobre ejemplos nuevos. Sin patrones, el sistema solo podría memorizar casos concretos.

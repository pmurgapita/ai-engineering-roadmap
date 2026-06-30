# Lesson 004 — Representing the World With Numbers

## Objetivo

Comprender por qué los modelos de Inteligencia Artificial necesitan representar el mundo mediante números y cómo aparecen conceptos fundamentales como escalares, vectores, matrices, tensores y embeddings.

---

## 1. Punto de partida

Una IA no trabaja directamente con objetos, palabras, imágenes o sonidos.

Trabaja con números.

Por eso, antes de que un modelo pueda aprender de algo, necesitamos convertir ese algo en una representación numérica.

```text
mundo real → representación numérica → modelo → salida
```

Ejemplos:

```text
imagen  → matriz o tensor de píxeles
texto   → tokens y embeddings
audio   → onda numérica
usuario → vector de características
```

---

## 2. Escalar

Un escalar es un único número.

Ejemplos:

```text
temperatura = 22
precio = 199
edad = 31
```

Un escalar tiene 0 dimensiones.

---

## 3. Vector

Un vector es una lista ordenada de números.

Ejemplo:

```text
[1, 4, 2]
```

Un vector tiene 1 dimensión.

Podemos representar una casa como un vector usando varias características:

```text
[metros_cuadrados, habitaciones, baños, distancia_al_centro]

[90, 3, 2, 4.5]
```

Aquí cada posición del vector tiene un significado.

---

## 4. Matriz

Una matriz es una tabla de números organizada en filas y columnas.

Ejemplo:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

Una matriz tiene 2 dimensiones.

Una imagen en blanco y negro puede representarse como una matriz, donde cada número indica la intensidad de un píxel.

```text
[
  [0, 0, 255],
  [0, 120, 255],
  [0, 0, 255]
]
```

---

## 5. Tensor

Un tensor es una estructura numérica de varias dimensiones.

Podemos entenderlo como una generalización de escalares, vectores y matrices.

```text
escalar → 0 dimensiones
vector  → 1 dimensión
matriz  → 2 dimensiones
tensor  → n dimensiones
```

Una imagen a color puede representarse como un tensor 3D:

```text
alto × ancho × canales
```

Por ejemplo:

```text
224 × 224 × 3
```

Los 3 canales suelen ser:

```text
R, G, B
```

Rojo, verde y azul.

Un conjunto de muchas imágenes puede representarse como un tensor 4D:

```text
número_de_imágenes × alto × ancho × canales
```

---

## 6. Ejemplo: viviendas

Imagina que queremos predecir el precio de viviendas.

Tenemos estos datos:

|  m² | habitaciones | baños | distancia al centro | precio |
| --: | -----------: | ----: | ------------------: | -----: |
|  50 |            1 |     1 |                 8.0 | 150000 |
|  80 |            3 |     2 |                 4.0 | 280000 |
| 120 |            4 |     2 |                 2.0 | 450000 |

Cada casa puede representarse como un vector:

```text
Casa 1 → [50, 1, 1, 8.0]
Casa 2 → [80, 3, 2, 4.0]
Casa 3 → [120, 4, 2, 2.0]
```

Todas las casas juntas forman una matriz:

```text
X = [
  [50, 1, 1, 8.0],
  [80, 3, 2, 4.0],
  [120, 4, 2, 2.0]
]
```

Los precios forman otro vector:

```text
y = [150000, 280000, 450000]
```

En Machine Learning veremos mucho esta notación:

```text
X → entradas
y → respuestas correctas
```

---

## 7. Por qué importa la representación

Un modelo no puede aprender directamente de “casas”, “bicicletas” o “emails”.

Tiene que aprender de representaciones numéricas.

Una parte fundamental de la IA consiste en decidir cómo representar un problema usando números.

Una mala representación puede hacer que un modelo falle incluso aunque el algoritmo sea bueno.

Una buena representación puede hacer que el modelo aprenda mucho mejor.

---

## 8. Texto y números

El texto es más complejo que las imágenes.

Una imagen puede convertirse directamente en píxeles.

Pero una frase como esta:

```text
Me gusta la inteligencia artificial
```

necesita transformarse de alguna manera.

Una primera idea ingenua sería asignar un número a cada palabra:

```text
Me = 1
gusta = 2
la = 3
inteligencia = 4
artificial = 5
```

Pero esto no captura significado.

El modelo no sabe que:

```text
rey
```

está más relacionado con:

```text
reina
```

que con:

```text
tornillo
```

Asignar números arbitrarios identifica palabras, pero no representa su relación semántica.

---

## 9. Embeddings

Un embedding es una representación numérica de algo, normalmente en forma de vector, que intenta conservar información útil sobre su significado.

Ejemplo:

```text
rey      → [0.21, -0.80, 1.34, 0.09, ...]
reina    → [0.25, -0.76, 1.29, 0.12, ...]
tornillo → [-1.90, 0.44, -0.12, 2.01, ...]
```

La idea importante no son los números concretos.

La idea importante es que elementos relacionados deberían quedar cerca en el espacio vectorial.

```text
rey ───── reina

tornillo ───────────────── lejos
```

---

## 10. Espacio vectorial

Cuando representamos elementos como vectores, podemos imaginarlos como puntos en un espacio.

En 2D sería fácil visualizarlo:

```text
          animales
             ↑
             |
     gato    |    perro
             |
-------------+------------→ objetos
             |
             |      silla
             |
```

En IA real, estos espacios pueden tener muchas dimensiones:

```text
128
384
768
1536
4096
```

No podemos dibujarlas fácilmente, pero matemáticamente siguen funcionando como espacios donde los vectores pueden estar más cerca o más lejos.

---

## 11. Embeddings y búsqueda semántica

Los embeddings son muy útiles para construir buscadores inteligentes.

Un buscador tradicional suele buscar coincidencias exactas de palabras.

Un buscador semántico busca significado parecido.

Ejemplo:

```text
Consulta: “cómo entrenar una IA”

Documento: “guía para ajustar modelos de machine learning”
```

Aunque no usen las mismas palabras exactas, están relacionados.

Con embeddings podemos comparar la cercanía entre la consulta y los documentos.

Esto será fundamental cuando estudiemos RAG.

---

## 12. Relación con sistemas RAG

Cuando un usuario sube un PDF a una aplicación con IA, normalmente no se entrega el PDF directamente al modelo.

Primero se procesa:

```text
PDF
 ↓
texto
 ↓
fragmentos
 ↓
embeddings
 ↓
base vectorial
 ↓
búsqueda semántica
 ↓
respuesta del modelo
```

Este flujo será muy importante más adelante.

Por ahora, lo importante es entender que la información debe transformarse en una representación numérica útil.

---

## 13. Idea fundamental

**La IA no aprende directamente del mundo; aprende de representaciones numéricas del mundo.**

---

## 14. Conceptos clave

* Escalar
* Vector
* Matriz
* Tensor
* Feature
* Representación numérica
* Embedding
* Espacio vectorial
* Búsqueda semántica
* RAG

---

## 15. Preguntas de repaso

1. ¿Por qué una IA necesita convertir el mundo en números?
2. ¿Cuál es la diferencia entre un escalar, un vector, una matriz y un tensor?
3. ¿Por qué una imagen a color puede representarse como un tensor?
4. ¿Por qué asignar `perro = 1` y `gato = 2` no captura significado?
5. ¿Qué es un embedding?
6. ¿Por qué los embeddings son útiles para buscar documentos por significado?

---

## 16. Errores comunes

### Error 1: pensar que una IA procesa conceptos directamente

Una IA no recibe conceptos puros.

Recibe representaciones numéricas.

---

### Error 2: pensar que cualquier número sirve como representación

No basta con convertir algo en un número.

La representación debe conservar información útil.

---

### Error 3: pensar que un tensor siempre tiene tres dimensiones

Un tensor puede tener cualquier número de dimensiones.

Un escalar, un vector y una matriz también pueden entenderse como casos particulares de tensores.

---

### Error 4: confundir identificación con significado

Asignar un número único a cada palabra identifica la palabra, pero no representa su significado.

Para representar relaciones semánticas necesitamos embeddings.

---

## 17. Pregunta del ingeniero

Si tuviera que construir un sistema inteligente, ¿qué problema intenta resolver la representación numérica?

Respuesta esperada:

La representación numérica permite convertir información del mundo real en una forma que el modelo pueda procesar matemáticamente. Si la representación es buena, el modelo puede aprender patrones útiles; si es mala, el modelo tendrá dificultades aunque el algoritmo sea potente.

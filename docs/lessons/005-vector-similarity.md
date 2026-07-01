# Lesson 005 — Distance, Similarity and Vector Spaces

## Objetivo

Comprender cómo se puede medir la similitud entre representaciones numéricas y por qué esto es fundamental para embeddings, búsqueda semántica, sistemas RAG, recomendadores y memoria en agentes.

---

## 1. Punto de partida

En la lección anterior vimos que la IA no trabaja directamente con el mundo, sino con representaciones numéricas del mundo.

Por ejemplo:

```text
palabra → embedding
imagen → tensor
documento → vector
usuario → vector de características
```

Esto nos lleva a una nueva pregunta:

**Si representamos cosas como vectores, ¿cómo sabemos si dos cosas se parecen?**

La respuesta está en medir su distancia o similitud dentro de un espacio vectorial.

---

## 2. Espacio vectorial

Un espacio vectorial es un espacio matemático donde podemos colocar vectores.

De forma intuitiva, podemos imaginar cada vector como un punto.

Ejemplo simplificado en 2 dimensiones:

```text
perro → [0.9, 0.8]
gato  → [0.8, 0.7]
mesa  → [-0.6, 0.1]
```

Si los dibujáramos, probablemente `perro` y `gato` quedarían cerca, mientras que `mesa` quedaría más lejos.

La cercanía entre vectores puede representar similitud.

En embeddings de lenguaje, esa similitud suele ser semántica.

---

## 3. ¿Qué significa que dos vectores estén cerca?

Decimos que dos vectores están cerca cuando la separación entre ellos es pequeña según una métrica concreta.

En el contexto de embeddings:

```text
vectores cercanos → significados parecidos
vectores lejanos → significados menos relacionados
```

Ejemplo:

```text
perro ↔ gato      → cercanos
perro ↔ lobo      → cercanos
perro ↔ tornillo  → lejanos
```

La IA no “entiende” esos conceptos como una persona.

Lo que hace es comparar representaciones numéricas.

---

## 4. Distancia euclídea

La distancia euclídea es la distancia geométrica clásica entre dos puntos.

Si tenemos:

```text
A = [1, 2]
B = [4, 6]
```

La distancia se calcula así:

```text
distancia = √((4 - 1)² + (6 - 2)²)
distancia = √(3² + 4²)
distancia = √(9 + 16)
distancia = √25
distancia = 5
```

Intuición:

```text
Distancia euclídea → ¿cuánto espacio hay entre dos puntos?
```

Cuanto menor sea la distancia, más cerca están los vectores.

---

## 5. Similitud coseno

En embeddings de texto se usa mucho la similitud coseno.

La similitud coseno no se centra tanto en la distancia absoluta entre puntos, sino en la dirección de los vectores.

Intuición:

```text
Similitud coseno → ¿apuntan los vectores en una dirección parecida?
```

Ejemplo visual:

```text
Vector A: →
Vector B: ↗
Vector C: ←
```

A y B son relativamente similares porque apuntan en direcciones parecidas.

A y C son muy diferentes porque apuntan en direcciones opuestas.

---

## 6. Diferencia intuitiva entre distancia euclídea y similitud coseno

```text
Distancia euclídea:
¿Cuánto espacio físico hay entre dos vectores?

Similitud coseno:
¿Hacia dónde apuntan esos vectores?
```

En muchos sistemas de embeddings, la dirección del vector es especialmente importante porque puede representar propiedades semánticas.

Por eso, en búsqueda semántica y RAG se usa con frecuencia la similitud coseno.

---

## 7. Ejemplo con documentos

Imagina estos tres documentos:

```text
Documento 1:
“Cómo entrenar modelos de inteligencia artificial.”

Documento 2:
“Guía para ajustar sistemas de machine learning.”

Documento 3:
“Receta de tortilla de patatas.”
```

Aunque los documentos 1 y 2 no usen exactamente las mismas palabras, tienen un significado parecido.

Si los convertimos en embeddings, esperaríamos que sus vectores estuvieran más cerca entre sí que respecto al documento 3.

```text
doc1 ≈ doc2

doc1 lejos de doc3
doc2 lejos de doc3
```

---

## 8. Búsqueda exacta vs búsqueda semántica

Un buscador exacto busca coincidencias literales de palabras.

Ejemplo:

```text
Consulta: “currículum”
```

Puede que no encuentre un documento que use la palabra:

```text
CV
```

aunque ambos conceptos estén relacionados.

Un buscador semántico, en cambio, compara significados.

```text
Consulta: “cómo mejorar mi currículum”

Documento: “guía para preparar un CV profesional”
```

Aunque no coincidan todas las palabras, el significado es parecido.

---

## 9. Por qué los embeddings son útiles para buscar documentos

Los embeddings permiten convertir tanto la consulta como los documentos en vectores.

Después podemos comparar esos vectores.

Flujo simplificado:

```text
consulta del usuario
        ↓
embedding de la consulta
        ↓
comparación con embeddings de documentos
        ↓
documentos más similares
```

Esto permite encontrar información relevante aunque esté escrita con palabras diferentes.

---

## 10. Espacios de muchas dimensiones

Nosotros podemos visualizar fácilmente 2 o 3 dimensiones.

Pero los embeddings reales pueden tener muchas más:

```text
384 dimensiones
768 dimensiones
1536 dimensiones
3072 dimensiones
```

Aunque no podamos dibujar un espacio de 1536 dimensiones, sí podemos operar matemáticamente con él.

Podemos calcular:

* distancias,
* similitudes,
* vecinos más cercanos,
* agrupaciones,
* direcciones.

Esta es una de las razones por las que los embeddings son tan potentes.

---

## 11. Relación con RAG

RAG significa Retrieval-Augmented Generation.

Lo estudiaremos en profundidad más adelante, pero su idea básica es:

```text
usuario hace una pregunta
        ↓
convertimos la pregunta en embedding
        ↓
buscamos documentos con embeddings parecidos
        ↓
recuperamos los fragmentos relevantes
        ↓
se los damos al modelo
        ↓
el modelo responde usando contexto
```

La parte central es:

```text
embedding de la pregunta ≈ embeddings de documentos relevantes
```

Sin similitud vectorial, RAG no funcionaría correctamente.

---

## 12. Ejemplo de RAG

Supón que un usuario pregunta:

```text
¿Cómo puedo hacer que mi modelo aprenda más rápido?
```

Y tienes un documento que dice:

```text
Técnicas para optimizar el entrenamiento de redes neuronales.
```

Un buscador exacto podría no encontrarlo, porque no aparecen exactamente las palabras “aprenda más rápido”.

Un sistema con embeddings puede detectar que ambos textos están relacionados semánticamente.

---

## 13. Idea fundamental

**Si representamos información como vectores, podemos medir similitud calculando la cercanía o dirección entre esos vectores.**

Esta idea es una base esencial de:

* embeddings,
* buscadores semánticos,
* sistemas RAG,
* recomendadores,
* clustering,
* memoria en agentes,
* recuperación de información.

---

## 14. Conceptos clave

* Vector
* Espacio vectorial
* Distancia
* Similitud
* Distancia euclídea
* Similitud coseno
* Embedding
* Búsqueda semántica
* Vecinos cercanos
* RAG

---

## 15. Preguntas de repaso

1. ¿Qué significa que dos vectores estén cerca?
2. ¿Por qué la cercanía entre embeddings puede representar similitud semántica?
3. ¿Qué mide la distancia euclídea?
4. ¿Qué mide la similitud coseno?
5. ¿Por qué la búsqueda semántica puede encontrar resultados que una búsqueda exacta no encuentra?
6. ¿Por qué la similitud vectorial es importante para RAG?

---

## 16. Errores comunes

### Error 1: pensar que cercanía siempre significa significado parecido

En embeddings bien entrenados, la cercanía suele reflejar similitud útil, pero esto depende del modelo y de los datos con los que fue entrenado.

---

### Error 2: confundir distancia euclídea con similitud coseno

La distancia euclídea mide separación entre puntos.

La similitud coseno mide alineación de dirección.

---

### Error 3: creer que las palabras deben coincidir exactamente

En búsqueda semántica, dos textos pueden estar relacionados aunque usen palabras diferentes.

---

### Error 4: pensar que no podemos trabajar con muchas dimensiones

Aunque no podamos visualizar espacios de cientos o miles de dimensiones, sí podemos calcular distancias y similitudes en ellos.

---

## 17. Pregunta del ingeniero

Si tuviera que construir un buscador inteligente, ¿qué problema intenta resolver la similitud vectorial?

Respuesta esperada:

La similitud vectorial permite encontrar información relacionada por significado, no solo por coincidencia exacta de palabras. Esto hace posible recuperar documentos relevantes aunque estén escritos con vocabulario diferente al de la consulta.

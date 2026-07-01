# Lesson 006 — Lab: Euclidean Distance and Cosine Similarity in Python

## Objetivo

Implementar desde cero las operaciones básicas que permiten comparar vectores:

* distancia euclídea,
* producto escalar,
* norma vectorial,
* similitud coseno,
* búsqueda del vector más similar.

El objetivo de esta práctica es comprender que un buscador semántico no es magia: convierte información en vectores y compara esos vectores matemáticamente.

---

## 1. Punto de partida

En lecciones anteriores vimos que la IA trabaja con representaciones numéricas.

También vimos que los embeddings representan palabras, documentos u otros objetos como vectores.

Si tenemos vectores, podemos compararlos.

La pregunta clave es:

**¿Cómo sabemos qué vector se parece más a otro?**

---

## 2. Código base

Archivo recomendado:

```text
code/fundamentals/vector_similarity.py
```

Código:

```python
import math


def euclidean_distance(a, b):
    """
    Calcula la distancia euclídea entre dos vectores.
    """
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        difference = a[i] - b[i]
        total += difference ** 2

    return math.sqrt(total)


def dot_product(a, b):
    """
    Calcula el producto escalar entre dos vectores.
    """
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def vector_norm(a):
    """
    Calcula la norma o longitud de un vector.
    """
    total = 0

    for value in a:
        total += value ** 2

    return math.sqrt(total)


def cosine_similarity(a, b):
    """
    Calcula la similitud coseno entre dos vectores.
    """
    product = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("No se puede calcular similitud coseno con un vector nulo")

    return product / (norm_a * norm_b)


perro = [0.9, 0.8]
gato = [0.8, 0.7]
mesa = [-0.6, 0.1]
lobo = [0.95, 0.85]

print("Distancia perro-gato:", euclidean_distance(perro, gato))
print("Distancia perro-mesa:", euclidean_distance(perro, mesa))
print("Distancia perro-lobo:", euclidean_distance(perro, lobo))

print("Coseno perro-gato:", cosine_similarity(perro, gato))
print("Coseno perro-mesa:", cosine_similarity(perro, mesa))
print("Coseno perro-lobo:", cosine_similarity(perro, lobo))
```

---

## 3. Resultados observados

Ejemplo de salida:

```text
Distancia perro-gato: 0.14142135623730953
Distancia perro-mesa: 1.6552945357246849
Coseno perro-gato: 0.9999694838187877
Coseno perro-mesa: -0.6280192682591459
```

Interpretación:

```text
perro-gato:
distancia baja
coseno cercano a 1
→ vectores muy similares

perro-mesa:
distancia alta
coseno negativo
→ vectores poco similares
```

---

## 4. Distancia euclídea

La distancia euclídea mide la separación geométrica entre dos vectores.

Intuición:

```text
¿Están cerca como puntos?
```

Si la distancia es baja, los vectores están cerca.

Si la distancia es alta, están lejos.

Ejemplo:

```text
perro = [0.9, 0.8]
gato  = [0.8, 0.7]

distancia baja → representaciones cercanas
```

---

## 5. Producto escalar

El producto escalar multiplica los componentes correspondientes de dos vectores y suma el resultado.

Ejemplo:

```text
[1, 2, 3]
[4, 5, 6]

1*4 + 2*5 + 3*6 = 32
```

El producto escalar será fundamental más adelante para entender:

* redes neuronales,
* capas lineales,
* embeddings,
* atención,
* Transformers.

Una red neuronal realiza productos escalares constantemente.

---

## 6. Norma de un vector

La norma representa la longitud de un vector.

Ejemplo:

```text
[3, 4] → longitud 5
```

La norma se utiliza para normalizar vectores y calcular similitud coseno.

---

## 7. Similitud coseno

La similitud coseno mide si dos vectores apuntan en una dirección parecida.

Valores habituales:

```text
1  → muy parecidos
0  → no relacionados o perpendiculares
-1 → direcciones opuestas
```

Intuición:

```text
No pregunta solo “cuánto espacio hay entre ellos”.
Pregunta “si apuntan hacia una dirección parecida”.
```

---

## 8. Buscar el elemento más similar

Podemos usar similitud coseno para encontrar el vector más parecido a una consulta.

```python
def most_similar(query_vector, items):
    best_name = None
    best_score = float("-inf")

    for name, vector in items.items():
        score = cosine_similarity(query_vector, vector)

        if score > best_score:
            best_name = name
            best_score = score

    return best_name, best_score


items = {
    "gato": [0.8, 0.7],
    "mesa": [-0.6, 0.1],
    "lobo": [0.95, 0.85],
}

result, score = most_similar(perro, items)

print("Elemento más similar:", result)
print("Similitud:", score)
```

Resultado esperado:

```text
Elemento más similar: lobo
```

---

## 9. Relación con buscadores semánticos

Este laboratorio es una versión muy pequeña de un buscador semántico.

Flujo general:

```text
consulta
   ↓
vector de la consulta
   ↓
comparación con vectores existentes
   ↓
elemento más similar
```

En un sistema real, los vectores no representarían manualmente palabras como `perro` o `gato`.

Representarían:

* frases,
* documentos,
* imágenes,
* productos,
* usuarios,
* fragmentos de PDF.

Y normalmente serían generados por modelos de embeddings.

---

## 10. Relación con RAG

Un sistema RAG usa una idea muy parecida:

```text
pregunta del usuario
        ↓
embedding de la pregunta
        ↓
comparación con embeddings de documentos
        ↓
fragmentos más similares
        ↓
contexto para el modelo de lenguaje
        ↓
respuesta final
```

Lo que hoy hemos implementado es una versión mínima del paso de recuperación.

---

## 11. Idea fundamental

**Un buscador semántico no es magia: convierte información en vectores y compara esos vectores matemáticamente.**

---

## 12. Conceptos clave

* Distancia euclídea
* Producto escalar
* Norma vectorial
* Similitud coseno
* Vector de consulta
* Elemento más similar
* Búsqueda semántica
* Recuperación de información
* RAG

---

## 13. Preguntas de repaso

1. ¿Qué mide la distancia euclídea?
2. ¿Qué mide la similitud coseno?
3. ¿Por qué necesitamos calcular la norma de un vector?
4. ¿Por qué `perro` y `gato` tienen mayor similitud que `perro` y `mesa` en el ejemplo?
5. ¿Qué relación tiene la función `most_similar` con un buscador semántico?
6. ¿Por qué este laboratorio es una versión mínima de un sistema RAG?

---

## 14. Errores comunes

### Error 1: pensar que la similitud sale de la palabra

En este laboratorio, la similitud sale de los vectores, no de las palabras.

Si cambiáramos los vectores, cambiarían las similitudes.

---

### Error 2: inicializar el mejor resultado con 0

Si todas las similitudes fueran negativas, inicializar con 0 podría hacer que no se eligiera ningún resultado.

Es mejor usar:

```python
best_score = float("-inf")
```

---

### Error 3: calcular dos veces lo mismo

Si calculamos la similitud dentro del `if` y luego otra vez para guardarla, hacemos trabajo duplicado.

Mejor:

```python
score = cosine_similarity(query_vector, vector)
```

y reutilizar `score`.

---

## 15. Pregunta del ingeniero

Si tuviera que construir un buscador semántico, ¿qué problema resuelve comparar vectores?

Respuesta esperada:

Comparar vectores permite encontrar el elemento más parecido a una consulta aunque no coincidan las palabras exactas. Esto permite recuperar información por significado y no solo por coincidencia literal.

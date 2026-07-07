# Lesson 021 — Tensors: The General Structure of AI

## Objetivo

Comprender qué es un tensor y por qué es la estructura general usada para representar datos en IA moderna.

Al terminar esta lección deberías entender:

* qué relación hay entre escalares, vectores, matrices y tensores,
* qué significa el shape de un tensor,
* cómo se representan imágenes como tensores,
* cómo se representan secuencias de texto como tensores,
* por qué el número de valores afecta al coste computacional,
* por qué dos tensores pueden tener el mismo shape pero significar cosas diferentes.

---

## 1. Punto de partida

Hasta ahora hemos visto:

```text
escalar → un número
vector  → una lista de números
matriz  → una tabla de números
```

Un tensor es la idea general:

```text
tensor → estructura de números con cualquier número de dimensiones
```

Los vectores y matrices también son tensores, pero de baja dimensión.

---

## 2. Por qué necesitamos tensores

Los vectores y matrices funcionan muy bien para muchos problemas, pero en IA moderna aparecen datos más complejos:

```text
imágenes
vídeo
audio
texto por batches
activaciones internas
pesos de redes neuronales
embeddings de secuencias
```

Una imagen RGB no es solo una lista de números.

Tiene:

```text
alto
ancho
canales de color
```

Un batch de imágenes tiene además:

```text
número de imágenes
alto
ancho
canales
```

Para representar este tipo de estructuras usamos tensores.

---

## 3. Qué es un tensor

Un tensor es una estructura numérica con una o más dimensiones.

Podemos verlo así:

```text
escalar → tensor de 0 dimensiones
vector  → tensor de 1 dimensión
matriz  → tensor de 2 dimensiones
tensor 3D, 4D, 5D...
```

Ejemplos:

```text
5
```

es un escalar.

```text
[1, 2, 3]
```

es un vector.

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

es una matriz.

```text
[
  [
    [1, 2],
    [3, 4]
  ],
  [
    [5, 6],
    [7, 8]
  ]
]
```

es un tensor 3D.

---

## 4. Número de dimensiones vs shape

Hay que distinguir dos ideas.

### Número de dimensiones

También se puede llamar rango del tensor.

Ejemplos:

```text
5                         → 0D
[1, 2, 3]                 → 1D
[[1, 2], [3, 4]]          → 2D
[[[1, 2], [3, 4]], ...]   → 3D
```

### Shape

El shape dice el tamaño en cada dimensión.

Ejemplo:

```python
tensor = [
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
]
```

Tiene shape:

```text
(2, 2, 2)
```

Interpretación:

```text
2 bloques
2 filas por bloque
2 columnas por fila
```

---

## 5. Tabla resumen

| Estructura | Ejemplo                            | Shape       |
| ---------- | ---------------------------------- | ----------- |
| Escalar    | `5`                                | `()`        |
| Vector     | `[1, 2, 3]`                        | `(3,)`      |
| Matriz     | `[[1, 2, 3], [4, 5, 6]]`           | `(2, 3)`    |
| Tensor 3D  | `[[[...], [...]], [[...], [...]]]` | `(2, 2, 2)` |

Idea importante:

```text
vector y matriz también son tensores
```

---

## 6. Tensores de imágenes

Una imagen en escala de grises puede representarse como una matriz:

```text
alto × ancho
```

Ejemplo:

```text
(28, 28)
```

Cada número representa la intensidad de un píxel.

Una imagen RGB tiene tres canales:

```text
rojo
verde
azul
```

Entonces su shape podría ser:

```text
(altura, anchura, canales)
```

Ejemplo:

```text
(224, 224, 3)
```

Interpretación:

```text
224 píxeles de alto
224 píxeles de ancho
3 canales de color
```

Esto es un tensor 3D.

---

## 7. Batch de imágenes

Una sola imagen RGB puede tener shape:

```text
(224, 224, 3)
```

Un batch de 32 imágenes RGB puede tener shape:

```text
(32, 224, 224, 3)
```

Interpretación:

```text
32 imágenes
224 alto
224 ancho
3 canales
```

Esto es un tensor 4D.

En Deep Learning, los batches son muy comunes.

---

## 8. Tensores de texto

Supón que tenemos una frase tokenizada:

```text
"Me gusta la IA"
```

Podríamos convertirla en tokens:

```text
["Me", "gusta", "la", "IA"]
```

Si cada token se representa con un embedding de 5 dimensiones:

```text
4 tokens
5 dimensiones por token
```

Shape:

```text
(4, 5)
```

Eso es una matriz.

Pero si tenemos un batch de 10 frases, cada una con 4 tokens y embeddings de 5 dimensiones:

```text
(10, 4, 5)
```

Interpretación:

```text
10 frases
4 tokens por frase
5 dimensiones por token
```

Eso es un tensor 3D.

---

## 9. Tensores en LLMs

En modelos de lenguaje modernos, una forma muy común es:

```text
(batch_size, sequence_length, embedding_dim)
```

Ejemplo:

```text
(8, 128, 768)
```

Interpretación:

```text
8 textos en el batch
128 tokens por texto
768 dimensiones por token
```

Esto significa que el modelo no procesa palabras como texto crudo.

Procesa tensores numéricos.

---

## 10. Tensores y redes neuronales

En redes neuronales, casi todo son tensores:

```text
entradas
pesos
sesgos
activaciones
gradientes
salidas
```

Una red neuronal puede verse como una cadena de transformaciones de tensores:

```text
tensor → transformación → tensor → transformación → tensor
```

Ejemplo:

```text
input tensor
↓
linear layer
↓
activation tensor
↓
next layer
↓
output tensor
```

---

## 11. Tensores y PyTorch

Cuando usemos PyTorch más adelante, veremos:

```python
torch.tensor(...)
```

Un tensor de PyTorch es una estructura numérica parecida a los arrays de NumPy, pero con capacidades especiales para Deep Learning:

```text
operaciones rápidas
GPU
gradientes automáticos
entrenamiento eficiente
```

Por ahora, lo importante es entender la idea general.

---

## 12. Shape como lenguaje de IA

En IA, entender shapes es casi como entender la gramática de los datos.

Ejemplos:

```text
(5, 3)
```

puede significar:

```text
5 ejemplos
3 features
```

```text
(32, 224, 224, 3)
```

puede significar:

```text
32 imágenes
224 alto
224 ancho
3 canales
```

```text
(8, 128, 768)
```

puede significar:

```text
8 textos
128 tokens
768 dimensiones de embedding
```

Cuando entiendes el shape, entiendes cómo se están organizando los datos.

---

## 13. Mismo shape, distinto significado

Un tensor con shape:

```text
(2, 2, 3)
```

podría representar:

```text
2 filas, 2 columnas, 3 canales de color
```

o:

```text
2 secuencias, 2 tokens, 3 dimensiones por token
```

Matemáticamente tienen la misma forma.

Semánticamente representan cosas distintas.

Idea clave:

```text
El shape describe la estructura, pero el contexto da el significado.
```

---

## 14. Error común: confundir dimensiones con valores del shape

Si tenemos:

```text
shape = (32, 224, 224, 3)
```

no significa que haya “3 dimensiones” porque aparece un 3 al final.

Significa que el tensor tiene 4 ejes:

```text
eje 1 → 32
eje 2 → 224
eje 3 → 224
eje 4 → 3
```

Número de dimensiones:

```text
4
```

Shape:

```text
(32, 224, 224, 3)
```

El número de dimensiones es cuántos números hay en el shape.

---

## 15. Número total de valores

Podemos calcular cuántos números contiene un tensor multiplicando las dimensiones del shape.

Ejemplo:

```text
shape = (2, 2, 3)
```

Número de valores:

```text
2 × 2 × 3 = 12
```

Ejemplo:

```text
shape = (32, 224, 224, 3)
```

Número de valores:

```text
32 × 224 × 224 × 3 = 4,816,896
```

Un batch pequeño de imágenes puede contener millones de números.

---

## 16. Código

Archivo recomendado:

```text
code/fundamentals/tensor_shapes.py
```

Código:

```python
def get_shape(data):
    if not isinstance(data, list):
        return ()

    if len(data) == 0:
        return (0,)

    return (len(data),) + get_shape(data[0])


def count_values(shape):
    total = 1

    for dimension in shape:
        total *= dimension

    return total


scalar = 5

vector = [1, 2, 3]

matrix = [
    [1, 2, 3],
    [4, 5, 6],
]

tensor_3d = [
    [
        [1, 2],
        [3, 4],
    ],
    [
        [5, 6],
        [7, 8],
    ],
]

image_rgb = [
    [
        [255, 0, 0],
        [0, 255, 0],
    ],
    [
        [0, 0, 255],
        [255, 255, 255],
    ],
]

batch_of_sequences = [
    [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ],
    [
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ],
]

print("Scalar shape:", get_shape(scalar))
print("Vector shape:", get_shape(vector))
print("Matrix shape:", get_shape(matrix))
print("Tensor 3D shape:", get_shape(tensor_3d))
print("RGB image shape:", get_shape(image_rgb))
print("Batch of sequences shape:", get_shape(batch_of_sequences))

print("Values in RGB image:", count_values(get_shape(image_rgb)))
print("Values in batch of sequences:", count_values(get_shape(batch_of_sequences)))

print(count_values((32, 224, 224, 3)))
print(count_values((8, 128, 768)))
```

---

## 17. Salida observada

```text
Scalar shape: ()
Vector shape: (3,)
Matrix shape: (2, 3)
Tensor 3D shape: (2, 2, 2)
RGB image shape: (2, 2, 3)
Batch of sequences shape: (2, 2, 3)
Values in RGB image: 12
Values in batch of sequences: 12
4816896
786432
```

---

## 18. Ejemplos resueltos

### Imagen RGB de 64×64

Shape:

```text
(64, 64, 3)
```

Interpretación:

```text
64 alto
64 ancho
3 canales
```

### Batch de 32 imágenes RGB de 64×64

Shape:

```text
(32, 64, 64, 3)
```

Interpretación:

```text
32 imágenes
64 alto
64 ancho
3 canales
```

### Tensor típico de LLM

Shape:

```text
(8, 128, 768)
```

Interpretación posible:

```text
8 textos
128 tokens por texto
768 dimensiones por token
```

---

## 19. Coste computacional

El número de valores afecta al coste de procesamiento.

Ejemplo:

```text
(32, 224, 224, 3) → 4,816,896 valores
(8, 128, 768)     → 786,432 valores
```

El primero contiene más valores.

Más valores suele implicar más:

```text
memoria
cálculo
tiempo
coste energético
```

Por eso en IA importan tanto:

```text
batch size
image resolution
sequence length
embedding dimension
```

---

## 20. Idea fundamental

**Un tensor es una estructura numérica general que permite representar datos con muchas dimensiones, como imágenes, texto por batches y activaciones de redes neuronales.**

---

## 21. Conceptos clave

* Tensor
* Escalar
* Vector
* Matriz
* Dimensión
* Rango
* Shape
* Eje
* Batch
* Canal
* Sequence length
* Embedding dimension
* Activación
* Coste computacional

---

## 22. Preguntas de repaso

1. ¿Qué es un tensor?
2. ¿Qué relación hay entre escalares, vectores, matrices y tensores?
3. ¿Qué diferencia hay entre número de dimensiones y shape?
4. ¿Qué shape puede tener una imagen RGB?
5. ¿Qué shape puede tener un batch de imágenes?
6. ¿Qué puede significar `(8, 128, 768)` en un LLM?
7. ¿Por qué dos tensores con el mismo shape pueden significar cosas distintas?
8. ¿Por qué el número de valores afecta al coste computacional?

---

## 23. Errores comunes

### Error 1: pensar que tensor significa siempre 3D

Un tensor puede tener 0, 1, 2, 3, 4 o más dimensiones.

---

### Error 2: olvidar que vector y matriz también son tensores

Un vector es un tensor 1D.

Una matriz es un tensor 2D.

---

### Error 3: confundir el número de dimensiones con un valor del shape

```text
(32, 224, 224, 3)
```

tiene 4 dimensiones, no 3.

---

### Error 4: pensar que el shape da todo el significado

El shape da la estructura.

El contexto da el significado.

---

## 24. Pregunta del ingeniero

Si tengo datos complejos como imágenes, secuencias de texto o batches, ¿qué problema resuelve un tensor?

Respuesta esperada:

Un tensor permite organizar datos numéricos con varios ejes, como batch, altura, anchura, canales, tokens o dimensiones de embedding. Esto hace posible representar y procesar estructuras complejas de forma matemática y eficiente.

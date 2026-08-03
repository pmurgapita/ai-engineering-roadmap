# Lesson 031 — NumPy Indexing, Slicing and Reshape

## Objetivo

Aprender a seleccionar partes de arrays y cambiar su organización dimensional sin cambiar sus datos.

## Indexación

En una matriz:

```python
X[row, column]
```

permite acceder a un elemento.

Por ejemplo:

```python
X[1, 2]
```

selecciona el elemento de la fila 1 y columna 2.

Un índice entero elimina el eje correspondiente.

```python
X[0]
```

puede transformar un array `(4,3)` en un vector `(3,)`.

## Slicing

La sintaxis:

```python
start:stop:step
```

permite seleccionar rangos.

El inicio se incluye y el final se excluye.

```python
X[1:4, 0:2]
```

selecciona las filas 1, 2 y 3 y las columnas 0 y 1.

## Conservar dimensiones

```python
X[:, 1]
```

produce:

```text
shape = (4,)
```

porque el índice entero elimina el eje de columnas.

En cambio:

```python
X[:, 1:2]
```

produce:

```text
shape = (4,1)
```

porque el slicing conserva la dimensión.

## Reshape

`reshape` cambia la organización de los datos pero conserva el número total de elementos.

Si tenemos 12 elementos:

```python
vector.reshape(3,4)
```

es válido porque:

```text
3 × 4 = 12
```

También:

```python
vector.reshape(4,3)
```

es válido.

Pero:

```python
vector.reshape(5,3)
```

no lo es porque requeriría 15 elementos.

## Dimensión automática con `-1`

NumPy puede deducir una dimensión:

```python
vector.reshape(-1, 1)
```

Si el vector contiene 12 elementos, NumPy obtiene:

```text
shape = (12,1)
```

También:

```python
vector.reshape(3,-1)
```

produce:

```text
shape = (3,4)
```

Solo puede utilizarse una dimensión `-1`.

## Vector, fila y columna

```text
(3,)   → vector 1D
(1,3)  → matriz fila
(3,1)  → matriz columna
```

Pueden contener los mismos valores, pero tienen estructuras diferentes.

## Views y copies

El slicing básico de NumPy suele devolver una vista.

```python
view = original[0:1]
```

`view` y `original` pueden compartir los mismos datos.

Por eso modificar la vista puede modificar el original.

Para obtener datos independientes:

```python
copy = original[0:1].copy()
```

## Separación de features y targets

Para:

```python
data
```

con varias columnas:

```python
X = data[:, 0:3]
y = data[:, 3]
```

podemos separar:

```text
X → features
y → targets
```

Los shapes pueden ser:

```text
X = (5,3)
y = (5,)
```

Si queremos conservar `y` como matriz columna:

```python
y = data[:, 3:4]
```

obtenemos:

```text
(5,1)
```

## Imágenes y reshape

Un tensor:

```text
(4,3,4)
```

puede representar:

```text
4 imágenes
3 filas
4 columnas
```

Podemos convertirlo en:

```python
images.reshape(4,-1)
```

Resultado:

```text
(4,12)
```

Cada imagen de `3×4` se transforma en un vector de 12 features.

Esto permite utilizar posteriormente operaciones como:

```text
X @ W
```

donde cada fila representa un ejemplo.

## Idea fundamental

**La indexación selecciona elementos, el slicing selecciona regiones y reshape cambia la organización dimensional manteniendo el número total de datos.**

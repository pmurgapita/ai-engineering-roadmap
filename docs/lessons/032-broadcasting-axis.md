# Lesson 032 — Broadcasting and Axis Operations

## Objetivo

Comprender cómo NumPy opera con arrays de shapes diferentes mediante broadcasting y cómo las operaciones de reducción utilizan `axis`.

---

## 1. Broadcasting

Broadcasting permite operar arrays cuyos shapes no son exactamente iguales cuando sus dimensiones son compatibles.

Ejemplo:

```python
X = np.array([
    [1,2,3],
    [4,5,6],
])

bias = np.array([10,20,30])

result = X + bias
```

Shapes:

```text
X    → (2,3)
bias →   (3,)
```

NumPy aplica el mismo vector a cada fila:

```text
[[1,2,3],      [10,20,30]
 [4,5,6]]   +  [10,20,30]
```

Resultado:

```text
[[11,22,33],
 [14,25,36]]
```

---

## 2. Reglas de broadcasting

NumPy compara las dimensiones desde la derecha.

Dos dimensiones son compatibles cuando:

```text
son iguales
```

o:

```text
una de ellas vale 1
```

Ejemplos:

```text
(2,3) + (3,)   → compatible
(2,3) + (1,3)  → compatible
(2,3) + (2,1)  → compatible
(2,3) + (2,)   → incompatible
```

El último caso falla porque:

```text
3 vs 2
```

no son iguales y ninguno vale `1`.

---

## 3. Broadcasting de una columna

```text
(2,3)
+
(2,1)
```

es compatible.

El array:

```text
[[100],
 [200]]
```

se comporta conceptualmente como:

```text
[[100,100,100],
 [200,200,200]]
```

El eje de tamaño `1` puede expandirse.

---

## 4. Una capa neuronal vectorizada

Tenemos:

```text
inputs shape  = (batch, features)
weights shape = (features, neurons)
biases shape  = (neurons,)
```

Por ejemplo:

```text
inputs  = (2,3)
weights = (3,4)
biases  = (4,)
```

La multiplicación:

```text
(2,3) @ (3,4)
```

produce:

```text
(2,4)
```

porque las dimensiones interiores coinciden.

Interpretación:

```text
2 ejemplos
4 neuronas por ejemplo
```

Después:

```text
(2,4) + (4,)
```

utiliza broadcasting para sumar los cuatro biases a todos los ejemplos.

La capa completa puede expresarse como:

```python
Z = X @ W + b
```

---

## 5. `axis`

En un array:

```text
shape = (2,3)
```

tenemos:

```text
axis 0 → dimensión de tamaño 2
axis 1 → dimensión de tamaño 3
```

Una operación de reducción sobre un eje elimina normalmente ese eje.

---

## 6. `axis=0`

```python
np.sum(X, axis=0)
```

Para:

```text
[[1,2,3],
 [4,5,6]]
```

produce:

```text
[5,7,9]
```

Sumamos a través de las filas.

Shape:

```text
(2,3) → (3,)
```

El eje 0 desaparece.

---

## 7. `axis=1`

```python
np.sum(X, axis=1)
```

produce:

```text
[6,15]
```

porque sumamos cada fila.

Shape:

```text
(2,3) → (2,)
```

El eje 1 desaparece.

---

## 8. Regla mental

En vez de memorizar:

```text
axis 0 = columnas
axis 1 = filas
```

es mejor pensar:

```text
axis=n → reduzco/eliminaré la dimensión n
```

Esto funciona también con tensores de más dimensiones.

---

## 9. `keepdims=True`

Normalmente:

```python
np.sum(X, axis=1)
```

produce:

```text
shape = (2,)
```

Con:

```python
np.sum(
    X,
    axis=1,
    keepdims=True,
)
```

produce:

```text
shape = (2,1)
```

El eje se conserva con tamaño `1`.

---

## 10. `keepdims` y broadcasting

Supongamos:

```text
X shape = (2,3)
```

Sin `keepdims`:

```text
row_sums shape = (2,)
```

La división:

```text
(2,3) / (2,)
```

no es compatible porque:

```text
3 vs 2
```

Con:

```text
row_sums shape = (2,1)
```

tenemos:

```text
(2,3) / (2,1)
```

que sí es compatible.

Así podemos dividir cada fila por su propia suma.

---

## 11. MSE vectorizado

Podemos calcular el error cuadrático medio de todo un batch:

```python
loss = np.mean(
    (predictions - targets) ** 2
)
```

Pasos:

```text
predictions - targets
→ error por ejemplo

error ** 2
→ error cuadrático por ejemplo

np.mean(...)
→ error cuadrático medio del batch
```

No necesitamos un bucle manual.

---

## 12. Ejemplo de capa

```python
X = np.array([
    [1.0,2.0],
    [3.0,4.0],
    [5.0,6.0],
])

W = np.array([
    [0.5,1.0,-1.0],
    [2.0,0.5,1.5],
])

b = np.array([1.0,-1.0,0.5])
```

Shapes:

```text
X = (3,2)
W = (2,3)
b = (3,)
```

Entonces:

```text
X @ W → (3,3)
```

y:

```text
X @ W + b → (3,3)
```

La primera fila de salida es:

```text
[5.5, 1.0, 2.5]
```

Cada fila representa un ejemplo y cada columna la salida de una neurona.

---

## 13. Idea fundamental

**Broadcasting permite reutilizar valores a través de dimensiones compatibles sin crear manualmente copias, mientras que `axis` indica qué dimensión se reduce en operaciones como sumas y medias.**

---

## 14. Aplicación en redes neuronales

Una capa completa puede expresarse como:

```python
Z = X @ W + b
```

donde:

```text
X → batch de ejemplos
W → pesos de todas las neuronas
b → bias de cada neurona
Z → activaciones lineales de todo el batch
```

Broadcasting permite aplicar los mismos biases a todos los ejemplos automáticamente.

---

## 15. Conceptos clave

* Broadcasting
* Compatibilidad de shapes
* Dimensión de tamaño 1
* Batch
* `axis`
* Reducción
* `keepdims`
* MSE vectorizado
* Bias
* Capa vectorizada

---

## 16. Errores comunes

### Pensar que cualquier vector puede hacer broadcasting

No basta con que sea un vector.

Los shapes deben ser compatibles desde la derecha.

### Pensar que los arrays necesitan el mismo número de dimensiones

Broadcasting permite operar arrays con distinto `ndim`.

Lo importante es la compatibilidad de sus dimensiones.

### Confundir `axis=0` con “sumar columnas”

Es más preciso pensar que `axis=0` elimina el eje 0 y deja un resultado por cada combinación de los ejes restantes.

### Olvidar `keepdims`

Mantener una dimensión con tamaño `1` puede hacer posible el broadcasting deseado.

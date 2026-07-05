# Lesson 015 — Feature Scaling and Normalization

## Objetivo

Comprender por qué la escala de las features afecta al entrenamiento de un modelo y cómo la normalización ayuda a que el aprendizaje sea más estable.

Al terminar esta lección deberías entender:

* por qué algunas features pueden dominar el entrenamiento,
* qué es la normalización min-max,
* qué es la estandarización,
* cómo escalar un dataset manualmente,
* por qué los pesos son menos interpretables después de escalar,
* qué es data leakage en el contexto del escalado.

---

## 1. Punto de partida

En la lección anterior entrenamos una regresión lineal con varias features.

Esperábamos que el modelo aprendiera algo parecido a:

```text
w ≈ [2.5, 12, -7]
b ≈ 40
```

Pero aprendió algo más parecido a:

```text
w ≈ [3.095, 0.113, -0.614]
b ≈ -0.026
```

Aun así, la predicción era bastante razonable.

Una de las razones principales fue que las features estaban en escalas muy diferentes.

---

## 2. El problema de la escala

Teníamos estas features:

```text
metros:        50 - 120
habitaciones:  1 - 4
distancia:     2 - 8
```

El modelo solo ve números.

No sabe que:

```text
120 metros
```

y:

```text
4 habitaciones
```

representan conceptos diferentes.

Solo ve que `120` es mucho mayor que `4`.

Por eso, durante el entrenamiento, la feature con números más grandes puede dominar los gradientes.

---

## 3. Relación entre escala y gradiente

En nuestro modelo, el gradiente de cada peso era:

```text
gradient_w = -2 × feature × error
```

Eso significa que, si el error es el mismo, una feature grande genera un gradiente más grande.

Ejemplo:

```text
metros       → feature = 100
habitaciones → feature = 3
distancia    → feature = 4
```

Entonces el gradiente asociado a `metros` puede ser mucho mayor que el de las otras features.

Esto hace que su peso cambie más rápido.

La feature puede parecer más importante solo por su escala, no porque realmente sea más útil.

---

## 4. Consecuencias de no escalar

Si las features tienen escalas muy diferentes, puede ocurrir:

```text
entrenamiento inestable
aprendizaje lento
pesos difíciles de interpretar
gradientes desbalanceados
dependencia excesiva de una feature
```

Por eso, en muchos modelos, es importante escalar o normalizar los datos antes de entrenar.

---

## 5. Normalización min-max

La normalización min-max transforma los valores de una feature para dejarlos normalmente entre `0` y `1`.

Fórmula:

```text
x_normalizado = (x - min) / (max - min)
```

Esto hace que:

```text
mínimo de la feature → 0
máximo de la feature → 1
valores intermedios → entre 0 y 1
```

Ejemplo:

```text
metros = [50, 80, 120]
```

Aquí:

```text
min = 50
max = 120
```

Para `80`:

```text
x_norm = (80 - 50) / (120 - 50)
x_norm = 30 / 70
x_norm ≈ 0.4286
```

Resultado:

```text
50  → 0
80  → 0.4286
120 → 1
```

---

## 6. Qué conserva la normalización

Normalizar no elimina la información esencial.

Si antes un valor era mayor que otro, después seguirá siendo mayor.

Ejemplo:

```text
120 metros > 80 metros > 50 metros
```

Después de normalizar:

```text
1 > 0.4286 > 0
```

Lo que cambia es la escala numérica, no el orden relativo.

---

## 7. Estandarización

Otra técnica común es la estandarización.

La estandarización transforma los datos para que tengan aproximadamente:

```text
media = 0
desviación estándar = 1
```

Fórmula:

```text
x_estandarizado = (x - media) / desviación_estándar
```

Ejemplo:

Si:

```text
media = 80
desviación = 20
```

Para:

```text
x = 100
```

entonces:

```text
x_std = (100 - 80) / 20
x_std = 1
```

Eso significa que `100` está una desviación estándar por encima de la media.

---

## 8. Min-max vs estandarización

### Min-max

Útil cuando queremos dejar los valores en un rango conocido.

Ejemplo:

```text
píxeles de imagen: 0-255 → 0-1
```

### Estandarización

Muy usada en Machine Learning clásico y redes neuronales cuando las features tienen distribuciones diferentes.

Ejemplo:

```text
metros, edad, ingresos, distancia...
```

Por ahora, lo importante es entender que ambas técnicas intentan resolver el mismo problema general:

```text
poner las features en escalas más comparables
```

---

## 9. Dataset del laboratorio

Usamos este dataset:

```python
houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]
```

Cada vivienda tiene:

```text
[metros, habitaciones, distancia]
```

Los mínimos y máximos de cada columna son:

```text
mins = [50, 1, 2]
maxs = [120, 4, 8]
```

---

## 10. Ejemplo manual

Queremos normalizar:

```text
[90, 3, 4]
```

Usando:

```text
mins = [50, 1, 2]
maxs = [120, 4, 8]
```

Cálculo:

```text
metros = (90 - 50) / (120 - 50)
metros = 40 / 70
metros = 0.5714
```

```text
habitaciones = (3 - 1) / (4 - 1)
habitaciones = 2 / 3
habitaciones = 0.6667
```

```text
distancia = (4 - 2) / (8 - 2)
distancia = 2 / 6
distancia = 0.3333
```

Resultado:

```text
[0.5714, 0.6667, 0.3333]
```

---

## 11. Código

Archivo recomendado:

```text
code/fundamentals/feature_scaling.py
```

Código:

```python
def get_columns(data):
    columns = []

    num_features = len(data[0])

    for feature_index in range(num_features):
        column = []

        for row in data:
            column.append(row[feature_index])

        columns.append(column)

    return columns


def min_max_scale(data):
    columns = get_columns(data)

    mins = []
    maxs = []

    for column in columns:
        mins.append(min(column))
        maxs.append(max(column))

    scaled_data = []

    for row in data:
        scaled_row = []

        for i, value in enumerate(row):
            min_value = mins[i]
            max_value = maxs[i]

            scaled_value = (value - min_value) / (max_value - min_value)
            scaled_row.append(scaled_value)

        scaled_data.append(scaled_row)

    return scaled_data, mins, maxs


houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

scaled_houses, mins, maxs = min_max_scale(houses)

print("Original houses:")
for house in houses:
    print(house)

print("\nMins:", mins)
print("Maxs:", maxs)

print("\nScaled houses:")
for house in scaled_houses:
    print([round(value, 4) for value in house])
```

---

## 12. Salida observada

```text
Original houses:
[50, 1, 8]
[80, 3, 5]
[120, 4, 2]
[60, 2, 7]
[100, 3, 3]

Mins: [50, 1, 2]
Maxs: [120, 4, 8]

Scaled houses:
[0.0, 0.0, 1.0]
[0.4286, 0.6667, 0.5]
[1.0, 1.0, 0.0]
[0.1429, 0.3333, 0.8333]
[0.7143, 0.6667, 0.1667]
```

---

## 13. Escalar una vivienda nueva

Para escalar una vivienda nueva, debemos usar los mismos `mins` y `maxs` aprendidos del dataset de entrenamiento.

Ejemplo:

```python
def scale_one(row, mins, maxs):
    scaled_row = []

    for i, value in enumerate(row):
        scaled_value = (value - mins[i]) / (maxs[i] - mins[i])
        scaled_row.append(scaled_value)

    return scaled_row


new_house = [90, 3, 4]
scaled_new_house = scale_one(new_house, mins, maxs)

print([round(value, 4) for value in scaled_new_house])
```

Salida esperada:

```text
[0.5714, 0.6667, 0.3333]
```

---

## 14. Interpretación de los pesos después de normalizar

Después de normalizar, los pesos ya no se interpretan directamente en unidades originales.

Antes, un peso podía interpretarse como:

```text
euros por metro
euros por habitación
euros por unidad de distancia
```

Después de normalizar, el modelo trabaja con valores entre `0` y `1`.

Por tanto, los pesos representan la influencia sobre la feature escalada, no sobre la unidad original.

Esto no significa que hayamos perdido la información importante.

Significa que hemos cambiado el sistema de medida.

---

## 15. Data leakage

Data leakage ocurre cuando información que debería estar reservada para evaluación se filtra en el entrenamiento.

En el contexto del escalado, el error sería calcular:

```text
min
max
media
desviación estándar
```

usando todos los datos, incluyendo los datos de test.

Eso permite que el entrenamiento use indirectamente información del conjunto de test.

La regla correcta es:

```text
El escalado se aprende con train y se aplica al test.
```

Es decir:

1. Calculamos `mins` y `maxs` solo con los datos de entrenamiento.
2. Usamos esos mismos `mins` y `maxs` para transformar train.
3. Usamos esos mismos `mins` y `maxs` para transformar test.

---

## 16. Idea fundamental

**Escalar features ayuda a que el modelo aprenda de forma más estable porque pone las entradas en rangos comparables.**

---

## 17. Conceptos clave

* Escalado
* Normalización
* Normalización min-max
* Estandarización
* Feature
* Escala de una feature
* Gradiente
* Data leakage
* Train
* Test

---

## 18. Preguntas de repaso

1. ¿Por qué una feature con números grandes puede dominar el entrenamiento?
2. ¿Qué hace la normalización min-max?
3. ¿Cuál es la fórmula de min-max?
4. ¿Qué diferencia hay entre normalización y estandarización?
5. ¿Por qué los pesos son menos interpretables después de normalizar?
6. ¿Qué es data leakage?
7. ¿Por qué el escalado debe aprenderse solo con el conjunto de entrenamiento?

---

## 19. Errores comunes

### Error 1: pensar que normalizar destruye la información

Normalizar cambia la escala, pero conserva el orden relativo entre valores.

---

### Error 2: comparar pesos normalizados como si estuvieran en unidades originales

Después de normalizar, los pesos ya no representan directamente euros por metro, euros por habitación, etc.

---

### Error 3: calcular min y max usando train y test juntos

Esto produce data leakage.

La evaluación deja de ser limpia porque el entrenamiento recibió información indirecta del test.

---

### Error 4: pensar que escalar siempre soluciona todo

Escalar ayuda mucho, pero no arregla datos malos, features insuficientes o modelos inadecuados.

---

## 20. Pregunta del ingeniero

Si tuviera que entrenar un modelo con features en escalas muy diferentes, ¿qué problema resuelve normalizar?

Respuesta esperada:

Normalizar pone las features en rangos comparables para evitar que una feature domine el entrenamiento solo por tener valores más grandes. Esto ayuda a que los gradientes sean más equilibrados y el aprendizaje sea más estable.

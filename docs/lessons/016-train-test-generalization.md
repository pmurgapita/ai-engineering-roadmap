# Lesson 016 — Train/Test Split and Generalization

## Objetivo

Comprender por qué no basta con que un modelo funcione bien en los datos de entrenamiento y cómo usamos train/test split para medir generalización.

Al terminar esta lección deberías entender:

* qué es el train set,
* qué es el test set,
* qué significa generalizar,
* qué es overfitting,
* qué es underfitting,
* cómo evitar data leakage al normalizar,
* por qué evaluar en datos nuevos es esencial.

---

## 1. Punto de partida

Hasta ahora hemos entrenado modelos y hemos observado si la pérdida bajaba.

Eso es importante, pero no suficiente.

La pregunta clave es:

```text
¿El modelo ha aprendido un patrón general o solo se ha ajustado a los ejemplos que ya vio?
```

Esta pregunta nos lleva a uno de los conceptos más importantes de Machine Learning:

```text
generalización
```

---

## 2. El problema

Imagina que entrenamos un modelo con estos datos:

```text
[50, 1, 8]   → 139
[80, 3, 5]   → 241
[120, 4, 2]  → 376
[60, 2, 7]   → 167
[100, 3, 3]  → 319
```

Si después evaluamos el modelo usando esos mismos datos, puede parecer que funciona bien.

Pero eso puede ser engañoso.

Quizá el modelo solo ha aprendido a responder bien a esos ejemplos concretos.

Lo que realmente queremos saber es:

```text
¿funciona con viviendas nuevas?
```

---

## 3. Memorizar vs generalizar

### Memorizar

El modelo funciona bien en ejemplos que ya vio.

```text
ejemplo visto → buena respuesta
ejemplo nuevo → mala respuesta
```

### Generalizar

El modelo aprende un patrón útil que también funciona en ejemplos nuevos.

```text
ejemplo visto → buena respuesta
ejemplo nuevo → buena respuesta razonable
```

En Machine Learning no buscamos solo memorizar.

Buscamos generalizar.

---

## 4. Train/Test Split

Para medir generalización, dividimos los datos en dos partes:

```text
train set
test set
```

### Train set

El train set contiene los datos usados para entrenar el modelo.

```text
el modelo puede ver estos datos durante el entrenamiento
```

### Test set

El test set contiene datos reservados para evaluar.

```text
el modelo no debe ver estos datos durante el entrenamiento
```

La idea es:

```text
entrenar con train
evaluar con test
```

---

## 5. Analogía del examen

Estudiar con unos ejercicios y examinarse con exactamente los mismos no demuestra comprensión real.

Podríamos haber memorizado las respuestas.

Una evaluación más justa es:

```text
estudiar con unos ejercicios
examinarse con ejercicios nuevos
```

En Machine Learning ocurre lo mismo.

El test set funciona como un examen con ejemplos nuevos.

---

## 6. Ejemplo de división

Si tenemos 10 ejemplos, una división común puede ser:

```text
80% train
20% test
```

Eso significa:

```text
8 ejemplos para entrenar
2 ejemplos para evaluar
```

Con datasets grandes, es común usar divisiones como:

```text
train: 70-80%
test: 20-30%
```

Con datasets pequeños, la evaluación es más inestable, pero la idea sigue siendo importante.

---

## 7. Train loss y test loss

Después de dividir los datos, podemos medir dos errores:

```text
train loss
test loss
```

### Train loss

Mide el error en datos que el modelo vio durante el entrenamiento.

### Test loss

Mide el error en datos nuevos que el modelo no vio durante el entrenamiento.

Lo ideal es:

```text
train loss baja
test loss también baja o se mantiene razonable
```

Una mala señal sería:

```text
train loss muy baja
test loss alta
```

Eso puede indicar mala generalización.

---

## 8. Overfitting

Overfitting, o sobreajuste, ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento y funciona mal en datos nuevos.

Señal típica:

```text
train loss baja
test loss alta
```

Ejemplo:

```text
train loss = 0.1
test loss = 200
```

El modelo parece muy bueno en entrenamiento, pero falla en datos nuevos.

Probablemente ha aprendido detalles específicos del train, no un patrón general.

---

## 9. Underfitting

Underfitting, o subajuste, ocurre cuando el modelo no aprende suficiente.

Falla tanto en train como en test.

Señal típica:

```text
train loss alta
test loss alta
```

Ejemplo:

```text
train loss = 500
test loss = 520
```

Aquí el modelo no está capturando bien el patrón ni siquiera en los datos que ya vio.

---

## 10. Resumen visual

```text
Buen modelo:
train loss baja
test loss baja

Overfitting:
train loss muy baja
test loss alta

Underfitting:
train loss alta
test loss alta
```

---

## 11. Código

Archivo recomendado:

```text
code/fundamentals/train_test_split.py
```

Código:

```python
def split_train_test(data, targets, test_size):
    if len(data) != len(targets):
        raise ValueError("data y targets deben tener la misma longitud")

    split_index = int(len(data) * (1 - test_size))

    x_train = data[:split_index]
    y_train = targets[:split_index]

    x_test = data[split_index:]
    y_test = targets[split_index:]

    return x_train, y_train, x_test, y_test


houses = [
    [50, 1, 8],
    [80, 3, 5],
    [120, 4, 2],
    [60, 2, 7],
    [100, 3, 3],
]

prices = [
    139,
    241,
    376,
    167,
    319,
]

x_train, y_train, x_test, y_test = split_train_test(
    houses,
    prices,
    test_size=0.4
)

print("X train:")
for row in x_train:
    print(row)

print("\ny train:")
print(y_train)

print("\nX test:")
for row in x_test:
    print(row)

print("\ny test:")
print(y_test)
```

---

## 12. Salida observada

Con `test_size=0.4`, usamos:

```text
60% train
40% test
```

Como tenemos 5 ejemplos:

```text
3 ejemplos para train
2 ejemplos para test
```

Salida:

```text
X train:
[50, 1, 8]
[80, 3, 5]
[120, 4, 2]

y train:
[139, 241, 376]

X test:
[60, 2, 7]
[100, 3, 3]

y test:
[167, 319]
```

---

## 13. Cuidado con el orden

Nuestro split actual separa los datos por orden:

```text
primeros ejemplos → train
últimos ejemplos → test
```

Esto es simple, pero puede ser peligroso.

Si los datos estuvieran ordenados por precio:

```text
baratas primero
caras después
```

el train tendría solo casas baratas y el test solo casas caras.

La evaluación sería poco representativa.

Por eso normalmente se mezclan los datos antes de dividirlos:

```text
shuffle → split
```

Lo veremos más adelante con librerías.

---

## 14. Relación con data leakage

En la lección anterior aprendimos:

```text
El escalado se aprende con train y se aplica al test.
```

Ahora podemos ver el orden correcto completo:

```text
1. dividir train/test
2. calcular min/max/media/desviación solo con train
3. transformar train usando esos valores
4. transformar test usando esos mismos valores
5. entrenar con train
6. evaluar con test
```

El orden incorrecto sería:

```text
1. calcular min/max usando todos los datos
2. escalar todo
3. dividir train/test
```

Ese orden produce data leakage, porque el entrenamiento recibe información indirecta del test.

---

## 15. Qué cambia en nuestra forma de pensar

Antes preguntábamos:

```text
¿baja la pérdida?
```

A partir de ahora preguntamos:

```text
¿baja la pérdida en train?
¿cómo se comporta en test?
¿está generalizando?
```

Esta es una mentalidad más profesional.

Un modelo no se evalúa solo por cómo funciona en los datos que ya vio.

Se evalúa por cómo funciona en datos nuevos.

---

## 16. Generalización

Un modelo generaliza bien cuando mantiene buen rendimiento en datos que no vio durante el entrenamiento.

No significa que acierte perfectamente todos los casos nuevos.

Significa que ha aprendido un patrón útil, no solo respuestas memorizadas.

---

## 17. Idea fundamental

**Un modelo no se evalúa por lo bien que recuerda el entrenamiento, sino por lo bien que funciona en datos nuevos.**

---

## 18. Conceptos clave

* Train set
* Test set
* Generalización
* Train loss
* Test loss
* Overfitting
* Underfitting
* Data leakage
* Shuffle
* Evaluación

---

## 19. Preguntas de repaso

1. ¿Qué es el train set?
2. ¿Qué es el test set?
3. ¿Por qué evaluar con los mismos datos de entrenamiento puede ser engañoso?
4. ¿Qué significa generalizar?
5. ¿Qué es overfitting?
6. ¿Qué es underfitting?
7. ¿Cuál es el orden correcto para evitar data leakage al normalizar?
8. ¿Por qué normalmente mezclamos los datos antes de dividirlos?

---

## 20. Errores comunes

### Error 1: pensar que train loss baja significa buen

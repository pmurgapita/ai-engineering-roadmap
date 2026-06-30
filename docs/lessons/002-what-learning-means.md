# Lesson 002 — What Does Learning Mean?

## Objetivo

Comprender qué significa aprender desde el punto de vista de la Inteligencia Artificial y por qué la generalización es uno de los conceptos más importantes de todo el Machine Learning.

---

## 1. Idea inicial

Aprender no significa memorizar.

Aprender significa modificar el comportamiento a partir de experiencias pasadas para obtener mejores resultados en situaciones futuras.

Una IA no aprende porque “entienda” el mundo como una persona, sino porque ajusta sus parámetros internos a partir de muchos ejemplos.

---

## 2. Definición práctica

**Aprender consiste en modificar el comportamiento utilizando experiencias pasadas para obtener mejores resultados en el futuro.**

Esta definición es importante porque se centra en tres ideas:

1. Experiencia previa.
2. Cambio de comportamiento.
3. Mejora futura.

Sin estas tres piezas, no estamos hablando realmente de aprendizaje.

---

## 3. Analogía: lanzar dardos

Imagina que lanzas un dardo y fallas.

Después corriges ligeramente la posición del brazo.

Vuelves a lanzar.

Fallaste menos.

Corriges de nuevo.

Después de muchos intentos, cada vez te acercas más al centro.

No has memorizado cada lanzamiento. Has ajustado tu comportamiento.

Una IA hace algo parecido: comete errores, mide esos errores y ajusta sus parámetros internos.

---

## 4. IA y parámetros

Una IA almacena conocimiento de forma indirecta mediante parámetros.

Podemos imaginar esos parámetros como millones o miles de millones de pequeños mandos que se ajustan durante el entrenamiento.

Cuando el modelo se equivoca, esos mandos cambian ligeramente.

Cuando acierta, se refuerza la configuración que le ha llevado a acertar.

Más adelante veremos que esos parámetros suelen ser pesos y sesgos dentro de una red neuronal.

---

## 5. Generalización

**Generalizar es ser capaz de resolver correctamente casos que nunca se han visto antes.**

Este es uno de los conceptos centrales de la Inteligencia Artificial.

Un modelo que memoriza no es realmente útil.

Un modelo útil debe aprender patrones generales.

Por ejemplo, una IA que reconoce bicicletas no debe memorizar únicamente imágenes concretas de bicicletas. Debe aprender características generales que le permitan reconocer bicicletas nuevas, en posiciones, colores y contextos diferentes.

---

## 6. Memorizar vs aprender

Un estudiante que memoriza preguntas de examen solo puede responder preguntas conocidas.

Un estudiante que entiende la materia puede resolver preguntas nuevas.

Lo mismo ocurre con una IA.

Memorizar ejemplos concretos no basta. El objetivo es descubrir reglas, patrones o estructuras que permitan actuar correctamente ante datos nuevos.

---

## 7. Ejemplo: bicicletas

Si entrenamos una IA solo con bicicletas rojas, puede aprender una asociación incorrecta:

Rojo → Bicicleta

Pero la relación correcta debería ser algo más general:

Forma, ruedas, manillar, estructura → Bicicleta

Este problema aparece cuando el conjunto de datos está sesgado o no representa bien la realidad.

---

## 8. Idea fundamental

**La inteligencia no consiste en memorizar el pasado; consiste en comportarse mejor ante situaciones nuevas gracias a la experiencia.**

---

## 9. Conceptos clave

* Aprendizaje
* Experiencia
* Parámetros
* Entrenamiento
* Generalización
* Memorización
* Patrón
* Sesgo de datos

---

## 10. Preguntas de repaso

1. ¿Por qué memorizar no es suficiente para que una IA sea útil?
2. ¿Qué significa generalizar?
3. ¿Por qué una IA necesita muchos ejemplos variados?
4. ¿Qué problema aparece si todas las imágenes de entrenamiento son demasiado parecidas?
5. ¿Cuál es la diferencia entre aprender una respuesta y aprender una regla?

---

## 11. Pregunta del ingeniero

Si tuviera que construir un sistema capaz de reconocer objetos nuevos, ¿qué problema intenta resolver la generalización?

Respuesta esperada:

La generalización permite que el sistema funcione con datos que nunca ha visto antes. Sin generalización, el sistema solo podría reconocer ejemplos memorizados y fallaría en situaciones nuevas.

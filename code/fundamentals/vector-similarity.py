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

print("Coseno perro-gato:", cosine_similarity(perro, gato))
print("Coseno perro-mesa:", cosine_similarity(perro, mesa))

print(euclidean_distance(perro, lobo))
print(cosine_similarity(perro, lobo))

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
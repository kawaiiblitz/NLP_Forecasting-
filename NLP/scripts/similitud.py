from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

"""
Este script implementa un enfoque de clasificación basado en embeddings de SentenceTransformers. 
Utiliza el modelo 'all-MiniLM-L6-v2' para convertir sentencias y etiquetas en representaciones vectoriales, 
y luego calcula la similitud coseno entre los embeddings para asignar etiquetas predichas a las oraciones.
La métrica principal utilizada para evaluar el desempeño del modelo es la exactitud (accuracy), 
acompañada de un reporte de clasificación detallado.
"""


# Sentencias y etiquetas verdaderas
sentences = [
    "The axolotl is more than a peculiar amphibian; in its natural environment, it plays an essential role in the ecological stability of the Xochimilco canals.",
    "Geoffrey Hinton, Yann LeCun and Yoshua Bengio are considered the ‘godfathers’ of an essential technique in artificial intelligence, called ‘deep learning’.",
    "Greenland is about to open up to adventure-seeking visitors. How many tourists will come is yet to be seen, but the three new airports will bring profound change.",
    "GitHub Copilot is an AI coding assistant that helps you write code faster and with less effort, allowing you to focus more energy on problem solving and collaboration.",
    "I have a problem with my laptop that needs to be resolved asap!!"
]

true_labels = [
    "animal",
    "artificial intelligence",
    "travel",
    "artificial intelligence",
    "urgent"
]

candidate_labels = ["urgent", "artificial intelligence", "computer", "travel", "animal", "fiction"]

# Descripciones de las etiquetas
label_descriptions = [
    "This text is about something urgent.",
    "This text is about artificial intelligence.",
    "This text is about computers and technology.",
    "This text is about travel and tourism.",
    "This text is about animals and wildlife.",
    "This text is a work of fiction."
]


# Cargar el modelo SentenceTransformer
# Este modelo convierte oraciones en representaciones vectoriales (embeddings) que capturan su semántica.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings para las sentencias y las descripciones de etiquetas
# Los embeddings son vectores que representan el significado semántico de los textos.
sentence_embeddings = model.encode(sentences)
label_embeddings = model.encode(label_descriptions)

# Inicializar listas para almacenar resultados
predicted_labels = []  # Etiquetas predichas para cada sentencia
predicted_probabilities = []  # Probabilidades asociadas a las etiquetas predichas

# Clasificar cada sentencia
for i, sentence_embedding in enumerate(sentence_embeddings):
    # Calcular similitud coseno con todas las etiquetas
    similarities = cosine_similarity([sentence_embedding], label_embeddings)[0]
    # Convertir similitudes a probabilidades
    probabilities = similarities / similarities.sum()
    # Obtener el índice de la etiqueta con mayor probabilidad
    best_label_idx = np.argmax(probabilities)
    predicted_label = candidate_labels[best_label_idx]
    predicted_labels.append(predicted_label)
    predicted_probabilities.append(probabilities[best_label_idx])
    
    # Imprimir resultados para cada sentencia
    print(f"Sentence {i+1}: \"{sentences[i]}\"")
    print(f"Predicted label: {predicted_label}")
    print(f"Probability: {probabilities[best_label_idx]:.4f}")
    print("-" * 80)

# Evaluación del modelo
# Calcula la exactitud comparando las etiquetas predichas con las verdaderas
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Generar un reporte detallado con métricas como precisión, recall y F1-score
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=candidate_labels, zero_division=0))

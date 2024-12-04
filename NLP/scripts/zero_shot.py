from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sentencias a clasificar y sus etiquetas verdaderas
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

# Etiquetas candidatas para clasificación
candidate_labels = ["urgent", "artificial intelligence", "computer", "travel", "animal", "fiction"]

# Configurar el clasificador utilizando Zero-Shot Classification
# Zero-shot classification permite clasificar texto en categorías no vistas previamente,
# basándose únicamente en una lista de etiquetas candidatas proporcionadas por el usuario.
# En este caso, usamos el modelo "facebook/bart-large-mnli", un modelo basado en BART.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Listas para almacenar las etiquetas predichas y las probabilidades asociadas
predicted_labels = []  # Etiqueta predicha para cada sentencia
predicted_scores = []  # Probabilidad asociada a la etiqueta predicha

# Clasificar las sentencias
for sentence in sentences:
    # Realizar clasificación para la sentencia actual
    result = classifier(sentence, candidate_labels=candidate_labels)
    
    # Extraer la etiqueta más probable y su probabilidad
    label = result['labels'][0]  # Etiqueta con la probabilidad más alta
    score = result['scores'][0]  # Probabilidad de la etiqueta
    
    # Almacenar resultados
    predicted_labels.append(label)
    predicted_scores.append(score)
    
    # Mostrar resultados de la clasificación para cada sentencia
    print(f"Sentence: {sentence}")
    print(f"Predicted label: {label}, Probability: {score:.4f}")
    print("-" * 80)

# Calcular la exactitud global de las predicciones
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Mostrar un informe de clasificación detallado
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=candidate_labels, zero_division=0))

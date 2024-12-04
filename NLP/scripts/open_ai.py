import os
from openai import OpenAI
import re
from sklearn.metrics import accuracy_score, classification_report

"""
Este script implementa un modelo de clasificación de texto utilizando la API de OpenAI para tareas de zero-shot classification. 
El enfoque principal es generar predicciones de categorías y probabilidades asociadas a partir de texto libre, utilizando el modelo GPT-4o.

Metodología:
- Las sentencias se clasifican en una de las etiquetas predefinidas proporcionadas como entrada.
- Se utiliza un formato estructurado para obtener predicciones de categoría y probabilidad directamente desde las respuestas de OpenAI.
- Se evalúa el desempeño del modelo utilizando métricas estándar como la exactitud (accuracy) y un informe detallado de clasificación.

Notas:
- El uso de una clave de API válida es obligatorio.
"""

# Configurar la clave de API de OpenAI desde una variable de entorno
client = OpenAI(api_key="sk-")



# Sentencias a clasificar
sentences = [
    "The axolotl is more than a peculiar amphibian; in its natural environment, it plays an essential role in the ecological stability of the Xochimilco canals.",
    "Geoffrey Hinton, Yann LeCun and Yoshua Bengio are considered the 'godfathers' of an essential technique in artificial intelligence, called 'deep learning'.",
    "Greenland is about to open up to adventure-seeking visitors. How many tourists will come is yet to be seen, but the three new airports will bring profound change.",
    "GitHub Copilot is an AI coding assistant that helps you write code faster and with less effort, allowing you to focus more energy on problem solving and collaboration.",
    "I have a problem with my laptop that needs to be resolved asap!!"
]

# Etiquetas candidatas
labels = ["urgent", "artificial intelligence", "computer", "travel", "animal", "fiction"]

# Etiquetas verdaderas
true_labels = ["animal", "artificial intelligence", "travel", "artificial intelligence", "urgent"]
true_labels = [label.lower() for label in true_labels]  # Normaliza las etiquetas verdaderas


def classify_sentence_chat(text, labels):
    prompt = f"""
Analyze the following sentence and classify it into one of the following categories: {', '.join(labels)}.
Also, provide a probability between 0 and 1 indicating how likely the sentence belongs to that category.

Sentence: "{text}"

Respond in the following format:

Category: <label>
Probability: <numeric value between 0 and 1>
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def parse_response(response_text):
    match = re.search(r"Category:\s*(.+?)\s*Probability:\s*([\d.]+)", response_text, re.IGNORECASE)
    if match:
        category = match.group(1).strip()
        probability = float(match.group(2))
        return category, probability
    else:
        print(f"Failed to parse response: {response_text}")
        return None, None

# Clasificar cada sentencia
predicted_labels = []
probabilities = []

for i, sentence in enumerate(sentences):
    response_text = classify_sentence_chat(sentence, labels)
    category, probability = parse_response(response_text)
    predicted_labels.append(category)
    probabilities.append(probability)
    print(f"Sentence {i+1}: {sentence}")
    print(f"Predicted label: {category}, Probability: {probability:.2f}")
    print("-" * 80)

predicted_labels = [label.lower() for label in predicted_labels]  # Normaliza las predicciones

# Calcular la exactitud
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Informe de clasificación
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, labels=labels, zero_division=0))


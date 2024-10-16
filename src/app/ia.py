# Código da IA que processa o texto do usuário

import torch
# Load model directly
from transformers import pipeline


def process_text(texto: str):
    try:
        # Load the model
        classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis", tokenizer="Zabihin/Symptom_to_Diagnosis")

        # Get the predicted label
        result = classifier(texto)

        # Print the predicted label
        if result and isinstance(result, list) and len(result) > 0 and 'label' in result[0]:
            return result[0]['label']
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = FastAPI()

# Carrega o BioBERT para análise de sintomas

# Load the model
classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis", tokenizer="Zabihin/Symptom_to_Diagnosis")

# Get the predicted label
result = classifier(request.text)

# Print the predicted label
predicted_label = result[0]['label']
print("Predicted Label:", predicted_label)

class TextRequest(BaseModel):
    text: str

@app.post("/")
def analyze_symptoms(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Você não enviou a requisição com texto solicitado")

    # Tokeniza e processa o texto com BioBERT
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True)
    
    # Realiza a inferência para obter os logits de sintomas
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Configuração do limiar para marcar sintomas
    logits = outputs.logits.squeeze()
    threshold = 0.5  # Limiar de probabilidade para sintomas relevantes
    symptoms_identified = []

    # Cria o checklist dinamicamente com base na probabilidade de cada sintoma
    for i, score in enumerate(logits):
        symptom_option = {
            "label": f"Sintoma {i + 1}",  # Ajuste conforme sintomas específicos
            "selected": torch.sigmoid(score).item() > threshold
        }
        symptoms_identified.append(symptom_option)

    # Retorno em JSON do checklist de sintomas
    response = {
        "symptom": request.text,
        "options": symptoms_identified
    }
    return response

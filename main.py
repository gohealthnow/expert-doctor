from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset 

app = FastAPI()

#! Carregar o dataset 
# ds = load_dataset("mohammad2928git/complete_medical_symptom_dataset")

# Carrega o modelo BioBERT
model_name = "dmis-lab/meerkat-7b-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)

class TextRequest(BaseModel):
    text: str

# Mapeamento de sintomas específicos
symptom_labels = [
    "fever",
    "cough",
    "shortness of breath",
    "loss of taste",
    "loss of smell",
    "sore throat",
    "fatigue",
    "headache",
    "chills",
    "chest pain",
    "diarrhea",
    "muscle aches",
    "runny nose",
    "nausea",
    "congestion",
    "vomiting",
    "abdominal pain",
    "confusion",
]

@app.post("/")
def analyze_symptoms(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Você não enviou a requisição com texto solicitado")

    # Tokeniza e processa o texto
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True)
    
    # Inferência com BioBERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtenha os logits e configure um limiar
    logits = outputs.logits.squeeze()
    threshold = 0.5
    symptoms_identified = []

    # Mapeie as saídas para sintomas reconhecíveis
    for i, score in enumerate(logits):
        symptom_option = {
            "label": symptom_labels[i],  # Usando o rótulo do dicionário
            "selected": torch.sigmoid(score).item() > threshold
        }
        symptoms_identified.append(symptom_option)

    # Retorno em JSON do checklist de sintomas
    response = {
        "symptoms": symptoms_identified,
        "original_text": request.text,
        "options": {
            "threshold": threshold,
            "model": model_name,
        }
    }

    return response

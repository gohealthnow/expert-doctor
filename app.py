from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Modelo e configurações
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

# Classe para requisição
class TextRequest(BaseModel):
    text: str

# Inicializar FastAPI
app = FastAPI()

# Carregar modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, from_tf=True)
classifier = pipeline("text-classification", model=MODEL_NAME)

# ! Esta função recebe um prompt do usuário e a IA retorna um checklist de sintomas. O usuário deve marcar os sintomas que está sentindo, conforme o texto que ele escreveu e que gerou o checklist.
@app.post("/symptoms")
async def predict_symptoms(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Você não enviou a requisição com texto solicitado")

    try:
        result = classifier(request.text)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Nenhum resultado obtido do classificador")
        
        symptoms = []
        for res in result:
            if isinstance(res, dict) and 'label' in res:
                symptoms.append(res['label'])
        
        if not symptoms:
            raise HTTPException(status_code=500, detail="Não foi possível extrair sintomas dos resultados")
        
        return symptoms

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

# ! Esta função irá enviar o checklist de sintomas marcados pelo usuário e a IA irá retornar um texto e uma descrição do que ele tem!
@app.post("/diagnosis")
def predict_diagnosis(request: TextRequest):
    # Verificar se há texto na requisição
    if not request.text:
        raise HTTPException(status_code=400, detail="Você não enviou a requisição com texto solicitado")

    try:
        # Realizar classificação com pipeline
        result = classifier(request.text)
        
        if not result or not isinstance(result, list) or not isinstance(result[0], dict):
            raise HTTPException(status_code=500, detail="Erro ao processar a análise de sintomas")
        
        # Preparar resposta
        prediction = {
            "diagnósticos": [
                res['label'] for res in result 
                if isinstance(res, dict) and 'label' in res
            ],
            "confianças": [
                float(res['score']) for res in result 
                if isinstance(res, dict) and 'score' in res
            ],
            "texto_original": request.text
        }
        
        # Validar se obtivemos resultados
        if not prediction["diagnósticos"]:
            raise HTTPException(
                status_code=500, 
                detail="Não foi possível extrair diagnósticos dos resultados"
            )
        
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
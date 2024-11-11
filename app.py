from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Classe para requisição
class TextRequest(BaseModel):
    text: str

# Inicializar FastAPI
app = FastAPI()

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# ! Esta função recebe um prompt do usuário e a IA retorna um checklist de sintomas. O usuário deve marcar os sintomas que está sentindo, conforme o texto que ele escreveu e que gerou o checklist.
@app.post("/symptoms")
async def predict_symptoms(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Você não enviou a requisição com texto solicitado")

    try:
        messages = [
            
            {"role": "system", "content": "Você é uma IA especializada em identificar sintomas a partir de descrições textuais fornecidas pelos usuários. Sua tarefa é gerar um checklist de sintomas baseado no texto fornecido, para que o usuário possa marcar os sintomas que está sentindo. A resposta deve ser um JSON válido contendo uma lista de sintomas."},
            {"role": "user", "content": request.text},
            {"role": "system", "content": "Por favor, forneça a resposta no formato JSON, com a chave 'sintomas' contendo uma lista de sintomas identificados."}
        ]

        outputs = pipe(messages,max_new_tokens=256,)
        
        if not outputs or not isinstance(outputs, list) or not isinstance(outputs[0], dict):
            raise HTTPException(status_code=500, detail="Erro ao processar a análise de sintomas")
        
        # Preparar resposta
        symptoms = [
            res['content'] for res in outputs 
            if isinstance(res, dict) and 'content' in res
        ]
        
        # Validar se obtivemos resultados
        if not symptoms:
            raise HTTPException(
                status_code=500, 
                detail="Não foi possível extrair sintomas dos resultados"
            )
            
        return {"sintomas": symptoms}
    
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
        result = pipe(request.text)
        
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
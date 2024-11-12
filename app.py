from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import re
import json

# Classe para requisição
class TextRequest(BaseModel):
    text: str

# Inicializar FastAPI
app = FastAPI()

# Inicializar o pipeline de geração de texto
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# Expressão regular para validar JSON com sintomas
json_regex = re.compile(r'^\{.*"sintomas":\s*\[.*\]\}$')

# ! Esta função recebe um prompt do usuário e a IA retorna um checklist de sintomas. O usuário deve marcar os sintomas que está sentindo, conforme o texto que ele escreveu e que gerou o checklist.
@app.post("/symptoms")
async def predict_symptoms(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Você não enviou a requisição com texto solicitado")
    
    try:
        # Prompt de entrada com estrutura {role, content}
        messages = [
            {"role": "system", "content": (
                "Você é uma IA especializada em identificar sintomas a partir de descrições textuais fornecidas pelos usuários. "
                "Sua tarefa é gerar um checklist de sintomas baseado no texto fornecido. Responda APENAS com um JSON compactado, sem quebras de linha, "
                "sem blocos de código e sem formatações adicionais. Use o formato: {\"sintomas\": [\"Sintoma 1\", \"Sintoma 2\", ...]}. "
                "Garanta que a resposta siga o regex: '^\\{\"sintomas\":\\s*\\[.*\\]\\}$'."
            )},
            {"role": "user", "content": request.text},
            {"role": "system", "content": (
                "Por favor, responda somente com o JSON em uma única linha, sem \n, \", ou blocos de código (```)."
            )}
        ]

        # Gerar texto usando o pipeline
        response = pipe(messages, max_new_tokens=256, pad_token_id=pipe.tokenizer.eos_token_id if pipe.tokenizer else 50256)

        # Obter o texto gerado pelo 'assistant'
        generated_text = response[0]['generated_text'][3]['content']
        
        print(response[0]['generated_text'][3]['content'])

        # Remover formatações extras como ``` e quebras de linha
        cleaned_text = re.sub(r'```|[\n\r]', '', generated_text).strip()

        # Validar o texto gerado com regex para garantir que seja um JSON válido
        if not re.match(json_regex, generated_text):
            raise HTTPException(status_code=500, detail="Resposta inválida, o formato JSON não foi detectado")

        # Converter para JSON
        result = json.loads(generated_text)

        # Verificar se a chave 'sintomas' está presente e é uma lista
        if not isinstance(result, dict) or "sintomas" not in result or not isinstance(result["sintomas"], list):
            raise HTTPException(status_code=500, detail="Formato de JSON inválido")

        return result
    
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
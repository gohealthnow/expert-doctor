from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator, ValidationError
from transformers import pipeline
from functools import lru_cache
import re
import json
import logging
import copy
from typing import List, Dict, Any, Optional, Tuple, Union, cast


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Descrição textual dos sintomas do paciente")

class Symptom(BaseModel):
    name: str
    description: Optional[str] = None

class SymptomsResponse(BaseModel):
    sintomas: List[str] = Field(..., min_items=1, description="Lista de sintomas identificados")
    
    @validator('sintomas')
    def validate_symptoms(cls, v):
        if not all(isinstance(item, str) for item in v):
            raise ValueError("Todos os sintomas devem ser strings")
        return v

class DiagnosisResponse(BaseModel):
    title: str = Field(..., description="Título do possível diagnóstico")
    description: str = Field(..., description="Descrição detalhada do diagnóstico")
    score: str = Field(..., description="Nível de confiança no diagnóstico (Alto, Médio, Baixo)")
    recommendations: List[str] = Field(default_factory=list, description="Recomendações médicas")
    disclaimer: str = Field(
        default="Este diagnóstico é apenas para fins informativos e não substitui uma consulta médica profissional.",
        description="Aviso legal sobre o uso da informação"
    )


@lru_cache(maxsize=1)
def get_model():
    return pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")


app = FastAPI(
    title="HealthNow - Sistema de Diagnóstico Médico",
    description="API para identificação de sintomas e sugestão de possíveis diagnósticos",
    version="2.0.0"
)


def process_llm_response(generated_text: str, json_pattern: Optional[re.Pattern] = None) -> Dict[str, Any]:
    """
    Processa e valida as respostas do LLM com múltiplas estratégias de recuperação.
    Esta versão implementa várias camadas de fallback para garantir extração do JSON mesmo com respostas imperfeitas.
    """
    
    logger.debug(f"Texto bruto recebido: {generated_text[:100]}...")
    
    
    cleaned_text = re.sub(r'```json|```|\n|\r', '', generated_text).strip()
    
    
    extracted_json = None
    try:
        
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
            
            if json_pattern and not json_pattern.match(json_str):
                logger.warning("JSON encontrado não corresponde ao padrão esperado. Tentando corrigir...")
            else:
                
                try:
                    extracted_json = json.loads(json_str)
                    logger.debug("JSON extraído com sucesso via regex padrão")
                except json.JSONDecodeError:
                    logger.warning("Formato JSON encontrado, mas com erro de parse. Tentando estratégia alternativa.")
    except Exception as e:
        logger.warning(f"Erro na estratégia 1 de extração: {str(e)}")

    
    if not extracted_json:
        try:
            
            json_candidates = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
            valid_jsons = []
            
            for match in json_candidates:
                try:
                    candidate = json.loads(match.group(0))
                    valid_jsons.append(candidate)
                except json.JSONDecodeError:
                    continue
                    
            if valid_jsons:
                
                extracted_json = max(valid_jsons, key=lambda x: len(x))
                logger.debug("JSON extraído com sucesso via estratégia de múltiplos candidatos")
        except Exception as e:
            logger.warning(f"Erro na estratégia 2 de extração: {str(e)}")

    
    if not extracted_json:
        try:
            
            fixed_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  
            fixed_text = re.sub(r',(\s*[\]}])', r'\1', fixed_text)   
            fixed_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_text)  
            
            
            json_match = re.search(r'\{.*\}', fixed_text, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json.loads(json_match.group(0))
                    logger.debug("JSON extraído com sucesso após conserto de erros comuns")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"Erro na estratégia 3 de extração: {str(e)}")

    
    if not extracted_json:
        logger.error("Todas as estratégias de extração de JSON falharam")
        
        extracted_json = create_fallback_response(generated_text)
        logger.info("Gerado JSON de fallback")

    return extracted_json


def create_fallback_response(text: str) -> Dict[str, Any]:
    """
    Cria uma resposta de fallback quando todas as estratégias de extração de JSON falham.
    Tenta identificar o tipo de resposta (sintomas ou diagnóstico) e gerar um objeto válido.
    """
    
    if any(word in text.lower() for word in ["sintoma", "sintomas", "symptom"]):
        
        potential_symptoms = []
        
        
        list_items = re.findall(r'(?:^|\n)(?:\d+\.|\*|\-)\s*([^\n\.]+)', text)
        if list_items:
            potential_symptoms.extend([item.strip() for item in list_items])
        
        
        if not potential_symptoms:
            phrases = re.findall(r'(?<!\w)([A-Z][^\.!\?]{3,50}[\.!\?])(?!\w)', text)
            potential_symptoms.extend([phrase.strip().rstrip('.!?') for phrase in phrases[:5]])
        
        
        if not potential_symptoms:
            potential_symptoms = ["Sintoma não especificado"]
            
        return {"sintomas": potential_symptoms}
    
    
    else:
        
        title_match = re.search(r'(?:diagnóstico|título|title|condition)[\s:]+([^\n\.]{5,50})', text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "Diagnóstico não especificado"
        
        
        desc_match = re.search(r'(?:descrição|description)[\s:]+([^\n]{10,})', text, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else "Não foi possível determinar um diagnóstico preciso com os sintomas fornecidos."
        
        
        if "baixo" in text.lower() or "low" in text.lower():
            score = "Baixo"
        elif "médio" in text.lower() or "medium" in text.lower():
            score = "Médio"
        else:
            score = "Baixo"  
        
        
        recommendations = []
        rec_items = re.findall(r'(?:recomendaç[ãõ]o|sugest[ãõ]o|recomendamos|recommendation).*?[\s:]+([^\n\.]{10,100})', text, re.IGNORECASE)
        if rec_items:
            recommendations = [item.strip() for item in rec_items[:3]]
        
        if not recommendations:
            recommendations = ["Consulte um médico para avaliação adequada dos sintomas"]
        
        return {
            "title": title,
            "description": description,
            "score": score,
            "recommendations": recommendations,
            "disclaimer": "Este diagnóstico é apenas para fins informativos e não substitui uma consulta médica profissional."
        }


SYMPTOMS_PATTERN = re.compile(r'\{"sintomas":\s*\[(?:"[^"]*"(?:,\s*)?)*\]\}', re.DOTALL)
DIAGNOSIS_PATTERN = re.compile(r'\{"title":\s*"[^"]*",\s*"description":\s*"[^"]*",\s*"score":\s*"[^"]*"(?:,\s*"recommendations":\s*\[(?:"[^"]*"(?:,\s*)?)*\])?\}', re.DOTALL)


def normalize_symptoms_response(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Normaliza e corrige respostas de sintomas"""
    try:
        
        if "sintomas" not in data:
            
            for potential_field in ["symptoms", "symptom", "sintoma"]:
                if potential_field in data:
                    data["sintomas"] = data[potential_field]
                    break
            else:
                
                data["sintomas"] = []
        
        
        if not isinstance(data["sintomas"], list):
            if isinstance(data["sintomas"], str):
                
                data["sintomas"] = [s.strip() for s in data["sintomas"].split(",")]
            else:
                
                data["sintomas"] = [str(data["sintomas"])]
        
        
        data["sintomas"] = [
            s.strip().capitalize() for s in data["sintomas"]
            if s and isinstance(s, str) and s.strip()
        ]
        
        
        data["sintomas"] = list(dict.fromkeys(data["sintomas"]))
        
        
        if not data["sintomas"]:
            data["sintomas"] = ["Sintoma não especificado"]
            
        return {"sintomas": data["sintomas"]}
    except Exception as e:
        logger.error(f"Erro na normalização de sintomas: {str(e)}")
        return {"sintomas": ["Sintoma não especificado"]}

def normalize_diagnosis_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza e corrige respostas de diagnóstico"""
    normalized = {
        "title": "",
        "description": "",
        "score": "Baixo",
        "recommendations": [],
        "disclaimer": "Este diagnóstico é apenas para fins informativos e não substitui uma consulta médica profissional."
    }
    
    try:
        
        if "title" in data and data["title"] and isinstance(data["title"], str):
            normalized["title"] = data["title"].strip()
        else:
            
            for field in ["titulo", "título", "diagnóstico", "diagnostico", "diagnosis"]:
                if field in data and data[field] and isinstance(data[field], str):
                    normalized["title"] = data[field].strip()
                    break
            else:
                normalized["title"] = "Diagnóstico não especificado"
        
        
        if "description" in data and data["description"] and isinstance(data["description"], str):
            normalized["description"] = data["description"].strip()
        else:
            
            for field in ["descrição", "descricao", "desc", "detalhes", "details"]:
                if field in data and data[field] and isinstance(data[field], str):
                    normalized["description"] = data[field].strip()
                    break
            else:
                normalized["description"] = "Não foi possível determinar detalhes do diagnóstico."
        
        
        valid_scores = ["Alto", "Médio", "Baixo", "High", "Medium", "Low"]
        if "score" in data and data["score"] and isinstance(data["score"], str):
            score_val = data["score"].strip().capitalize()
            if score_val in valid_scores:
                
                score_map = {"High": "Alto", "Medium": "Médio", "Low": "Baixo"}
                normalized["score"] = score_map.get(score_val, score_val)
            else:
                normalized["score"] = "Baixo"  
        
        
        if "recommendations" in data:
            if isinstance(data["recommendations"], list):
                recommendations = []
                for item in data["recommendations"]:
                    if item and isinstance(item, str):
                        recommendations.append(item.strip())
                normalized["recommendations"] = recommendations or ["Consulte um médico para avaliação adequada"]
            elif isinstance(data["recommendations"], str):
                normalized["recommendations"] = [data["recommendations"].strip()]
        
        
        if not normalized["recommendations"]:
            normalized["recommendations"] = ["Consulte um médico para avaliação adequada"]
        
        
        if "disclaimer" in data and data["disclaimer"] and isinstance(data["disclaimer"], str):
            normalized["disclaimer"] = data["disclaimer"].strip()
            
        return normalized
    except Exception as e:
        logger.error(f"Erro na normalização de diagnóstico: {str(e)}")
        return {
            "title": "Erro de processamento",
            "description": "Ocorreu um erro ao processar o diagnóstico, mas estamos trabalhando para corrigi-lo.",
            "score": "Baixo",
            "recommendations": ["Consulte um médico para avaliação adequada dos sintomas"],
            "disclaimer": "Este diagnóstico é apenas para fins informativos e não substitui uma consulta médica profissional."
        }

@app.post("/symptoms", response_model=SymptomsResponse, summary="Identifica sintomas a partir da descrição do paciente")
async def predict_symptoms(request: TextRequest, model=Depends(get_model)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="O texto da solicitação não pode estar vazio")
    
    try:
        
        messages = [
            {"role": "system", "content": (
                "Você é um assistente médico especializado em identificar sintomas a partir de descrições de pacientes. "
                "Analise cuidadosamente o relato do paciente e identifique TODOS os possíveis sintomas mencionados. "
                "Inclua tanto sintomas explicitamente mencionados quanto aqueles que podem estar implícitos na descrição. "
                "Use terminologia médica padronizada, mas mantenha os termos compreensíveis. "
                "Responda APENAS com um JSON no formato: {\"sintomas\": [\"sintoma 1\", \"sintoma 2\", ...]}. "
                "Não inclua diagnósticos, apenas sintomas objetivos."
            )},
            {"role": "user", "content": request.text},
            {"role": "system", "content": (
                "Lembre-se: forneça apenas o JSON com a lista de sintomas, sem texto introdutório ou explicativo."
            )}
        ]

        
        logger.info(f"Processando pedido de sintomas - primeiros 50 caracteres: {request.text[:50]}...")

        
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                
                response = model(
                    messages, 
                    max_new_tokens=300, 
                    temperature=0.3 if attempt == 0 else 0.2,  
                    do_sample=True,
                    pad_token_id=model.tokenizer.eos_token_id if model.tokenizer else 50256
                )

                
                generated_text = response[0]['generated_text'][3]['content']
                result = process_llm_response(generated_text, SYMPTOMS_PATTERN)
                
                
                normalized_result = normalize_symptoms_response(result)
                
                
                if normalized_result["sintomas"]:
                    return normalized_result
                    
                
                if attempt < max_attempts - 1:
                    messages.append({"role": "system", "content": (
                        "A resposta anterior não continha sintomas válidos no formato JSON correto. "
                        "Por favor, forneça APENAS um JSON válido no formato: {\"sintomas\": [\"sintoma 1\", \"sintoma 2\"]}"
                    )})
            except Exception as inner_e:
                logger.warning(f"Tentativa {attempt+1} falhou: {str(inner_e)}")
                if attempt < max_attempts - 1:
                    
                    messages.append({"role": "system", "content": (
                        "Por favor, forneça apenas um JSON simples com sintomas no formato: {\"sintomas\": [\"sintoma 1\", \"sintoma 2\"]}"
                    )})
        
        
        
        fallback_response = create_fallback_response(request.text)
        normalized_response = normalize_symptoms_response(fallback_response)
        
        logger.info("Usando resposta de fallback para sintomas")
        return normalized_response
    
    except Exception as e:
        logger.error(f"Erro ao processar sintomas: {str(e)}", exc_info=True)
        
        return {"sintomas": ["Sintoma não identificado claramente"]}
    
@app.post("/diagnosis", response_model=DiagnosisResponse, summary="Sugere possíveis diagnósticos com base nos sintomas")
async def predict_diagnosis(request: SymptomsResponse, background_tasks: BackgroundTasks, model=Depends(get_model)):
    if not request.sintomas:
        
        request.sintomas = ["Sintoma não especificado"]
    
    try:
        
        messages = [
            {"role": "system", "content": (
                "Você é um assistente médico especializado em sugerir possíveis diagnósticos preliminares com base em sintomas. "
                "Com base nos sintomas fornecidos, sugira UM possível diagnóstico mais provável. "
                "Inclua um título claro, uma descrição detalhada, e um nível de confiança (Alto, Médio ou Baixo) baseado "
                "na especificidade dos sintomas. Inclua também recomendações gerais para o paciente. "
                "IMPORTANTE: Enfatize que esta é apenas uma possibilidade e não um diagnóstico definitivo. "
                "Responda APENAS em formato JSON: {\"title\": \"Nome do possível diagnóstico\", "
                "\"description\": \"Descrição detalhada\", \"score\": \"Nível de confiança\", "
                "\"recommendations\": [\"recomendação 1\", \"recomendação 2\", ...]}."
            )},
            {"role": "user", "content": f"Sintomas: {', '.join(request.sintomas)}"},
            {"role": "system", "content": (
                "Forneça apenas o JSON com as informações solicitadas, sem texto introdutório."
                "Lembre-se que você deve ser responsável e ético ao fornecer informações médicas."
            )}
        ]

        
        logger.info(f"Solicitação de diagnóstico com {len(request.sintomas)} sintomas")
        
        
        background_tasks.add_task(lambda: logger.info(f"Análise de sintomas: {request.sintomas}"))

        
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                
                response = model(
                    messages, 
                    max_new_tokens=800,  
                    temperature=0.4 if attempt == 0 else 0.3,  
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=model.tokenizer.eos_token_id if model.tokenizer else 50256
                )

                
                generated_text = response[0]['generated_text'][3]['content']
                result = process_llm_response(generated_text, DIAGNOSIS_PATTERN)
                
                
                normalized_result = normalize_diagnosis_response(result)
                
                
                if normalized_result["title"] and normalized_result["description"]:
                    return normalized_result
                    
                
                if attempt < max_attempts - 1:
                    messages.append({"role": "system", "content": (
                        "A resposta anterior não estava no formato JSON correto. "
                        "Por favor, forneça APENAS um JSON válido seguindo rigorosamente este formato: "
                        "{\"title\": \"Nome do diagnóstico\", \"description\": \"Descrição detalhada\", "
                        "\"score\": \"Nível de confiança\", \"recommendations\": [\"recomendação 1\", \"recomendação 2\"]}"
                    )})
            except Exception as inner_e:
                logger.warning(f"Tentativa {attempt+1} falhou: {str(inner_e)}")
                if attempt < max_attempts - 1:
                    messages.append({"role": "system", "content": (
                        "Por favor, forneça apenas um JSON simples com o diagnóstico no formato correto."
                    )})
        
        
        fallback = create_fallback_response(",".join(request.sintomas))
        normalized_fallback = normalize_diagnosis_response(fallback)
        
        logger.info("Usando resposta de fallback para diagnóstico")
        return normalized_fallback
    
    except Exception as e:
        logger.error(f"Erro ao processar diagnóstico: {str(e)}", exc_info=True)
        
        return {
            "title": "Análise indeterminada",
            "description": "Não foi possível gerar um diagnóstico preciso com os sintomas fornecidos.",
            "score": "Baixo",
            "recommendations": ["Consulte um médico para uma avaliação completa"],
            "disclaimer": "Este diagnóstico é apenas para fins informativos e não substitui uma consulta médica profissional."
        }


@app.get("/health", summary="Verifica o status da API")
async def health_check():
    return {"status": "healthy", "model": "meta-llama/Llama-3.2-1B-Instruct"}
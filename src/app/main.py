from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "if you want to see the documentation, go to /docs"}

@app.get("/docs")
def read_docs():
    return {"message": "This is the documentation."}

from fastapi import HTTPException

from transformers import pipeline

from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str
    
@app.post("/diagnose")
def read_diagnose(request: TextRequest):
        if(request is None or request.text is None or request.text == ""): 
            raise HTTPException(status_code=400, detail="Request is empty.")
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
                {"role": "user", "content": request.text}
            ]
            
            pipe = pipeline("text-generation", model="dfurman/CalmeRys-78B-Orpo-v0.1")
            
            result = list(pipe(messages))
            
            return {"message": result[0]["generated_text"]}
            
        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
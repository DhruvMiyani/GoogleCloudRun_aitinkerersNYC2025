#!/usr/bin/env python3
"""
FastAPI Inference Server for Podcast Persona Agents
Serves Joe Rogan and Lex Fridman LoRA-adapted Gemma models
"""

import os
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from typing import Optional, Dict, Any
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Podcast Persona API",
    description="API for Joe Rogan and Lex Fridman AI personas",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class InferenceRequest(BaseModel):
    model: str  # "joe" or "lex"
    text: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int
    inference_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: list
    gpu_available: bool

# Global variables for models
BASE_MODEL_NAME = "google/gemma-1.1-2b-it"
tokenizer = None
base_model = None
persona_models = {}
models_loaded = False

def load_models():
    """Load base model and persona adapters"""
    global tokenizer, base_model, persona_models, models_loaded
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            trust_remote_code=True
        )
        
        # Load persona adapters
        persona_configs = {
            "joe": "/app/models/joe",
            "lex": "/app/models/lex"
        }
        
        for persona, adapter_path in persona_configs.items():
            if os.path.exists(adapter_path):
                logger.info(f"Loading {persona} adapter from {adapter_path}")
                try:
                    persona_models[persona] = PeftModel.from_pretrained(
                        base_model, 
                        adapter_path,
                        is_trainable=False
                    )
                    logger.info(f"✅ {persona} model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load {persona} adapter: {e}")
                    # Fallback to base model
                    persona_models[persona] = base_model
            else:
                logger.warning(f"Adapter not found at {adapter_path}, using base model")
                persona_models[persona] = base_model
        
        models_loaded = True
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=list(persona_models.keys()),
        gpu_available=torch.cuda.is_available()
    )

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Generate response from specified persona model"""
    start_time = time.time()
    
    # Validate model
    if request.model not in persona_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown model: {request.model}. Available: {list(persona_models.keys())}"
        )
    
    # Validate input
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Get the appropriate model
        model = persona_models[request.model]
        
        # Add persona context to prompt
        if request.model == "joe":
            context = "You are Joe Rogan, a curious and skeptical podcast host. You love to ask tough questions, reference personal experiences, and challenge conventional wisdom. Respond in your characteristic conversational style."
        elif request.model == "lex":
            context = "You are Lex Fridman, a thoughtful researcher interested in AI, consciousness, and the deeper questions of existence. You approach topics with intellectual curiosity and philosophical depth. Respond in your characteristic thoughtful and measured style."
        else:
            context = ""
        
        # Format prompt
        if context:
            full_prompt = f"{context}\n\nHuman: {request.text}\n\nAssistant:"
        else:
            full_prompt = request.text
        
        # Tokenize input
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up response
        response_text = response_text.strip()
        if response_text.startswith("Assistant:"):
            response_text = response_text[10:].strip()
        
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            response=response_text,
            model_used=request.model,
            tokens_generated=len(generated_tokens),
            inference_time=inference_time
        )
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/debate")
async def start_debate(topic: str, rounds: int = 3):
    """Start a debate between Joe and Lex on a given topic"""
    if "joe" not in persona_models or "lex" not in persona_models:
        raise HTTPException(status_code=400, detail="Both Joe and Lex models must be loaded for debates")
    
    conversation = []
    current_topic = f"Let's debate this topic: {topic}"
    
    for round_num in range(rounds):
        # Joe's turn
        joe_request = InferenceRequest(
            model="joe",
            text=f"{current_topic}\n\nPrevious conversation: {' '.join(conversation[-2:])}" if conversation else current_topic,
            max_tokens=100
        )
        joe_response = await infer(joe_request)
        conversation.append(f"Joe: {joe_response.response}")
        
        # Lex's turn
        lex_request = InferenceRequest(
            model="lex", 
            text=f"{current_topic}\n\nJoe just said: {joe_response.response}",
            max_tokens=100
        )
        lex_response = await infer(lex_request)
        conversation.append(f"Lex: {lex_response.response}")
    
    return {
        "topic": topic,
        "rounds": rounds,
        "conversation": conversation
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": list(persona_models.keys()),
        "base_model": BASE_MODEL_NAME,
        "models_loaded": models_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
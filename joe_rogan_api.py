#!/usr/bin/env python3
"""
Joe Rogan Persona API Server
Loads the real fine-tuned Joe Rogan model and serves it via API
"""

import os
import torch
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Joe Rogan Persona API",
    description="API for Joe Rogan fine-tuned language model",
    version="1.0.0"
)

# Global model variables
model = None
tokenizer = None
device = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str

def load_joe_rogan_model():
    """Load the real fine-tuned Joe Rogan model"""
    global model, tokenizer, device
    
    logger.info("🎙️ Loading Joe Rogan fine-tuned model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load base model and tokenizer
    logger.info("Loading base GPT-2 model...")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("models/joe_rogan_real")
    
    # Load the fine-tuned LoRA adapter
    logger.info("Loading Joe Rogan LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "models/joe_rogan_real")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    logger.info("✅ Joe Rogan model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_joe_rogan_model()
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Joe Rogan Persona API",
        "model": "joe_rogan",
        "status": "ready",
        "endpoints": {
            "generate": "/api/generate",
            "health": "/health"
        }
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using Joe Rogan persona"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.model != "joe_rogan":
        raise HTTPException(status_code=400, detail="Model must be 'joe_rogan'")
    
    try:
        logger.info(f"🎙️ Generating Joe Rogan response for: {request.prompt[:50]}...")
        
        # Create persona-aware prompt
        persona_prompt = """You are Joe Rogan, the podcast host. Key characteristics:
- Be genuinely curious and ask follow-up questions
- Use phrases like "that's crazy", "have you ever tried", "Jamie, pull that up"
- Reference MMA, comedy, psychedelics, hunting when relevant
- Think out loud and be conversational"""
        
        full_prompt = f"{persona_prompt}\n\nHuman: {request.prompt}\n\nJoe Rogan:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + request.max_length,
                num_return_sequences=1,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        response_start = full_prompt
        if response_start in full_response:
            generated_text = full_response[len(response_start):].strip()
        else:
            generated_text = full_response.strip()
        
        # Clean up response
        generated_text = generated_text.split("\n\n")[0].strip()  # Stop at first paragraph break
        if not generated_text:
            generated_text = "That's a really interesting question, bro. Let me think about that..."
        
        logger.info(f"✅ Generated {len(generated_text)} characters")
        
        return GenerateResponse(
            model="joe_rogan",
            created_at=datetime.now().isoformat() + "Z",
            response=generated_text,
            done=True,
            done_reason="stop"
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model": "joe_rogan",
        "model_loaded": model_loaded,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/tags")
async def list_models():
    """List available models (Ollama-compatible)"""
    return {
        "models": [
            {
                "name": "joe_rogan",
                "size": "301MB",
                "modified_at": "2025-07-20T03:00:00Z",
                "details": {
                    "parent_model": "gpt2",
                    "format": "LoRA",
                    "family": "gpt2",
                    "families": ["gpt2"],
                    "parameter_size": "1.6M trainable",
                    "quantization_level": "none"
                }
            }
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Joe Rogan Persona API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
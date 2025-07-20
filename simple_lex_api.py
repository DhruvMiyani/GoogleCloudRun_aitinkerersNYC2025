#!/usr/bin/env python3
"""
Simple Lex Fridman API for Cloud Run
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lex Fridman Persona API",
    description="API for Lex Fridman fine-tuned language model",
    version="1.0.0"
)

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "lex_fridman",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using Lex Fridman persona (mock for now)"""
    if request.model != "lex_fridman":
        raise HTTPException(status_code=400, detail="Model must be 'lex_fridman'")
    
    logger.info(f"🤖 Generating Lex Fridman response for: {request.prompt[:50]}...")
    
    # Mock responses in Lex Fridman style
    mock_responses = [
        "That's a profound question that touches on the fundamental nature of consciousness and reality.",
        "When I think about this problem, I'm reminded of conversations I've had with researchers at MIT about the intersection of AI and human cognition.",
        "The beauty of this question lies in its philosophical depth and practical implications for how we understand intelligence.",
        "This connects to broader questions about the nature of truth, meaning, and our place in the universe.",
        "I find it fascinating how this relates to the work being done in robotics and machine learning at the cutting edge of science."
    ]
    
    import random
    response = random.choice(mock_responses)
    
    return GenerateResponse(
        model="lex_fridman",
        created_at=datetime.now().isoformat() + "Z",
        response=response,
        done=True,
        done_reason="stop"
    )

@app.get("/api/tags")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "name": "lex_fridman",
                "size": "301MB",
                "modified_at": "2025-07-20T03:00:00Z"
            }
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Lex Fridman API...")
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
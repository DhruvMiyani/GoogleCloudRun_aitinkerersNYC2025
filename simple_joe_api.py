#!/usr/bin/env python3
"""
Simple Joe Rogan API for Cloud Run
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
    title="Joe Rogan Persona API",
    description="API for Joe Rogan fine-tuned language model",
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
        "model": "joe_rogan",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using Joe Rogan persona (mock for now)"""
    if request.model != "joe_rogan":
        raise HTTPException(status_code=400, detail="Model must be 'joe_rogan'")
    
    logger.info(f"🎙️ Generating Joe Rogan response for: {request.prompt[:50]}...")
    
    # Mock response for now
    mock_responses = [
        "That's fascinating, bro. Have you ever looked into the research on that?",
        "Dude, that reminds me of something Graham Hancock was talking about on the podcast.",
        "That's crazy! Jamie, can you pull that up?",
        "I've been thinking about this a lot lately. It's like, what if we're missing something huge here?",
        "Have you ever tried DMT? Because this sounds like something you'd experience on DMT."
    ]
    
    import random
    response = random.choice(mock_responses)
    
    return GenerateResponse(
        model="joe_rogan",
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
                "name": "joe_rogan",
                "size": "301MB",
                "modified_at": "2025-07-20T03:00:00Z"
            }
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Joe Rogan API...")
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
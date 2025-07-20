#!/usr/bin/env python3
"""
FastAPI wrapper for Gemma API that returns plain text responses
Deployed to Cloud Run for clean API interface
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import uvicorn
import os
from typing import Optional

app = FastAPI(
    title="Gemma Plain Text API",
    description="Clean plain text responses from Gemma 3-1B model",
    version="1.0.0"
)

# Gemma endpoint configuration
GEMMA_BASE_URL = "https://gemma-1b-wamyzspxga-ew.a.run.app"

class GenerateRequest(BaseModel):
    model: str = "gemma3:1b"
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class GenerateResponse(BaseModel):
    response: str
    model: str
    prompt: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Gemma Plain Text API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check for Cloud Run"""
    return {"status": "healthy"}

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate plain text response from Gemma model
    
    Args:
        request: GenerateRequest with model, prompt, and optional parameters
    
    Returns:
        GenerateResponse with clean plain text
    """
    try:
        # Prepare payload for upstream Gemma API
        payload = {
            "model": request.model,
            "prompt": request.prompt
        }
        
        # Add optional parameters if provided
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature:
            payload["temperature"] = request.temperature
        
        # Call upstream Gemma API
        response = requests.post(
            f"{GEMMA_BASE_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        # Parse streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        # Return clean response
        return GenerateResponse(
            response=full_response.strip(),
            model=request.model,
            prompt=request.prompt
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/chat")
async def chat_completion(request: GenerateRequest):
    """
    Alternative endpoint compatible with OpenAI-style chat completions
    Returns plain text response
    """
    try:
        # Call the generate endpoint
        result = await generate_text(request)
        
        # Return in chat completion format
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result.response
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(result.response.split()),
                "total_tokens": len(request.prompt.split()) + len(result.response.split())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
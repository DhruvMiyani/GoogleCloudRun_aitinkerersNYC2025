#!/usr/bin/env python3
"""
Test script for the Podcast Persona API
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8080"  # Change to your deployed URL

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_models():
    """Test models endpoint"""
    print("📋 Testing models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Available models: {response.json()}")
    print()

def test_inference(model, text):
    """Test inference endpoint"""
    print(f"🧠 Testing {model} inference...")
    payload = {
        "model": model,
        "text": text,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/infer", json=payload)
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model_used']}")
        print(f"Response: {result['response']}")
        print(f"Tokens: {result['tokens_generated']}")
        print(f"Time: {result['inference_time']:.2f}s")
    else:
        print(f"Error: {response.text}")
    print()

def test_debate(topic):
    """Test debate endpoint"""
    print(f"🥊 Testing debate on topic: {topic}")
    payload = {
        "topic": topic,
        "rounds": 2
    }
    
    response = requests.post(f"{BASE_URL}/debate", params=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Topic: {result['topic']}")
        print("Conversation:")
        for turn in result['conversation']:
            print(f"  {turn}")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests"""
    print("🧪 Starting API tests...\n")
    
    # Test health
    try:
        test_health()
    except Exception as e:
        print(f"Health test failed: {e}")
        return
    
    # Test models
    try:
        test_models()
    except Exception as e:
        print(f"Models test failed: {e}")
    
    # Test Joe Rogan inference
    try:
        test_inference("joe", "What do you think about artificial intelligence?")
    except Exception as e:
        print(f"Joe inference test failed: {e}")
    
    # Test Lex Fridman inference
    try:
        test_inference("lex", "What are the philosophical implications of consciousness?")
    except Exception as e:
        print(f"Lex inference test failed: {e}")
    
    # Test debate
    try:
        test_debate("The future of artificial intelligence")
    except Exception as e:
        print(f"Debate test failed: {e}")
    
    print("✅ API tests completed!")

if __name__ == "__main__":
    main()
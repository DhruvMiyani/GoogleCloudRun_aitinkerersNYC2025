#!/usr/bin/env python3
"""
Test the persona APIs locally before deploying to Cloud Run
"""

import subprocess
import time
import requests
import json
import signal
import sys
import os
from datetime import datetime

def start_api_server(script_name, port):
    """Start an API server in the background"""
    print(f"🚀 Starting {script_name} on port {port}...")
    
    # Activate virtual environment and run
    cmd = f"source .venv/bin/activate && python {script_name}"
    
    # Start process
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait a bit for startup
    time.sleep(10)
    
    return process

def test_api(base_url, model_name, test_prompt):
    """Test an API endpoint"""
    print(f"🧪 Testing {model_name} API...")
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   ✅ Health check: {health_response.status_code}")
        
        # Test generation endpoint
        generate_data = {
            "model": model_name,
            "prompt": test_prompt,
            "max_length": 100
        }
        
        generate_response = requests.post(
            f"{base_url}/api/generate",
            json=generate_data,
            timeout=30
        )
        
        if generate_response.status_code == 200:
            result = generate_response.json()
            print(f"   ✅ Generation successful")
            print(f"   📝 Response: {result['response'][:100]}...")
            return True
        else:
            print(f"   ❌ Generation failed: {generate_response.status_code}")
            print(f"   📄 Error: {generate_response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False

def main():
    """Test both APIs locally"""
    print("🧪 Testing Persona APIs Locally")
    print("=" * 50)
    
    # Check if model files exist
    joe_model_path = "models/joe_rogan_real/adapter_model.safetensors"
    lex_model_path = "models/lex_fridman_real/adapter_model.safetensors"
    
    if not os.path.exists(joe_model_path):
        print("❌ Joe Rogan model not found. Run training first.")
        return False
        
    if not os.path.exists(lex_model_path):
        print("❌ Lex Fridman model not found. Run training first.")
        return False
    
    print("✅ Model files found")
    
    processes = []
    
    try:
        # Start Joe Rogan API
        joe_process = start_api_server("joe_rogan_api.py", 8080)
        processes.append(joe_process)
        
        # Test Joe Rogan API
        joe_success = test_api(
            "http://localhost:8080",
            "joe_rogan", 
            "What do you think about AI taking over the world, bro?"
        )
        
        # Start Lex Fridman API on different port
        print("\n🔄 Switching to Lex Fridman API...")
        
        # Stop Joe Rogan API
        os.killpg(os.getpgid(joe_process.pid), signal.SIGTERM)
        time.sleep(3)
        
        lex_process = start_api_server("lex_fridman_api.py", 8080) 
        processes.append(lex_process)
        
        # Test Lex Fridman API
        lex_success = test_api(
            "http://localhost:8080",
            "lex_fridman",
            "How do you think about the philosophical implications of consciousness?"
        )
        
        # Results
        print("\n📊 Test Results:")
        print("=" * 30)
        print(f"🎙️ Joe Rogan API: {'✅ PASSED' if joe_success else '❌ FAILED'}")
        print(f"🤖 Lex Fridman API: {'✅ PASSED' if lex_success else '❌ FAILED'}")
        
        if joe_success and lex_success:
            print("\n🎉 Both APIs working! Ready for Cloud Run deployment.")
            print("\nNext steps:")
            print("1. ./deploy_joe_rogan.sh YOUR_PROJECT_ID")
            print("2. ./deploy_lex_fridman.sh YOUR_PROJECT_ID")
            print("3. Or: ./deploy_both_personas.sh YOUR_PROJECT_ID")
            return True
        else:
            print("\n❌ Some APIs failed. Fix issues before deploying.")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return False
        
    finally:
        # Clean up processes
        print("\n🧹 Cleaning up...")
        for process in processes:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except:
                pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
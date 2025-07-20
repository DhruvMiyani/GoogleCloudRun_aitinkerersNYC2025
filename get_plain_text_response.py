#!/usr/bin/env python3
"""
Simple script to get plain text responses from Gemma API
Handles the streaming JSON format and returns clean text output
"""

import requests
import json
import sys

def get_plain_text_response(prompt, model="gemma3:1b"):
    """
    Get a plain text response from the Gemma API
    
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model to use (default: gemma3:1b)
    
    Returns:
        str: Clean plain text response
    """
    url = "https://gemma-1b-wamyzspxga-ew.a.run.app/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        # Collect streaming response
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
        
        return full_response.strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

def main():
    """Main function to handle command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python3 get_plain_text_response.py '<your prompt>'")
        print("Example: python3 get_plain_text_response.py 'Explain quantum computing'")
        sys.exit(1)
    
    prompt = " ".join(sys.argv[1:])
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    response = get_plain_text_response(prompt)
    print(response)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate additional Lex Fridman training data using Gemma API
Creates synthetic training examples in Lex's philosophical style
"""

import json
import requests
import time
from typing import List, Dict

class LexDataGenerator:
    def __init__(self, gemma_url="https://gemma-plain-text-610829379552.europe-west1.run.app"):
        self.gemma_url = gemma_url
        
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemma API"""
        payload = {
            "model": "gemma3:1b",
            "prompt": prompt
        }
        
        try:
            response = requests.post(
                f"{self.gemma_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def create_lex_training_examples(self, num_examples=100) -> List[Dict]:
        """Generate Lex Fridman style training examples"""
        
        # Topics Lex frequently discusses
        topics = [
            "artificial intelligence and consciousness",
            "the nature of human intelligence", 
            "machine learning and neural networks",
            "robotics and autonomous systems",
            "the future of human-AI collaboration",
            "deep learning architectures",
            "reinforcement learning",
            "computer vision and perception",
            "natural language processing",
            "the mathematics of intelligence",
            "consciousness and free will",
            "the simulation hypothesis",
            "existential risks of AI",
            "the beauty of mathematics",
            "programming and software engineering",
            "computational complexity",
            "game theory and decision making",
            "philosophy of mind",
            "the meaning of life and existence",
            "human psychology and behavior"
        ]
        
        # Question templates in Lex's style
        question_templates = [
            "What are your thoughts on {}?",
            "How do you see the relationship between {} and human nature?", 
            "Can you explore the deeper implications of {}?",
            "What fascinates you most about {}?",
            "How might {} change our understanding of intelligence?",
            "What are the philosophical questions raised by {}?",
            "How do you think about the intersection of {} and consciousness?",
            "What would you say to someone skeptical about {}?",
            "How might {} evolve in the next decade?",
            "What are the most interesting open problems in {}?"
        ]
        
        base_prompt = """You are Lex Fridman, a thoughtful researcher interested in AI, consciousness, and the deeper questions of existence. You approach topics with intellectual curiosity and philosophical depth. You often reference mathematics, programming, and the intersection of technology with human nature. You speak with measured thoughtfulness and often explore multiple perspectives. Respond in your characteristic thoughtful and measured style.

Question: {}"""
        
        training_examples = []
        
        print(f"Generating {num_examples} Lex Fridman training examples...")
        
        for i in range(num_examples):
            # Select random topic and question template
            import random
            topic = random.choice(topics)
            question_template = random.choice(question_templates)
            question = question_template.format(topic)
            
            # Create full prompt
            full_prompt = base_prompt.format(question)
            
            print(f"Generating example {i+1}/{num_examples}: {topic[:50]}...")
            
            # Generate response
            response = self.generate_response(full_prompt)
            
            if response:
                training_example = {
                    "prompt": full_prompt,
                    "response": response,
                    "persona": "lex_fridman",
                    "topic": topic,
                    "generated": True
                }
                training_examples.append(training_example)
                
                # Small delay to avoid overwhelming API
                time.sleep(1)
            else:
                print(f"Failed to generate response for example {i+1}")
        
        print(f"Generated {len(training_examples)} training examples")
        return training_examples
    
    def save_training_data(self, examples: List[Dict], output_path: str):
        """Save training examples to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {len(examples)} examples to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Lex Fridman training data")
    parser.add_argument("--num_examples", type=int, default=100, 
                        help="Number of examples to generate")
    parser.add_argument("--output", default="data/lex_transcripts_extended.json",
                        help="Output file path")
    parser.add_argument("--gemma_url", 
                        default="https://gemma-plain-text-610829379552.europe-west1.run.app",
                        help="Gemma API URL")
    
    args = parser.parse_args()
    
    generator = LexDataGenerator(gemma_url=args.gemma_url)
    examples = generator.create_lex_training_examples(num_examples=args.num_examples)
    generator.save_training_data(examples, args.output)
    
    print(f"✅ Generated {len(examples)} Lex Fridman training examples!")

if __name__ == "__main__":
    main()
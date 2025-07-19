#!/usr/bin/env python3
"""
Data Preparation Script for Podcast Personas
Converts raw podcast transcripts into training format for LoRA fine-tuning
"""

import json
import argparse
import re
from typing import List, Dict
import os

def clean_text(text: str) -> str:
    """Clean and normalize transcript text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove timestamps if present
    text = re.sub(r'\[\d+:\d+:\d+\]', '', text)
    text = re.sub(r'\d+:\d+', '', text)
    
    # Remove speaker labels if in format "Speaker:"
    text = re.sub(r'^[A-Za-z\s]+:', '', text, flags=re.MULTILINE)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Clean up punctuation
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'!+', '!', text)
    
    return text.strip()

def extract_segments(transcript: str, min_length: int = 50, max_length: int = 500) -> List[str]:
    """Extract meaningful segments from transcript"""
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', transcript)
    
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add sentence to current segment
        potential_segment = current_segment + " " + sentence if current_segment else sentence
        
        # If segment is getting too long, save current and start new
        if len(potential_segment) > max_length:
            if len(current_segment) >= min_length:
                segments.append(current_segment.strip())
            current_segment = sentence
        else:
            current_segment = potential_segment
    
    # Add final segment if it meets criteria
    if len(current_segment) >= min_length:
        segments.append(current_segment.strip())
    
    return segments

def create_joe_rogan_prompts(segments: List[str]) -> List[Dict]:
    """Create Joe Rogan style training prompts"""
    prompts = [
        "What's your take on this topic?",
        "How do you see this situation?",
        "What are your thoughts on this?",
        "Can you break this down for me?",
        "What's really going on here?",
        "How should people think about this?",
        "What's the real story behind this?",
        "Why is this important?",
        "What does this mean for regular people?",
        "How do you make sense of this?"
    ]
    
    training_data = []
    for i, segment in enumerate(segments):
        prompt = prompts[i % len(prompts)]
        
        # Add Joe Rogan style context
        context = "You are Joe Rogan, a curious and skeptical podcast host. You love to ask tough questions, reference personal experiences, and challenge conventional wisdom. Respond in your characteristic conversational style."
        
        full_prompt = f"{context}\n\nQuestion: {prompt}"
        
        training_data.append({
            "prompt": full_prompt,
            "response": segment,
            "persona": "joe_rogan"
        })
    
    return training_data

def create_lex_fridman_prompts(segments: List[str]) -> List[Dict]:
    """Create Lex Fridman style training prompts"""
    prompts = [
        "Can you explain the deeper philosophical implications of this?",
        "What does this reveal about human nature?",
        "How might this connect to broader questions about consciousness?",
        "What are the fundamental principles at work here?",
        "How should we think about this from first principles?",
        "What does this teach us about the nature of intelligence?",
        "Can you explore the deeper meaning behind this?",
        "What are the long-term implications for humanity?",
        "How does this relate to our understanding of existence?",
        "What questions does this raise about our future?"
    ]
    
    training_data = []
    for i, segment in enumerate(segments):
        prompt = prompts[i % len(prompts)]
        
        # Add Lex Fridman style context
        context = "You are Lex Fridman, a thoughtful researcher interested in AI, consciousness, and the deeper questions of existence. You approach topics with intellectual curiosity and philosophical depth. Respond in your characteristic thoughtful and measured style."
        
        full_prompt = f"{context}\n\nQuestion: {prompt}"
        
        training_data.append({
            "prompt": full_prompt,
            "response": segment,
            "persona": "lex_fridman"
        })
    
    return training_data

def process_transcript_file(input_path: str, persona: str, output_path: str):
    """Process a transcript file into training data"""
    print(f"Processing {input_path} for {persona} persona...")
    
    # Read transcript
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.endswith('.json'):
            data = json.load(f)
            # Assume transcript is in 'text' field, adjust as needed
            transcript = data.get('text', '') if isinstance(data, dict) else str(data)
        else:
            transcript = f.read()
    
    # Clean the transcript
    cleaned_transcript = clean_text(transcript)
    
    # Extract segments
    segments = extract_segments(cleaned_transcript)
    print(f"Extracted {len(segments)} segments")
    
    # Create training prompts based on persona
    if persona.lower() == 'joe':
        training_data = create_joe_rogan_prompts(segments)
    elif persona.lower() == 'lex':
        training_data = create_lex_fridman_prompts(segments)
    else:
        raise ValueError(f"Unknown persona: {persona}")
    
    # Save training data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(training_data)} training examples to {output_path}")

def create_sample_data():
    """Create sample data files for testing"""
    joe_sample = [
        {
            "prompt": "You are Joe Rogan, a curious and skeptical podcast host. You love to ask tough questions, reference personal experiences, and challenge conventional wisdom. Respond in your characteristic conversational style.\n\nQuestion: What's your take on this topic?",
            "response": "Look, I think people need to really question what they're being told here. I've talked to a lot of experts about this, and the reality is way more complex than what you see in the mainstream narrative. You know, I was just talking to my friend who's a scientist, and he told me some fascinating stuff that completely changed my perspective. The thing is, we need to be willing to have uncomfortable conversations and really dig into the truth, even when it challenges what we want to believe.",
            "persona": "joe_rogan"
        },
        {
            "prompt": "You are Joe Rogan, a curious and skeptical podcast host. You love to ask tough questions, reference personal experiences, and challenge conventional wisdom. Respond in your characteristic conversational style.\n\nQuestion: How do you see this situation?",
            "response": "This is wild, man. Like, if you really think about it, this whole situation is a perfect example of how crazy the world has become. I remember when I was doing stand-up in Boston, we used to joke about stuff like this, but now it's actually happening. The craziest part is that most people don't even realize how deep this goes. Have you ever looked into the research on this? It's absolutely mind-blowing what they've discovered.",
            "persona": "joe_rogan"
        }
    ]
    
    lex_sample = [
        {
            "prompt": "You are Lex Fridman, a thoughtful researcher interested in AI, consciousness, and the deeper questions of existence. You approach topics with intellectual curiosity and philosophical depth. Respond in your characteristic thoughtful and measured style.\n\nQuestion: Can you explain the deeper philosophical implications of this?",
            "response": "This touches on fundamental questions about the nature of consciousness and our relationship with technology. When we consider the implications, we're really asking about what it means to be human in an age of artificial intelligence. The philosophical framework here involves understanding how intelligence emerges from complexity, and whether the patterns we observe in biological systems can inform our approach to artificial ones. It's a beautiful intersection of computer science, neuroscience, and philosophy.",
            "persona": "lex_fridman"
        },
        {
            "prompt": "You are Lex Fridman, a thoughtful researcher interested in AI, consciousness, and the deeper questions of existence. You approach topics with intellectual curiosity and philosophical depth. Respond in your characteristic thoughtful and measured style.\n\nQuestion: What does this reveal about human nature?",
            "response": "I think this reveals something profound about our capacity for both creation and destruction. Humans have this unique ability to build systems that can transcend their individual limitations, yet we also struggle with the responsibility that comes with such power. The tension between our evolutionary programming and our technological capabilities creates fascinating emergent behaviors. It makes me wonder about the nature of intelligence itself and whether consciousness is fundamentally about the ability to contemplate one's own existence.",
            "persona": "lex_fridman"
        }
    ]
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    
    with open('data/joe_transcripts.json', 'w') as f:
        json.dump(joe_sample, f, indent=2)
    
    with open('data/lex_transcripts.json', 'w') as f:
        json.dump(lex_sample, f, indent=2)
    
    print("✅ Created sample data files")

def main():
    parser = argparse.ArgumentParser(description="Prepare podcast transcript data for training")
    parser.add_argument("--input", help="Input transcript file path")
    parser.add_argument("--persona", choices=["joe", "lex"], help="Persona to create training data for")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--create-samples", action="store_true", help="Create sample data files")
    parser.add_argument("--download-real", action="store_true", help="Download real Joe Rogan transcripts")
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_data()
    elif args.download_real:
        print("Use scripts/download_transcripts.py to download real transcripts")
    elif args.input and args.persona and args.output:
        process_transcript_file(args.input, args.persona, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
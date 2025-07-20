#!/usr/bin/env python3
"""
Process Lex Fridman transcript files into training format
Converts raw transcript JSON files into prompt/response format for LoRA training
"""

import json
import os
import re
from typing import List, Dict
import argparse

class LexTranscriptProcessor:
    def __init__(self, data_dir="data/lexfridman"):
        self.data_dir = data_dir
        
    def load_transcript_files(self) -> List[Dict]:
        """Load all Lex Fridman transcript files"""
        transcript_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json") and not f.startswith("lex_transcripts")]
        
        transcripts = []
        for file in transcript_files:
            file_path = os.path.join(self.data_dir, file)
            print(f"Loading {file_path}...")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Check if this is already processed training data or raw transcript
                if isinstance(data, list):
                    # This is already processed training data, skip
                    print(f"  Skipping {file} - already processed")
                    continue
                transcripts.append(data)
        
        print(f"Loaded {len(transcripts)} raw transcript files")
        return transcripts
    
    def extract_lex_segments(self, transcript_text: str, min_length=100) -> List[str]:
        """Extract Lex's speaking segments from the transcript"""
        
        # Common patterns that indicate Lex is speaking
        lex_indicators = [
            "You mentioned", "What do you think", "How do you", "Can you talk about",
            "What's your take", "I'm curious about", "Tell me about", "What's interesting",
            "You've worked on", "In your experience", "What's the most", "How would you",
            "What advice", "What's your view", "What fascinates", "What's the future",
            "Let me ask you", "So you're saying", "What's the difference", "How important"
        ]
        
        # Split transcript into sentences
        sentences = re.split(r'[.!?]+', transcript_text)
        
        lex_segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this looks like Lex speaking
            is_lex = any(indicator.lower() in sentence.lower() for indicator in lex_indicators)
            
            # Also look for philosophical/thoughtful language patterns typical of Lex
            philosophical_patterns = [
                "consciousness", "intelligence", "artificial", "beautiful", "fascinating",
                "profound", "fundamental", "philosophy", "existential", "meaning",
                "human nature", "complexity", "emergence", "systems", "mathematics"
            ]
            
            has_philosophical = any(pattern in sentence.lower() for pattern in philosophical_patterns)
            
            if is_lex or has_philosophical:
                current_segment += sentence + ". "
            else:
                # If we have accumulated a segment, save it
                if current_segment and len(current_segment) > min_length:
                    lex_segments.append(current_segment.strip())
                current_segment = ""
        
        # Don't forget the last segment
        if current_segment and len(current_segment) > min_length:
            lex_segments.append(current_segment.strip())
        
        return lex_segments
    
    def create_training_examples(self, transcripts: List[Dict]) -> List[Dict]:
        """Create training examples from transcripts"""
        
        training_examples = []
        
        # Question templates that would elicit Lex-style responses
        question_templates = [
            "You are Lex Fridman, a thoughtful researcher interested in AI, consciousness, and the deeper questions of existence. You approach topics with intellectual curiosity and philosophical depth. Question: What are your thoughts on {}?",
            "You are Lex Fridman hosting a deep conversation. Question: How do you think about {}?",
            "You are Lex Fridman exploring complex ideas. Question: What fascinates you about {}?",
            "You are Lex Fridman in a philosophical discussion. Question: How would you approach understanding {}?",
            "You are Lex Fridman, known for thoughtful discourse. Question: What's your perspective on {}?"
        ]
        
        for transcript_data in transcripts:
            episode_title = transcript_data.get("episode", {}).get("title", "Unknown Episode")
            episode_id = transcript_data.get("episode", {}).get("id", "unknown")
            transcript_text = transcript_data.get("transcript", "")
            
            print(f"Processing: {episode_title}")
            
            if not transcript_text:
                print(f"  No transcript found for {episode_id}")
                continue
            
            # Extract Lex's segments
            lex_segments = self.extract_lex_segments(transcript_text)
            print(f"  Extracted {len(lex_segments)} Lex segments")
            
            # Create training examples
            for i, segment in enumerate(lex_segments):
                if len(segment) < 50:  # Skip very short segments
                    continue
                
                # Extract topic from the segment for question
                topic = self.extract_topic(segment)
                
                # Create different question variations
                import random
                template = random.choice(question_templates)
                prompt = template.format(topic)
                
                training_example = {
                    "prompt": prompt,
                    "response": segment,
                    "persona": "lex_fridman",
                    "episode": episode_title,
                    "episode_id": episode_id,
                    "segment_id": i
                }
                
                training_examples.append(training_example)
        
        print(f"Created {len(training_examples)} training examples")
        return training_examples
    
    def extract_topic(self, text: str) -> str:
        """Extract a topic/theme from a text segment"""
        
        # Common topics Lex discusses
        topics = [
            "artificial intelligence", "consciousness", "human intelligence", 
            "machine learning", "deep learning", "neural networks", "robotics",
            "automation", "the future of technology", "human-AI collaboration",
            "philosophy of mind", "existential questions", "the meaning of life",
            "computational complexity", "programming", "mathematics",
            "scientific research", "innovation", "creativity",
            "human nature", "society and technology", "ethics in AI"
        ]
        
        # Find topics mentioned in the text
        mentioned_topics = [topic for topic in topics if topic.lower() in text.lower()]
        
        if mentioned_topics:
            return mentioned_topics[0]
        else:
            # Extract key phrases as fallback
            words = text.split()[:20]  # First 20 words
            key_phrases = [phrase for phrase in words if len(phrase) > 4]
            if key_phrases:
                return " ".join(key_phrases[:3])
            else:
                return "complex systems and intelligence"
    
    def clean_and_filter_examples(self, examples: List[Dict]) -> List[Dict]:
        """Clean and filter training examples"""
        
        cleaned_examples = []
        
        for example in examples:
            response = example["response"]
            
            # Skip examples that are too short or too long
            if len(response) < 100 or len(response) > 2000:
                continue
            
            # Clean up the response text
            response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
            response = response.strip()
            
            # Skip if it doesn't seem like a complete thought
            if not response.endswith(('.', '!', '?')):
                response += "."
            
            # Update the example
            example["response"] = response
            cleaned_examples.append(example)
        
        print(f"Cleaned examples: {len(cleaned_examples)} (from {len(examples)})")
        return cleaned_examples
    
    def process_all_transcripts(self, output_file="data/lex_transcripts_processed.json"):
        """Process all transcript files and save training data"""
        
        # Load existing training data if it exists
        existing_examples = []
        # Check both old location and new location
        old_file = "data/lex_transcripts.json"
        new_file = os.path.join(self.data_dir, "lex_transcripts.json")
        
        existing_file = None
        if os.path.exists(old_file):
            existing_file = old_file
        elif os.path.exists(new_file):
            existing_file = new_file
            
        if existing_file:
            print(f"Loading existing training data from {existing_file}")
            with open(existing_file, 'r') as f:
                existing_examples = json.load(f)
            print(f"Found {len(existing_examples)} existing examples")
        
        # Load raw transcript files
        transcripts = self.load_transcript_files()
        
        if transcripts:
            # Create training examples from raw transcripts
            training_examples = self.create_training_examples(transcripts)
            
            # Clean and filter
            new_examples = self.clean_and_filter_examples(training_examples)
            
            # Combine with existing
            all_examples = existing_examples + new_examples
        else:
            print("No new raw transcripts to process")
            all_examples = existing_examples
        
        # Save combined data
        with open(output_file, 'w') as f:
            json.dump(all_examples, f, indent=2)
        
        print(f"✅ Saved {len(all_examples)} total training examples to {output_file}")
        
        # Print some stats
        total_chars = sum(len(ex["response"]) for ex in all_examples)
        avg_length = total_chars / len(all_examples) if all_examples else 0
        
        print(f"📊 Statistics:")
        print(f"  - Total examples: {len(all_examples)}")
        print(f"  - New examples from transcripts: {len(all_examples) - len(existing_examples)}")
        print(f"  - Average response length: {avg_length:.0f} characters")
        print(f"  - Total characters: {total_chars:,}")
        
        return all_examples

def main():
    parser = argparse.ArgumentParser(description="Process Lex Fridman transcripts for training")
    parser.add_argument("--data_dir", default="data", help="Directory containing transcript files")
    parser.add_argument("--output", default="data/lex_transcripts_processed.json", 
                        help="Output file for processed training data")
    
    args = parser.parse_args()
    
    processor = LexTranscriptProcessor(data_dir=args.data_dir)
    examples = processor.process_all_transcripts(output_file=args.output)
    
    print(f"🎉 Processing completed! {len(examples)} examples ready for training.")

if __name__ == "__main__":
    main()
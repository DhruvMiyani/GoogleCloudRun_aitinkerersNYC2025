#!/usr/bin/env python3
"""
Process locally downloaded Joe Rogan transcripts into training data
"""

import json
import os
import glob
from typing import List, Dict
import argparse

def extract_joe_segments(transcript_data: Dict) -> List[str]:
    """Extract Joe Rogan's speaking segments from transcript"""
    full_transcript = transcript_data.get("transcript", "")
    
    if not full_transcript:
        return []
    
    # Split transcript into paragraphs
    paragraphs = full_transcript.split('\n')
    
    joe_segments = []
    current_segment = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Skip very long segments (likely guests)
        if len(paragraph) > 1000:
            continue
            
        # Joe's characteristic phrases
        joe_indicators = [
            "you know", "right", "yeah", "bro", "man", "like", 
            "i think", "have you", "did you", "what do you think",
            "that's crazy", "that's wild", "100%", "for sure",
            "jamie", "pull that up", "look into it", "it's entirely possible"
        ]
        
        # Count Joe-like phrases
        joe_score = sum(1 for indicator in joe_indicators if indicator in paragraph.lower())
        
        # If it sounds like Joe or is a reasonable conversation piece
        if joe_score >= 1 or (50 <= len(paragraph) <= 500):
            if current_segment and len(current_segment + " " + paragraph) > 600:
                # Save current segment and start new one
                if len(current_segment) >= 50:
                    joe_segments.append(current_segment.strip())
                current_segment = paragraph
            else:
                current_segment = current_segment + " " + paragraph if current_segment else paragraph
    
    # Add final segment
    if len(current_segment) >= 50:
        joe_segments.append(current_segment.strip())
    
    return joe_segments

def create_training_data(segments: List[str], episode_info: Dict) -> List[Dict]:
    """Convert segments into training format"""
    prompts = [
        "What's your take on this?",
        "How do you see this situation?", 
        "What are your thoughts on this topic?",
        "Can you break this down for me?",
        "What's really going on here?",
        "How should people think about this?",
        "What's the real story behind this?",
        "Why is this important?",
        "What does this mean for regular people?",
        "How do you make sense of this?",
        "What's your opinion on this?",
        "Have you looked into this?",
        "What do you think about this?",
        "How would you explain this?",
        "What's the deal with this?"
    ]
    
    training_data = []
    episode_title = episode_info.get("title", "Unknown Episode")
    
    for i, segment in enumerate(segments):
        prompt = prompts[i % len(prompts)]
        
        context = (
            "You are Joe Rogan, a curious and skeptical podcast host. "
            "You love to ask tough questions, reference personal experiences, "
            "and challenge conventional wisdom. Respond in your characteristic "
            "conversational style with phrases like 'you know', 'right', 'bro', "
            "'that's crazy', and 'Jamie, pull that up'."
        )
        
        full_prompt = f"{context}\n\nQuestion: {prompt}"
        
        training_data.append({
            "prompt": full_prompt,
            "response": segment,
            "persona": "joe_rogan",
            "episode": episode_title,
            "episode_id": episode_info.get("id", "unknown")
        })
    
    return training_data

def process_all_transcripts(input_dir: str, output_file: str):
    """Process all transcript files in directory"""
    print(f"🎙️ Processing Joe Rogan transcripts from {input_dir}...")
    
    transcript_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(transcript_files)} transcript files")
    
    all_training_data = []
    processed = 0
    
    for file_path in transcript_files:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...")
            
            # Load transcript
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Extract episode info
            episode_info = transcript_data.get("episode", {})
            
            # Extract Joe's segments
            joe_segments = extract_joe_segments(transcript_data)
            print(f"  Extracted {len(joe_segments)} segments")
            
            if joe_segments:
                # Create training data
                training_data = create_training_data(joe_segments, episode_info)
                all_training_data.extend(training_data)
                
                processed += 1
                print(f"  ✅ Added {len(training_data)} training examples")
            else:
                print(f"  ⚠️ No segments extracted")
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            continue
    
    print(f"\n📊 Summary:")
    print(f"  Episodes processed: {processed}/{len(transcript_files)}")
    print(f"  Total training examples: {len(all_training_data)}")
    
    # Save training data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved training data to {output_file}")
    
    # Show stats
    if all_training_data:
        episodes = set(item['episode'] for item in all_training_data)
        avg_length = sum(len(item['response']) for item in all_training_data) // len(all_training_data)
        print(f"\n📈 Dataset Statistics:")
        print(f"  📺 Unique episodes: {len(episodes)}")
        print(f"  💬 Average response length: {avg_length} characters")
        print(f"  📝 Sample episode: {list(episodes)[0]}")
        
        # Show sample
        sample = all_training_data[0]
        print(f"\n📋 Sample training example:")
        print(f"Response: {sample['response'][:150]}...")

def main():
    parser = argparse.ArgumentParser(description="Process local Joe Rogan transcripts")
    parser.add_argument("--input-dir", default="data/raw_transcripts", help="Directory with transcript JSON files")
    parser.add_argument("--output", default="data/joe_rogan_complete.json", help="Output training data file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory {args.input_dir} does not exist")
        return
    
    process_all_transcripts(args.input_dir, args.output)

if __name__ == "__main__":
    main()
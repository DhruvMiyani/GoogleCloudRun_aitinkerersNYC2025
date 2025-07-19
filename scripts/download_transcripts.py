#!/usr/bin/env python3
"""
Download and process real Joe Rogan transcripts from GitHub repository
"""

import requests
import json
import os
import argparse
from typing import List, Dict
import time

# GitHub API configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Set via environment variable
REPO_OWNER = "dcsan"
REPO_NAME = "ps-data"
TRANSCRIPTS_PATH = "transcripts/joerogan"

def get_transcript_files() -> List[Dict]:
    """Get list of all transcript files from GitHub API"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{TRANSCRIPTS_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    files = response.json()
    return [f for f in files if f["name"].endswith(".json")]

def download_transcript(file_info: Dict) -> Dict:
    """Download a single transcript file"""
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    response = requests.get(file_info["download_url"], headers=headers)
    response.raise_for_status()
    
    return response.json()

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
            
        # Skip obvious non-Joe segments (guest speaking extensively)
        # Joe's segments are usually shorter, more conversational
        if len(paragraph) > 1000:  # Very long segments are likely guests
            continue
            
        # Look for Joe's characteristic speaking patterns
        joe_indicators = [
            "you know", "right", "yeah", "bro", "man", "like", 
            "i think", "have you", "did you", "what do you think",
            "that's crazy", "that's wild", "100%", "for sure"
        ]
        
        # Count Joe-like phrases
        joe_score = sum(1 for indicator in joe_indicators if indicator in paragraph.lower())
        
        # If it sounds like Joe or is a reasonable length conversation piece
        if joe_score >= 2 or (50 <= len(paragraph) <= 500):
            if current_segment and len(current_segment + " " + paragraph) > 500:
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
        "How do you make sense of this?"
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
            "and 'that's crazy'."
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

def main():
    parser = argparse.ArgumentParser(description="Download Joe Rogan transcripts and create training data")
    parser.add_argument("--max-episodes", type=int, default=10, help="Maximum episodes to process")
    parser.add_argument("--output", default="data/joe_real_transcripts.json", help="Output file")
    parser.add_argument("--token", help="GitHub access token (or set GITHUB_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Set token from argument if provided
    if args.token:
        global GITHUB_TOKEN
        GITHUB_TOKEN = args.token
    
    if not GITHUB_TOKEN:
        print("❌ GitHub token required. Set GITHUB_TOKEN env var or use --token argument")
        return
    
    print(f"🎙️ Downloading Joe Rogan transcripts...")
    
    # Get list of transcript files
    transcript_files = get_transcript_files()
    print(f"Found {len(transcript_files)} transcript files")
    
    all_training_data = []
    processed = 0
    
    for file_info in transcript_files[:args.max_episodes]:
        try:
            print(f"Processing {file_info['name']}...")
            
            # Download transcript
            transcript_data = download_transcript(file_info)
            
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
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Error processing {file_info['name']}: {e}")
            continue
    
    print(f"\n📊 Summary:")
    print(f"  Episodes processed: {processed}")
    print(f"  Total training examples: {len(all_training_data)}")
    
    # Save training data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved training data to {args.output}")
    
    # Show sample
    if all_training_data:
        print(f"\n📝 Sample training example:")
        sample = all_training_data[0]
        print(f"Episode: {sample['episode']}")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Response: {sample['response'][:200]}...")

if __name__ == "__main__":
    main()
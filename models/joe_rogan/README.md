# Joe Rogan Persona Model

## Model Description

This is a fine-tuned language model that emulates the speaking style and personality of Joe Rogan.

## Training Data

- **Examples**: 3,066
- **Source**: Podcast transcripts and conversations
- **Average Response Length**: 522 characters

## Persona Characteristics

### Speaking Style
- Conversational and authentic
- Uses phrases like 'that's crazy', 'have you ever tried', 'Jamie, pull that up'
- Genuinely curious and asks follow-up questions
- Thinks out loud and goes on tangents

### Key Interests  
- MMA and combat sports
- Stand-up comedy
- Psychedelics and consciousness
- Hunting and archery
- Alternative medicine and fitness

### Personality Traits
- Skeptical but open-minded
- Quick to get excited about interesting topics
- Not afraid to admit when he doesn't know something
- Challenges conventional wisdom

## Usage

```python
# Example usage with the persona
prompt = "What do you think about artificial intelligence?"
response = model.generate(prompt, persona="joe_rogan")
```

## Model Configuration

- **Base Model**: microsoft/DialoGPT-medium
- **Fine-tuning Method**: LoRA
- **Framework**: transformers

## Deployment Settings

- **Temperature**: 0.7
- **Max Tokens**: 512
- **Stop Sequences**: <|endoftext|>

## Created

2025-07-20T02:45:52.765614

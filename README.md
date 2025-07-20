# Podcast Persona Debates - Agentic AI Hackathon

> **AI agents trained on podcast personalities engage in multi-agent debates with real-time fact-checking**

## 🎯 Project Overview

This project creates AI agents based on popular podcast host personalities (Joe Rogan, Ezra Klein, Lex Fridman) that can engage in dynamic debates. Each agent is fine-tuned on podcast transcripts using Gemma models and orchestrated through FastAPI for multi-agent interactions.

## 🧠 Core Concept

- **Fine-tune lightweight Gemma models** on podcast transcripts to capture each host's unique style, opinions, and speaking patterns
- **Multi-agent orchestration** using FastAPI to facilitate natural debates between personalities  
- **Real-time fact-checking** using structured podcast verification data as a critic agent
- **Chain-of-Thought reasoning** for complex discussions and nuanced position-taking
- **Optional voice synthesis** to bring the debates to life

## 🏗️ Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ Joe Agent  │     │ Ezra Agent │     │ Lex Agent  │
│ (Gemma +   │     │ (Gemma +   │     │ (Gemma +   │
│ LoRA-Joe)  │     │ LoRA-Ezra) │     │ LoRA-Lex)  │
└────┬───────┘     └────┬───────┘     └────┬───────┘
     │                  │                  │
 Personality         Personality       Personality
 Prompting           Prompting         Prompting
     │                  │                  │
 └──────────────[ FastAPI Server ]──────────────┘
                         ↕
                Fact-checking / Critic Agent
                         ↕
              [ Voice Synthesis (Optional) ]
```

## 🛠️ Tech Stack

### Models & Training
- **Base Model**: Google Gemma 1.1 (2B/7B instruction-tuned)
- **Fine-tuning**: LoRA/QLoRA adapters for memory efficiency
- **Training Framework**: 🤗 Transformers + PEFT + bitsandbytes

### Infrastructure 
- **Deployment**: Google Cloud Run with NVIDIA L4 GPUs
- **API Framework**: FastAPI for multi-agent workflows
- **Monitoring**: Opik by Comet for agent evaluation and safety
- **Voice**: Coqui TTS/XTTS (optional)

### Data
- **3,066 training examples** from 21 real Joe Rogan podcast episodes
- Structured fact-checking data for verification
- Chain-of-thought reasoning datasets

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers datasets peft accelerate bitsandbytes trl opik
```

### 🎉 Live Gemma API Endpoint

Our pre-deployed Gemma 3-1B model is ready for use:

- **Service URL**: https://gemma-1b-wamyzspxga-ew.a.run.app
- **Model**: gemma3:1b
- **Region**: europe-west1
- **Status**: ✅ Working (tested with "Why is the sky blue?" query)

#### 🔧 API Usage
```bash
curl -X POST "https://gemma-1b-wamyzspxga-ew.a.run.app/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:1b",
    "prompt": "Your question here"
  }'
```

#### Example Usage
```bash
# Test the API with a simple question
curl -X POST "https://gemma-1b-wamyzspxga-ew.a.run.app/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:1b",
    "prompt": "Explain quantum computing in simple terms"
  }'
```

### Quick Start with Real Data
```bash
# 1. Data is already processed and ready
ls data/joe_transcripts.json  # 3066 training examples

# 2. Test training pipeline (dry run)
python3 test_training_dry_run.py

# 3. Train models (requires GPU/ML packages)
./scripts/run_training.sh

# 4. Test locally
uvicorn api.server:app --host 0.0.0.0 --port 8080
python3 test_api.py

# 5. Deploy to Cloud Run
./scripts/deploy_cloudrun.sh
```

### Training Options
```bash
# Quick test with small dataset
python3 scripts/train_lora.py --persona joe --data data/joe_transcripts.json --output models/joe --epochs 1

# Full training
./scripts/run_training.sh
```

## 📊 Dataset Structure

Real Joe Rogan training data format:
```json
{
  "prompt": "You are Joe Rogan, a curious and skeptical podcast host...\n\nQuestion: What's your take on this?",
  "response": "You know, I think people need to really question what they're being told here...",
  "persona": "joe_rogan",
  "episode": "Joe Rogan Experience #2349 - Danny Jones",
  "episode_id": "89IEFOiW-Z0"
}
```

## 🎯 Hackathon Goals

- [x] **Team Formation** (2-5 people, ML/multi-agent expertise)
- [x] **Real Data Collection** - 3,066 examples from 21 Joe Rogan episodes
- [x] **Training Pipeline** - LoRA fine-tuning scripts ready
- [x] **API Server** - FastAPI with inference and debate endpoints
- [x] **Testing Framework** - Dry run validation without GPU requirements
- [ ] **Model Training** - Fine-tune Gemma agents on Google Cloud GPU
- [ ] **Agent Orchestration** - Multi-agent debate implementation
- [ ] **Demo Preparation** - Live debate showcase with optional voice
- [ ] **Submission** - Deploy on Cloud Run, submit by 10:00 AM Day 2

## 🧪 Testing

The project includes comprehensive testing without requiring ML packages:

```bash
# Test training pipeline logic
python3 test_training_dry_run.py

# Test with full dataset
python3 test_training_dry_run.py --data data/joe_transcripts.json

# Process local transcripts
python3 scripts/process_local_transcripts.py
```

## 🏆 Demo Scenarios

1. **Cross-ideological Debates** - "What would Joe Rogan say to Ezra Klein about AI regulation?"
2. **Multi-perspective Analysis** - Three agents analyzing complex topics from different angles
3. **Fact-checked Discussions** - Real-time verification of claims during debates
4. **Voice-enabled Conversations** - Audio output for immersive experience

## 📁 Project Structure

```
├── data/
│   ├── joe_transcripts.json     # 3066 Joe Rogan training examples
│   ├── lex_transcripts.json     # Sample Lex Fridman data
│   └── joe_rogan_complete.json  # Full processed dataset
├── models/
│   ├── joe/                     # Joe Rogan LoRA adapter
│   └── lex/                     # Lex Fridman LoRA adapter
├── api/
│   └── server.py                # FastAPI inference server
├── scripts/
│   ├── train_lora.py            # LoRA training script
│   ├── process_local_transcripts.py # Real data processing
│   ├── run_training.sh          # Training automation
│   └── deploy_cloudrun.sh       # Cloud Run deployment
├── test_training_dry_run.py     # Training pipeline validation
├── test_api.py                  # API testing script
├── Dockerfile                   # Cloud Run deployment
└── requirements.txt             # Dependencies
```

## 📈 Dataset Statistics

- **Episodes**: 21 Joe Rogan podcast episodes
- **Training Examples**: 3,066 real conversation segments
- **Average Length**: 522 characters per response
- **Characteristic Phrases**: "you know", "bro", "that's crazy", "Jamie, pull that up"
- **Content Types**: Conversations, debates, interviews, monologues

## 👥 Team

Looking for teammates with:
- **ML/Fine-tuning experience** (LoRA, model training)
- **Multi-agent systems** (FastAPI, agent orchestration)  
- **Voice synthesis** (TTS, voice cloning)
- **Cloud deployment** (Docker, Cloud Run)

## 📚 Resources

- [Google Cloud Run GPU Handbook](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Opik Agent Observability](https://github.com/comet-ml/opik)
- [Gemma Model Hub](https://huggingface.co/google/gemma-1.1-2b-it)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built for the Agentic AI App Hackathon** | **Powered by Google Cloud Run + NVIDIA L4 GPUs**
# Podcast Persona Debates - Agentic AI Hackathon

> **AI agents trained on podcast personalities engage in multi-agent debates with real-time fact-checking**

## 🎯 Project Overview

This project creates AI agents based on popular podcast host personalities (Joe Rogan, Ezra Klein, Lex Fridman) that can engage in dynamic debates. Each agent is fine-tuned on podcast transcripts using Gemma models and orchestrated through the A2A SDK for multi-agent interactions.

## 🧠 Core Concept

- **Fine-tune lightweight Gemma models** on podcast transcripts to capture each host's unique style, opinions, and speaking patterns
- **Multi-agent orchestration** using A2A SDK to facilitate natural debates between personalities  
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
 └──────────────[ A2A SDK Orchestration ]──────────────┘
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
- **Orchestration**: A2A SDK for multi-agent workflows
- **Monitoring**: Opik by Comet for agent evaluation and safety
- **Voice**: Coqui TTS/XTTS (optional)

### Data
- Podcast transcript archives for personality training
- Structured fact-checking data for verification
- Chain-of-thought reasoning datasets

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch transformers datasets peft accelerate bitsandbytes trl opik
```

### Training Persona Agents
1. **Prepare datasets** - Process podcast transcripts into instruction format
2. **Fine-tune adapters** - Train LoRA adapters for each personality
3. **Deploy inference** - Set up Cloud Run GPU endpoints
4. **Orchestrate debates** - Use A2A SDK for multi-agent interactions

### Quick Start
```python
# Load persona agent
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it")
joe_agent = PeftModel.from_pretrained(base_model, "adapters/joe_rogan")

# Generate response
prompt = "What do you think about AI regulation?"
response = joe_agent.generate(prompt, max_new_tokens=200)
```

## 📊 Dataset Structure

```json
{
  "instruction": "What are your thoughts on X topic?",
  "input": "Context: [relevant background or debate setup]", 
  "output": "Response as [Personality]: [characteristic answer with reasoning]"
}
```

## 🎯 Hackathon Goals

- [x] **Team Formation** (2-5 people, ML/multi-agent expertise)
- [ ] **Data Collection** - Gather podcast transcripts and fact-check data
- [ ] **Model Training** - Fine-tune Gemma agents on Google Cloud GPU
- [ ] **Agent Orchestration** - Implement A2A debate framework
- [ ] **Fact Integration** - Add critic agent for real-time verification
- [ ] **Demo Preparation** - Live debate showcase with optional voice
- [ ] **Submission** - Deploy on Cloud Run, submit by 10:00 AM Day 2

## 🏆 Demo Scenarios

1. **Cross-ideological Debates** - "What would Joe Rogan say to Ezra Klein about AI regulation?"
2. **Multi-perspective Analysis** - Three agents analyzing complex topics from different angles
3. **Fact-checked Discussions** - Real-time verification of claims during debates
4. **Voice-enabled Conversations** - Audio output for immersive experience

## 📁 Project Structure

```
├── data/
│   ├── transcripts/          # Podcast transcript archives
│   ├── fact_checks/          # Structured verification data
│   └── processed/            # Training datasets
├── models/
│   ├── adapters/             # LoRA fine-tuned adapters
│   └── base/                 # Base Gemma models
├── src/
│   ├── training/             # Fine-tuning scripts
│   ├── agents/               # Agent implementations
│   └── orchestration/        # A2A debate logic
├── deployment/
│   ├── Dockerfile            # Cloud Run deployment
│   └── requirements.txt      # Dependencies
└── demos/                    # Example debates and notebooks
```

## 👥 Team

Looking for teammates with:
- **ML/Fine-tuning experience** (LoRA, model training)
- **Multi-agent systems** (A2A SDK, LangGraph)  
- **Voice synthesis** (TTS, voice cloning)
- **Cloud deployment** (Docker, Cloud Run)

## 📚 Resources

- [Google Cloud Run GPU Handbook](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Opik Agent Observability](https://github.com/comet-ml/opik)
- [A2A SDK Documentation](https://docs.a2a.ai)
- [Gemma Model Hub](https://huggingface.co/google/gemma-1.1-2b-it)

---

**Built for the Agentic AI App Hackathon** | **Powered by Google Cloud Run + NVIDIA L4 GPUs**
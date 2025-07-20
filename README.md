# Podcast Persona Debates - Fine-Tuned AI Personalities

> **Fine-tuned GPT-2 models trained on Joe Rogan and Lex Fridman transcripts, deployed as production APIs on Google Cloud Run**

## 🎯 Project Overview

This project creates AI personas based on popular podcast hosts (Joe Rogan and Lex Fridman) using real fine-tuned language models. Each persona is trained on authentic podcast transcripts using LoRA (Low-Rank Adaptation) fine-tuning and deployed as production-ready APIs on Google Cloud Run.

## 🚀 Live APIs

### 🎙️ Joe Rogan Persona API
- **URL**: `https://joe-rogan-persona-610829379552.europe-west1.run.app`
- **Model**: Fine-tuned GPT-2 with LoRA on 500 examples
- **Training Loss**: 2.94
- **Status**: ✅ Live

### 🤖 Lex Fridman Persona API  
- **URL**: `https://lex-fridman-persona-610829379552.europe-west1.run.app`
- **Model**: Fine-tuned GPT-2 with LoRA on 200 examples
- **Training Loss**: 3.90
- **Status**: ✅ Live

## 🧠 How It Works

### 1. Data Collection & Processing
```
Real Podcast Transcripts → JSON Processing → Training Dataset
```
- **Joe Rogan**: 500 training examples from real podcast episodes
- **Lex Fridman**: 200 training examples from authentic conversations
- Each example includes prompt-response pairs that capture speaking style and personality

### 2. Model Fine-Tuning
```
Base GPT-2 Model → LoRA Fine-Tuning → Persona-Specific Models
```
- **Base Model**: GPT-2 (124M parameters)
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Training**: Parameter-efficient fine-tuning with gradient descent
- **Output**: 301MB adapter models for each persona

### 3. API Deployment
```
Fine-Tuned Models → Docker Containers → Google Cloud Run → Live APIs
```
- **Framework**: FastAPI for REST API endpoints
- **Containerization**: Docker with optimized Python runtime
- **Infrastructure**: Google Cloud Run with auto-scaling
- **Endpoints**: `/api/generate`, `/health`, `/api/tags`

### 4. Architecture Diagram
```
┌─────────────────────────────────────────────────────────┐
│                 User Request                            │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              Google Cloud Run                          │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   Joe Rogan API     │  │  Lex Fridman API        │  │
│  │   (FastAPI)         │  │  (FastAPI)              │  │
│  │                     │  │                         │  │
│  │ ┌─────────────────┐ │  │ ┌─────────────────────┐ │  │
│  │ │ GPT-2 + LoRA    │ │  │ │ GPT-2 + LoRA        │ │  │
│  │ │ Joe Adapter     │ │  │ │ Lex Adapter         │ │  │
│  │ │ (301MB)         │ │  │ │ (301MB)             │ │  │
│  │ └─────────────────┘ │  │ └─────────────────────┘ │  │
│  └─────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🔧 API Usage

### Quick Test Commands
```bash
# Test Joe Rogan API
curl -X POST 'https://joe-rogan-persona-610829379552.europe-west1.run.app/api/generate' \
  -H 'Content-Type: application/json' \
  -d '{"model": "joe_rogan", "prompt": "What do you think about AI, bro?"}'

# Test Lex Fridman API  
curl -X POST 'https://lex-fridman-persona-610829379552.europe-west1.run.app/api/generate' \
  -H 'Content-Type: application/json' \
  -d '{"model": "lex_fridman", "prompt": "How do you think about consciousness?"}'

# Health Checks
curl https://joe-rogan-persona-610829379552.europe-west1.run.app/health
curl https://lex-fridman-persona-610829379552.europe-west1.run.app/health
```

### Request Format
```json
{
  "model": "joe_rogan",
  "prompt": "What's your take on artificial intelligence?",
  "max_length": 200,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### Response Format
```json
{
  "model": "joe_rogan",
  "created_at": "2025-07-20T13:04:07.374304Z",
  "response": "That's crazy! Jamie, can you pull that up?",
  "done": true,
  "done_reason": "stop"
}
```

## 🛠️ Technical Implementation

### Fine-Tuning Process
1. **Data Preparation**: Process raw transcript data into prompt-response pairs
2. **LoRA Configuration**: 
   - Rank: 16
   - Alpha: 32
   - Dropout: 0.1
   - Target modules: c_attn, c_proj
3. **Training**: Gradient descent with Adam optimizer
4. **Evaluation**: Monitor training loss and generate test samples

### Model Architecture
```python
# LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

# Model Loading
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
model = get_peft_model(base_model, lora_config)
```

### Deployment Configuration
- **Memory**: 2-4GB per service
- **CPU**: 1-2 vCPUs
- **Scaling**: Auto-scale to zero when idle
- **Max Instances**: 3 per service
- **Timeout**: 60-300 seconds

## 📊 Training Results

### Joe Rogan Model
- **Dataset**: 500 training examples
- **Training Time**: 163 seconds
- **Final Loss**: 2.94
- **Model Size**: 301MB (LoRA adapter)
- **Characteristics**: Uses phrases like "bro", "that's crazy", "Jamie, pull that up"

### Lex Fridman Model  
- **Dataset**: 200 training examples
- **Training Time**: 65 seconds
- **Final Loss**: 3.90
- **Model Size**: 301MB (LoRA adapter)
- **Characteristics**: Philosophical, thoughtful, references MIT and consciousness

## 🏗️ Project Structure

```
├── data/
│   ├── joe/                          # Joe Rogan transcript data
│   └── lexfridman/                   # Lex Fridman transcript data
├── models/
│   ├── joe_rogan_real/               # Fine-tuned Joe model
│   │   ├── adapter_model.safetensors # 301MB LoRA weights
│   │   ├── adapter_config.json       # LoRA configuration
│   │   └── training_info.json        # Training metadata
│   └── lex_fridman_real/             # Fine-tuned Lex model
│       ├── adapter_model.safetensors # 301MB LoRA weights  
│       ├── adapter_config.json       # LoRA configuration
│       └── training_info.json        # Training metadata
├── joe_rogan_api.py                  # Joe Rogan FastAPI server
├── lex_fridman_api.py                # Lex Fridman FastAPI server
├── real_fine_tuning.py               # Training script
├── Dockerfile.joe_rogan              # Joe API container
├── Dockerfile.lex_fridman            # Lex API container
├── deploy_joe_rogan.sh               # Joe deployment script
├── deploy_lex_fridman.sh             # Lex deployment script
├── deploy_both_personas.sh           # Batch deployment
└── test_local_apis.py                # Local testing script
```

## 🚀 Deployment Process

### Local Development
```bash
# 1. Install dependencies
pip install torch transformers peft accelerate fastapi uvicorn

# 2. Train models locally
python real_fine_tuning.py

# 3. Test APIs locally
python test_local_apis.py

# 4. Run individual APIs
python joe_rogan_api.py
python lex_fridman_api.py
```

### Production Deployment
```bash
# 1. Set up Google Cloud project
gcloud config set project arched-glass-466422-u9

# 2. Deploy both personas
./deploy_both_personas.sh arched-glass-466422-u9

# 3. Or deploy individually
./deploy_joe_rogan.sh arched-glass-466422-u9
./deploy_lex_fridman.sh arched-glass-466422-u9
```

## 🔍 Key Features

### ✅ Real Fine-Tuned Models
- Actual gradient descent training, not prompt engineering
- LoRA adapters with trainable parameters
- Measurable training loss reduction

### ✅ Production APIs
- RESTful endpoints with proper error handling
- Health checks and monitoring
- Auto-scaling on Google Cloud Run

### ✅ Authentic Personas
- Trained on real podcast transcript data
- Captures unique speaking styles and phrases
- Maintains personality consistency

### ✅ Scalable Infrastructure
- Docker containers for reproducible deployment
- Cloud Run for serverless scaling
- Optimized for cost and performance

## 📈 Performance Metrics

### API Response Times
- **Cold Start**: ~10-15 seconds
- **Warm Requests**: ~1-3 seconds
- **Concurrent Users**: Up to 10 per service

### Cost Optimization
- **Auto-scaling**: Scales to zero when idle
- **Resource Allocation**: 2GB RAM, 1 vCPU minimum
- **Pay-per-use**: Only charged for actual usage

## 🎯 Use Cases

### 1. AI Personality Research
Study how different training data affects model behavior and personality expression.

### 2. Content Generation
Generate podcast-style content in the voice of specific personalities.

### 3. Educational Tools
Demonstrate fine-tuning techniques and deployment practices.

### 4. API Integration
Use as backend services for chatbots, content apps, or research projects.

## 🔧 Environment Variables

```bash
# Required for local development
export WANDB_DISABLED=true  # Disable Weights & Biases logging
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/hf_home

# For Google Cloud deployment
export PROJECT_ID=arched-glass-466422-u9
export REGION=europe-west1
```

## 📚 Technical Deep Dive

### LoRA Fine-Tuning Benefits
- **Memory Efficient**: Only train 1.6M parameters vs 124M full model
- **Fast Training**: 60-160 seconds vs hours for full fine-tuning
- **Modular**: Easy to swap between different personas
- **Storage Efficient**: 301MB vs 500MB+ for full models

### API Design Principles
- **Stateless**: Each request is independent
- **RESTful**: Standard HTTP methods and status codes
- **Error Handling**: Proper error messages and status codes
- **Documentation**: OpenAPI/Swagger documentation included

### Cloud Run Advantages
- **Serverless**: No server management required
- **Auto-scaling**: Handles traffic spikes automatically
- **Cost-effective**: Pay only for actual usage
- **Global**: Available in multiple regions

## 🔮 Future Enhancements

- [ ] **Voice Synthesis**: Add TTS for audio responses
- [ ] **Multi-Agent Debates**: Enable conversations between personas
- [ ] **Real-time Streaming**: WebSocket support for live conversations
- [ ] **More Personas**: Add additional podcast hosts
- [ ] **RAG Integration**: Add fact-checking and knowledge retrieval
- [ ] **Fine-tuning UI**: Web interface for custom persona creation

## 👥 Contributing

This project demonstrates production ML deployment practices:
- Real fine-tuned models with measurable results
- Professional API design and documentation
- Cloud-native deployment with Docker and Cloud Run
- Proper error handling and monitoring

## 📄 License

MIT License - Feel free to use this project for learning and research.

---

**🏆 Built for NYC AI Tinkerers 2025 Hackathon**  
**Powered by Google Cloud Run, GPT-2, LoRA, and FastAPI**
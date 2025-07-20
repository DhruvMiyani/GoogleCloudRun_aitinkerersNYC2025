#!/usr/bin/env python3
"""
Joe Rogan Persona Training Script for Cloud Run GPU
Dedicated training instance for Joe Rogan LoRA adapter
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train Joe Rogan persona model"""
    logger.info("🎙️ Starting Joe Rogan Persona Training on GPU")
    logger.info("=" * 60)
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️ No GPU detected - training will be slow")
    except ImportError:
        logger.error("❌ PyTorch not available")
        sys.exit(1)
    
    # Import training modules
    sys.path.append('/app/scripts')
    from train_persona_lora import PersonaTrainer
    
    # Training configuration
    config = {
        "persona": "joe_rogan",
        "data_path": "/app/data/joe_transcripts.json",
        "output_dir": "/app/models/joe_rogan",
        "base_model": "google/gemma-1.1-2b-it",
        "epochs": 3,
        "batch_size": 8,  # Optimized for L4 GPU
        "learning_rate": 2e-4,
        "max_length": 512,
        "lora_rank": 16,
        "lora_alpha": 32,
        "use_4bit": True
    }
    
    # Check training data
    if not os.path.exists(config["data_path"]):
        logger.error(f"❌ Training data not found: {config['data_path']}")
        sys.exit(1)
    
    with open(config["data_path"], 'r') as f:
        data = json.load(f)
        logger.info(f"📊 Training examples: {len(data)}")
        avg_length = sum(len(ex['response']) for ex in data) / len(data)
        logger.info(f"📏 Average response length: {avg_length:.0f} characters")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Start training
    start_time = time.time()
    logger.info("🏋️ Starting LoRA fine-tuning...")
    
    try:
        # Create trainer
        trainer = PersonaTrainer(
            base_model=config["base_model"],
            use_4bit=config["use_4bit"]
        )
        
        # Train the model
        model_path = trainer.train(
            data_path=config["data_path"],
            output_dir=config["output_dir"],
            persona_name=config["persona"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            max_length=config["max_length"],
            save_steps=200,
            logging_steps=20
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info("🎉 Training Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        logger.info(f"📁 Model saved to: {model_path}")
        
        # Save training results
        results = {
            "persona": config["persona"],
            "model_path": model_path,
            "training_time_seconds": training_time,
            "training_examples": len(data),
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        results_file = "/app/joe_rogan_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📝 Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        sys.exit(1)
    
    # Keep container running for model access
    logger.info("🔄 Starting web server for model access...")
    
    try:
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(title="Joe Rogan Training Results")
        
        @app.get("/")
        def get_status():
            return {
                "persona": "joe_rogan",
                "status": "training_completed",
                "model_path": model_path,
                "training_time": f"{training_time:.2f} seconds"
            }
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "persona": "joe_rogan"}
        
        logger.info("🌐 Web server starting on port 8080...")
        uvicorn.run(app, host="0.0.0.0", port=8080)
        
    except ImportError:
        logger.info("📡 Keeping container alive...")
        while True:
            time.sleep(60)

if __name__ == "__main__":
    main()
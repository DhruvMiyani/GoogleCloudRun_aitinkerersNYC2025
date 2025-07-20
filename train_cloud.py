#!/usr/bin/env python3
"""
Cloud Run GPU Training Script for Persona Models
Trains both Joe Rogan and Lex Fridman models on GPU
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup training environment"""
    logger.info("🚀 Starting Cloud GPU Training")
    logger.info(f"CUDA Available: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Import training modules
    sys.path.append('/app/scripts')
    from train_persona_lora import PersonaTrainer
    
    return PersonaTrainer

def train_persona(trainer_class, persona_name, data_path, output_dir):
    """Train a single persona model"""
    logger.info(f"🎭 Training {persona_name} persona...")
    
    start_time = time.time()
    
    try:
        # Create trainer with GPU acceleration
        trainer = trainer_class(
            base_model="google/gemma-1.1-2b-it",
            use_4bit=True  # Use 4-bit quantization for memory efficiency
        )
        
        # Train the model
        model_path = trainer.train(
            data_path=data_path,
            output_dir=output_dir,
            persona_name=persona_name,
            epochs=3,
            batch_size=8,  # Increase batch size for GPU
            learning_rate=2e-4,
            max_length=512,
            save_steps=100,
            logging_steps=10
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"✅ {persona_name} training completed in {training_time:.2f} seconds")
        logger.info(f"📁 Model saved to: {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Training failed for {persona_name}: {str(e)}")
        raise

def main():
    """Main training function"""
    logger.info("🎯 Cloud Run GPU Training Started")
    
    # Setup environment
    PersonaTrainer = setup_environment()
    
    # Training configurations
    training_jobs = [
        {
            "persona": "joe_rogan",
            "data_path": "/app/data/joe_transcripts.json",
            "output_dir": "/app/models/joe_rogan"
        },
        {
            "persona": "lex_fridman", 
            "data_path": "/app/data/lex_transcripts_final.json",
            "output_dir": "/app/models/lex_fridman"
        }
    ]
    
    # Check data availability
    for job in training_jobs:
        if not os.path.exists(job["data_path"]):
            logger.error(f"❌ Training data not found: {job['data_path']}")
            sys.exit(1)
        
        with open(job["data_path"], 'r') as f:
            data = json.load(f)
            logger.info(f"📊 {job['persona']}: {len(data)} training examples")
    
    # Train both models
    total_start = time.time()
    trained_models = []
    
    for job in training_jobs:
        try:
            model_path = train_persona(
                trainer_class=PersonaTrainer,
                persona_name=job["persona"],
                data_path=job["data_path"],
                output_dir=job["output_dir"]
            )
            trained_models.append({
                "persona": job["persona"],
                "model_path": model_path,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"❌ Failed to train {job['persona']}: {str(e)}")
            trained_models.append({
                "persona": job["persona"],
                "model_path": None,
                "status": "failed",
                "error": str(e)
            })
    
    total_time = time.time() - total_start
    
    # Summary
    logger.info("🎉 Training Summary")
    logger.info("=" * 50)
    logger.info(f"⏱️  Total training time: {total_time:.2f} seconds")
    
    for result in trained_models:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        logger.info(f"{status_emoji} {result['persona']}: {result['status']}")
        if result["status"] == "success":
            logger.info(f"   📁 {result['model_path']}")
    
    # Save training results
    results_file = "/app/training_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "models": trained_models
        }, f, indent=2)
    
    logger.info(f"📝 Results saved to: {results_file}")
    
    # Keep container running for model download
    logger.info("🔄 Training complete! Container will stay alive for model access...")
    logger.info("💡 You can now download the models or test them via API")
    
    # Simple web server for model access
    try:
        import uvicorn
        from fastapi import FastAPI
        
        app = FastAPI(title="Training Results")
        
        @app.get("/")
        def get_results():
            return {"message": "Training completed", "models": trained_models}
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "training": "completed"}
        
        logger.info("🌐 Starting web server on port 8080...")
        uvicorn.run(app, host="0.0.0.0", port=8080)
        
    except ImportError:
        # Fallback: just keep container alive
        logger.info("📡 Web server not available, keeping container alive...")
        while True:
            time.sleep(60)
            logger.info("⏰ Container still running... (training completed)")

if __name__ == "__main__":
    main()
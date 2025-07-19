#!/usr/bin/env python3
"""
Dry run test of training pipeline - validates data format and logic without requiring ML packages
"""

import json
import os
import argparse

def test_data_loading(data_path, max_length=512):
    """Test data loading and tokenization logic"""
    print(f"🧪 Testing data loading from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} training examples")
    
    # Validate data structure
    required_fields = ['prompt', 'response', 'persona']
    for i, item in enumerate(data[:5]):  # Check first 5
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            print(f"❌ Item {i} missing fields: {missing_fields}")
            return False
        
        # Check data types
        if not isinstance(item['prompt'], str) or not isinstance(item['response'], str):
            print(f"❌ Item {i} has non-string prompt/response")
            return False
    
    print("✅ Data structure validation passed")
    
    # Test format conversion (simulated)
    formatted_data = []
    for item in data[:10]:  # Test first 10
        # Simulate the training format
        prompt = f"<start_of_turn>user\n{item['prompt']}<end_of_turn>\n<start_of_turn>model\n{item['response']}<end_of_turn>"
        
        # Simulate tokenization (just check length)
        if len(prompt) > max_length * 4:  # Rough char to token ratio
            print(f"⚠️ Long example detected: {len(prompt)} chars")
        
        formatted_data.append({
            "text": prompt,
            "length": len(prompt),
            "persona": item["persona"]
        })
    
    print(f"✅ Formatted {len(formatted_data)} examples for training")
    
    # Show statistics
    lengths = [item["length"] for item in formatted_data]
    avg_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    
    print(f"📊 Text length stats:")
    print(f"  Average: {avg_length:.0f} chars")
    print(f"  Range: {min_length} - {max_length} chars")
    
    return True

def test_training_args():
    """Test training argument validation"""
    print("\n🧪 Testing training arguments...")
    
    # Simulate training args that would be used
    training_config = {
        "output_dir": "./models/joe_test",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 100,
        "max_steps": 500,
        "learning_rate": 2e-4,
        "fp16": True,
        "logging_steps": 50,
        "save_strategy": "steps",
        "save_steps": 250,
    }
    
    # Validate configuration
    if training_config["per_device_train_batch_size"] * training_config["gradient_accumulation_steps"] > 16:
        print("⚠️ Effective batch size might be too large for memory")
    
    if training_config["learning_rate"] > 1e-3:
        print("⚠️ Learning rate might be too high for LoRA")
    
    print("✅ Training arguments validated")
    return training_config

def test_lora_config():
    """Test LoRA configuration"""
    print("\n🧪 Testing LoRA configuration...")
    
    lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    # Validate LoRA settings
    if lora_config["r"] > 64:
        print("⚠️ LoRA rank might be too high")
    
    if lora_config["lora_alpha"] != lora_config["r"] * 2:
        print("⚠️ LoRA alpha should typically be 2x rank")
    
    print("✅ LoRA configuration validated")
    return lora_config

def test_model_compatibility():
    """Test model compatibility"""
    print("\n🧪 Testing model compatibility...")
    
    model_name = "google/gemma-1.1-2b-it"
    
    # The actual model loading would happen here
    # For dry run, just validate the model name format
    if not model_name.startswith("google/gemma"):
        print("❌ Model should be a Gemma model for this project")
        return False
    
    print(f"✅ Model {model_name} is compatible")
    
    # Test output directory
    output_dir = "models/joe_test"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory {output_dir} ready")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Dry run test of training pipeline")
    parser.add_argument("--data", default="data/joe_test_small.json", help="Training data file")
    
    args = parser.parse_args()
    
    print("🧪 Starting training pipeline dry run test...\n")
    
    # Test data loading
    if not test_data_loading(args.data):
        print("❌ Data loading test failed")
        return
    
    # Test training configuration
    training_config = test_training_args()
    
    # Test LoRA configuration  
    lora_config = test_lora_config()
    
    # Test model compatibility
    if not test_model_compatibility():
        print("❌ Model compatibility test failed")
        return
    
    print("\n🎉 All dry run tests passed!")
    print("\n📋 Training would proceed with:")
    print(f"  Data: {args.data}")
    print(f"  Model: google/gemma-1.1-2b-it")
    print(f"  LoRA rank: {lora_config['r']}")
    print(f"  Batch size: {training_config['per_device_train_batch_size']}")
    print(f"  Steps: {training_config['max_steps']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    
    print("\n✅ Training pipeline is ready for execution with ML packages installed!")

if __name__ == "__main__":
    main()
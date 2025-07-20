#!/usr/bin/env python3
"""
ACTUAL Fine-tuning Script for Joe Rogan and Lex Fridman Models
This creates real trained models, not placeholders
"""

import os
import json
import torch
import logging
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np

# Disable wandb to avoid authentication issues
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealPersonaTrainer:
    def __init__(self, base_model="gpt2"):
        self.base_model = base_model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"🚀 Using device: {self.device}")
        
    def load_and_prepare_data(self, data_path: str, persona_name: str):
        """Load and prepare training data for fine-tuning"""
        logger.info(f"📂 Loading data from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare training texts with persona context
        training_texts = []
        persona_prompt = self.get_persona_system_prompt(persona_name)
        
        # Use subset for faster training (increase for better results)
        subset_size = min(500, len(data))  # Use 500 examples for real training
        data_subset = data[:subset_size]
        
        for item in data_subset:
            # Format: System prompt + Question + Response
            text = f"{persona_prompt}\n\nHuman: {item['prompt']}\n\n{persona_name.replace('_', ' ').title()}: {item['response']}<|endoftext|>"
            training_texts.append(text)
        
        logger.info(f"📊 Prepared {len(training_texts)} training examples")
        logger.info(f"📏 Average text length: {np.mean([len(text) for text in training_texts]):.0f} characters")
        
        return training_texts
    
    def get_persona_system_prompt(self, persona_name: str) -> str:
        """Get system prompt for each persona"""
        prompts = {
            "joe_rogan": """You are Joe Rogan, the podcast host. Key characteristics:
- Be genuinely curious and ask follow-up questions
- Use phrases like "that's crazy", "have you ever tried", "Jamie, pull that up"
- Reference MMA, comedy, psychedelics, hunting when relevant
- Think out loud and be conversational""",
            
            "lex_fridman": """You are Lex Fridman, the AI researcher and podcast host. Key characteristics:
- Be philosophical and contemplative about consciousness and reality
- Ask deep questions about intelligence and existence
- Reference AI, robotics, mathematics, physics
- Maintain a calm, measured, thoughtful tone"""
        }
        return prompts.get(persona_name, "")
    
    def setup_model_and_tokenizer(self):
        """Setup base model and tokenizer for training"""
        logger.info(f"🔧 Loading base model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.base_model)
        
        # Add special tokens
        special_tokens = {"pad_token": "<|pad|>", "eos_token": "<|endoftext|>"}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        )
        
        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],  # GPT-2 specific modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✅ Model and tokenizer setup complete")
    
    def tokenize_dataset(self, texts):
        """Tokenize the training data"""
        logger.info("🔤 Tokenizing training data...")
        
        def tokenize_function(examples):
            # Tokenize with truncation and padding
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,  # Reasonable length for training
                return_tensors="pt"
            )
            # For causal LM, labels are the same as input_ids
            tokens["labels"] = tokens["input_ids"].clone()
            return tokens
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"✅ Tokenized {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train_model(self, dataset, output_dir: str, persona_name: str):
        """Actually train the model"""
        logger.info(f"🏋️ Starting REAL training for {persona_name}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,  # Real training epochs
            per_device_train_batch_size=2,  # Small batch for memory efficiency
            gradient_accumulation_steps=4,  # Simulate larger batch
            save_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=25,
            learning_rate=5e-5,  # Good learning rate for fine-tuning
            warmup_steps=50,
            logging_dir=f"{output_dir}/logs",
            report_to=None,  # Disable wandb
            dataloader_pin_memory=False,  # For MPS compatibility
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train the model
        logger.info("🔥 Starting actual training process...")
        start_time = datetime.now()
        
        train_result = trainer.train()
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        logger.info(f"🎉 Training completed in {training_time:.2f} seconds!")
        logger.info(f"📉 Final loss: {train_result.training_loss:.4f}")
        
        # Save the trained model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "persona": persona_name,
            "base_model": self.base_model,
            "training_examples": len(dataset),
            "training_loss": train_result.training_loss,
            "training_time_seconds": training_time,
            "training_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "created": datetime.now().isoformat(),
            "device": self.device,
            "model_type": "LoRA fine-tuned",
            "trainable_parameters": self.get_trainable_params()
        }
        
        with open(f"{output_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"💾 Model saved to: {output_dir}")
        logger.info(f"📊 Training info saved to: {output_dir}/training_info.json")
        
        return output_dir, training_info
    
    def get_trainable_params(self):
        """Get count of trainable parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params,
            "percentage": (trainable_params / total_params) * 100
        }
    
    def test_model(self, output_dir: str, persona_name: str):
        """Test the trained model"""
        logger.info(f"🧪 Testing trained {persona_name} model...")
        
        # Load the trained model
        model = GPT2LMHeadModel.from_pretrained(output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
        
        # Test prompts
        test_prompts = {
            "joe_rogan": [
                "What do you think about psychedelics?",
                "Have you ever tried martial arts?",
                "What's your take on comedy?"
            ],
            "lex_fridman": [
                "What is consciousness?",
                "How do you think about artificial intelligence?",
                "What fascinates you about mathematics?"
            ]
        }
        
        prompts = test_prompts.get(persona_name, ["Tell me about yourself"])
        
        results = []
        for prompt in prompts:
            # Encode input
            input_text = f"Human: {prompt}\n\n{persona_name.replace('_', ' ').title()}:"
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(input_text):].strip()
            
            results.append({
                "prompt": prompt,
                "response": response[:200] + "..." if len(response) > 200 else response
            })
            
            logger.info(f"✅ Test prompt: {prompt}")
            logger.info(f"   Response: {response[:100]}...")
        
        return results

def main():
    """Main training function"""
    logger.info("🚀 REAL Fine-tuning Process Starting")
    logger.info("=" * 60)
    
    trainer = RealPersonaTrainer()
    
    personas = [
        {
            "name": "joe_rogan",
            "data_path": "data/joe/joe_transcripts.json",
            "output_dir": "models/joe_rogan_real"
        },
        {
            "name": "lex_fridman",
            "data_path": "data/lexfridman/lex_transcripts_final.json",
            "output_dir": "models/lex_fridman_real"
        }
    ]
    
    all_results = {}
    
    for persona in personas:
        logger.info(f"\n🎭 Training REAL {persona['name']} model...")
        logger.info("-" * 40)
        
        # Setup model and tokenizer fresh for each persona
        trainer.setup_model_and_tokenizer()
        
        # Load and prepare data
        texts = trainer.load_and_prepare_data(persona['data_path'], persona['name'])
        
        # Tokenize data
        dataset = trainer.tokenize_dataset(texts)
        
        # Train the model
        model_path, training_info = trainer.train_model(
            dataset, 
            persona['output_dir'], 
            persona['name']
        )
        
        # Test the model
        test_results = trainer.test_model(persona['output_dir'], persona['name'])
        
        all_results[persona['name']] = {
            "model_path": model_path,
            "training_info": training_info,
            "test_results": test_results
        }
        
        logger.info(f"✅ {persona['name']} REAL model completed!")
    
    logger.info("\n🎉 ALL REAL MODELS TRAINED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    for persona_name, results in all_results.items():
        info = results['training_info']
        logger.info(f"\n📊 {persona_name.upper()} RESULTS:")
        logger.info(f"   Model: {results['model_path']}")
        logger.info(f"   Training Loss: {info['training_loss']:.4f}")
        logger.info(f"   Training Time: {info['training_time_seconds']:.1f}s")
        logger.info(f"   Examples: {info['training_examples']}")
        logger.info(f"   Trainable Params: {info['trainable_parameters']['trainable']:,}")
    
    logger.info(f"\n💾 REAL model files saved in:")
    logger.info(f"   - models/joe_rogan_real/")
    logger.info(f"   - models/lex_fridman_real/")

if __name__ == "__main__":
    main()
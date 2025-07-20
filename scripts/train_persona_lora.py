#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Persona Training
Supports training custom models for different personas (Joe Rogan, Lex Fridman, etc.)
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import wandb
from datetime import datetime

class PersonaTrainer:
    def __init__(self, base_model="google/gemma-1.1-2b-it", use_4bit=True):
        self.base_model = base_model
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with optional 4-bit quantization"""
        print(f"Loading model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
    
    def setup_lora_config(self, rank=16, alpha=32, dropout=0.05):
        """Configure LoRA parameters"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            bias="none"
        )
        return lora_config
    
    def load_and_prepare_data(self, data_path, max_length=512):
        """Load and prepare training data"""
        print(f"Loading data from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} training examples")
        
        # Format data for training
        formatted_data = []
        for item in data:
            # Combine prompt and response
            text = f"{item['prompt']}\n\nResponse: {item['response']}"
            formatted_data.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, 
              data_path,
              output_dir,
              persona_name,
              epochs=3,
              batch_size=4,
              learning_rate=2e-4,
              max_length=512,
              save_steps=500,
              logging_steps=10):
        """Train the LoRA adapter"""
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Load and prepare data
        train_dataset = self.load_and_prepare_data(data_path, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            push_to_hub=False,
            report_to="none"  # Disable wandb for now
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            max_seq_length=max_length,
            packing=False
        )
        
        # Train
        print(f"Starting training for {persona_name} persona...")
        trainer.train()
        
        # Save model
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training completed for {persona_name}!")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for personas")
    parser.add_argument("--persona", required=True, choices=["joe_rogan", "lex_fridman"], 
                        help="Persona to train")
    parser.add_argument("--data_path", required=True, help="Path to training data JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--base_model", default="google/gemma-1.1-2b-it", help="Base model")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PersonaTrainer(base_model=args.base_model, use_4bit=args.use_4bit)
    
    # Train model
    model_path = trainer.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        persona_name=args.persona,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    print(f"✅ Training completed! Model saved to: {model_path}")

if __name__ == "__main__":
    main()
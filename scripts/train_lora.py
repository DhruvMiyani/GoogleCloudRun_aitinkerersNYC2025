#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Podcast Personas
Trains Gemma models with LoRA adapters for Joe Rogan and Lex Fridman personalities
"""

import argparse
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import os

def load_and_format_data(data_path, tokenizer, max_length=512):
    """Load JSON data and format for training"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Format as instruction-response pairs
        prompt = f"<start_of_turn>user\n{item['prompt']}<end_of_turn>\n<start_of_turn>model\n{item['response']}<end_of_turn>"
        
        # Tokenize
        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        formatted_data.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()  # For causal LM
        })
    
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer(model_id="google/gemma-1.1-2b-it"):
    """Load and prepare model for LoRA training"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def train_persona(persona_name, data_path, output_dir, epochs=3, batch_size=4):
    """Train a persona-specific LoRA adapter"""
    print(f"Training {persona_name} persona...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Load and format data
    dataset = load_and_format_data(data_path, tokenizer)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=500,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=250,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save the adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ {persona_name} adapter saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train persona LoRA adapters")
    parser.add_argument("--persona", choices=["joe", "lex"], required=True, help="Persona to train")
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument("--output", required=True, help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train the specified persona
    train_persona(
        persona_name=args.persona,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
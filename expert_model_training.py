"""
Expert Model Fine-tuning System
Uses QLoRA to fine-tune specialist language models on expert-specific datasets
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertTrainingConfig:
    """Configuration for expert model training"""
    
    # Base model configuration
    model_name: str = "microsoft/DialoGPT-small"  # 117M parameters
    model_cache_dir: str = "models"
    
    # QLoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc"])
    
    # Quantization configuration
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Training configuration
    output_dir: str = "expert_models"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_length: int = 512
    
    # Resource configuration
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Expert types
    expert_types: List[str] = field(default_factory=lambda: [
        "tiling", "extraction", "mapping", "symmetry", "rule"
    ])

class ExpertModelTrainer:
    """Trains specialist expert models using QLoRA"""
    
    def __init__(self, config: ExpertTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        Path(config.output_dir).mkdir(exist_ok=True)
        Path(config.model_cache_dir).mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = self._initialize_tokenizer()
        
        # Track trained models
        self.trained_models = {}
        
    def _initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize and configure tokenizer"""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration for 4-bit loading"""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
    
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
    
    def _load_base_model(self):
        """Load and prepare base model for QLoRA training"""
        logger.info(f"Loading base model: {self.config.model_name}")
        
        # Create quantization config
        quant_config = self._create_quantization_config()
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_config,
            cache_dir=self.config.model_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters
        lora_config = self._create_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def _load_expert_dataset(self, expert_type: str) -> Dataset:
        """Load expert-specific dataset"""
        dataset_path = f"expert_datasets/{expert_type}_expert_dataset.json"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Expert dataset not found: {dataset_path}")
        
        logger.info(f"Loading {expert_type} expert dataset from {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Convert to HuggingFace dataset format
        dataset = Dataset.from_list(data)
        
        logger.info(f"Loaded {len(dataset)} examples for {expert_type} expert")
        return dataset
    
    def _preprocess_dataset(self, dataset: Dataset, expert_type: str) -> Dataset:
        """Preprocess dataset for training"""
        
        def tokenize_function(examples):
            # Combine instruction and output for causal LM training
            texts = []
            for instruction, output in zip(examples['instruction'], examples['output']):
                # Format as instruction-following conversation
                text = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}{self.tokenizer.eos_token}"
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"Preprocessed {expert_type} dataset: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def _create_training_arguments(self, expert_type: str) -> TrainingArguments:
        """Create training arguments for specific expert"""
        output_dir = os.path.join(self.config.output_dir, f"{expert_type}_expert")
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=["wandb"] if wandb.run is not None else [],
            run_name=f"arc_{expert_type}_expert",
        )
    
    def _split_dataset(self, dataset: Dataset, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset]:
        """Split dataset into train/validation"""
        train_size = int(len(dataset) * train_ratio)
        
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
    
    def train_expert(self, expert_type: str) -> Dict[str, Any]:
        """Train a single expert model"""
        logger.info(f"\\n🚀 Training {expert_type} expert model")
        logger.info("=" * 50)
        
        try:
            # Load and preprocess dataset
            dataset = self._load_expert_dataset(expert_type)
            processed_dataset = self._preprocess_dataset(dataset, expert_type)
            train_dataset, eval_dataset = self._split_dataset(processed_dataset)
            
            # Load model
            model = self._load_base_model()
            
            # Create training arguments
            training_args = self._create_training_arguments(expert_type)
            
            # Create data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=model,
                padding=True,
                return_tensors="pt"
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # Train model
            logger.info(f"Starting training for {expert_type} expert...")
            train_result = trainer.train()
            
            # Save model
            logger.info(f"Saving {expert_type} expert model...")
            trainer.save_model()
            trainer.save_state()
            
            # Evaluate model
            logger.info(f"Evaluating {expert_type} expert model...")
            eval_result = trainer.evaluate()
            
            # Collect results
            results = {
                'expert_type': expert_type,
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
                'model_path': training_args.output_dir,
                'training_args': training_args.to_dict()
            }
            
            self.trained_models[expert_type] = results
            
            logger.info(f"✅ {expert_type} expert training completed!")
            logger.info(f"   Train loss: {train_result.training_loss:.4f}")
            logger.info(f"   Eval loss: {eval_result['eval_loss']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to train {expert_type} expert: {e}")
            raise e
    
    def train_all_experts(self) -> Dict[str, Any]:
        """Train all expert models"""
        logger.info("🎯 Starting training for all expert models")
        logger.info("=" * 60)
        
        all_results = {}
        
        for expert_type in self.config.expert_types:
            try:
                result = self.train_expert(expert_type)
                all_results[expert_type] = result
                
                # Clear GPU memory between models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to train {expert_type}: {e}")
                all_results[expert_type] = {'error': str(e)}
                continue
        
        # Save training summary
        summary_path = os.path.join(self.config.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\\n🎉 Expert model training completed!")
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Print summary
        successful_models = [k for k, v in all_results.items() if 'error' not in v]
        failed_models = [k for k, v in all_results.items() if 'error' in v]
        
        logger.info(f"\\n📊 Training Summary:")
        logger.info(f"   ✅ Successful: {len(successful_models)} models")
        logger.info(f"   ❌ Failed: {len(failed_models)} models")
        
        if successful_models:
            logger.info(f"   Successful models: {', '.join(successful_models)}")
        if failed_models:
            logger.info(f"   Failed models: {', '.join(failed_models)}")
        
        return all_results

class ExpertModelInference:
    """Handle inference with trained expert models"""
    
    def __init__(self, config: ExpertTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = {}
        self.tokenizer = None
    
    def _load_tokenizer(self):
        """Load tokenizer if not already loaded"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.model_cache_dir
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_expert_model(self, expert_type: str):
        """Load a trained expert model"""
        if expert_type in self.loaded_models:
            return self.loaded_models[expert_type]
        
        model_path = os.path.join(self.config.output_dir, f"{expert_type}_expert")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expert model not found: {model_path}")
        
        logger.info(f"Loading {expert_type} expert model from {model_path}")
        
        # Load base model with quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.loaded_models[expert_type] = model
        return model
    
    def generate_response(self, expert_type: str, instruction: str, max_length: int = 256) -> str:
        """Generate response using expert model"""
        self._load_tokenizer()
        model = self.load_expert_model(expert_type)
        
        # Format input
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:\\n")[-1].strip()
        
        return response

def main():
    """Main training function"""
    print("🎯 Expert Model Fine-tuning System")
    print("=" * 40)
    
    # Initialize configuration
    config = ExpertTrainingConfig()
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (will be very slow)")
    
    # Initialize trainer
    trainer = ExpertModelTrainer(config)
    
    # Check if expert datasets exist
    missing_datasets = []
    for expert_type in config.expert_types:
        dataset_path = f"expert_datasets/{expert_type}_expert_dataset.json"
        if not os.path.exists(dataset_path):
            missing_datasets.append(expert_type)
    
    if missing_datasets:
        print(f"❌ Missing expert datasets: {missing_datasets}")
        print("   Please run create_expert_datasets.py first")
        return
    
    print(f"\\n📚 Found datasets for {len(config.expert_types)} expert types")
    
    try:
        # Train all expert models
        results = trainer.train_all_experts()
        
        print("\\n🎉 Expert model training pipeline completed!")
        print(f"Models saved in: {config.output_dir}")
        
        # Display final summary
        successful = sum(1 for r in results.values() if 'error' not in r)
        total = len(results)
        print(f"\\n📈 Final Results: {successful}/{total} models trained successfully")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
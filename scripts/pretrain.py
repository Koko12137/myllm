import os
import json
import warnings

import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, TrainingArguments, Trainer

from models.configuration import MyLLMConfig
from models.modeling import MyLLMForCausalLM
from utils.dataset import MyLLMPreTrainDataset, MyLLMPreTrainCollator
from utils.io import io_operation
from utils.inspect import model_inspect


warnings.filterwarnings('ignore')


def pretrain_model() -> None:
    # Set random seed
    torch.manual_seed(1234)
    
    # Read pretrain config from json file
    config: dict = json.load(open('configs/pretrain.json'))
    
    # IO operations
    io_operation(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        local_files_only=True, 
    )

    # Setup Language Model Config
    lm_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),        # FIXED: vocab_size of qwen2 is 151643, however the len(tokenizer.vocab_size) is 151665, this will cause error when indexing the embedding
        ignore_index=tokenizer.pad_token_id, 
        num_logits_to_keep=100, 
    )
    
    # Initialize the model
    model = MyLLMForCausalLM(lm_config, debug=False)
    model_inspect(model)
    
    # Setup training arguments
    training_args = TrainingArguments(
        save_total_limit=2, 
        save_strategy='steps', 
        save_steps=500, 
        output_dir=config['output'], 
        # Training arguments
        do_train=True, 
        do_eval=False, 
        # Logging arguments
        logging_strategy='steps', 
        logging_steps=10, 
        report_to='none', 
        # Other arguments
        **config['training_args'], 
    )
    
    # Load dataset
    train_ds = MyLLMPreTrainDataset(config['dataset'])
    
    collator = MyLLMPreTrainCollator(tokenizer, padding_side='left')
    # Initialize the trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        data_collator=collator, 
    )
    
    # Train the model
    trainer.train()
    
    # Save the model, tokenizer and config
    model.save_pretrained(config['output'])
    lm_config.save_pretrained(config['output'])
    tokenizer.save_pretrained(config['output'])
    
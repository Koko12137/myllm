import json
import warnings

import torch
import pyarrow.parquet as pq
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, Qwen2Config
)

from test.modeling_qwen import Qwen2ForCausalLM
from utils.dataset import PretrainDataset
from utils.io import io_operation


warnings.filterwarnings('ignore')


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    xs = []
    ys = []
    masks = []
    for sample in batch:
        xs.append(sample['input_ids'])
        ys.append(sample['labels'])
        masks.append(sample['attention_mask'])
        
    # Concatenate the samples
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    masks = torch.cat(masks, dim=0)
    # Create ids
    ids = torch.range(0, xs.size(1) - 1, dtype=torch.long).unsqueeze(0)
    # Repeat ids
    ids = ids.repeat(xs.size(0), 1)
    return {'input_ids': xs, 'labels': ys, 'attention_mask': masks, 'position_ids': ids}


def pretrain_model() -> None:
    # Set random seed
    torch.manual_seed(1234)
    
    # Read pretrain config from json file
    config = json.load(open('configs/pretrain.json'))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

    # Setup Language Model Config
    lm_config = Qwen2Config(
        num_hidden_layers=16, 
        hidden_size=1024, 
        intermediate_size=4096, 
        num_attention_heads=16, 
        attention_head_dim=128, 
        max_position_embeddings=512, 
        vocab_size=tokenizer.vocab_size, 
        num_key_value_heads=8, 
    )
    
    # IO operations
    io_operation(config)
    
    # Initialize the model
    model = Qwen2ForCausalLM(lm_config)
    
    # Print the trainable parameters
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(model)
    
    # Setup training arguments
    training_args = TrainingArguments(
        save_total_limit=1, 
        save_strategy='steps', 
        save_steps=200, 
        output_dir=config['output'],
        # Training arguments
        do_train=True, 
        do_eval=False, 
        max_grad_norm=1.0, 
        # Logging arguments
        logging_strategy='steps', 
        logging_steps=10, 
        report_to='none',
        # Other arguments
        **config['training_args'], 
    )
    
    # Load dataset
    table = pq.read_table(config['dataset'])
    train_ds = PretrainDataset(table, tokenizer, max_length=lm_config.max_position_embeddings)
    
    # Initialize the trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        data_collator=collate_fn, 
    )
    
    # Train the model
    trainer.train()
    
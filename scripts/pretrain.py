import json
import warnings

import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, TrainingArguments, Trainer

from models.configuration import MyLLMConfig
from models.modeling import MyLLMForCausalLM as MyLLM
from utils.dataset import PretrainDataset
from utils.io import io_operation


warnings.filterwarnings('ignore')


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    masks = []
    for sample in batch:
        xs.append(sample['input_ids'])
        ys.append(sample['labels'])
        masks.append(sample['attention_mask'])
        
    # Convert the list of tensors to a single tensor
    xs = torch.tensor(xs)   # shape: (batch_size, max_length)
    ys = torch.tensor(ys)
    masks = torch.tensor(masks)
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

    # Setup Language Model Config
    lm_config = MyLLMConfig(num_hidden_layers=16)
    
    # IO operations
    io_operation(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    
    # Initialize the model
    model = MyLLM(lm_config)
    
    # Setup training arguments
    training_args = TrainingArguments(
        save_total_limit=1, 
        save_strategy='steps', 
        save_steps=200, 
        output_dir=config['output'],
        # Training arguments
        do_train=True, 
        do_eval=False, 
        # Logging arguments
        logging_strategy='steps', 
        logging_steps=20, 
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
    
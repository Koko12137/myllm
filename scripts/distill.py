import json
import random
import warnings

import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalPrediction
from transformers import AutoTokenizer, TrainingArguments, Trainer

from models.configuration import MyLLMConfig
from models.distill import MyLLMDistillWrapper
from utils.dataset import InMemoryDataset, MyLLMDataset, MyLLMPreTrainCollator
from utils.inspect import model_inspect
from metrics.model_metrics import perplexity


warnings.filterwarnings('ignore')


def metric_fn(eval_outputs: EvalPrediction) -> dict:
    r"""Compute the perplexity of the model predictions."""
    # Get the predictions and references
    predictions = eval_outputs.predictions[0]
    references = eval_outputs.label_ids
    # Convert the predictions to logits
    perplexity_value = perplexity(references, predictions)
    return {'perplexity': perplexity_value}

def distill_model() -> None:
    # Set random seed
    torch.manual_seed(1234)
    
    # Read pretrain config from json file
    config: dict = json.load(open('configs/distill.json'))
    
    from_pretrained = config.get("from_pretrained")
    distill_args = config.get("distill_args")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        local_files_only=True, 
    )
    
    # Load model config
    lm_config = MyLLMConfig.from_pretrained(from_pretrained)
    
    # Initialize the model
    model = MyLLMDistillWrapper("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", lm_config, **distill_args)
    model.student_from_pretrained(from_pretrained, debug=False)
    model_inspect(model.student)
    model_inspect(model.teacher)
    
    # Setup training arguments
    training_args = TrainingArguments(
        save_total_limit=2, 
        save_strategy='steps', 
        save_steps=10, 
        output_dir=config['output'], 
        # Training arguments
        do_train=True, 
        do_eval=False, 
        warmup_steps=4000, 
        # Evaluation arguments
        eval_strategy='steps',
        eval_steps=5, 
        per_gpu_eval_batch_size=2, 
        eval_on_start=True, 
        # Logging arguments
        logging_strategy='steps', 
        logging_steps=1, 
        report_to='none', 
        # Other arguments
        **config['training_args'], 
    )
    
    # Load dataset
    train_ds = MyLLMDataset(config['dataset'])
    # Randomly sample a subset of the dataset for perplexity calculation
    random_ids = random.sample(range(len(train_ds)), 10)
    test_ds = InMemoryDataset(train_ds.get_batch(random_ids))
    
    collator = MyLLMPreTrainCollator(tokenizer, padding_side='left')
    # Initialize the trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        eval_dataset=test_ds, 
        compute_metrics=metric_fn, 
        data_collator=collator, 
    )
    
    # Train the model
    trainer.train()
    
    # Save the model, tokenizer and config
    model.student.save_pretrained(config['output'], safe_serialization=False)
    model.student.config.save_pretrained(config['output'])
    tokenizer.save_pretrained(config['output'])
    
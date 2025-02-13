import os
import json
import random 
from typing import Iterable

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import AutoTokenizer

from utils.io import io_operation


MAX_LENGTH = 32768
CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<system>\\n' + system_message + '</system>\\n' }}{% else %}{{ '<system>\\nYou are an helpful assistent, please think step by step and answer the questions.</system>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<user>\\n' + content + '</user>' + '\\n<assistant>\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</assistant>\\n' }}{% endif %}{% endfor %}"


def load_jsonl(file_path: str) -> Iterable:
    with open(file_path, "r", encoding='utf8') as f:
        for line in f:
            yield json.loads(line)['text']

    
def eval_tokenizer(config: dict) -> None:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['output'], add_suffix_space=False)
    
    # Check the length of the vocabulary
    assert len(tokenizer) == config['vocab_size'], "The length of the vocabulary is not correct."
    print("The length of the vocabulary is correct.")
    
    # Dummy message test
    messages = [
            {"role": "system", "content": "你是个智能问答助手，所有回答需要用中文进行。"},
            {"role": "user", "content": "I am looking for a book to read."}, 
            {"role": "assistant", "content": "读书是好的，长脑子。"}, 
    ]
    
    # Apply the chat template 
    template = tokenizer.apply_chat_template(messages, tokenize=False)
    print(template)
    
    # Check the length of the template before and after tokenization
    input_ids = tokenizer(template, add_special_tokens=False)['input_ids']
    response = tokenizer.decode(input_ids)
    print(response)
    # BUG: Extra whitespace will be added after each special token
    assert len(template) == len(response), "The length of the template before and after tokenization is not the same."
    print("The length of the template before and after tokenization is the same.")
    

def train_tokenizer(config: dict) -> None:
    # Set random seed
    random.seed(1234)
    
    # Convert relative paths to absolute paths
    config['corpus_file'] = os.path.abspath(config['corpus_file'])
    config['output'] = os.path.abspath(config['output'])
    
    vocab_size = config['vocab_size']
    corpus_file = config['corpus_file']
    output_path = config['output']
    
    # IO operations
    io_operation(config)
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE())
    # Initialize the pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    # Set special tokens
    special_tokens = [
        "<pad>", "<unk>", "<system>", "<user>", "<assistant>", "</system>", "</user>", "</assistant>"
    ]
    
    # Initialize the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=special_tokens, 
        show_progress=True, 
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), 
    )
    
    # Initialize the decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Load the corpus
    corpus = load_jsonl(corpus_file)
    
    # Train the tokenizer
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    # Check the index of the special tokens
    assert tokenizer.token_to_id("<pad>") == 0, "The index of <pad> is not 0"
    assert tokenizer.token_to_id("<unk>") == 1, "The index of <unk> is not 1"
    assert tokenizer.token_to_id("<system>") == 2, "The index of <system> is not 2"
    assert tokenizer.token_to_id("<user>") == 3, "The index of <user> is not 3"
    assert tokenizer.token_to_id("<assistant>") == 4, "The index of <assistant> is not 4"
    assert tokenizer.token_to_id("</system>") == 5, "The index of </system> is not 5"
    assert tokenizer.token_to_id("</user>") == 6, "The index of </user> is not 6"
    assert tokenizer.token_to_id("</assistant>") == 7, "The index of </assistant> is not 7"
    
    # Save the tokenizer
    tokenizer.save(os.path.join(output_path, "tokenizer.json"))
    tokenizer.model.save(output_path)
    
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<system>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }, 
            "3": {
                "content": "<user>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }, 
            "4": {
                "content": "<assistant>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }, 
            "5": {
                "content": "</system>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "6": {
                "content": "</user>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "7": {
                "content": "</assistant>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<assistant>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</assistant>",
        "legacy": False,
        "model_max_length": MAX_LENGTH, 
        "padding_side": "left", 
        "pad_token": "<pad>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>", 
        "chat_template": CHAT_TEMPLATE, 
    }
    
    # Save the tokenizer config
    with open(os.path.join(output_path, "tokenizer_config.json"), "w", encoding="utf8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)
    
    # Evaluate the tokenizer
    eval_tokenizer(config)
    
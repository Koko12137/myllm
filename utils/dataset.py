import os

import torch
import pyarrow as pa
from torch.utils.data import Dataset
from tokenizers import Tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    
    def __init__(self, table: pa.Table, tokenizer: Tokenizer, field: str = 'text', max_length: int = 512) -> None:
        super().__init__()
        self.table = table
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.field = field

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        # Get the sample from the table
        text = self.table[self.field][idx].as_py()
        
        # Tokenize the sample
        encodings = self.tokenizer(
            text, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=self.max_length + 1, 
            padding='max_length', 
            return_tensors='pt', 
        )

        # Get the input and target
        input_ids = encodings['input_ids'][:, :-1]
        attention_mask = encodings['attention_mask'][:, :-1]
        labels = encodings['input_ids'][:, 1:]
        # Mask the labels
        labels[attention_mask == 0] = self.tokenizer.pad_token_id
        
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
    
import os
from collections.abc import Sequence

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class InMemoryDataset(Sequence, Dataset):
    
    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]
    
    
class MyLLMDataset(Sequence, Dataset):
    r"""This Dataset that can load multiple parquet files

    Attributes:
        path (`str`): 
            The path to the directory containing the parquet files.
        field (`str`): 
            The field to extract from the parquet files.
        table (`pa.Table`): 
            The table containing all the text data.
    """
    path: str
    field: str
    table: pa.Table
    
    def __init__(self, path: str, field: str = 'text') -> None:
        super().__init__()
        self.path = path
        self.field = field
        
        # Get all parquet files
        data_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.parquet'):
                    data_files.append(os.path.join(root, file))
                # elif file.endswith('.csv'):
                #     data_files.append(os.path.join(root, file))
                    
        tables = [pq.read_table(file, columns=[field]) for file in data_files]
        # Concatenate tables
        self.table = pa.concat_tables(tables)
        
    def __len__(self) -> int:
        return self.table.shape[0]

    def __getitem__(self, idx: int) -> str:
        """Delay the computation of the text until it is needed

        Args:
            idx (`int`): 
                The index of the text to get.

        Returns:
            `str`: 
                The text at the given index. 
        """
        # Get the text
        text = self.table[self.field][idx].as_py()
        return text
    
    def get_batch(self, indices: list[int]) -> list[str]:
        """Get a batch of text from the dataset

        Args:
            indices (`list[int]`): 
                The indices of the text to get.

        Returns:
            `list[str]`: 
                The text at the given indices. 
        """
        # Get the text
        texts = [self.table[self.field][idx].as_py() for idx in indices]
        return texts
    

class MyLLMPreTrainCollator:
    r"""This collator will set the last token of the input as the target token.
    
    Attributes:
        tokenizer (`AutoTokenizer`): 
            The tokenizer to use.
        padding_side (`str`):
            The side to pad the inputs.
        max_length (`int`):
            The maximum length of the inputs.
    """
    tokenizer: AutoTokenizer
    padding_side: str
    max_length: int
    
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        padding_side: str = 'left', 
        max_length: int = 512, 
    ) -> None:
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.max_length = max_length
    
    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        # Convert batch of str to batch of encodings
        encodings: dict[str, torch.Tensor] = self.tokenizer(
            batch, 
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


class MyLLMTemplateCollator(MyLLMPreTrainCollator):
    r"""Collator for the model. This collator will apply the chat template to the input text."""
    
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        padding_side: str = 'left', 
        max_length: int = 512, 
    ) -> None:
        super().__init__(tokenizer, padding_side, max_length)
        
        # Check if the tokenizer has the chat template
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise ValueError("The tokenizer does not have the apply_chat_template method.")
        
    
    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        # Apply the chat template to the input text without encoding
        batch = self.tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        
        # Convert batch of str to batch of encodings
        return super().__call__(batch)

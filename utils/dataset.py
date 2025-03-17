import os

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MyLLMPreTrainDataset(Dataset):
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
    

class MyLLMPreTrainCollator:
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
    
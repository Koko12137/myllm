import pyarrow.parquet as pq
from transformers import AutoTokenizer

from utils.dataset import PretrainDataset


def test_dataset() -> None:
    tokenizer = AutoTokenizer.from_pretrained('tokenizer')
    table = pq.read_table('datasets/pretrain/pretrain.parquet')
    dataset = PretrainDataset(table, tokenizer)
    print(len(dataset))
    print(dataset[1])
    
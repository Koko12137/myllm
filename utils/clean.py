import re
from multiprocessing import Pool

import pandas as pd


def replace_white_space(text: str) -> str:
    # Replace the continuous white spaces with a single white space
    white_space_pattern = re.compile(r'\s+')
    text = white_space_pattern.sub(' ', text)
    t_pattern = re.compile(r'\t+')
    text = t_pattern.sub(' ', text)
    n_pattern = re.compile(r'\n+')
    text = n_pattern.sub('\n', text)
    return text
    

def replace_consecutive_punctuation(text: str) -> str:
    # Replace the consecutive punctuation with a single punctuation
    punctuation_pattern = re.compile(r'([,.?!-=+#$%^&*<>/;:|`])\1+')
    text = punctuation_pattern.sub(r'\1', text)
    return text


def replace_consecutive_tokens(input_ids: list[int]) -> list[int]:
    # Replace the consecutive tokens with a single token
    new_input_ids = [input_ids[0]]
    for i in input_ids[1:]:
        if i != new_input_ids[-1]:
            new_input_ids.append(i)
    return new_input_ids

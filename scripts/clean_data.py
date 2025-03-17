import os
import random
import json
from hashlib import sha256
from functools import partial
from multiprocessing.pool import Pool

import pandas as pd


SEED = 42


def process_clean(files: list[str], output: str) -> None:
    
    for file in files:
        print("Processing: ", file)
        f_path =  file.split("/")
        itype, language, name = f_path[-4], f_path[-3], f_path[-1]
        
        # Check if the file is already processed
        f_path = f"{output}/{itype}-{language}-{name}"
        if os.path.exists(f_path):
            print("Already processed: ", f_path)
            continue
        
        table: pd.DataFrame = pd.read_parquet(file)
        # Discard the rows that length of `text` is less than 300 
        table = table[table['text'].str.len() > 300]
        # Drop all the columns except `text`
        cols = table.columns.difference(['text'])
        table = table.drop(columns=cols)
        # Drop the duplicates using hash of the `text`
        table['hash'] = table['text'].apply(lambda x: sha256(x.encode()).hexdigest())
        table = table.drop_duplicates(subset=['hash'])
        table = table.drop(columns=['hash'])
            
        # Save the processed data
        table.to_parquet(f_path, index=False)
        print("##  Processed: ", f_path)
        
        # Close file and release memory
        table = None
    
    
def clean() -> None: 
    # Read the configuration file
    with open("configs/preprocess.json", "r") as file:
        config = json.load(file)
    
    files = []
    for root, dirs, file in os.walk(config['dataset']):
        for f in file:
            if "high" in root:
                files.append(os.path.join(root, f))
                
    n = config['n_jobs']
    # If the n is -1, use all the available cores
    if n == -1:
        n = os.cpu_count()
    
    # Split the files into n chunks
    chunk_size = len(files) // n
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    # Initialize processing pool
    pool = Pool(n)
    
    # Partial function to pass the output directory
    process_fn = partial(process_clean, output=config['output'])
    # Process the chunks
    data = pool.map_async(process_fn, chunks)
    
    # Wait for the pool to finish
    pool.close()
    pool.join()
    print("Data cleaning done.")
    

def process_merge(files: list[str], output: str, sample: int = -1, seed: int = 42) -> None:
    # Set seed
    random.seed(seed)
    
    # Industry type
    f_name =  files[0].split("/")[-1]
    itype, language = f_name.split("-")[0], f_name.split("-")[1]
    itype = f"{itype}-{language}"
    output = f"{output}/{itype}.parquet"
    
    merge = None
    for file in files:
        print("Processing: ", file)
        # Read the file
        table: pd.DataFrame = pd.read_parquet(file)
        # Merge
        if merge is None:
            merge = table
        else:
            merge = pd.concat([merge, table])
        # Close file and release memory
        table = None
        
    # Check if sample
    if sample != -1 and sample < len(merge):
        # Sample the data
        merge = merge.sample(sample)
        
    # Save the merged data
    merge.to_parquet(output, index=False)
    print("##  Processed: ", output)
    # Close file and release memory
    merge = None


def merge() -> None:
    # Read the configuration file
    with open("configs/preprocess.json", "r") as file:
        config = json.load(file)
        
    files: list[str] = []
    for root, dirs, file in os.walk(config['output']):
        for f in file:
            files.append(os.path.join(root, f))
            
    # Split by industry type
    industry = {}
    for file in files:
        f_name =  file.split("/")[-1]
        itype, language = f_name.split("-")[0], f_name.split("-")[1]
        itype = f"{itype}-{language}"
        if itype not in industry:
            industry[itype] = []
        industry[itype].append(file)
                
    n = config['n_jobs']
    # If the n is -1, use all the available cores
    if n == -1:
        n = os.cpu_count()
    
    # Initialize processing pool
    pool = Pool(n)
    
    # Partial function to pass the output directory
    process_fn = partial(process_merge, output=f"{config['output']}/merge", sample=100000, seed=SEED)
    # Process the chunks
    data = pool.map_async(process_fn, industry.values())
    
    # Wait for the pool to finish
    pool.close()
    pool.join()
    print("Data cleaning done.")
        
    
if __name__ == "__main__":
    random.seed(SEED)
    clean()
    merge()
    
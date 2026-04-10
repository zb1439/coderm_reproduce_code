import os
import json
from tqdm import tqdm

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(filename, dataset):
    with open(filename, 'w', encoding='UTF-8') as fp:
        for data in tqdm(dataset):
            fp.write(json.dumps(data, ensure_ascii=False) + '\n')

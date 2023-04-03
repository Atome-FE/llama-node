import json
import torch
import os

file_path = os.path.abspath(os.path.dirname(__file__))


def read_json_file(json_file: str):
    path = os.path.join(file_path, json_file)
    with open(path, 'r') as f:
        data = json.load(f)
    return torch.tensor(data)


dog1 = read_json_file('./dog1.json')
dog2 = read_json_file('./dog2.json')
cat1 = read_json_file('./cat1.json')


def compute_similarity(v1, v2):
    return torch.cosine_similarity(v1, v2, dim=0)


print(compute_similarity(dog1, dog2)) # 0.8430
print(compute_similarity(dog1, cat1)) # 0.9792

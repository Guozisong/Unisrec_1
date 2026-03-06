import os
import torch
from transformers import AutoModel, AutoTokenizer


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def load_plm(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained("./bert-base-uncased/")
    model = AutoModel.from_pretrained("./bert-base-uncased/")
    return tokenizer, model



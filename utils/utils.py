import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Any
from transformers import T5Tokenizer, T5ForSequenceClassification

def load_t5_model(model_name: str, n_sets: int, dtype: torch.dtype = torch.float32) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'
    model_kwargs['num_labels'] = n_sets

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name,**model_kwargs)
    return tokenizer, model

def get_ll2_model(model_name: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.eval()
    return tokenizer, model

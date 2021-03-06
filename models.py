'''
Define model architectures for NMT
'''
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn

class T5Based(nn.Module):
    def __init__(self, size='google/mt5-base'):
        super().__init__()
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(size)
            self.model = MT5ForConditionalGeneration.from_pretrained(size)
        except:
            raise ValueError(f"T5 size {size} unsupported")

    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
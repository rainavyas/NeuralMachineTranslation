'''
Prepare WMT18 data as tensors
'''

import torch
from datasets import load_dataset

class DataTensorLoader():
    def __init__(self, tokenizer, subset='cs-en', lang_flip=True, arch='T5'):
        self.tokenizer = tokenizer
        self.arch = arch
        self.dataset = load_dataset('wmt18', subset)

        langs = subset.split('-')
        if lang_flip:
            self.source = langs[1]
            self.target = langs[0]
        else:
            self.source = langs[0]
            self.target = langs[1]
        self._get_prefix()

    def _get_data(self, data):
        source_sentences = [item['translation'][self.source] for item in data]
        target_sentences = [item['translation'][self.target] for item in data]

        source_sentences = [self.prefix+sen for sen in source_sentences]

        # prep input tensors - source
        encoded_inputs = self.tokenizer(source_sentences, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = encoded_inputs['input_ids']
        input_mask = encoded_inputs['attention_mask']

        # prep output tensors - target -> use '-100' for masked positions (T5)
        encoded_inputs = self.tokenizer(target_sentences, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        output_ids = encoded_inputs['input_ids']
        mask = encoded_inputs['attention_mask']
        output_ids[mask==0] = -100

        return input_ids, input_mask, output_ids
    
    def _get_prefix(self):
        code_to_lang = {
            'en' : 'English',
            'de' : 'German',
            'et' : 'Estonian',
            'fi' : 'Finnish',
            'ru' : 'Russian',
            'tr' : 'Turkish',
            'zh' : 'Chinese'
        }
        self.prefix = f'translate {code_to_lang[self.source]} to {code_to_lang[self.target]}: '
        
    def get_train(self):
        return self._get_data(self.dataset['train'])

    def get_validation(self):
        return self._get_data(self.dataset['validation'])
    
    def get_test(self):
        return self._get_data(self.dataset['test'])
        


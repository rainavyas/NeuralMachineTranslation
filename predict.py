'''
Apply NMT model at inference time
Generates a JSON file with contents being a list of length == number of data points
Each item in list is a dict
Each dict has keys 'source', 'target', 'prediction'
'''

import json
from data_prep import DataTensorLoader
import sys
import os
import argparse
import torch
from models import T5Based
from tools import set_seeds

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='Specify trained model th file')
    commandLineParser.add_argument('OUT', type=str, help='Specify output .json file')
    commandLineParser.add_argument('--subset', type=str, default='cs-en', help="Specify translation")
    commandLineParser.add_argument('--arch', type=str, default='T5', help="Specify model architecture")
    commandLineParser.add_argument('--size', type=str, default='t5-base', help="Specify model size")
    commandLineParser.add_argument('--num_beams', type=int, default=3, help="Specify number of decoding beams")
    args = commandLineParser.parse_args()

    set_seeds(args.seed)

    # Save the command run
    text = ' '.join(sys.argv)+'\n'
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(text)
    print(text)

    # Load model
    model = T5Based(args.size)
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load the data
    dataloader = DataTensorLoader(model.tokenizer, subset=args.subset, lang_flip=True, arch=args.arch)
    input_ids, input_mask, _, source_sentences, target_sentences = dataloader.get_test(return_sentences=True)

    # Decode ids into sentences
    sentences = []
    for i, (inp_id, mask, source, target) in enumerate(zip(input_ids, input_mask, source_sentences, target_sentences)):
        # Generate prediction ids
        prediction_ids = model.generate(
        input_ids = inp_id.unsqueeze(dim=0),
        attention_mask = mask.unsqueeze(dim=0),
        num_beams = args.num_beams,
        do_sample = False,
        max_length = 512,
        length_penalty = 1.0,
        early_stopping = True,
        use_cache = True,
        num_return_sequences = 1
        )
        print(f'Decoding {i}/{len(input_ids)}')
        prediction = model.tokenizer.decode(prediction_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        sentences.append({
            'source' : source,
            'target' : target,
            'prediction' : prediction
        })

    # Save to file
    with open(args.OUT, 'w') as f:
        f.write(sentences)


# Objective

Training and evaluation for Neural Machine Translation. 

# Data

This work uses the [WMT18 dataset](https://www.statmt.org/wmt18/). This supports the following language translations:

- English-Chinese and Chinese-English
- English-Czech and Czech-English
- English-Estonian and Estonian-English
- English-Finnish and Finnish-English
- English-German and German-English
- English-Russian and Russian-English
- English-Turkish and Turkish-English

# Model Architectures

The following model architectures are supported:
 - [T5](https://huggingface.co/docs/transformers/model_doc/t5)
 - [BART](https://huggingface.co/docs/transformers/model_doc/bart)
 - RNN-based

# Requirements

python3.7 and above

## Install with PyPI

`pip install datasets transformers torch sentencepiece`

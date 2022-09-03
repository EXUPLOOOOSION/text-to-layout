import sys
import io
import os
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import numpy as np
from transformers import *
#from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import transformers


class SentenceEncoder(nn.Module):
    def __init__(self, freeze=False, verbose=False):
        """
        Args:
            freeze: Whether to freeze the model weights. (Even when calling train())
            verbose: when initializing the model, whether to supress model initialization messages.
        """

        super(SentenceEncoder, self).__init__()
        self.freeze = freeze
        self.verbose = verbose
        if not verbose:
            transformers.logging.set_verbosity_error()
        # Build the model
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    def eval(self):
        for param in self.parameters():
            param.requires_grad = False
    def train(self):
        if not freeze:
            for param in self.parameters():
                param.requires_grad = True
        

    def forward(self, captions):
        """
        Applies a all-mpnet-base-v2 transformer to an input sequence, giving a single embedding for each sequence.

        Args:
            captions: dictionary containing 2 keys: 'input_ids': python list of tokens of each sentence and 
                                                    'attention_mask': list of 0s and 1s to mask out padded input.
        Returns:
            Tensor of size (batch_size,768) # where batch_size = number of sentences given.
        """
        outputs = self.model(captions)['sentence_embedding']
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        return outputs
#quick example and tests
if __name__ == '__main__':
    sentences = ["Hello, world!", "Hello"]
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    model = SentenceEncoder()
    embeddings = model.model(encoded_input)
    print(embeddings.size())
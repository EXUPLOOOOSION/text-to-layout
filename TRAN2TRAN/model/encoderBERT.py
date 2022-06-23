import sys

import torch
import torch.nn as nn
import numpy as np
from transformers import *
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

class EncoderBERT(nn.Module):
    def __init__(self, pretrained_path=None, device=torch.cuda.current_device()):
        super(EncoderBERT, self).__init__()
        self.device = device
        self.pretrained_path = pretrained_path # The path of the pretrained model (None if BERT-base)
        
        # Build the model
        self.model = None
        if self.pretrained_path == None:
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_path)

    def forward(self, input_ids, attention_mask):
        """
        Applies a BERT transformer to an input sequence

        Args:
            input_ids (batch, seq_len): tensor containing the tokens of the input sequence.
            attention_mask (batch, seq_len): tensor of 0s and 1s to mask out padded input.
        Returns: Tensor of size (batch_size, seq_len, 768).  Features for each token.

        """

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        result = outputs.last_hidden_state
        return result

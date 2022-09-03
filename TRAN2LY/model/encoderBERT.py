import sys

import torch
import torch.nn as nn
import numpy as np
from transformers import *
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import transformers
class EncoderBERT(nn.Module):
    def __init__(self, pretrained_path=None, freeze=False, pooling_type='avg', device=torch.cuda.current_device(), verbose=False):
        """
        Args:
            pooling_type: way of combining each resulting vector of a sequence into a single vector.
                            'max': for each feature vector, take the highest value among all result vectors
                            'avg': for each feature vector, take the average value among all result vectors
                            'cls': take the cls result vector as the representation for the entire sentence
            freeze: Whether to freeze the model weights. (Even when calling train())
            verbose: when initializing the model, whether to supress model initialization messages.
        """

        super(EncoderBERT, self).__init__()
        self.device = device
        self.pretrained_path = pretrained_path # The path of the pretrained model (None if BERT-base)
        self.freeze = freeze 
        self.pooling_type = pooling_type
        self.verbose = verbose
        
        if not verbose:
            transformers.logging.set_verbosity_error()
        
        # Build the model
        self.model = None
        if self.pretrained_path == None:
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.pretrained_path)

    def forward(self, input_ids, attention_mask):
        """
        Applies a BERT transformer to an input sequence and returns a single feature vector for each sequence 
        (by aggregating BERT's vectors depending on the pooling_type)

        Args:
            input_ids (batch, seq_len): tensor containing the tokens of the input sequence.
            attention_mask (batch, seq_len): tensor of 0s and 1s to mask out padded input.
        Returns: Tensor of size (batch_size, 768). Features for each sentence.
        """

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if self.pooling_type == 'cls':
            result = outputs.hidden_states[-1].permute(1, 0, 2)[0]
        elif self.pooling_type =='avg' or self.pooling_type == 'average':
            #mask with size [batch_size, padded_sentence_size, embed_size] where each word's new dimension is filled with that word's value
            #i.e if in the first sentence the first word had a mask of 1, all of perElementMask[0,0,:] will be 1
            perElementMask = attention_mask[:,:,None].expand(outputs.last_hidden_state.size())
            #size of outputs.last_hidden_state : [batch_size, length_of_caption, vector_length]
            #end vector of each token
            output_vectors = outputs.last_hidden_state
            #take out the cls from each caption. Only want the average from the word tokens
            output_vectors = output_vectors[:,1:,:]
            perElementMask = perElementMask[:,1:,:]
            #average vectors from each word token to a single vector
            sentence_sums = torch.sum(output_vectors * perElementMask,dim=1)
            denominator = torch.sum(perElementMask, dim=1) #number of valid (not masked) for each sentence. All values in the last dimension should be the same as they are all from the same sentence.
            result = sentence_sums / denominator
        elif self.pooling_type == 'max' or self.pooling_type == 'max_pooling':
            #words to mask have value -inf and words to keep have value 0
            reversedMask = attention_mask.clone().float()
            reversedMask[attention_mask==0] = np.float("-inf")
            reversedMask[attention_mask!=0] = 0
            #mask with size [batch_size, padded_sentence_size, embed_size] where each word's new dimension is filled with that word's value
            #i.e if in the first sentence the first word had a mask of 0, all of perElementMask[0,0,:] will be 0
            perElementMask = reversedMask[:,:,None].expand(outputs.last_hidden_state.size())
            #size of outputs.last_hidden_state : [batch_size, length_of_caption, vector_length]
            #end vector of each token
            output_vectors = outputs.last_hidden_state
            #take out the cls from each caption. Only want the average from the word tokens
            output_vectors = output_vectors[:,1:,:]
            perElementMask = perElementMask[:,1:,:]

            masked_output = output_vectors+perElementMask
            #print(masked_output)
            result = masked_output.max(dim=1).values
        else:
            raise NotImplementedError('Incorrect pooling type for Transformer encoder')
        return result
#scuffed tests
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    captions = ["Hello, world!", "Hello"]
    encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
    all_inputs_ids = encoding['input_ids']
    all_attention_masks = encoding['attention_mask']
    encoder = EncoderBERT(pooling_type='max')
    a = encoder(all_inputs_ids, all_attention_masks)
    print(a)
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.attention import Attention

import sys

class DecoderRNN(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, is_training, bbox_dimension=128, dropout_p=0.2, use_attention=False, bidirectional=False, xy_distribution_size=16, temperature=0.4, device=torch.cuda.current_device()):

        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.output_size = vocab_size
        self.use_attention = use_attention
        self.bbox_dimension = bbox_dimension
        self.is_training = is_training
        self.xy_distribution_size = xy_distribution_size
        self.temperature = temperature
        # Class
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(p=dropout_p)

        # Bounding box
        self.xy_input = nn.Linear(2, self.bbox_dimension)
        self.xy_dropout = nn.Dropout(p=dropout_p)

        self.wh_input = nn.Linear(2, self.bbox_dimension)
        self.wh_dropout = nn.Dropout(p=dropout_p)

        # ninput, nhidden, nlayers,
        if self.use_attention:
            # Attention
            self.attention = Attention(self.hidden_size)
        self.rnn = nn.LSTM(
            (2 if self.use_attention else 1) * self.hidden_size + 2 * self.bbox_dimension, 
            self.hidden_size, 
            1)
        
        # Class prediction
        self.class_out = nn.Linear((3 if self.use_attention else 2) * self.hidden_size, self.output_size)

        # Bbox prediction
        self.bbox_xywh_out = build_mlp([(2 if self.use_attention else 1) * self.hidden_size + self.output_size, 512, self.xy_distribution_size ** 2])

        self.next_xy_input = nn.Linear(2, self.bbox_dimension)
        self.next_xy_dropout = nn.Dropout(p=dropout_p)

        self.bbox_wh_out = build_mlp([(2 if self.use_attention else 1) * self.hidden_size + self.output_size + self.bbox_dimension, 512, 2], final_nonlinearity=False)
        self.bbox_wh_sigmoid = nn.Sigmoid()

    def forward(self, target_l, target_x, target_y, target_w, target_h, hidden, encoder_output, next_l=None, next_xy=None):

        # Unsqueeze the input
        target_l = target_l.unsqueeze(0)
        target_x = target_x.unsqueeze(1)
        target_y = target_y.unsqueeze(1)
        target_w = target_w.unsqueeze(1)
        target_h = target_h.unsqueeze(1)
        # target_l = [1, batch_size]
        # target_x, y, w, h = [batch_size, 1]

        
        # Concatenate the [x, y, w, h] coordinates
        target_xy = torch.cat((target_x, target_y), dim=1)
        target_wh = torch.cat((target_w, target_h), dim=1)
        # target_xy,wh = [batch_size,2]

        embedded_l = self.embedding(target_l)
        # embedded_l = [1, batch size, emb dim]
        embedded_l = self.embedding_dropout(embedded_l)
        # embedded_l = [1, batch size, emb dim]
        
        xy_decoder_input = self.xy_dropout(self.xy_input(target_xy))
        wh_decoder_input = self.wh_dropout(self.wh_input(target_wh))
        # xy,wh_decoder input = [batch_size, bbox_dimension]
        
        xy_decoder_input = xy_decoder_input.unsqueeze(0)
        wh_decoder_input = wh_decoder_input.unsqueeze(0)
        # xy,wh_decoder input = [1, batch_size, bbox_dimension]

        if self.use_attention:
            attn = self.attention(hidden[0].squeeze(0), encoder_output).unsqueeze(1)
            # attn = [batch_size, 1, src_len]
            context = attn.bmm(encoder_output).permute(1, 0, 2)
            # context = [1, batch_size, hidden_size]
            l_and_xywh = torch.cat((xy_decoder_input, wh_decoder_input, embedded_l, context), dim=2)
            # l_and_xywh = [1, batch_size, 2*hidden_size + 2 * self.bbox_dimension]
        else:
            l_and_xywh = torch.cat((xy_decoder_input, wh_decoder_input, embedded_l), dim=2)
            # l_and_xywh = [1, batch_size, hidden_size + 2 * self.bbox_dimension]
        
        outputRNN, hiddenRNN = self.rnn(l_and_xywh, hidden)
        # output = [1, batch size, hidden_size * n directions]
        # hidden = [n layers * n directions, batch size, hidden_size]
        # cell = [n layers * n directions, batch size, hidden_size]

        if self.use_attention:
            class_input = torch.cat((outputRNN.squeeze(0), embedded_l.squeeze(0), context.squeeze(0)), dim=1)
            # class_input = [batch_size, hidden_size*3]
        else:
            class_input = torch.cat((outputRNN.squeeze(0), embedded_l.squeeze(0)), dim=1)
            # class_input = [batch_size, hidden_size*2]
            
        class_prediction = self.class_out(class_input)
        # class_prediction = [batch size, output_dim]

        if self.is_training and next_l != None:
            if self.use_attention:
                xy_hidden = torch.cat((hiddenRNN[0].squeeze(0), next_l, context.squeeze(0)), dim=1)
                # xy_hidden = [batch_size, output_dim + hidden_size*2]
            else:
                xy_hidden = torch.cat((hiddenRNN[0].squeeze(0), next_l), dim=1)
                # xy_hidden = [batch_size, output_dim + hidden_size]
        else:
            # Create a one hot encoding vector
            predicted_class = F.softmax(class_prediction, dim=1).clamp(1e-5, 1)
            one_hot = torch.zeros((predicted_class.shape[0], predicted_class.shape[1]))
            predicted_class_argmax = predicted_class.argmax(1)
            for i in range(predicted_class.shape[0]):
                one_hot[i, predicted_class_argmax[i]] = 1
            if torch.cuda.is_available():
                one_hot = one_hot.cuda()
            if self.use_attention:
                xy_hidden = torch.cat((hiddenRNN[0].squeeze(0), one_hot, context.squeeze(0)), dim=1)
                # xy_hidden = [batch_size, output_dim + hidden_size*2]
            else:
                xy_hidden = torch.cat((hiddenRNN[0].squeeze(0), one_hot), dim=1)
                # xy_hidden = [batch_size, output_dim + hidden_size]
        
        # generate a probability distribution for [x, y]
        xy_out = self.bbox_xywh_out(xy_hidden)
        # xy_out = [batch_size, xy_distribution_size**2]
        
        # When training we use the real xy values
        # When evaluating we sample the xy values
        topi = None
        if self.is_training and next_xy != None:
            xy_distance = xy_out.div(self.temperature).exp().clamp(min=1e-5, max=torch.finfo(torch.float32).max)
            #xy_distance = F.log_softmax(xy_out.div(self.temperature), dim=1)
            xy_topi = torch.multinomial(xy_distance, 1)
            topi = self.convert_to_coordinates(xy_topi)
            
            next_xy_decoder_input = self.next_xy_dropout(self.next_xy_input(next_xy))
            # next_xy_decoder_input = [batch_size, bbox_dimension]
        else:
            # Sample
            #xy_distance = F.normalize(xy_out,dim=1)
            xy_distance = xy_out.div(self.temperature).exp().clamp(min=1e-5, max=torch.finfo(torch.float32).max)
            #xy_distance = F.log_softmax(xy_out.div(self.temperature), dim=1)
            xy_topi = torch.multinomial(xy_distance, 1)
            topi = self.convert_to_coordinates(xy_topi)
            next_xy_decoder_input = self.next_xy_dropout(self.next_xy_input(topi))
            # next_xy_decoder_input = [batch_size, bbox_dimension]

        wh_hidden = torch.cat((xy_hidden, next_xy_decoder_input), dim=1)
        # wh_hidden = [batch_size, output_dim + hidden_size*2 + bbox_dimension]
        wh_out = self.bbox_wh_out(wh_hidden)
        # wh_out = [batch_size, 2]
        wh_out = self.bbox_wh_sigmoid(wh_out)
        # wh_out = [batch_size, 2]

        return class_prediction, xy_out, wh_out, hiddenRNN, topi

    def convert_to_coordinates(self, input_coordinates):
        """
        Function to convert the input coordinates to a x,y value.
        The input coordinate is a value between [0...., xy_distribution_size**2]
        """
        number_of_sectors = self.xy_distribution_size

        # First obtain the coordinates of the matrix
        x, y = input_coordinates % number_of_sectors, input_coordinates.div(number_of_sectors,rounding_mode='trunc')

        # Obtain the [x,y] value in [0, 1] range
        x_value = x.true_divide(number_of_sectors)
        y_value = y.true_divide(number_of_sectors)

        return torch.cat((x_value, y_value), dim=1)

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import random

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, vocab_size, pad_token, is_training, max_len=12, teacher_learning=True, pretrained_encoder=False, freeze_encoder=False, cap_lang=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token = pad_token
        self.teacher_learning = teacher_learning
        self.output_l_size = vocab_size
        self.is_training = is_training
        self.max_len = max_len
        self.temperature = 0.4
        self.xy_distribution_size = self.decoder.xy_distribution_size
        self.pretrained_encoder = pretrained_encoder
        self.freeze_encoder = freeze_encoder
        self.cap_lang = cap_lang

        if self.pretrained_encoder and self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
               
    def change_is_training(self, is_training):
        """
        Function to update is_training value
        """
        self.is_training = is_training
        if is_training:
            self.train()
        else:
            self.eval()
    def train(self):
        self.is_training = True
        for param in self.parameters():
            param.requires_grad = True
        self.decoder.train()
        if not self.freeze_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def eval(self):
        self.is_training = False
        self.decoder.eval()
        self.encoder.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, captions_padded, captions_length, target_l=None, target_x=None, target_y=None, target_w=None, target_h=None):
        """
        print("sizes:")
        print("captions_padded", captions_padded.size())
        print("captions_length", captions_length.size())
        print("target_l", target_l.size())
        print("target_x", target_x.size())
        print("target_y", target_y.size())
        print("target_w", target_w.size())
        print("target_h", target_h.size())
        """
        # Store the matches
        l_match, total = 0, 0
        # Encode the input
        encoder_output, encoder_hidden, include = self.encoder(captions_padded, captions_length)
        
        # Concatenate directions of the bidirectional lstm
        decoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        # Obtain the batch size
        batch_size = decoder_hidden[0].size(1)

        # Establish the longest length of the input
        trg_len = target_l.size(1)

        # tensors to store the outputs of the decoder
        outputs_class = torch.zeros(trg_len, target_l.size(0), self.decoder.output_size)
        outputs_bbox = torch.zeros(trg_len, target_l.size(0), 4)
        outputs_bbox_xy = torch.zeros(trg_len, target_l.size(0), self.xy_distribution_size ** 2)
        
        
        # Convert to cuda
        if torch.cuda.is_available():
            outputs_class = outputs_class.cuda()
            outputs_bbox = outputs_bbox.cuda()
            outputs_bbox_xy = outputs_bbox_xy.cuda()

        # Create the <sos> token
        trg_l = torch.ones(batch_size, dtype=torch.long)
        if torch.cuda.is_available():
            trg_l = trg_l.cuda()
        
        # Bbox for <sos> (x, y, w, h) = (0, 0, 0, 0)
        trg_x = torch.zeros(batch_size, dtype=torch.float)
        trg_y = torch.zeros(batch_size, dtype=torch.float)
        trg_w = torch.zeros(batch_size, dtype=torch.float)
        trg_h = torch.zeros(batch_size, dtype=torch.float)
        if torch.cuda.is_available():
            trg_x = trg_x.cuda()
            trg_y = trg_y.cuda()
            trg_w = trg_w.cuda()
            trg_h = trg_h.cuda()

        # When training retrieve the next ground truth class and bbox to apply teacher forcing
        # When validating we don't have this information so we set next_l and next_xy to None
        if self.is_training and self.teacher_learning:
            next_l = torch.FloatTensor(target_l.size(0), self.output_l_size)
            if torch.cuda.is_available():
                next_l = next_l.cuda()
                
            # Prediction of the next class (softmax). Only used for training.
            next_l[next_l != 0] = 0
            for batch_index in range(target_l.size(0)):
                next_l[batch_index, int(target_l[batch_index, 1])] = 1
            
            next_xy = torch.cat((target_x[:, 1].unsqueeze(1), target_y[:, 1].unsqueeze(1)), dim=1)
        else:
            next_l = None
            next_xy = None
        #xy_out_sample = np.empty((batch_size,trg_len,1024))
        # Column by column
        for di in range(1, trg_len):
            # Obtain the output of the decoder
            class_prediction, xy_out, wh_out, decoder_hidden, xy_coordinates = self.decoder(trg_l, trg_x, trg_y, trg_w, trg_h, decoder_hidden, encoder_output, next_l=next_l, next_xy=next_xy)

            # Save the prediction
            outputs_class[di] = class_prediction
            outputs_bbox[di, :, 2:] = wh_out
            #cpu_xy_out = xy_out.cpu()
            #xy_out_sample[:,di,:] = cpu_xy_out

            # Sample if the xy_coordinates are not calculated
            if xy_coordinates == None:
                #xy_distance = F.normalize(xy_out, dim=1)
                xy_distance = xy_out.div(self.temperature).exp()
                xy_topi = torch.multinomial(xy_distance, 1)
                xy_coordinates = self.convert_to_coordinates(xy_topi)

            # Save the prediction
            outputs_bbox[di, :, :2] = xy_coordinates # (x, y) coordinates
            outputs_bbox_xy[di] = xy_out # xy probability distribution
            
            # When training use the real values of the step for the next one
            # When validating use the predicted values of the step for the next one
            top1 = class_prediction.argmax(1)
            if self.is_training and self.teacher_learning:
                trg_l, trg_x, trg_y, trg_w, trg_h = target_l[:, di], target_x[:, di], target_y[:, di], target_w[:, di], target_h[:, di]
            else:
                trg_l = top1
                trg_x = xy_coordinates[:, 0]
                trg_y = xy_coordinates[:, 1]
                trg_w = wh_out[:, 0]
                trg_h = wh_out[:, 1]
            # When training retrieve the next ground truth class and bbox to apply teacher forcing
            # When validating we don't have this information so we set next_l and next_xy to None
            if self.is_training and self.teacher_learning:
                if di == trg_len-1:
                    next_l = None
                    next_xy = None
                else:
                    # Prediction of the next class (softmax). Only used for training
                    if next_l == None:
                        next_l = torch.FloatTensor(target_l.size(0), self.output_l_size)
                        if torch.cuda.is_available():
                            next_l = next_l.cuda()
                    next_l[next_l != 0] = 0
                    for batch_index in range(target_l.size(0)):
                        next_l[batch_index, int(target_l[batch_index, di+1])] = 1

                    next_xy = torch.cat((target_x[:, di+1].unsqueeze(1), target_y[:, di+1].unsqueeze(1)), dim=1)
            else:
                next_l = None
                next_xy = None

            if self.is_training:
                # Calculate some stats about the output (correct matching without taking into account the padding)                target_tensor = target_l[:, di]
                target_tensor = target_l[:, di]
                non_padding = target_tensor.ne(self.pad_token)
                l_correct = top1.view(-1).eq(target_tensor).masked_select(non_padding).sum().item()
                l_match += l_correct
                total += non_padding.sum().item()
        #xy_out_sample.dump("xy_out_samples.txt")
        #raise NotImplementedError('Inference without teacher learning not implemented in seq2seq for TRAN decoder')
        # Return all the information
        final_output = {
            "output_class": outputs_class,
            "output_bbox": outputs_bbox,
            "outputs_bbox_xy": outputs_bbox_xy,
            "l_match": l_match,
            "total": total
        }
        return final_output
    def indexes_from_sentence(self, sentence):
        # Tokenize a sentece
        if "<sos>" in self.cap_lang.word2index:
            seq = [self.cap_lang.word2index["<sos>"]]
        else:
            seq = []
        if type(sentence) is list:
            words = sentence
        else:
            words = sentence.split(' ')
        for word in words:
            if word in self.cap_lang.word2index:
                seq.append(self.cap_lang.word2index[word])
        if "<eos>" in self.cap_lang.word2index:
            seq.append(self.cap_lang.word2index["<eos>"])
        return seq
    def pad_seq(self, seq, max_length, pad_token):
        # Add padding to the sequence with a maximum of max_length
        seq = torch.cat((seq, torch.LongTensor([pad_token for i in range(max_length - len(seq))], device=seq.device)))
        return seq
    def generate(self, raw_captions):
        if self.cap_lang is None:
            print("Cap lang not initialized. Can't tokenize raw captions for further processing.")
            return None
        encodings = [self.indexes_from_sentence(raw_caption) for raw_caption in raw_captions]
        lengths = [len(encoding) for encoding in encodings]
        max_length = max(lengths)
        for encoding in encodings:
            encoding += [0]*(max_length-len(encoding))

        padded_encodings = torch.LongTensor(encodings)
        lengths = torch.LongTensor(lengths)
        #for some reason the model isn't made to work without targets even in eval mode.
        target_l = torch.zeros((padded_encodings.size(0), self.max_len))
        target_x,target_y,target_w,target_h=target_l,target_l,target_l,target_l

        if torch.cuda.is_available():
            padded_encodings = padded_encodings.cuda()
            lengths = lengths.cuda()
            target_l,target_x,target_y,target_w,target_h = target_l.cuda(),target_x.cuda(),target_y.cuda(),target_w.cuda(),target_h.cuda()
        output = self(padded_encodings, lengths, target_l= target_l,target_x= target_x,target_y= target_y,target_w= target_w,target_h= target_h)
        simplified_output = {} #only need actual output, not l_match or total. Clean it if necessary (make class selections, swap dimensions to be batch_size, seq_len)...
        simplified_output['output_class'] = torch.argmax(output['output_class'], dim=2).permute(1,0)
        simplified_output['output_bbox'] = output['output_bbox'].permute(1,0,2)
        return simplified_output
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if h.shape[0] == 2:
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def convert_from_coordinates(self, input_coordinates):
        distribution = torch.zeros((input_coordinates.shape[0], self.xy_distribution_size, self.xy_distribution_size), device=input_coordinates.device, dtype=torch.int)
        # We make a matrix to make the operations easier
        # distribution = [batch_size, xy_distribution_size, xy_distribution_size]

        # Convert coordinates from range [0, 1] to range [0, xy_distribution_size]
        input_coordinates[:, 0] = (input_coordinates[: ,0] * self.xy_distribution_size).clamp(0, self.xy_distribution_size-1)
        input_coordinates[:, 1] = (input_coordinates[: ,1] * self.xy_distribution_size).clamp(0, self.xy_distribution_size-1)

        # Obtain the distribution. All zeros except the coordinate that is a 1
        # Ex.
        # Suppose that this is our matrix
        # 0 | 1 | 2 | 3
        # - - - - - - -
        # 4 | 5 | 6 | 7
        # - - - - - - -   
        # 8 | 9 | 10 | 11
        # - - - - - - -
        # 12 | 13 | 14 | 15
        #
        # The output for (0.2, 0.2)
        # 0 | 0 | 0 | 0
        # - - - - - - -
        # 0 | 1 | 0 | 0
        # - - - - - - -
        # 0 | 0 | 0 | 0
        # - - - - - - -
        # 0 | 0 | 0 | 0
        input_coordinates = input_coordinates.long()
        for i in range(input_coordinates.shape[0]):
            distribution[i, input_coordinates[i, 1], input_coordinates[i , 0]] = 1

        return torch.flatten(distribution, 1)

    def convert_to_coordinates(self, input_coordinates):
        """
        Function to convert the input coordinates to a x,y value.
        The input coordinate is a value between [0...., xy_distribution_size**2]
        """
        # To obtain y coordinate -> (input_coordinates[i] / number of sectors
        # ### Check if there is an easier way to calculate x
        # To obtain x coordinate -> ((input_coordinates[i] * number_of_sectors) % (number_of_sectors ** 2)) /  number_of_sectors

        # Examples
        # 0 | 1 | 2 | 3
        # - - - - - - -
        # 4 | 5 | 6 | 7
        # - - - - - - -
        # 8 | 9 | 10 | 11
        # - - - - - - -
        # 12 | 13 | 14 | 15
        # 
        # number_of_sectors = 4 # 4 rows and 4 columns
        # 
        # Ex.
        # input coordinate: 3
        # y = (3 / 4) = 0.75 = int(0.75) = 0 -> row 0
        # x = ((3 * 4) % 16) / 4 = ((12) % 16) / 4 = 12 / 4 = 3 -> col 3
        # 
        # input coordinate: 6
        # y = (6 / 4) = 1.5 = int(1.5) = 1 -> row 1 
        # x = ((6 * 4) % 16) / 4 = ((24) % 16) / 4 = 8 / 4 = 2 -> col 2
        # 
        # input coordinate: 12
        # y = (12 / 4) = 3.0 = int(3.0) = 3 -> row 3
        # x = ((12*4) % 16) / 4 = ((48) % 16) / 4 = 0 / 4 = 0 -> col 0
        number_of_sectors = self.xy_distribution_size

        # First obtain the coordinates of the matrix
        x, y = input_coordinates % number_of_sectors, input_coordinates.div(number_of_sectors,rounding_mode='trunc')

        # Obtain the [x,y] value in [0, 1] range
        x_value = x.true_divide(number_of_sectors)
        y_value = y.true_divide(number_of_sectors)

        return torch.cat((x_value, y_value), dim=1)

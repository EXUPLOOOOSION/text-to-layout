import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

import random
import time
from transformers import AutoTokenizer
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, num_classes, is_training, pad_token=0, max_len=12, freeze_encoder=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token = pad_token
        self.output_l_size = num_classes
        self.is_training = is_training
        self.max_len = max_len
        self.temperature = 0.4
        self.xy_distribution_size = self.decoder.xy_distribution_size
        self.freeze_encoder = freeze_encoder
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
            self.encoder.model.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def eval(self):
        self.is_training = False
        self.decoder.eval()
        self.encoder.eval()
        self.encoder.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    #TODO target_l esta solo aqui porque seq2seq calcula l_match. Mover eso a trainer
    def forward(self, input_ids, attention_mask, input_l=None, input_x=None, input_y=None, input_w=None, input_h=None, input_key_mask=None, target_key_mask=None, target_l = None):
        #TODO? mabe make these parameters
        SOS_class = 1
        EOS_class = 2
        #print("decoder's decoder: ", self.decoder.decoder.is_training)
        # Store the matches
        l_match, total = 0, 0
        if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
        encoder_output = self.encoder(input_ids, attention_mask)
        # Obtain the batch size
        batch_size = encoder_output.size(0)

        
        if self.is_training:
            
            #calculate outputs (automatically made with teacher learning)
            
            outputs_class, class_predictions, xy_out, wh_out, decoder_hidden, xy_coordinates = self.decoder(input_l, input_x, input_y, input_w, input_h, input_key_mask, encoder_output, attention_mask)
            
            
            outputs_bbox = torch.cat((xy_coordinates,wh_out),dim=2)
            xy_prob_dist = xy_out
            #calculate some accuracy stats when training
            #target_key_mask has True for the ones to ignore and viceversa. Masked selec ignores Falses and viceversa. So we inverse target_key_mask to work as we want to with masked_select
            non_padding = torch.logical_not(target_key_mask)
            #taking into account only non-masked answers, how many are equal to the ground truth
            l_correct = class_predictions.eq(target_l).masked_select(non_padding).sum().item()
            l_match += l_correct
            total += non_padding.sum().item()
        else:#evaluation
            with torch.no_grad():
                #initialize targets with all padding. These are not ground truth
                in_l = torch.full((batch_size,self.max_len-1), self.pad_token)
                in_x = torch.zeros(batch_size,self.max_len-1)
                in_y = torch.zeros(batch_size,self.max_len-1)
                in_w = torch.zeros(batch_size,self.max_len-1)
                in_h = torch.zeros(batch_size,self.max_len-1)
                
                #max_len=max_objects+2 (<sos>,<eos>)

                in_l[:,0] = SOS_class

                #initialization of mask. All set to True --> be ignored
                input_key_mask = torch.ones(batch_size, self.max_len-1).bool()
                #dont ignore <sos> :)
                input_key_mask[:, 0] = False
                #to save class probability and xy probability outputs
                outputs_class = torch.zeros(batch_size, self.max_len-1, self.output_l_size)
                xy_prob_dist = torch.zeros(batch_size, self.max_len-1, self.xy_distribution_size * self.xy_distribution_size)
                outputs_bbox = torch.zeros(batch_size, self.max_len-1, 4)#xywh
                class_predictions = torch.zeros(batch_size, self.max_len-1).long()
                unfinished_sequences = torch.ones(batch_size).bool()
                if torch.cuda.is_available():
                    in_l = in_l.cuda()
                    in_x = in_x.cuda()
                    in_y = in_y.cuda()
                    in_w = in_w.cuda()
                    in_h = in_h.cuda()
                    input_key_mask = input_key_mask.cuda()
                    
                    outputs_class = outputs_class.cuda()
                    xy_prob_dist = xy_prob_dist.cuda()
                    outputs_bbox = outputs_bbox.cuda()
                    class_predictions = class_predictions.cuda()
                    unfinished_sequences = unfinished_sequences.cuda()
                    

                for i in range(1,self.max_len-1):# 1 because sos is already entered. max_len-1 because <eos> will forcefully set at the end. so at most we can enter up to max_len-1
                    class_prob, current_class_predictions, xy_prob, predicted_whs, decoder_emb, predicted_xy_coords = self.decoder(in_l, in_x, in_y, in_w, in_h, input_key_mask, encoder_output, attention_mask)
                    class_predictions[unfinished_sequences,i-1] = current_class_predictions[unfinished_sequences,i-1]
                    xy_prob_dist[unfinished_sequences,i-1,:] = xy_prob[unfinished_sequences,i-1,:]
                    outputs_class[unfinished_sequences,i-1,:] = class_prob[unfinished_sequences,i-1,:]
                    in_l[unfinished_sequences,i] = current_class_predictions[unfinished_sequences,i-1]
                    in_x[unfinished_sequences,i] = predicted_xy_coords[unfinished_sequences,i-1,0]
                    in_y[unfinished_sequences,i] = predicted_xy_coords[unfinished_sequences,i-1,1]
                    in_w[unfinished_sequences,i] = predicted_whs[unfinished_sequences,i-1,0]
                    in_h[unfinished_sequences,i] = predicted_whs[unfinished_sequences,i-1,1]
                    #dont ignore the newly added object
                    input_key_mask[:, i] = False
                    #mark any finished sequence as such
                    for j in range(batch_size):
                        if current_class_predictions[j,i] == EOS_class:
                            unfinished_sequences[j] =False
                #if the model hasn't outputted <eos> we put it manually at the end.
                for j in range(batch_size):
                    if class_predictions[j,-2] != EOS_class and class_predictions[j,-2]!= self.pad_token:
                        outputs_class[j,-1,EOS_class] = 1
                        class_predictions[j,-1] = EOS_class
                outputs_bbox[:,:-1,0] = in_x[:,1:]
                outputs_bbox[:,:-1,1] = in_y[:,1:]
                outputs_bbox[:,:-1,2] = in_w[:,1:]
                outputs_bbox[:,:-1,3] = in_h[:,1:]
        # Return all the information
        final_output = {
            "output_class": outputs_class,
            "output_bbox": outputs_bbox,
            "xy_prob_dist": xy_prob_dist,
            "predicted_class":class_predictions,
            "l_match": l_match,
            "total": total
        }
        return final_output
    def generate(self, raw_captions):
        SOS_class = 1
        EOS_class = 2
        encoding = self.tokenizer(raw_captions, return_tensors='pt', padding=True, truncation=True)
        all_inputs_ids = encoding['input_ids']
        all_attention_masks = encoding['attention_mask']
        if torch.cuda.is_available():
            all_inputs_ids = all_inputs_ids.cuda()
            all_attention_masks = all_attention_masks.cuda()
        output = self(all_inputs_ids, all_attention_masks)
        simplified_output = {} #only need actual output, not l_match or total. Clean it if necessary (make class selections, swap dimensions to be batch_size, seq_len)...
        simplified_output['output_class'] = torch.empty(output['output_class'].size(0), self.max_len)
        simplified_output['output_bbox'] = torch.empty(output['output_bbox'].size(0), self.max_len, 4)
        
        simplified_output['output_class'][:,0] = SOS_class
        simplified_output['output_bbox'][:,0,:] = torch.Tensor((0,0,0,0))

        simplified_output['output_class'][:,1:] = torch.argmax(output['output_class'], dim=2)
        simplified_output['output_bbox'][:,1:,:] = output['output_bbox']
        return simplified_output
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if h.shape[0] == 2:
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
    def convert_from_coordinates_to_flat_idx(self,x_coordinates, y_coordinates):
        """
        maps each x,y pair to its corresponding index in a flattened self.xy_distribution_size, self.xy_distribution_size grid
        example: assuming self.xy_distribution_size = 32, (0.5, 0.25) is in the grid coordinates (16,8).
        After flattening the grid ir will have a index of
        Parameters:
            x_coordinates, y_coordinates tensors with the same dimensions. These dimensions are arbitrary. values [0,1]
        Returns:
            Tensor with the same dimension as x_coordinates and y_coordinates with values [0,self.xy_distribution_size**2)
        """
        x_coordinates = torch.mul(x_coordinates, self.xy_distribution_size).floor()# get them from [0,1] to int([0,32])
        y_coordinates = torch.mul(y_coordinates, self.xy_distribution_size).floor()# get them from [0,1] to int([0,32])
        return torch.add(x_coordinates, y_coordinates.mul(self.xy_distribution_size))
    def convert_from_coordinates_flat(self, x_coordinates, y_coordinates):
        #one hot version of convert_from_coordinates_to_flat_idx where output is of size [k,self.xy_distribution_size**2] and the last dimension is filled with 0's except a 1 in that coordinate's index
        idx = self.convert_from_coordinates_to_flat_idx(x_coordinates, y_coordinates).unsqueeze(1)
        ind = np.indices(idx.size())
        ind[-1] = idx
        result[ind] = 1
        return result

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
"""previous evaluation code:
else:#evaluation
            #initialize targets with all padding. These are not ground truth
            in_l = torch.full((1,self.max_len), self.pad_token)
            in_x = torch.zeros(1,self.max_len)
            in_y = torch.zeros(1,self.max_len)
            in_w = torch.zeros(1,self.max_len)
            in_h = torch.zeros(1,self.max_len)
            
            #max_len=max_objects+2 (<sos>,<eos>)

            in_l[:,0] = SOS_class

            #initialization of padding. All set to True --> be ignored
            target_key_mask = torch.ones(1, self.max_len).bool()
            #dont ignore <sos> :)
            target_key_mask[:, 0] = False
            #to save class probability and xy probability outputs
            outputs_class = torch.zeros(batch_size, self.max_len, self.output_l_size)
            xy_prob_dist = torch.zeros(batch_size, self.max_len, self.xy_distribution_size * self.xy_distribution_size)
            outputs_bbox = torch.zeros(batch_size, self.max_len, 4)#xywh

            if torch.cuda.is_available():
                in_l = in_l.cuda()
                in_x = in_x.cuda()
                in_y = in_y.cuda()
                in_w = in_w.cuda()
                in_h = in_h.cuda()
                target_key_mask = target_key_mask.cuda()
                
                outputs_class = outputs_class.cuda()
                xy_prob_dist = xy_prob_dist.cuda()
                outputs_bbox = outputs_bbox.cuda()
                

            for seq in range(batch_size):
                for i in range(1,self.max_len-1):# 1 because sos is already entered. max_len-1 because <eos> will forcefully set at the end. so at most we can enter up to max_len-1
                    
                    class_prob, xy_prob, predicted_whs, decoder_emb, predicted_xy_coords = self.decoder(in_l, in_x, in_y, in_w, in_h, target_key_mask, encoder_output[seq].unsqueeze(0), attention_mask[seq].unsqueeze(0))
                    #save output and set last output as new object in target
                    class_predictions = class_prob.argmax(2)
                    xy_prob_dist[seq,i,:] = xy_prob[:,-1,:]
                    outputs_class[seq,i,:] = class_prob[:,-1,:]
                    in_l[:,i] = class_predictions[:,-1]
                    in_x[:,i] = predicted_xy_coords[:,-1,0]
                    in_y[:,i] = predicted_xy_coords[:,-1,1]
                    in_w[:,i] = predicted_whs[:,-1,0]
                    in_h[:,i] = predicted_whs[:,-1,1]
                    #dont ignore the newly added object
                    target_key_mask[:, i] = False
                #if the model hasn't outputted <eos> we put it manually at the end.
                if i == self.max_len-2 and in_l[0,i] != EOS_class:#if the loop ends without breaking out, i will have self.max_len-2 value.
                    in_l[0,-1] = EOS_class # equal to in_l[0,self.max_len-1] = EOS_class
                #reset temporary tgt staorages for next sequence
                outputs_bbox[seq,:,0] = in_x
                outputs_bbox[seq,:,1] = in_y
                outputs_bbox[seq,:,2] = in_w
                outputs_bbox[seq,:,3] = in_h
                in_l[1:] = self.pad_token
                in_x[1:] = 0
                in_y[1:] = 0
                in_w[1:] = 0
                in_h[1:] = 0
                target_key_mask[1:] = True
"""
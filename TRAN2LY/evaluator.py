from tqdm import tqdm

import torch
from torch.nn import functional as F

import json

import collections

import numpy as np

class Evaluator():
    def __init__(self, seq2seq, loss_class, loss_bbox_xy, loss_bbox_wh, vocab, gaussian_dict=None, validator_output_path="./", save_output=False, verbose=False, name="DEVELOPMENT"):
        self.seq2seq = seq2seq
        self.loss_class = loss_class
        self.loss_bbox_xy = loss_bbox_xy
        self.loss_bbox_wh = loss_bbox_wh
        self.vocab = vocab
        self.VERBOSE = verbose
        self.save_output = save_output
        self.validator_output_path = validator_output_path if validator_output_path[-1] != "/" else validator_output_path[:-1]
        self.gaussian_dict = gaussian_dict
        self.name = name
    
    def convert_objects_to_list(self, coco_boxes, coco_ids, coco_to_img):
        """
        Function to reorganize the bounding boxes and classes
        """
        # Longest sequence. The minimum value is 3 <sos> class <eos>
        
        target_l_variables_list, target_x_variables_list, target_y_variables_list, target_w_variables_list, target_h_variables_list = [], [], [], [], []
        
        target_coco_to_img_list = []
        # First split the class/bounding box in list according to which image belongs each one
        for i in range(coco_to_img.max()+1):
            object_idx = np.where(coco_to_img.cpu().numpy()==i)[0]
            target_l_variables_list.append(coco_ids[object_idx[0]: object_idx[-1]+1])
            target_x_variables_list.append(coco_boxes[:, 0][object_idx[0]: object_idx[-1]+1])
            target_y_variables_list.append(coco_boxes[:, 1][object_idx[0]: object_idx[-1]+1])
            target_w_variables_list.append(coco_boxes[:, 2][object_idx[0]: object_idx[-1]+1])
            target_h_variables_list.append(coco_boxes[:, 3][object_idx[0]: object_idx[-1]+1])
            target_coco_to_img_list.append(coco_to_img[object_idx[0]: object_idx[-1]+1])
        maximum = self.seq2seq.max_len
        # here we add padding according to the caption that has the maximum number of objects
        target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables = [], [], [], [], []
        target_coco_to_img = []
        for i in range(len(target_l_variables_list)):
            s = len(target_l_variables_list[i])
            target_l, target_x, target_y, target_w, target_h, target_c_t_i = torch.zeros(maximum, dtype=torch.int64), torch.zeros(maximum), torch.zeros(maximum), torch.zeros(maximum), torch.zeros(maximum), torch.empty(maximum).fill_(i)
            target_l[:s], target_x[:s], target_y[:s], target_w[:s], target_h[:s], target_c_t_i[:s] = target_l_variables_list[i], target_x_variables_list[i], target_y_variables_list[i], target_w_variables_list[i], target_h_variables_list[i], target_coco_to_img_list[i]
            target_l_variables.append(target_l)
            target_x_variables.append(target_x)
            target_y_variables.append(target_y)
            target_w_variables.append(target_w)
            target_h_variables.append(target_h)
            target_coco_to_img.append(target_c_t_i)
        target_l_variables = torch.stack(target_l_variables)
        target_x_variables = torch.stack(target_x_variables)
        target_y_variables = torch.stack(target_y_variables)
        target_w_variables = torch.stack(target_w_variables)
        target_h_variables = torch.stack(target_h_variables)
        target_coco_to_img = torch.stack(target_coco_to_img)
        # Convert the x, y coordinates to coordinates in the grid
        target_x_coordinates = torch.zeros((target_x_variables.shape[0], target_x_variables.shape[1]))
        target_y_coordinates = torch.zeros((target_y_variables.shape[0], target_y_variables.shape[1]))
        for i in range(target_x_variables.shape[1]):
            coordinates = self.convert_to_coordinates(self.convert_from_coordinates(torch.cat((target_x_variables[:, i].unsqueeze(1), target_y_variables[:, i].unsqueeze(1)), dim=1)).argmax(1).unsqueeze(1))

            target_x_coordinates[:, i] = coordinates[:, 0]
            target_y_coordinates[:, i] = coordinates[:, 1]

        target_x_variables = target_x_coordinates
        target_y_variables = target_y_coordinates
        # Convert to cuda   
        if torch.cuda.is_available():
            target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables, target_coco_to_img = target_l_variables.cuda(), target_x_variables.cuda(), target_y_variables.cuda(), target_w_variables.cuda(), target_h_variables.cuda(), target_coco_to_img.cuda()
        return target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables, target_coco_to_img

    def convert_index_to_word(self, lxywh):
        """
        This function converts indexes of coco to the corresponding word
        """
        return [(self.vocab['index2wordCoco'][l[-1]]) for l in lxywh]
    
    def clean_output(self, l, x, y, w, h):
        """
        This function cleans the output.
        1. Converts tensors to float
        2. Remove padding, <sos>, <eos>, <ukn> tokens
        """
        output = []
        for i in range(len(l)):
            if l[i] in [0, 1, 2, 3]:
                break
            output.append([torch.clamp(x[i], 0, 1).item(), torch.clamp(y[i], 0, 1).item(), torch.clamp(w[i], 0, 1).item(), torch.clamp(h[i], 0, 1).item(), int(self.vocab['index2wordint'][int(l[i])])])
        return output
    
    def filter_redundant_labels(self, predicted):
        """
        This function filters redundant labels using the gaussian dict.
        """
        ls = np.array([i[-1] for i in predicted])
        # filter redundant labels
        counter = collections.Counter(ls)
        unique_labels, label_counts = list(counter.keys()), list(counter.values())
        kept_indices = []
        for label_index in range(len(unique_labels)):
            label = unique_labels[label_index]
            label_num = label_counts[label_index]
            # sample an upper-bound threshold for this label
            mu, sigma = self.gaussian_dict[label]
            threshold = max(int(np.random.normal(mu, sigma, 1)), 2)
            old_indices = np.where(ls == label)[0].tolist()
            new_indices = old_indices
            if threshold < len(old_indices):
                new_indices = old_indices[:threshold]
            kept_indices += new_indices
        kept_indices.sort()
        final_output = []
        for i in range(len(predicted)):
            if i in kept_indices:
                final_output.append(predicted[i])
        return final_output
        
    @torch.no_grad()
    def evaluate(self, dl, ds, epoch, output_folder):
        
        # Set the model to validation
        self.seq2seq.eval()
        self.seq2seq.teacher_learning = False

        epoch_loss, epoch_lloss, epoch_bloss_xy, epoch_bloss_wh, epoch_bloss_xy_MSE = 0, 0, 0, 0, 0
        
        bbox_and_ls_outputs = {}
        step = 0
        for batch in tqdm(dl):
            
            captions_padded, captions_length, coco_boxes, coco_ids, coco_to_img, all_idx = batch
            target_l_raw, target_x_raw, target_y_raw, target_w_raw, target_h_raw, target_coco_to_img = self.convert_objects_to_list(coco_boxes, coco_ids, coco_to_img)
            outputs = self.seq2seq(captions_padded, captions_length, target_l=target_l_raw, target_x=target_x_raw, target_y=target_y_raw, target_w=target_w_raw, target_h=target_h_raw)
            
            # Obtain the loss of the class
            output_dim = outputs['output_class'].shape[-1]
            output_class_clean = outputs['output_class'][1:]
            outputs_class =  output_class_clean.view(-1, output_dim)
            # output_class = [trg len*batch size, output dim]
            
            target_l = target_l_raw[:, 1:]
            target_l_cleaned = torch.transpose(target_l, 0, 1).contiguous().view(-1)
            # target_l = [batch size*output dim]

            class_loss = self.loss_class(outputs_class, target_l_cleaned)

            # Obtain the loss of the the bounding box
            # cross entropy xy
            output_xy_dim = outputs['outputs_bbox_xy'].shape[-1]
            output_xy_clean = outputs['outputs_bbox_xy'][1:]
            outputs_xy = output_xy_clean.view(-1, output_xy_dim)

            target_xy = self.convert_from_coordinates_to_flat_idx(target_x_raw, target_y_raw).transpose(0,1)[1:].contiguous().view(-1).long()
            xy_prob_loss = self.loss_bbox_xy(outputs_xy, target_xy)
            
            # MSE of xywh
            output_dim = outputs['output_bbox'].shape[-1]
            output_bbox_clean = outputs['output_bbox'][1:]
            outputs_bbox =  output_bbox_clean.view(-1, output_dim)

            target_x, target_y, target_w, target_h, target_coco_to_img = target_x_raw[:, 1:], target_y_raw[:, 1:], target_w_raw[:, 1:], target_h_raw[:, 1:], target_coco_to_img[:, 1:]

            # Concatenate the [x, y, w, h] coordinates
            target_xywh = torch.zeros(outputs_bbox.shape)
            target_coco = torch.zeros(outputs_bbox.shape[0])

            if torch.cuda.is_available():
                target_xywh = target_xywh.cuda()
                target_coco = target_coco.cuda()
            trg_len = target_l_raw.shape[1]
            batch_size = target_l_raw.shape[0]
            for di in range(trg_len-1):
                target_xywh[di*batch_size:di*batch_size+batch_size, 0] = target_x[:, di]
                target_xywh[di*batch_size:di*batch_size+batch_size, 1] = target_y[:, di]
                target_xywh[di*batch_size:di*batch_size+batch_size, 2] = target_w[:, di]
                target_xywh[di*batch_size:di*batch_size+batch_size, 3] = target_h[:, di]
                target_coco[di*batch_size:di*batch_size+batch_size] = target_coco_to_img[:, di]

            output_compare = torch.zeros(len(output_class_clean), batch_size)
            for di in range(len(output_class_clean)):
                # Step di
                output_compare[di] = output_class_clean[di].argmax(1)

            # After finding <pad>, <ukn>, <sos> or <eos> token convert the remaining numbers to 0
            for di in range(len(output_class_clean)):
                for dj in range(batch_size):
                    if output_compare[di, dj] <= 3:
                        output_compare[di:, dj] = 0

            if torch.cuda.is_available():
                output_compare = output_compare.cuda()

            # mask for padding and <eos> of target_xywh and output_bbox
            flatten = torch.flatten(output_compare)

            mask = (flatten != 0).float() * (flatten != 2).float()

            # Convert the mask to cuda
            if torch.cuda.is_available():
                mask = mask.cuda()

            # Obtain the losses
            wh_loss, xy_loss = self.loss_bbox_wh(outputs_bbox, target_xywh, mask=mask)
            wh_loss, xy_loss =  wh_loss * 10, xy_loss * 10
            
            # Total loss
            loss = class_loss + wh_loss + xy_prob_loss
            
            epoch_loss += loss.item()
            epoch_lloss += class_loss.item()
            epoch_bloss_wh += wh_loss.item()
            epoch_bloss_xy += xy_prob_loss.item()
            epoch_bloss_xy_MSE += xy_loss.item()
            
            # if save_output clean the prediction and add the output to a dictionary
            if self.VERBOSE or self.save_output:                
                for idx in range(max(coco_to_img)+1):
                    image_id = ds.get_image_id(all_idx[idx])

                    cleaned_output_original = self.clean_output(target_l[idx], target_x[idx], target_y[idx], target_w[idx], target_h[idx])
                    cleaned_output_predicted = self.clean_output(output_compare[:, idx], output_bbox_clean[:, idx, 0], output_bbox_clean[:, idx, 1], output_bbox_clean[:, idx, 2], output_bbox_clean[:, idx, 3])
                    if self.gaussian_dict != None:
                        cleaned_output_predicted = self.filter_redundant_labels(cleaned_output_predicted)
                    
                    if self.VERBOSE:
                        print("Results")
                        print("Caption", ds.image_id_to_caption[image_id])
                        print("Triples", ds.image_id_to_triples[image_id])
                        print("Original",  self.convert_index_to_word(cleaned_output_original))
                        print("predicted2", self.convert_index_to_word(cleaned_output_predicted))
                        print(" ")
                    
                    # If more than one objects are predicted save the prediction
                    if self.save_output:
                        if len(cleaned_output_predicted) > 0:
                            bbox_and_ls_outputs[image_id] = cleaned_output_predicted
            step += 1
        # Save the average losses
        with open(output_folder + "/" + self.name + "losses" + str(epoch)+ ".txt", "w") as f:
            info = "{} {} {} {}".format(str(epoch_lloss/(len(dl))), str(epoch_bloss_xy/(len(dl))), str(epoch_bloss_wh/(len(dl))), str(epoch_bloss_xy_MSE/(len(dl))))
            f.write(info)
        
        # Save the output (bbox and class for each picture) in a json file 
        if self.save_output: 
            with open(self.validator_output_path + "/" + self.name + "epoch" + str(epoch)+".json", "w") as f:
                json.dump(bbox_and_ls_outputs, f)
        
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
        x_coordinates = torch.mul(x_coordinates, self.seq2seq.xy_distribution_size).floor()# get them from [0,1] to int([0,32])
        y_coordinates = torch.mul(y_coordinates, self.seq2seq.xy_distribution_size).floor()# get them from [0,1] to int([0,32])
        return torch.add(x_coordinates, y_coordinates.mul(self.seq2seq.xy_distribution_size))
    def convert_from_coordinates(self, input_coordinates):
        distribution = torch.zeros((input_coordinates.shape[0], self.seq2seq.xy_distribution_size, self.seq2seq.xy_distribution_size), device=input_coordinates.device, dtype=torch.int)
        # We make a matrix to make the operations easier
        # distribution = [batch_size, xy_distribution_size, xy_distribution_size]

        # Convert coordinates from range [0, 1] to range [0, xy_distribution_size]
        input_coordinates[:, 0] = (input_coordinates[: ,0] * self.seq2seq.xy_distribution_size).clamp(0, self.seq2seq.xy_distribution_size-1)
        input_coordinates[:, 1] = (input_coordinates[: ,1] * self.seq2seq.xy_distribution_size).clamp(0, self.seq2seq.xy_distribution_size-1)

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
        number_of_sectors = self.seq2seq.xy_distribution_size

        # First obtain the coordinates of the matrix
        x, y = input_coordinates % number_of_sectors, input_coordinates.div(number_of_sectors,rounding_mode='trunc')
        # Obtain the [x,y] value in [0, 1] range
        x_value = x.true_divide(number_of_sectors)
        y_value = y.true_divide(number_of_sectors)

        return torch.cat((x_value, y_value), dim=1)
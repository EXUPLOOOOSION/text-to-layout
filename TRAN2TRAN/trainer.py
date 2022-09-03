from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from evaluator import Evaluator

from transformers import get_linear_schedule_with_warmup
import numpy as np
import traceback
def remove_eos_unk(target_l):
        EOS_class = 2
        UNK_class = 3
        new_target_l = target_l.clone()
        new_target_l[new_target_l == EOS_class] = 0
        new_target_l[new_target_l == UNK_class] = 0
        return new_target_l
class SupervisedTrainer():

    def __init__(self, seq2seq, vocab, epochs, print_every, loss_class, loss_bbox_xy, loss_bbox_wh, batch_size, hidden_size, max_lr, opt_func, steps_per_epoch, checkpoints_path="./checkpoints", gaussian_dict=None, validator_output_path="./evaluator_output", save_output=False):
        self.seq2seq = seq2seq
        self.max_len = self.seq2seq.max_len
        self.vocab = vocab
        self.epochs = epochs
        self.print_every = print_every
        self.loss_class = loss_class
        self.loss_bbox_xy = loss_bbox_xy
        self.loss_bbox_wh = loss_bbox_wh
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_lr = max_lr
        self.opt_func = opt_func
        self.checkpoints_path = checkpoints_path if checkpoints_path[-1] != "/" else checkpoints_path[:-1]
        self.pad_token = self.seq2seq.pad_token
        self.xy_distribution_size = self.seq2seq.xy_distribution_size
        # Optimizer and scheduler
        # self.optimizer = self.opt_func(self.seq2seq.parameters(), self.max_lr, weight_decay=1e-4)
        # self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.max_lr, epochs=self.epochs, steps_per_epoch=steps_per_epoch)
        self.optimizer = self.opt_func(self.seq2seq.parameters(), lr=self.max_lr, eps=1e-8)
        self.sched = get_linear_schedule_with_warmup(self.optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=steps_per_epoch*self.epochs)

        # Generate the evaluator
        self.gaussian_dict = gaussian_dict
        self.validator_output_path = validator_output_path
        self.save_output = save_output
        self.evaluator = Evaluator(self.seq2seq, self.loss_class, self.loss_bbox_xy, self.loss_bbox_wh, self.vocab, gaussian_dict=self.gaussian_dict, validator_output_path=self.validator_output_path, save_output=self.save_output, verbose=False)
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

        # here we add padding according to the caption that has the self.max_len number of objects
        target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables = [], [], [], [], []
        target_coco_to_img = []
        target_key_mask_variables = []
        for i in range(len(target_l_variables_list)):
            s = len(target_l_variables_list[i])
            #if s>2:#filter out sentences without objects #TODO this should be done in dataset
            target_l, target_x, target_y, target_w, target_h, target_c_t_i = torch.full((self.max_len,), self.pad_token, dtype=torch.int64), torch.zeros(self.max_len), torch.zeros(self.max_len), torch.zeros(self.max_len), torch.zeros(self.max_len), torch.empty(self.max_len).fill_(i)
            target_l[:s], target_x[:s], target_y[:s], target_w[:s], target_h[:s], target_c_t_i[:s] = target_l_variables_list[i], target_x_variables_list[i], target_y_variables_list[i], target_w_variables_list[i], target_h_variables_list[i], target_coco_to_img_list[i]
            #ones --> True to ignore
            target_key_mask = torch.ones(self.max_len).bool()
            #False to have them into account. both <sos> and <eos> are attended
            
            target_key_mask[:s] = False
            target_l_variables.append(target_l)
            target_x_variables.append(target_x)
            target_y_variables.append(target_y)
            target_w_variables.append(target_w)
            target_h_variables.append(target_h)
            target_coco_to_img.append(target_c_t_i)
            target_key_mask_variables.append(target_key_mask)
        target_l_variables = torch.stack(target_l_variables)
        target_x_variables = torch.stack(target_x_variables)
        target_y_variables = torch.stack(target_y_variables)
        target_w_variables = torch.stack(target_w_variables)
        target_h_variables = torch.stack(target_h_variables)
        target_key_mask_variables = torch.stack(target_key_mask_variables)
        target_coco_to_img = torch.stack(target_coco_to_img)
        
        # Convert the x, y coordinates to coordinates in the grid
        #round coordinates from any number in [0,1) to {0/32, 1/32, 2/32,...,31/32} (32 possibilities) where 32 = xy_distribution_size
        #example: 0.23658 --> 7/32 = 0.21875. Steps: 0.23658*32 = 7.57056 --floor (and clamp)--> 7 --div--> 7/32
        
        target_x_variables = torch.mul(target_x_variables,self.xy_distribution_size).floor().clamp(0, self.seq2seq.xy_distribution_size-1).div(self.xy_distribution_size)
        target_y_variables = torch.mul(target_y_variables,self.xy_distribution_size).floor().clamp(0, self.seq2seq.xy_distribution_size-1).div(self.xy_distribution_size)
        
        # Convert to cuda   
        if torch.cuda.is_available():
            target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables, target_coco_to_img,target_key_mask_variables = target_l_variables.cuda(), target_x_variables.cuda(), target_y_variables.cuda(), target_w_variables.cuda(), target_h_variables.cuda(), target_coco_to_img.cuda(),target_key_mask_variables.cuda()
        return target_l_variables, target_x_variables, target_y_variables, target_w_variables, target_h_variables, target_coco_to_img, target_key_mask_variables

    def train_batch(self, batch_step, input_ids, attention_mask, coco_boxes, coco_ids, coco_to_img):
        """
        This function trains a batch
        """
        # Obtain the targets
        data_l, data_x, data_y, data_w, data_h, target_coco_to_img, data_key_mask = self.convert_objects_to_list(coco_boxes, coco_ids, coco_to_img)
        #evaluation_obj_mask only masks padded ground truth
        #target_key_mask masks padded input AND <eos>. This is used as the decoder target_key_mask because <eos> is never an input to the decoder, the last input is the object before <eos>
        target_l_raw, target_x_raw, target_y_raw, target_w_raw, target_h_raw = data_l[:,1:], data_x[:,1:], data_y[:,1:], data_w[:,1:], data_h[:,1:]
        input_l, input_x, input_y, input_w, input_h = data_l[:,:-1], data_x[:,:-1], data_y[:,:-1], data_w[:,:-1], data_h[:,:-1]
        input_key_mask = data_key_mask[:,:-1]
        target_key_mask_raw = data_key_mask[:,1:]
        
        # calculate ground truth distribution of xy probabilities
        target_xy_raw = self.convert_from_coordinates_to_flat_idx(target_x_raw, target_y_raw)
        if torch.cuda.is_available():
            target_xy_raw=target_xy_raw.cuda()
            input_l, input_x, input_y, input_w, input_h, input_key_mask = input_l.cuda(), input_x.cuda(), input_y.cuda(), input_w.cuda(), input_h.cuda(), input_key_mask.cuda()
            target_l_raw, target_x_raw, target_y_raw, target_w_raw, target_h_raw, target_key_mask_raw = target_l_raw.cuda(), target_x_raw.cuda(), target_y_raw.cuda(), target_w_raw.cuda(), target_h_raw.cuda(), target_key_mask_raw.cuda()
        #get predictions
        outputs = self.seq2seq(input_ids, attention_mask,input_l=input_l, input_x=input_x, input_y=input_y, input_w=input_w, input_h=input_h, input_key_mask=input_key_mask,target_key_mask=target_key_mask_raw, target_l=target_l_raw)
        
        # final_output keys = [
        #   "output_class": [batch_size, seq_len, self.num_classes] probability distribution of class
        #   "output_bbox": [batch_size, seq_len, 4] xywh of each bbox
        #   "xy_prob_dist": [batch_size, seq_len, self.xy_distribution_size**2] probability distribution of positioning of bbox
        #   "predicted_class": [batch_size, seq_len] ...
        #   "l_match", "total": int.
        #   "coco_to_img"]
        # Obtain the accuracy of correctly predicted classes
        total, l_match = outputs['total'], outputs['l_match']
        if batch_step % self.print_every == 0:
            if total == 0:
                l_accuracy = float('nan')
            else:
                l_accuracy = l_match / total
            print('l_accuracy: {}. l_match: {}. total: {}'.format(l_accuracy, l_match, total))
        
        # Obtain the loss of the class
        output_dim = outputs['output_class'].shape[-1]
        outputs_class = outputs['output_class']
        outputs_class = outputs_class.reshape(-1, output_dim)
        # output_class = [trg len*batch size, output dim]
        target_l = target_l_raw.reshape(-1)
        # target_l = [batch size*trg len]
        #if you dont remove <eos> it just spams <eos>'s ):
        target_l = remove_eos_unk(target_l)
        class_loss = self.loss_class(outputs_class, target_l)
        
        # Obtain the loss of the the bounding box
        # cross entropy xy
        outputs_xy_prob_dist_dim = outputs['xy_prob_dist'].shape[-1] #should be 1024 for xy_distribution_size=32

        outputs_xy_prob_dist = outputs['xy_prob_dist']
        outputs_xy_prob_dist = outputs_xy_prob_dist.reshape(-1, outputs_xy_prob_dist_dim)

        target_xy = target_xy_raw.reshape(-1).long()
        xy_prob_loss = self.loss_bbox_xy(outputs_xy_prob_dist, target_xy)
        

        # MSE of xywh
        outputs_bbox_dim = outputs['output_bbox'].shape[-1]
        outputs_bbox = outputs['output_bbox']
        outputs_bbox =  outputs_bbox.reshape(-1, outputs_bbox_dim)
        target_x, target_y, target_w, target_h = target_x_raw, target_y_raw, target_w_raw, target_h_raw
        
        # Concatenate the [x, y, w, h] coordinates
        target_bbox = torch.stack((target_x, target_y, target_w, target_h), dim=2)
        target_bbox = target_bbox.reshape(-1,4)
        
        if torch.cuda.is_available():
            target_bbox = target_bbox.cuda()

        # mask for padding of target_bbox and outputs_bbox
        mask = target_key_mask_raw == False # target_key_mask has True for ignore and False for attend. We need the opposite
        mask = mask.reshape(-1)
        # Convert the mask to cuda
        if torch.cuda.is_available():
            mask = mask.cuda()
        # Obtain the losses
        wh_loss, xy_loss = self.loss_bbox_wh(outputs_bbox, target_bbox, mask=mask)
        wh_loss, xy_loss =  wh_loss * 10, xy_loss * 10

        # Total loss
        loss = class_loss + wh_loss + xy_prob_loss

        self.optimizer.zero_grad()
        
        loss.backward()
        
        nn.utils.clip_grad_value_(self.seq2seq.parameters(), 0.1)

        self.optimizer.step()
        
        self.sched.step()

        return class_loss.item(), wh_loss.item(), xy_prob_loss.item(), xy_loss.item()

    def train_epoches(self, train_loader, train_ds, val_loader, val_ds, start_epoch=0):
        
        steps_per_epoch = int(round(len(train_loader)))
        step = 0
        total_steps = steps_per_epoch * self.epochs
        
        for epoch in range(start_epoch, start_epoch + self.epochs):
            torch.cuda.empty_cache()
            # Set the model to training. seq2seq has into account whether the encoder is frozen or not
            self.seq2seq.train()
            
            
            epoch_loss_total = 0  # Reset every epoch
            epoch_lloss_total = 0
            epoch_bloss_xy_total = 0
            epoch_bloss_wh_total = 0
            epoch_bloss_xy_MSE_total = 0

            print("Training epoch", epoch+1)
            # Train the model
            for batch in tqdm(train_loader):
                input_ids, attention_mask, coco_boxes, coco_ids, coco_to_img, all_idx = batch

                lloss, bloss_wh, bloss_xy, bloss_xy_MSE = self.train_batch(step, input_ids, attention_mask, coco_boxes, coco_ids, coco_to_img)
                
                epoch_loss_total += lloss + bloss_xy + bloss_wh
                epoch_lloss_total += lloss
                epoch_bloss_xy_total += bloss_xy
                epoch_bloss_wh_total += bloss_wh
                epoch_bloss_xy_MSE_total += bloss_xy_MSE
                
                if step % self.print_every == 0:
                    print('step: ', step)
                    
                    print_lloss_total = 0
                    print_bloss_xy_total = 0
                    print_bloss_wh_total = 0
                    print_loss_total = 0
                    print_bloss_xy_MSE_total = 0
                    print('{}/{} Progress: {}. {}: {}. {}:{}. {}:{}. {}:{}'.format(
                        step,
                        steps_per_epoch,
                        step / total_steps * 100,
                        'Cross Entropy Loss',
                        lloss,
                        'bbox XY loss',
                        bloss_xy,
                        'bbox WH loss',
                        bloss_wh,
                        "bbox XY MSE loss",
                        bloss_xy_MSE
                    ))
                step += 1
            
            # Save the epoch
            torch.save(self.seq2seq.state_dict(), self.checkpoints_path + '/amr-gan'+str(epoch)+'.pth')
            
            # Save the average losses
            with open(self.checkpoints_path + "/TRAININGlosses" + str(epoch)+ ".txt", "w") as f:
                info = "{} {} {} {}".format(str(epoch_lloss_total/(len(train_loader))), str(epoch_bloss_xy_total/(len(train_loader))), str(epoch_bloss_wh_total/(len(train_loader))), str(epoch_bloss_xy_MSE_total/(len(train_loader))))
                f.write(info)
                
            # Evaluate. 99% of the works but sometimes when sampling we get +inf and throws and exception. This is just to keep it training.
            try:
                self.evaluator.evaluate(val_loader, val_ds, epoch, self.checkpoints_path)
            except:
                traceback.print_exc()
                pass
    def convert_from_coordinates_to_flat_idx(self,x_coordinates, y_coordinates):
        """
        maps each x,y pair to its corresponding index in a flattened self.xy_distribution_size, self.xy_distribution_size grid
        example: assuming self.xy_distribution_size = 32, (0.5, 0.25) is (16,8) in the grid coordinates.
        After flattening the grid it will have a index of 8*32+16 = 272.
        The y valueis multiplied by 32 because it is the height and thus has to "skip" y rows.
        Parameters:
            x_coordinates, y_coordinates tensors with the same dimensions. These dimensions are arbitrary. values [0,1]
        Returns:
            Tensor with the same dimension as x_coordinates and y_coordinates with values [0,self.xy_distribution_size**2)
        """
        x_coordinates = torch.mul(x_coordinates, self.xy_distribution_size).floor().clamp(0, self.xy_distribution_size-1)# get them from [0,1] to int([0,31])
        y_coordinates = torch.mul(y_coordinates, self.xy_distribution_size).floor().clamp(0, self.xy_distribution_size-1)# get them from [0,1] to int([0,31])
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
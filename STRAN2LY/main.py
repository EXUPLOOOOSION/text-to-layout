from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from torch.optim import AdamW

from model.sentenceEncoder import SentenceEncoder
from model.decoderRNN import DecoderRNN
from model.seq2seq import Seq2Seq
from model import config_loader
from evaluator import Evaluator
from loss import bbox_loss

from dataset import CocoDataset, Collator
from gpu import DeviceDataLoader, get_default_device, to_device
from trainer import SupervisedTrainer

import math

import numpy as np

from tqdm import tqdm

import collections

import os
import traceback

import pickle

import shutil


# Dataloader
BATCH_SIZE = 32
NUM_WORKERS = 4
SHUFFLE = True
PIN_MEMORY = True

#encoder-decoder
CONFIG_FILE_PATH = './checkpoints/unfrozen/model_config.json'#'./checkpoints/current_config.json'
GPU_INDEX = 0

# Dataset hyperparameters
UQ_CAP = True # Use one caption or all the captions. Values: False -> All the captions. True -> One caption
NORMALIZE_INPUT = True # Normalize the pictures to range [0, 1].

# Training
STARTING_EPOCH = -1 #epoch to start  training back from. The STARTING_EPOCH th model, ie. model with index STARTING_EPOCH-1 will be loaded from CHECKPOINTS_PATH. If STARTING_EPOCH=0 or less no epoch will be loaded.
EPOCHS = 100 # Number of epochs to train
PRINT_EVERY = 500 # Print information about the model every n steps
IS_TRAINING = False # Set the model to training or validation. Values: True -> Training mode. False -> Validation mode
CHECKPOINTS_PATH = "./checkpoints/unfrozen" # Path to save the epochs and average losses
LEARNING_RATE = 5e-5

# Validation
CALCULATE_GAUSS_DICT = True # Gauss dictionary with means and std for the objects in the dataset. Values: True -> calculates and saves the gaussian dict. False -> Uses the file located at GAUSS_DICT_PATH  
GAUSS_DICT_PATH = "./data/gaussian_dict_full.npy" # Path to the gauss dict
VALIDATION_OUTPUT = "./evaluator_output/unfrozen" # Path to save the output (bbox and class for each picture)
SAVE_OUTPUT = True # Whether to save or not the output (bbox and class for each picture) when validating. Values: True -> the output is saved. False -> The output is not saved
EPOCHS_VALIDATION = [99]#list(range(0,EPOCHS,3)) # Number of the epoch to validate

#Paths to the training, development and validation dataset
DATASET_PATH_TRAIN = "./data/datasets/AMR2014train-dev-test/GraphTrain.json"
#DATASET_PATH_TRAIN = "./data/datasets/COCO/annotations/captions_train2014.json"
INSTAN_PATH_TRAIN = "./data/datasets/COCO/annotations/instances_train2014.json"
WHITE_LIST_PATH_TRAIN = None #"./data/datasets/COCO/annotations/train_white_list.json"

DATASET_PATH_DEV = "./data/datasets/AMR2014train-dev-test/GraphDev.json"
#DATASET_PATH_DEV = "./data/datasets/COCO/annotations/captions_train2014.json"
INSTAN_PATH_DEV = "./data/datasets/COCO/annotations/instances_train2014.json"
WHITE_LIST_PATH_DEV = None #"./data/datasets/COCO/annotations/dev_white_list.json"

DATASET_PATH_VAL = "./data/datasets/AMR2014train-dev-test/GraphTest.json"
INSTAN_PATH_VAL = "./data/datasets/COCO/annotations/instances_val2014.json"

def generate_dataset(dataset_path_train, instan_path_train, white_list_train, dataset_path_test, instan_path_test, white_list_test,
                    normalize_input=True, uq_cap=False, max_objects=10,
                    shuffle=True, num_workers=4, pin_memory=True, batch_size=16, idx2word=None, word2idx=None):

    # Create the dataset
    print("Loading training dataset")
    train_ds = CocoDataset(dataset_path_train, instan_path_train, white_list=white_list_train, normalize=normalize_input, uq_cap=uq_cap, max_objects=max_objects)

    if IS_TRAINING:
        print("Counting valid objects...")
        for i in tqdm(range(len(train_ds))):
            for j in train_ds.get_coco_objects_tensor(i)[1]:
                if j == 1 or j == 2:
                    continue
                train_ds.vocab['word2count'][train_ds.vocab['index2word'][j.item()]] += 1
                
    print("Loading validation dataset")
    val_ds = CocoDataset(dataset_path_test, instan_path_test, white_list=white_list_test, normalize=normalize_input, vocab=train_ds.vocab, uq_cap=uq_cap, max_objects=max_objects)

    print("Train length:", len(train_ds))
    print("Validation length:", len(val_ds))

    coco_collate_fn = Collator()
    # Generate data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=coco_collate_fn)
    val_dl = DataLoader(val_ds, batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=coco_collate_fn)

    # send dataset to GPU if available
    device = get_default_device()
    #train_dl = DeviceDataLoader(train_dl, device)
    #val_dl = DeviceDataLoader(val_dl, device)

    return train_dl, val_dl, train_ds.vocab, train_ds, val_ds

def generate_losses(vocab, train_ds):
    """
    Function to generate the losses
    """
    # Class loss
    total_sum = sum(vocab['word2count'].values())
    weight = torch.zeros(len(vocab['word2index']))
    for word in vocab['word2index']:    
        index = vocab['word2index'][word]
        weight[index] = (1 - (vocab['word2count'][word]/total_sum))
    
    weight[0], weight[3] = 0, 0
    
    lloss = nn.CrossEntropyLoss(weight, ignore_index=0)
    
    # bbox loss
    bloss_xy = nn.CrossEntropyLoss()
    bloss_wh = bbox_loss

    # send losses to GPU if available
    device = get_default_device()
    lloss, bloss_xy, bloss_wh = to_device(lloss, device), to_device(bloss_xy, device), bloss_wh

    return lloss, bloss_xy, bloss_wh

def calculate_gaussian_dict(train_ds):
    """
    Function to calculate the gaussian dictionary.
    """
    print("Getting class stats")
    sta_dict, gaussian_dict = {}, {}
    for i in tqdm(range(len(train_ds))):
        labels = train_ds.get_coco_objects_tensor(i)[1].tolist()[1:-1] # Remove first and last object <sos> and <eos>
        counter = collections.Counter(labels)
        unique_labels, label_counts = list(counter.keys()), list(counter.values())
        for label_index in range(len(unique_labels)):
            label = train_ds.vocab['index2word'][unique_labels[label_index]]
            label = train_ds.vocab['word2indexCoco'][label]
            count = label_counts[label_index]
            if label not in sta_dict:
                sta_dict[label] = []
                sta_dict[label].append(count)
            else:
                sta_dict[label].append(count)
    for label in sta_dict:
        tmp_mean = np.mean(np.array(sta_dict[label]))
        tmp_std = np.std(np.array(sta_dict[label]))
        gaussian_dict[label] = (tmp_mean, tmp_std)
    np.save(GAUSS_DICT_PATH, gaussian_dict)
def select_device():
    if GPU_INDEX is None:
        device = get_default_device()
        print("USING DEVICE", device)
    else:
        
        device = torch.device('cuda:'+str(GPU_INDEX))
        torch.cuda.device(device)
        print("USING DEVICE",torch.cuda.get_device_name(device), device)
        
    return device
def makedirs():
    """
    If needed creates the checkpoints path folder and the evaluator output path folders. Also saves the model configuration.
    """
    try:
        os.makedirs(CHECKPOINTS_PATH+"/")
        print("Checkpoints path "+CHECKPOINTS_PATH +" created")
    except FileExistsError:
        print("Checkpoints path "+CHECKPOINTS_PATH +" already existed")
    except Exception as e:
        print("Error while trying to create checkpoints path "+CHECKPOINTS_PATH, file=os.stderr)
        print(e,file=os.stderr)
    overwrite = 'y'
    files_in_checkpoint = os.listdir(CHECKPOINTS_PATH+"/")
    if 'model_config.json' in files_in_checkpoint:
        overwrite = input("model_config.json already exists in checkpoints path. Would you like to overwrite it?[y/n]")
    if overwrite == 'y' or overwrite == 'Y':
        shutil.copyfile(CONFIG_FILE_PATH, CHECKPOINTS_PATH+"/model_config.json")
        print("Current configuration saved to checkpoint folder")
    else:
        print("Continuing with current configuration but without permanently saving it.")

    #if we're saving evaluator's output, make the folder for it
    try:
        if SAVE_OUTPUT:
            os.makedirs(VALIDATION_OUTPUT+"/")
            print("Evaluator_output path "+VALIDATION_OUTPUT+" created")
    except FileExistsError:
        print("Evaluator_output path "+VALIDATION_OUTPUT+" already existed")
    except Exception as e:
        print("Error while trying to create evaluator_output path "+VALIDATION_OUTPUT, file=os.stderr)
        print(e,file=os.stderr)
if __name__ == "__main__":

    device = select_device()
    with torch.cuda.device(torch.device(device)):
        # Generate the dataset
        if IS_TRAINING:
            valg = DATASET_PATH_DEV
            vali = INSTAN_PATH_DEV
            valw = WHITE_LIST_PATH_DEV
        else:
            valg = DATASET_PATH_VAL
            vali = INSTAN_PATH_VAL
            valw = None
        #load model
        seq2seq, model_configuration = config_loader.load_config(CONFIG_FILE_PATH)
        seq2seq = to_device(seq2seq, device)
        #save model configuration (and create checkpoint folder if necessary)
        makedirs()
        

        # Generate the dataset
        train_dl, val_dl, vocab, train_ds, val_ds = generate_dataset(DATASET_PATH_TRAIN, INSTAN_PATH_TRAIN, WHITE_LIST_PATH_TRAIN, valg, vali, valw, uq_cap=UQ_CAP, batch_size=BATCH_SIZE, max_objects=model_configuration['max_objs'])

        # Generate the losses
        lloss, bloss_xy, bloss_wh = generate_losses(vocab, train_ds)

        # Calculate gaussian dict and open the file
        if CALCULATE_GAUSS_DICT:
            calculate_gaussian_dict(train_ds)
        gaussian_dict = np.load(GAUSS_DICT_PATH, allow_pickle=True).item()

        # Train or validate
        if IS_TRAINING:
            train = SupervisedTrainer(seq2seq, vocab, EPOCHS, PRINT_EVERY, lloss, bloss_xy, bloss_wh, BATCH_SIZE, model_configuration['hidden_size'], LEARNING_RATE, AdamW, len(train_dl), checkpoints_path=CHECKPOINTS_PATH, gaussian_dict=gaussian_dict, validator_output_path=VALIDATION_OUTPUT, save_output=SAVE_OUTPUT)        
            if STARTING_EPOCH>0:
                print("Loading '"+CHECKPOINTS_PATH + "/amr-gan" + str(STARTING_EPOCH-1) + ".pth'"+" as initial model weighs.")
                seq2seq.load_state_dict(torch.load(CHECKPOINTS_PATH + "/amr-gan" + str(STARTING_EPOCH-1) + ".pth", map_location=device))
                seq2seq.train()
                train.train_epoches(train_dl, train_ds, val_dl, val_ds, start_epoch=STARTING_EPOCH)
            else:
                seq2seq.train()
                train.train_epoches(train_dl, train_ds, val_dl, val_ds)
        else:
            # Epoch to validate
            print("evaluation results will be saved to "+ VALIDATION_OUTPUT)
            for epoch in EPOCHS_VALIDATION:
                print("evaluating" + CHECKPOINTS_PATH + "/amr-gan" + str(epoch) + ".pth")
                seq2seq.load_state_dict(torch.load(CHECKPOINTS_PATH + "/amr-gan" + str(epoch) + ".pth", map_location=device))

                evaluator = Evaluator(seq2seq, lloss, bloss_xy, bloss_wh, vocab, gaussian_dict=gaussian_dict, validator_output_path=VALIDATION_OUTPUT, save_output=True, verbose=False, name="TESTING")
                evaluator.evaluate(val_dl, val_ds, epoch, CHECKPOINTS_PATH)
                
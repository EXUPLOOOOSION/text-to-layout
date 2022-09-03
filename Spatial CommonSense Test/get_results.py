import os
import json
import sys
import torch
from copy import deepcopy
def clear_output(classes, bboxes):
    """
    returns a copy of classes and bboxes but its in a python list
    and each sequence has SOS removed and is cut at EOS (EOS is also removed)
    """
    SOS_class = 1
    EOS_class = 2
    PAD_class = 0
    #classes shape: [batch_size, seq_len] seq_len should be max_obj+2
    #bboxes shape: [batch_size, seq_len, 4]
    classes = classes.cpu()
    bboxes = bboxes.cpu()
    batch_size = classes.size(0)
    seq_len = classes.size(1)
    clean_classes = []
    clean_bboxes = []
    for i in range(batch_size):
        clean_seq_classes = []
        clean_seq_bboxes = []
        for j in range(seq_len):
            if classes[i][j] == SOS_class or classes[i][j]==PAD_class: #ignore initial SOS_class
                continue
            if classes[i][j] == EOS_class: #stop reading the current sequence if we get to EOS
                break
            clean_seq_classes.append(classes[i][j].item())
            clean_seq_bboxes.append(bboxes[i][j].tolist())
        clean_classes.append(clean_seq_classes)
        clean_bboxes.append(clean_seq_bboxes)
    return clean_classes, clean_bboxes


import argparse


#config
model_name = 'STRAN2LY'
checkpoint_name = 'unfrozen'
MODEL_CODE_PATH = 'G:/TFG/text-to-layout/'+model_name #
MODEL_CHECKPOINTS_PATH = 'G:/TFG/text-to-layout/'+model_name+'/checkpoints/'+checkpoint_name
MODEL_VERSIONS = range(100) #version of model to check with the test. Index.
PRINT_OUTPUT = False#True
RESULT_SAVE_PATH = './results/'+model_name+checkpoint_name
TEST_NAME = model_name+'.'+checkpoint_name# for the saved result name. It will be 'TEST_NAME_TEST_TYPE.json
TEST_TYPE = 'all' # 'posrel' 'size' 'height
DATASET_FOLDER = "./filtered_spatial_commonsense/"


parser = argparse.ArgumentParser()
parser.add_argument('v', nargs='?', type=int, const=None)
args = parser.parse_args()
if args.v is not None:
    MODEL_VERSIONS = [args.v]

#so it can import the corresponding config_loader for the  model
sys.path.insert(0, MODEL_CODE_PATH)
if TEST_TYPE == 'all':
    test_types = ['size', 'height', 'posrel']
else:
    test_types = [TEST_TYPE]
#read dataset
data_lists = []
for test_type in test_types:
    data_list = []
    print("Reading data from"+DATASET_FOLDER+test_type+"_coco.json...")
    with open(DATASET_FOLDER+test_type+'_coco.json') as f:
        for jsonObj in f:
            data_point = json.loads(jsonObj)
            data_list.append(data_point)
    data_lists.append(data_list)
#initialize model
from model import config_loader
print("Initializing model...")
model, model_configuration = config_loader.load_config(MODEL_CHECKPOINTS_PATH+"/model_config.json")
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
for MODEL_VERSION in MODEL_VERSIONS:
    print("Loading model weights from "+MODEL_CHECKPOINTS_PATH + "/amr-gan" + str(MODEL_VERSION) + ".pth...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINTS_PATH + "/amr-gan" + str(MODEL_VERSION) + ".pth", map_location='cuda:0'))

    print("Calculating Results...")
    clean_classes_list = []
    clean_bboxes_list = []
    for data_list in data_lists:
        #input
        captions = []
        for data_point in data_list:
            #normalize captions. No punctuation. All lower case.
            text = data_point['text']
            text = text.replace('. ', '')
            text = text.replace('.', '')
            text = text.lower()
            captions.append(text)
        #get output
        output = model.generate(captions)

        clean_classes, clean_bboxes = clear_output(output['output_class'], output['output_bbox'])
        clean_classes_list.append(clean_classes)
        clean_bboxes_list.append(clean_bboxes)
    try:
        os.mkdir(RESULT_SAVE_PATH)
    except FileExistsError:
        pass
    for test_idx in range(len(test_types)):
        batch_size = len(data_lists[test_idx])
        print("Saving results in "+RESULT_SAVE_PATH+"/"+TEST_NAME+"_"+str(MODEL_VERSION)+"_"+test_types[test_idx]+".json...")
        for i in range(batch_size):
            data_point = data_lists[test_idx][i]
            classes = clean_classes_list[test_idx][i]
            bboxes = clean_bboxes_list[test_idx][i]
            data_point['classes'] = classes
            data_point['bboxes'] = bboxes
            widths = [bbox[2] for bbox in bboxes]
            heights = [bbox[3] for bbox in bboxes]
            data_point['widths'] = widths
            data_point['heights'] = heights
            sizes =  [widths[i]*heights[i] for i in range(len(classes))]#all widths* all heights = all sizes (area)
            data_point['sizes'] = sizes
            max_dims = [widths[i] if widths[i] > heights[i] else heights[i] for i in range(len(classes))]
            data_point['max_dims'] = max_dims
            positions = [(bbox[0],bbox[1]) for bbox in bboxes]
            data_point['positions'] = positions

        with open(RESULT_SAVE_PATH+"/"+TEST_NAME+"_"+str(MODEL_VERSION)+"_"+test_types[test_idx]+".json","w") as f:
            for data_point in data_lists[test_idx]:
                json.dump(data_point, f)
                print(file=f)

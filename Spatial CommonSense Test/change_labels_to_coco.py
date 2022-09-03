import json
import copy
INPUT_DATASET_FILES = ['filtered_spatial_commonsense/posrel.json', 'filtered_spatial_commonsense/height.json', 'filtered_spatial_commonsense/size.json']
OUTPUT_DATASET_FILES = ['filtered_spatial_commonsense/posrel_coco.json', 'filtered_spatial_commonsense/height_coco.json', 'filtered_spatial_commonsense/size_coco.json']
LABEL_TO_COCO_LABEL_GENERATION_FOLDER = 'filtered_spatial_commonsense'
DATASET_LABEL_TO_COCO_LABEL = 'filtered_spatial_commonsense/dataset_label_to_coco_label.json'
def help_generate_class_to_coco_class():
    """
    This function will generate a json file containing every object class in all INPUT_DATASET_FILES as keys.
    The idea is that the user is to then fill this file's values in order for this program to be able to make the change from the datasets' classes to coco classes.
    So this function will for example leave "Man" or "Girl" as keys and the user would have to fill "person" as the appropiate coco class to them.
    """
    print("As no DATASET_LABEL_TO_LABEL_CLASS was given, a file helping the process of making it will be made in "+LABEL_TO_COCO_LABEL_GENERATION_FOLDER+"/empty_dataset_label_to_coco_label.json")
    print("Every label present in the datasets provided will be saved as a json file's keys. The user is then expecteds to fill these with their respective coco object label for the translation to be made in a later execution.")
    print("If a dataset label doesn't have a coco label counterpart, the entire key should be deleted. When using the file later the ammount of data points deleted because no coco label  existed for them will be reported in the statistics.")
    print()
    all_labels = []
    for dataset_path in INPUT_DATASET_FILES:
        with open(dataset_path, "r") as dataset_file:
            for jsonObj in dataset_file:
                data_point = json.loads(jsonObj)
                all_labels.append(data_point['obj_a'])
                all_labels.append(data_point['obj_b'])
    all_labels = set(all_labels)
    label_dict = {}#will contain every label as keys and an empty string as value. It will be saved and the user is supposed to fill the values with the corresponding coco label
    for label in all_labels:
        label_dict[label] = ""
    with open(LABEL_TO_COCO_LABEL_GENERATION_FOLDER+"/empty_dataset_label_to_coco_label.json","w") as f:
        json.dump(label_dict, f, indent=4, sort_keys=True)
    print("all datset labels correctly saved")
    show_coco_labels = input("would you like to get the list of all coco labels?[y/n]")
    if show_coco_labels == 'y' or show_coco_labels=='Y':
        all_coco_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        all_coco_labels = sorted(all_coco_labels) #order alphabetically
        print(all_coco_labels)
def change_labels():
    print("initilizing object label translation for:")
    for dataset_path in INPUT_DATASET_FILES:
        print(" "+dataset_path)
    print("resuts will be saved in:")
    for dataset_path in OUTPUT_DATASET_FILES:
        print(" "+dataset_path)
    print("stats will be saved in "+LABEL_TO_COCO_LABEL_GENERATION_FOLDER+"/label_translation_stats.json")
    #load label translator from any label to coco label
    with open(DATASET_LABEL_TO_COCO_LABEL, 'r') as f:
        label_to_coco_label = json.load(f)
    scs_datasets = []
    #load all datasets
    for dataset_path in INPUT_DATASET_FILES:
        current_dataset = []
        with open(dataset_path, "r") as dataset_file:
            for jsonObj in dataset_file:
                data_point = json.loads(jsonObj)
                current_dataset.append(data_point)
        scs_datasets.append(current_dataset)
    #time to do the actual filtering:
    #if any object's label is not in the translation dict, that data point will be removed.
    #when removing a data_point, we save in stats what label caused that removal and the data_point's label removed

    stats = {}
    stats['removed_labels'] = []
    for i,dataset in enumerate(scs_datasets):
        #for this dataset initialize stats and future dataset
        filtered_dataset = []
        stats[INPUT_DATASET_FILES[i]] = {}
        stats[INPUT_DATASET_FILES[i]]['n_removed_labels'] = {}
        stats[INPUT_DATASET_FILES[i]]['previous_total'] = len(dataset)
        stats[INPUT_DATASET_FILES[i]]['n_removed_labels']['total'] = 0
        for data_point in dataset:#initialize every possible data point label count to 0
            if data_point['label'] not in stats[INPUT_DATASET_FILES[i]]['n_removed_labels'].keys():
                stats[INPUT_DATASET_FILES[i]]['n_removed_labels'][data_point['label']]=0
        for data_point in dataset:
            data_point = copy.deepcopy(data_point)
            remove = False
            #check whether both object labels are in the translation dict.
            #if they are, trasnlate them individually
            if data_point['obj_a'] not in label_to_coco_label.keys():
                stats['removed_labels'].append(data_point['obj_a'])
                remove=True
            else:
                data_point['obj_a'] = label_to_coco_label[data_point['obj_a']]
            if data_point['obj_b'] not in label_to_coco_label.keys():
                stats['removed_labels'].append(data_point['obj_b'])
                remove=True
            else:
                data_point['obj_b'] = label_to_coco_label[data_point['obj_b']]
            if not remove:#if both object labels are in the translation dict
                filtered_dataset.append(data_point)
            else:# if they're not, it will be removed and stats must be updated
                stats[INPUT_DATASET_FILES[i]]['n_removed_labels'][data_point['label']] += 1
                stats[INPUT_DATASET_FILES[i]]['n_removed_labels']['total'] +=1
        stats[INPUT_DATASET_FILES[i]]['new_total'] = len(filtered_dataset)
        #save new dataset. (stats for all the datsets will be saved in the same place)
        with open(OUTPUT_DATASET_FILES[i],"w") as output_file:
            for data_point in filtered_dataset:
                json.dump(data_point, output_file)
                print(file=output_file)
    #save stats
    stats['removed_labels'] = list(set(stats['removed_labels']))
    with open(LABEL_TO_COCO_LABEL_GENERATION_FOLDER+"/label_translation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
def main():
    if DATASET_LABEL_TO_COCO_LABEL is None:
        help_generate_class_to_coco_class()
    else:
        change_labels()
if __name__ == '__main__':
    main()
import copy
import json
SCS_FILE = 'spatial-commonsense-main/data/posrel/data.json'
OUTPUT_FILE = 'filtered_spatial_commonsense/posrel.json'
STATS_OUTPUT_FILE = 'filtered_spatial_commonsense/posrel_removal_stats.json'
COCO_CAPTIONS = "../TRAN2LY/data/datasets/AMR2014train-dev-test/GraphTrain.json"


#RNN2LY code
def normalize_string(s):
        # Now included in the dataset
        """
        cap = s.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())

        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
        return tokens_new
        """
        return s.split(" ")

def main():
    """
    This program deletes datapoints that contain words that aren't in any COCO dataset captions*.
    *it only has the first caption of each image into account
    """
    print("loading coco captions...",end="")
    with open(COCO_CAPTIONS, "r") as captions_file:
        coco_captions_dict = json.load(captions_file)
    print("done.")
    print("loading scs captions...",end="")
    with open(SCS_FILE) as f:
        scs_data = []
        for jsonObj in f:
            data_point = json.loads(jsonObj)
            scs_data.append(data_point)
    print("done.")
    print("agregating and normalizing captions...",end="")
    coco_captions = []
    for caption_info in coco_captions_dict.values():
        caption = caption_info['graphs'][0]['caption']
        coco_captions.append(caption)
    coco_words = []#every word in coco dataset
    for caption in coco_captions:
        coco_words += normalize_string(caption)
    coco_words = set(coco_words)
    scs_captions = []
    for datapoint in scs_data:
        scs_caption = copy.deepcopy(datapoint['text'])
        scs_caption = scs_caption.replace(". ", "")
        scs_caption = scs_caption.replace(".", "")
        scs_caption = scs_caption.lower()
        scs_captions.append(scs_caption)
    print("done.")
    print("checking dataset captions for words not in coco...",end="")
    stats = {}
    stats['initial_n_captions'] = len(scs_data)
    stats['removed_words'] = []
    stats['removed_data_numbers'] = {}
    for data_point in scs_data:
        label = data_point['label']
        if label not in stats['removed_data_numbers'].keys():
            stats['removed_data_numbers'][label] = 0
    filtered_scs_data = []
    for i, scs_caption in enumerate(scs_captions):
        all_words_in_coco = True
        for word in scs_caption.split(" "):
            if word not in coco_words:
                all_words_in_coco = False
                stats['removed_words'].append(word)
                caption_label = scs_data[i]['label']
                stats['removed_data_numbers'][caption_label]+=1
                break
        if all_words_in_coco:
            filtered_scs_data.append(scs_data[i])#scs_captions and scs_data share indexes. scs_captions[i] is the normalized caption of scs_data[i]
    stats['final_n_captions'] = len(filtered_scs_data)
    stats['removed_words'] = list(set(stats['removed_words']))
    print("done")
    print("saving results...",end="")
    with open(OUTPUT_FILE ,"w") as f:
        for data_point in filtered_scs_data:
            json.dump(data_point, f)
            print(file=f)
    with open(STATS_OUTPUT_FILE, "w") as f:
        json.dump(stats, f)

    print("done.")
    

if __name__ == '__main__':
    main()
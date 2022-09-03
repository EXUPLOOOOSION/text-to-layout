import json
import os
from openpyxl import Workbook
import glob
import math
import re
RESULTS_FOLDER = 'results/STRAN2LYunfrozen'
OUTPUT_NAME = 'STRAN2LYunfrozen'
MODEL_NAME = 'STRAN2LY'
def select_highest_size(result, class1, class2):
    """
     returns the index of the bboxs with class1 and class2 with the highest size for each class.
     example:
     class1: 1 class2: 4
     classes:   [0, 2, 4, 6, 1 , 4, 0, 1, 2]
     sizes:     [5, 1, 2, 4, 56, 6, 7, 8, 9]
     returns: 4, 5
    """
    highest_size1 = 0
    highest_size2 = 0
    index1 = None
    index2 = None
    sizes = result['sizes']
    classes = result['classes']
    for i in range(len(sizes)):
        if classes[i] == class1 and sizes[i]>highest_size1:
            index1=i
            highest_size1 = sizes[i]
        if classes[i] == class2 and sizes[i]>highest_size2:
            index2=i
            highest_size2 = sizes[i]
    return index1, index2
def test_size(result):
    """
    given a size result instance (with each bbox's size and class along with the correct answer label and obj_a, obj_b class),
    return whether its missclassed (-1 or -2), wrong (0) or correct (1).
    Missclassed being at least one of the classes is not present 
    and the wrong/correct being, if both classes are present whether the size is correct.
    """
    output_classes, output_sizes, obj_a_class, obj_b_class, result_label = result['classes'], result['sizes'], result['obj_a'], result['obj_b'], result['label']
    seq_len = len(output_classes) # number of bboxes outputted.
    
    obj_a_class = name_to_model_index[obj_a_class]
    obj_b_class = name_to_model_index[obj_b_class]
    #for each class, select the bbox with highest size
    obj_a_index, obj_b_index = select_highest_size(result, obj_a_class, obj_b_class)
    #check if both classes had at least one bbox with that class
    not_present_classes = 0
    if obj_a_index is None:
        not_present_classes+=1
    if obj_b_index is None:
        not_present_classes+=1
    if not_present_classes>0:
        return -not_present_classes
    #finally, depending on the label, determine if the answer is correct.
    obj_a_size = output_sizes[obj_a_index]
    obj_b_size = output_sizes[obj_b_index]
    
    if result_label == 0 and obj_a_size < obj_b_size:#correct
        return 1
    elif result_label == 0 and obj_a_size >= obj_b_size:#wrong
        return 0
    elif result_label == 1 and obj_a_size > obj_b_size:#correct
        return 1
    elif result_label == 1 and obj_a_size <= obj_b_size:#wrong
        return 0
    else:
        print("caso que no se ha tenido en cuenta")
        return None
def test_height(result):
    """
    given a height result instance (with each bbox's height, size and class along with the correct answer label and obj_a, obj_b class),
    return whether its missclassed (-1 or -2), wrong (0) or correct (1).
    Missclassed being at least one of the classes is not present 
    and the others being, if both classes are present whether the height is correct.
    """
    output_classes, output_heights, obj_a_class, obj_b_class, result_label = result['classes'], result['heights'], result['obj_a'], result['obj_b'], result['label']
    seq_len = len(output_classes) # number of bboxes outputted.
    
    obj_a_class = name_to_model_index[obj_a_class]
    obj_b_class = name_to_model_index[obj_b_class]
    #for each class, select the bbox with highest size
    obj_a_index, obj_b_index = select_highest_size(result, obj_a_class, obj_b_class)
    #check if both classes had at least one bbox with that class
    not_present_classes = 0
    if obj_a_index is None:
        not_present_classes+=1
    if obj_b_index is None:
        not_present_classes+=1
    if not_present_classes>0:
        return -not_present_classes
    #finally, depending on the label, determine if the answer is correct.
    obj_a_height = output_heights[obj_a_index]
    obj_b_height = output_heights[obj_b_index]
    
    if result_label == 0 and obj_a_height < obj_b_height:#correct
        return 1
    elif result_label == 0 and obj_a_height >= obj_b_height:#wrong
        return 0
    elif result_label == 1 and obj_a_height > obj_b_height:#correct
        return 1
    elif result_label == 1 and obj_a_height <= obj_b_height:#wrong
        return 0
    else:
        print("caso que no se ha tenido en cuenta")
        return None
def is_inside(pos1, pos2, width1, width2, height1, height2):
    """
    returns whether de bbox in pos1 defined by width1 and height1 is inside the bbox
    in pos2 and defined by width2 height2
    """
    #check if left side of first object is outside (to the left) of the second object.
    if (pos1[0] - width1/2) < (pos2[0] - width2/2):
        return False
    #check if right side of first object is outside (to the right) of the second object.
    if (pos1[0] + width1/2) > (pos2[0] + width2/2):
        return False
    #check if bottom side of first object is outside (below) of the second object.
    if (pos1[1] + height1/2) > (pos2[1] + height2/2):
        return False
    #check if top side of first object is outside (above) of the second object.
    if (pos1[1] + height1/2) > (pos2[1] + height2/2):
        return False
    #if none of the sides are outside, everything is inside.
    return True
def get_angle(pos1, pos2):
    """
    This  first angle is:
                          90
                135                45
    
           180          Origin           0
    
               -135                -45
    
                         -90
    """
    #atan2 returns in radians and is converted to degrees with *180/pi
    angle = math.atan2((pos2[1]-pos1[1]),(pos2[0]-pos1[0])) * 180 / math.pi
    angle *=-1 #because the smaller the y position the higher it is (reverse from normal human beings) so we have to reverse angle.
    return angle
def is_beside(pos1, pos2):
    """
    returns whether pos1 is "beside" pos2. 
    a.k.a. whether The angle between the pos1 and pos2 
    lies between 315º and 45º or 135º and 225º.
    """
    angle = get_angle(pos1, pos2)  # (-180,180]
    return abs(angle) <= 45 or abs(angle)>=135
def is_below(pos1, pos2):
    """
    returns whether pos1 is "beside" pos2. 
    a.k.a. whether The angle between the pos1 and pos2 
    lies between 45◦ and 135◦.
    """
    angle = get_angle(pos1, pos2)  # (-180,180]
    return 45 < angle < 135
def is_above(pos1, pos2):
    """
    returns whether pos1 is "beside" pos2. 
    a.k.a. whether The angle between the pos1 and pos2 
    lies between 225◦ and 315◦
    """
    angle = get_angle(pos1, pos2)  # (-180,180]
    return -45 > angle > -135
def test_posrel(result):
    output_classes, output_poss, output_ws, output_hs, obj_a_class, obj_b_class, result_label = result['classes'], result['positions'], result['widths'],result['heights'],result['obj_a'], result['obj_b'], result['label']
    seq_len = len(output_classes) # number of bboxes outputted.
    
    obj_a_class = name_to_model_index[obj_a_class]
    obj_b_class = name_to_model_index[obj_b_class]
    #for each class, select the bbox with highest size
    obj_a_index, obj_b_index = select_highest_size(result, obj_a_class, obj_b_class)
    #check if both classes had at least one bbox with that class
    not_present_classes = 0
    if obj_a_index is None:
        not_present_classes+=1
    if obj_b_index is None:
        not_present_classes+=1
    if not_present_classes>0:
        return -not_present_classes
    if result_label == 0:#obj_a is inside obj_b
        return int(is_inside(output_poss[obj_a_index], output_poss[obj_b_index], output_ws[obj_a_index],output_ws[obj_b_index],output_hs[obj_a_index],output_hs[obj_b_index]))
    elif result_label == 1:#obj_a is above obj_b
        return int(is_above(output_poss[obj_a_index], output_poss[obj_b_index]))
    elif result_label == 2:#obj_a is below obj_b
        return int(is_below(output_poss[obj_a_index], output_poss[obj_b_index]))
        
    elif result_label == 3:#obj_a is beside obj_b
        return int(is_beside(output_poss[obj_a_index], output_poss[obj_b_index]))
    else:
        print("Unknown label: "+output_label+ " for posrel test",file=2)
    return -2
def get_result_set_stats(results, test_type):
    """
    given a list of results, get the performance stats.
    Usually used to get stats of how one model did in one dataset.
    Returns:
        number of results with not obj_A nor obj_b present in the answer.
        number of results with only obj_A or obj_b not present in the answer.
        number of results with both obj_a and obj_b class present but wrong bbox relation between them
        number of results with both obj_a and obj_b class present and correct bbox relation between them
        total number of results
    """
    results_without_class = 0
    results_with_only_one_class = 0
    results_correct = 0
    results_wrong = 0
    posrel_wrong_by_label = [0]*4
    posrel_total_by_label = [0]*4
    for result in results:
        if test_type == 'size':
            answer = test_size(result)
        elif test_type == 'height':
            answer = test_height(result)
        elif test_type == 'posrel':
            answer = test_posrel(result)
            if answer != 1:
                posrel_wrong_by_label[result['label']]+=1
            posrel_total_by_label[result['label']]+=1
        else:
            print("unrecognized test type: "+test_type)
            continue
        if answer == -2:#both classes are missing
            results_without_class+=1
        elif answer == -1:# only one class missing :)
            results_with_only_one_class+=1
        elif answer == 0:# both classes are present but bboxes are wrong
            results_wrong+=1
        elif answer == 1:# correct:both classes are present and bboxes are right
            results_correct+=1
        else:
            print("Unexpected answer from assesment function: "+answer+".Skipped.")
    if posrel_total_by_label[0] > 0:
        """
        print("wrong:", posrel_wrong_by_label)
        print("totals:", posrel_total_by_label)
        """
        print("wrong%", [100*(posrel_wrong_by_label[i] / posrel_total_by_label[i])for i in range(4)])
    return results_without_class, results_with_only_one_class, results_wrong, results_correct, len(results)


files = list(glob.glob(RESULTS_FOLDER+"/*.json"))
print(len(files), "files found")
"""
They're not actually dictionaries, they're just lists but the indexes match
test_types: [type1, type2, type3,...]
file_lists: (num_types, num_files)
    [
        type1: [file1, file2, file3,...]
        type2: [file1, file2, file3,...]
        ...
    ]
result_lists: (num_types, num_files, num_stats)
    [
        type1: [
            file1 : [stat1, stat2,...]
            file2 : [stat1, stat2,...]
            ...
        ],
        type2: [
            file1 : [stat1, stat2,...]
            file2 : [stat1, stat2,...]
            ...
        ],...
    ]
"""
#dictionary from normal human word to model indexes (4,5,6...) (which should be COCO index+3 basically)
with open("name_to_model_index.json") as f:
    name_to_model_index = json.load(f)

test_types = ['size', 'height', 'posrel']
file_lists = []

for test_type in test_types:
    file_lists.append(sorted([f for f in files if test_type in f], key=lambda x: len(x)))
epoch_lists = []
for file_list in file_lists:
    epoch_lists.append([])
    for file_name in file_list:
        new_string = file_name.replace(MODEL_NAME, '')
        file_epoch_string = ''
        for character in new_string:
            if character.isdigit():
                file_epoch_string+=character
        epoch_lists[-1].append(int(file_epoch_string))
result_lists = []
for i in range(len(test_types)):
    result_lists.append([])
    for fil in file_lists[i]:
        current_results = []
        with open(fil, 'r') as results_file:
            for a in results_file:
                current_results.append(json.loads(a))
        stats = get_result_set_stats(current_results, test_types[i])
        result_lists[i].append(stats)

for i, test_type in enumerate(test_types):
    workbook = Workbook()
    sheet = workbook.active
    sheet['A1'] = 'model'
    sheet['B1'] = 'epoch'
    sheet['C1'] = 'no classes'
    sheet['D1'] = 'only 1 class'
    sheet['E1'] = 'wrong'
    sheet['F1'] = 'correct'
    sheet['G1'] = 'total'
    
    for file_ind in range(len(file_lists[i])):
        sheet['A'+str(file_ind+2)] = file_lists[i][file_ind].split('/')[-1].split('\\')[-1].replace('.json','')
        sheet['B'+str(file_ind+2)] = epoch_lists[i][file_ind]
        sheet['C'+str(file_ind+2)] = result_lists[i][file_ind][0]
        sheet['D'+str(file_ind+2)] = result_lists[i][file_ind][1]
        sheet['E'+str(file_ind+2)] = result_lists[i][file_ind][2]
        sheet['F'+str(file_ind+2)] = result_lists[i][file_ind][3]
        sheet['G'+str(file_ind+2)] = result_lists[i][file_ind][4]
    workbook.save(filename = RESULTS_FOLDER+'/'+OUTPUT_NAME+'_'+test_type+'.xlsx')

leters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
workbook = Workbook()
sheet = workbook.active
#poner los nombres y epoch de cada modelo testeado
for file_ind in range(len(file_lists[0])):
    sheet['A'+str(file_ind+2)] = file_lists[i][file_ind].split('/')[-1].split('\\')[-1].replace('.json','')
    sheet['B'+str(file_ind+2)] = epoch_lists[i][file_ind]
# por cada test, elegir su columna y para cada modelo poner su % de acierto en ese test.
for i, test_type in enumerate(test_types):
    leter = leters[i+2]
    for file_ind in range(len(file_lists[i])):
        sheet[leter+str(file_ind+2)] = round((result_lists[i][file_ind][3] / result_lists[i][file_ind][4])*100, 2)
    sheet[leter+'1'] = test_type
workbook.save(filename = RESULTS_FOLDER+'/'+OUTPUT_NAME+'_resumen.xlsx')
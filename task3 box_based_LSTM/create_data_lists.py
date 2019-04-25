import json
import os
import difflib
from PIL import Image

import string
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
import numpy as np
import copy

# Label map
label_map = {'background': 0, 'company': 1, 'address': 2, 'date': 3, 'total': 4}
rev_label_map = {0: 'background', 1: 'company', 2: 'address', 3: 'date', 4: 'total'}

def parse_annotation(ICDAR_path, id, len_max):
    # print(id)

    texts = list()
    labels = list()

    f_task1 = open(os.path.join(ICDAR_path, '0325updated.task1train(626p)/0325updated.task1train(626p)/' + id))
    f_task3 = open(os.path.join(ICDAR_path, '0325updated.task2train(626p)/0325updated.task2train(626p)/' + id))
    f_img = os.path.join(ICDAR_path, '0325updated.task1train(626p)/0325updated.task1train(626p)/' + id.strip('txt') + 'jpg')

    im = Image.open(f_img)
    im_width = im.size[0]
    im_height = im.size[1]

    # Read task3 label as dict
    label_text = f_task3.readline()
    dict_label = {}

    label_text = f_task3.readline()
    while label_text:
        if label_text != '}':

            if label_text.find('company') !=-1:
                dict_label[label_text.strip().split("\"")[3].replace(' ','').replace(',','').replace('.','')] = 'company'
            elif label_text.find('address') !=-1:
                dict_label[label_text.strip().split("\"")[3].replace(' ','').replace(',','').replace('.','')] = 'address'
            elif label_text.find('date') !=-1:
                dict_label[label_text.strip().split("\"")[3].replace(' ','').replace(',','').replace('.','')] = 'date'
            elif label_text.find('total') !=-1:
                dict_label[label_text.strip().split("\"")[3].replace(' ','').replace(',','').replace('.','')] = 'total'

        label_text = f_task3.readline()


    # print(dict_label)
    rev_dict_label = {v: k for k, v in dict_label.items()}

    # In ICDAR case, the first line is our ROI coordinate (xmin, ymin)
    line_txt = f_task1.readline()
    coor = line_txt.split(',')
    ROI_x = int(coor[0].strip('\''))
    ROI_y = int(coor[1].strip('\''))

    label_num = 0
    final_total = 0

    while line_txt:
        line_txt = f_task1.readline()
        coor = line_txt.split(',')
        # print(coor)

        if coor[0] !='"\r\n' and  coor[0] !='"\n' and coor[0] !='':

            xmin = float(( int(coor[0].strip('\'')) - ROI_x )) / float(im_width)
            ymin = float(( int(coor[1].strip('\'')) - ROI_y )) / float(im_height)
            xmax = float(( int(coor[4].strip('\'')) - ROI_x )) / float(im_width)
            ymax = float(( int(coor[5].strip('\'')) - ROI_y )) / float(im_height)

            text = coor[8:]

            # 'ori_text' pretains special signs which block the following comparison but are useful in encoding
            ori_text = copy.deepcopy(text)
            ori_text = ','.join(ori_text)
            ori_text = list(ori_text)
            text_ascii = [ord(c) for c in ori_text]

            data = [xmin, ymin, xmax, ymax]
            data.extend(text_ascii)

            # 'text' is for comparison between task1 and task2, to find 4 labels
            text = ''.join(text)
            text = text.strip('\n').strip('\'').strip('\r')
            text = text.replace(' ','').replace(',','').replace('.','')
            # print(text)

            label = label_map['background']

            # company, address
            if 'company' in rev_dict_label:
                if (rev_dict_label['company'].find(text) !=-1 or text.find(rev_dict_label['company']) !=-1 or get_equal_rate_1(text, rev_dict_label['company'])>0.8) and len(text)>3:
                    label = label_map['company']

            if 'address' in rev_dict_label:
                if ( rev_dict_label['address'].find(text) !=-1 or text.find(rev_dict_label['address']) !=-1 or get_equal_rate_1(text, rev_dict_label['address'])>0.8) and len(text)>3 and text != 'SALE':
                    label = label_map['address']
                # since address has many lines, directly compare the similarity is impractical
                if len(text)>3 and text != 'SALE':
                    for k in range(len(rev_dict_label['address']) - len(text) +1):
                        # print(rev_dict_label['address'][k:k+len(text)])
                        # print(text)
                        compare = get_equal_rate_1(rev_dict_label['address'][k:k+len(text)], text)
                        if compare > 0.9:
                            label = label_map['address']

            # date
            if 'date' in rev_dict_label:
                if (rev_dict_label['date'].find(text) !=-1 or  text.find(rev_dict_label['date']) !=-1 or get_equal_rate_1(text, rev_dict_label['date'])>0.8) and len(text)>3:
                    label = label_map['date']

            # total
            if 'total' in rev_dict_label:
                if (text.find(rev_dict_label['total']) !=-1) and len(text)>2: #rev_dict_label['total'] .find(text) !=-1 or
                    label = label_map['total']
                    final_total = label_num

            data_pad = [0] * len_max
            if len(data) <= len_max:
                data_pad[:len(data)] = data
            else:
                data_pad = data[:len_max]

            # print(label)

            texts.append(data_pad)
            labels.append(label)
            label_num += 1

    # Padding for the same number of boxes
    # texts_pad = np.zeros([box_max, len_max])
    # texts = np.array(texts)
    # texts_pad[:texts.shape[0], :texts.shape[1]] = texts
    # texts_pad = texts_pad.tolist()
    #
    # labels_pad = [0] * box_max
    # labels_pad[:len(labels)] = labels
    #
    # print(texts_pad)
    # print(labels_pad)


    # only pretain the last total
    # if 4 in labels:
    #     for i, v in enumerate(labels):
    #         if v == 4:
    #             labels[i] = 0
    #     labels[final_total] = 4

    return {'texts': texts, 'labels': labels}, [ROI_x,ROI_y, im_width, im_height]



def get_equal_rate_1(str1, str2):
   return difflib.SequenceMatcher(None, str1, str2).quick_ratio()


def create_data_lists(ICDAR_path, output_folder):
    """
    :param ICDAR_path: path to the 'ICDAR' folder
    :param output_folder: folder where the JSONs must be saved
    """

    ICDAR_path = os.path.abspath(ICDAR_path)

    train_objects = list()
    n_objects = 0

    len_max = 50
    box_max = 180

    # Training data
    for path in [ICDAR_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'task3/train.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's txt file
            objects, d = parse_annotation(path, id, len_max)
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)


    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)  # store objects path
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_objects), n_objects, os.path.abspath(output_folder)))



    test_objects = list()
    n_objects = 0

    # Training data
    for path in [ICDAR_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'task3/test.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's txt file
            objects, d = parse_annotation(path, id, len_max)
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            test_objects.append(objects)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)  # store objects path

    print('\nThere are %d testing images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_objects), n_objects, os.path.abspath(output_folder)))


if __name__ == '__main__':
    create_data_lists(
        ICDAR_path='../ICDAR_Dataset/',
        output_folder='../ICDAR_Dataset/task3/')

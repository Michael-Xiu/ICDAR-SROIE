import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_annotation(annotation_path):
    boxes = list()

    f_txt = open(annotation_path)
    line_txt = f_txt.readline()
    while line_txt:
        coor = line_txt.split(',')
        xmin = int(coor[0].strip('\''))
        ymin = int(coor[1].strip('\''))
        xmax = int(coor[4].strip('\''))
        ymax = int(coor[5].strip('\''))
        text = coor[8].strip('\n').strip('\'')
        boxes.append([xmin, ymin, xmax, ymax])

        line_txt = f_txt.readline()

    return {'boxes': boxes}


def create_data_lists(ICDAR_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param ICDAR_path: path to the 'ICDAR' folder
    :param output_folder: folder where the JSONs must be saved
    """
    ICDAR_path = os.path.abspath(ICDAR_path)  # 返回绝对路径

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in [ICDAR_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'train1/train.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's txt file
            objects = parse_annotation(
                os.path.join(path, 'train1/' + id + '.txt'))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'train1/' + id + '.jpg'))

    assert len(train_objects) == len(train_images)

    #Save to json file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)   # store image path
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)  # store objects path


    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    test_images = list()
    test_objects = list()
    n_objects = 0

    # Training data
    for path in [ICDAR_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'test1/test.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's txt file
            objects = parse_annotation(
                os.path.join(path, 'test1/' + id + '.txt'))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            test_objects.append(objects)
            test_images.append(os.path.join(path, 'test1/' + id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to json file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


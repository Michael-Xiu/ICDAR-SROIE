import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # to specify the GPU_id in the remote server

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from create_data_lists import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # to specify the GPU_id in the remote server

import time
import torch.backends.cudnn as cudnn
import torch
import torch.optim
import torch.nn
import torch.utils.data
from model import LSTMClassifier
from datasets import ICDARDataset #, PadSequence
from utils import *
import numpy as np
from focalloss import *
from create_data_lists import parse_annotation
from train import val_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint ='BEST_checkpoint_ssd300.pth.tar' #    #' #'best.tar' # #
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_F1 = checkpoint['best_F1']
print('\nLoaded checkpoint from epoch %d. Best F1 so far is %.3f.\n' % (start_epoch, best_F1))
model = checkpoint['model']
model = model.to(device)
model.eval()

# # Transforms
# resize = transforms.Resize((300, 300))
# to_tensor = transforms.ToTensor()
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])


def detect(ICDAR_path, output_folder):
    ICDAR_path = os.path.abspath(ICDAR_path)

    train_objects = list()
    n_objects = 0

    len_max = 50

    # Training data
    for path in [ICDAR_path]:

        # Find IDs of images in testing data
        with open(os.path.join(path, 'task3/test.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            print(id)
            # Parse annotation's txt file
            objects, [ROI_x,ROI_y, im_width, im_height] = parse_annotation(ICDAR_path, id, len_max)
            texts = objects['texts']
            labels = objects['labels']

            texts = torch.FloatTensor(texts)
            labels = torch.LongTensor(labels)

            # inference pre processing
            texts = torch.unsqueeze(texts, dim=0)  # (1 batch  x n x 50) for LSTM

            # Move to default device
            texts = texts.to(device)
            labels = labels.to(device)

            # Forward prop.
            predicted_scores = model(texts)  # (1 batch x n x 5)
            predicted_scores = torch.squeeze(predicted_scores, dim=0)

            if device != "cpu":
                texts.to('cpu')
                labels.to('cpu')
                predicted_scores.to('cpu')

            # Find predicted label
            acc, F1, PRE, REC, predict_labels = val_accuracy(predicted_scores, labels)
            print(predict_labels)
            print(labels)


            # Draw image
            img_path = ICDAR_path + '/0325updated.task1train(626p)/0325updated.task1train(626p)/' + id.strip('.txt') + '.jpg'
            original_image = Image.open(img_path, mode='r')
            original_image = original_image.convert('RGBA')

            annotated_image = original_image
            tmp = Image.new('RGBA', annotated_image.size, (0, 0, 0, 0))  # for RGBA painting
            draw = ImageDraw.Draw(tmp)
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc", 15)


            # Restore coordinate
            coor = texts[:,:, :4].clone()

            coor = torch.squeeze(coor, dim=0)


            coor[:, 0] = (coor[:, 0] ) * float(im_width)
            coor[:, 1] = (coor[:, 1] ) * float(im_height)
            coor[:, 2] = (coor[:, 2] ) * float(im_width)
            coor[:, 3] = (coor[:, 3] ) * float(im_height)

            coor = torch.round(coor)

            coor = coor.type(torch.LongTensor)

            coor1 = coor.clone()

            coor1[:, 0:2] = coor1[:, 0:2] - 3
            coor1[:, 2:4] = coor1[:, 2:4] + 3



            #print(predict_labels)
            for i in range(len(predict_labels)):
                xy = coor1[i, :].tolist()
                xy2 = (coor1[i, :2] - 20).tolist()
                xy3 = coor[i, :].tolist()
                # print(xy)
                if labels[i] != 0:
                    draw.rectangle(xy=xy3, fill=(0, 120, 0, 72))  # really stubborn, should manually changle its value when labels change

                if predict_labels[i] != 0:
                    if predict_labels[i] == 1:
                        draw.rectangle(xy=xy, outline='red', width = 4)
                        draw.text(xy=xy2, text='company', fill='red', font=font)
                    elif predict_labels[i] == 2:
                        draw.rectangle(xy=xy, outline='yellow', width=4)
                        draw.text(xy=xy2, text='address', fill='yellow',font=font)
                    elif predict_labels[i] == 3:
                        draw.rectangle(xy=xy, outline='blue', width = 4)
                        draw.text(xy=xy2, text='date', fill='blue', font=font)
                    elif predict_labels[i] == 4:
                        draw.rectangle(xy=xy, outline='green', width = 4)
                        draw.text(xy=xy2, text='total', fill='green', font=font)


            annotated_image = Image.alpha_composite(annotated_image, tmp)
            annotated_image = annotated_image.convert("RGB")


            # save annotated image
            image_path = output_folder +  id.strip('.txt') + '.jpg'
            annotated_image.save(image_path)

if __name__ == '__main__':

    ICDAR_path = '../ICDAR_Dataset/'
    output_folder = '../ICDAR_Dataset/task3/result/'

    detect(ICDAR_path, output_folder)

    #
    # img_path = '../ICDAR_Dataset/0325updated.task1train(626p)/0325updated.task1train(626p)/X00016469623.jpg'
    # original_image = Image.open(img_path, mode='r')
    # original_image = original_image.convert('RGB')
    # out_image = detect(original_image, min_score=min_score, max_overlap=max_overlap, top_k=top_k, max_OCR_overlap=max_OCR_overlap, max_OCR_ratio=max_OCR_ratio)  # .show()
    # img_save_path = './test1.jpg'
    # out_image.save(img_save_path)


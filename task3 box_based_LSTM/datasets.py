import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
from torch.nn.utils.rnn import *
import numpy as np


class ICDARDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # Read data files
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        self.len = len(self.objects)


    def __getitem__(self, i):

        # Read objects
        objects = self.objects[i]

        texts = torch.FloatTensor(objects['texts'])  # (n_objects, len(text)) 2D array
        labels = torch.LongTensor(objects['labels'])  # (n_objects)

        return texts, labels

    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        texts = list()
        labels = list()

        for b in batch:
            texts.append(b[0])
            labels.append(b[1])

        return texts, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each

# def pad_tensor(vec, pad, dim):
#     """
#     args:
#         vec - tensor to pad
#         pad - the size to pad to
#         dim - dimension to pad
#
#     return:
#         a new tensor padded to 'pad' in dimension 'dim'
#     """
#     pad_size = list(vec.shape)
#     pad_size[dim] = pad - vec.size(dim)
#     return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


# class PadCollate:
#     """
#     a variant of callate_fn that pads according to the longest sequence in
#     a batch of sequences
#     """
#
#     def __init__(self, dim=0):
#         """
#         args:
#             dim - the dimension to be padded (dimension of time in sequences)
#         """
#         self.dim = dim
#
#     def pad_collate(self, batch):
#         """
#         args:
#             batch - list of (tensor, label)
#
#         reutrn:
#             xs - a tensor of all examples in 'batch' after padding
#             ys - a LongTensor of all labels in batch
#         """
#         # find longest sequence
#         max_len = max(map(lambda x: x[0].shape[self.dim], batch))
#         # pad according to max_len
#         batch = map(lambda (x, y):
#                     (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
#         # stack all
#         xs = torch.stack(map(lambda x: x[0], batch), dim=0)
#         ys = torch.LongTensor(map(lambda x: x[1], batch))
#         return xs, ys
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)

# class PadSequence:
#     def __call__(self, batch):
#         # Let's assume that each element in "batch" is a tuple (data, label).
#         # Sort the batch in the descending order
#         sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
#         # Get each sequence and pad it
#         sequences = [x[0] for x in sorted_batch]
#         sequences_padded = pad_sequence(sequences, batch_first=True)
#         # Also need to store the length of each sequence
#         # This is later needed in order to unpad the sequences
#         lengths = torch.LongTensor([len(x) for x in sequences])
#         # Don't forget to grab the labels of the *sorted* batch
#         labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
#         return sequences_padded, labels











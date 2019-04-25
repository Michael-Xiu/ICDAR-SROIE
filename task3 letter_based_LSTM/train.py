import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # to specify the GPU_id in the remote server

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn
import torch.utils.data
from model import LSTMClassifier
from datasets import ICDARDataset #, PadSequence
from utils import *
import numpy as np
from focalloss import *

# Data parameters
data_folder = '../ICDAR_Dataset/task3/'  # folder with data files

# Model parameters
n_classes = len(label_map)  # number of different types of objects
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

# Learning parameters
checkpoint =  'BEST_checkpoint_ssd300.pth.tar' #None # # None
batch_size = 10  # batch size
start_epoch = 0  # start at this epoch
epochs = 3000  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
best_F1 = 0.  # assume a high loss at first
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 10  # print training or validation status every __ batches
lr = 1e-4  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training and validation.
    """
    global epochs_since_improvement, start_epoch, label_map, best_F1, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = LSTMClassifier()
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    #lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_F1 = checkpoint['best_F1']
        print('\nLoaded checkpoint from epoch %d. Best F1 so far is %.3f.\n' % (start_epoch, best_F1))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    print(model)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss()

    # Custom dataloaders
    train_dataset = ICDARDataset(data_folder,
                                     split='train')
    val_dataset = ICDARDataset(data_folder,
                                   split='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn,
                                               num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=val_dataset.collate_fn,
                                             num_workers=workers, pin_memory=True)


    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        val_loss, accuracy, F1 = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)

        # Did validation loss improve?
        # is_best = train_loss < best_loss
        # best_loss = min(train_loss, best_loss)


        # Did validation loss improve?
        is_best = F1 > best_F1
        best_F1 = max(F1, best_F1)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_F1, is_best)

        with open('log.txt', 'a+') as f:
            f.write('epoch:'+ str(epoch) + '  train loss:' + str(train_loss)+ '  val loss:' + str(val_loss)+ 'accuracy:' + str(accuracy) + '\n')


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (texts, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)


        texts = torch.cat(texts, dim=0)  # a batch(list) of tensor -> tensor  (N*texts_each_txt  x 100)
        labels = torch.cat(labels, dim=0)  # a batch(list) of tensor -> tensor (N*texts_each_txt  x 1)

        texts = torch.unsqueeze(texts, dim=2) # (N*texts_each_txt  x 100 x 1) for LSTM

        # Move to default device
        texts = texts.to(device)
        labels = labels.to(device)

        # Forward prop.
        predicted_scores = model(texts)  # (N*texts_each_txt, 5)

        # Loss
        loss = criterion(predicted_scores, labels)  # scalar

        # acc = val_accuracy(predicted_scores, labels)
        # print(acc)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), len(labels))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

    del predicted_scores, texts, labels  # free some memory since their histories may be stored

    return losses.avg

def validate(val_loader, model, criterion):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()
    F1s = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i,  (texts, labels) in enumerate(val_loader):

            # print(texts)
            batch_shape = texts[0].size()[0]

            texts = torch.cat(texts, dim=0)  # a batch(list) of tensor -> tensor  (N*texts_each_txt  x 100)
            labels = torch.cat(labels, dim=0)  # a batch(list) of tensor -> tensor (N*texts_each_txt  x 1)

            texts = torch.unsqueeze(texts, dim=2)  # (N*texts_each_txt  x 100 x 1) for LSTM

            # Move to default device
            texts = texts.to(device)
            labels = labels.to(device)

            # Forward prop.
            predicted_scores = model(texts)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_scores, labels)  # scalar

            # Accuracy
            acc, F1, PRE, REC, predict_labels = val_accuracy(predicted_scores, labels)

            losses.update(loss.item(), batch_shape)
            F1s.update(F1, batch_shape)
            batch_time.update(time.time() - start)

            start = time.time()


            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' '\nAccuracy {accuracy:.3f}%\t' 'F1 {F1:.3f}\t' 
                      'Precision {Precision:.3f}\t' 'Recall {Recall:.3f}' .format(i, len(val_loader), batch_time=batch_time,
                       loss=losses, accuracy=acc*100, F1=F1, Precision=PRE, Recall=REC))

    print('\n * LOSS - {loss.avg:.3f}'.format(loss=losses))
    print("\n * F1 score - %.3f" % F1s.avg)

    return losses.avg, accuracy, F1s.avg

def val_accuracy(predicted_scores, labels):
    predict_labels = torch.max(predicted_scores, 1)[1]  # max index
    if device != "cpu":
        predict_labels = predict_labels.cpu()
        labels = labels.cpu()
    predict_labels = predict_labels.data.numpy()

    labels = labels.numpy()
    acc = float(sum(predict_labels == labels)) / float(np.size(predict_labels, 0))

    predict_labels_bi = predict_labels.copy()

    predict_labels_bi[predict_labels>1] = 1
    labels[labels>1] = 1

    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0

    for n in range(np.shape(predict_labels_bi)[0]):
        if labels[n]  == 1:
            if predict_labels_bi[n] == 1:
                TP+=1
            else:
                FN+=1
        else:
            if predict_labels_bi[n] == 1:
                FP+=1
            else:
                TN+=1
    PRE = TP/(TP+FP+0.0001)
    REC = TP/(FN+TP+0.0001)
    F1 = 2*PRE*REC/(PRE+REC+0.0001)

    return acc, F1, PRE, REC, predict_labels


if __name__ == '__main__':
    main()

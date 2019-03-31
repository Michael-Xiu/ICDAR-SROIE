# ICDAR-SROIE

[ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction](http://rrc.cvc.uab.es/?ch=13)

This code is based on [SSD PyTorch tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) for our specific task. Thank ***Sagar Vinodababu*** for permission and support.


## Our Target

Download the ICDAR-SROIE dataset:
- [2019 _ICDAR-SROIE_](https://pan.baidu.com/s/1a57eKCSq8SV8Njz8-jO4Ww#list/path=%2FSROIE2019&parentPath=%2F) (542MB)

Ensure the dataset or the soft link is stored in `src` file and run [`boxing.py`](https://github.com/Michael-Xiu/ICDAR-SROIE/blob/master/src/boxing.py)

<div align=center><img src="./img/boxing.PNG" width="450" height="436"/></div>


## Prepare Data for training

### Train/Test Data Split 
Open [`split_train_test.py`](https://github.com/Michael-Xiu/ICDAR-SROIE/blob/master/src/split_train_test.py) and set **train_test_ratio**. In this case it is set to 4.

In `.\ICDAR_Dataset` two files `train1` and `test1` are created. Images and labels are split into these two files in sequence.


### Pack Data
Run [`create_data_lists.py`](https://github.com/Michael-Xiu/ICDAR-SROIE/blob/master/src/create_data_lists.py) in order to pack all the image paths and labels into json files for futher operations. These json files are named `TEST_images.json` `TEST_objects.json` `TRAIN_images.json` `TRAIN_objects.json`.

<div align=center><img src="./img/json_list.PNG" width="300" height="110"/></div>

## ..be continued

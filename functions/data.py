from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
import sys


if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)

import mmsdk
import os
import re
import numpy as np
from  mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError


import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict





def download_data(DATASET):
    
        
    # create folders for storing the data
    if not os.path.exists(DATA_PATH): #./data/
        check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)    


    # download highlevel features, low-level (raw) data and labels for the dataset MOSI
    # if the files are already present, instead of downloading it you just load it yourself.
    # here we use CMU_MOSI dataset as example.
    
    
    try:
        md.mmdataset(DATASET.highlevel, DATA_PATH)
    except RuntimeError:
        print("High-level features have been downloaded previously.")
    
    try:
        md.mmdataset(DATASET.raw, DATA_PATH)
    except RuntimeError:
        print("Raw data have been downloaded previously.")
        
    try:
        md.mmdataset(DATASET.labels, DATA_PATH)
    except RuntimeError:
        print("Labels have been downloaded previously.")
        
    
    #Inspecting the download dataset    
    # list the directory contents... let's see what features there are
    data_files = os.listdir(DATA_PATH)
    print('\n'.join(data_files))
    
    
def load_dataset(visual_field,acoustic_field,text_field):
    
    
    features = [
        text_field, 
        visual_field, 
        acoustic_field
    ]
    
    recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
    
    
    dataset = md.mmdataset(recipe)
    
    #Just look at its data
    print(list(dataset.keys()))
    print("=" * 80)
    print(list(dataset[visual_field].keys())[:10])
    print(list(dataset[text_field].keys())[:10])
    print("=" * 80)
    some_id = list(dataset[visual_field].keys())[15]
    print(list(dataset[visual_field][some_id].keys()))
    print("=" * 80)
    print('Interval dimention is 2 since each step has the start and end timestamp'  )
    print('Visual:',list(dataset[visual_field][some_id]['intervals'].shape))
    print('Text:',list(dataset[text_field][some_id]['intervals'].shape))
    print('Accoustic:',list(dataset[acoustic_field][some_id]['intervals'].shape))
    print("=" * 80)
    print('ID:', some_id)
    print(list(dataset[visual_field][some_id]['features'].shape))
    print(list(dataset[text_field][some_id]['features'].shape))
    print(list(dataset[acoustic_field][some_id]['features'].shape))
    print("Different modalities have different number of time steps!")
    
    
    return dataset


#We define a simple averaging function that does not depend on intervals
def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features
    
    
    
    
def split_dataset(DATASET):
     
   # obtain the train/dev/test splits - these splits are based on video IDs
   train_split = DATASET.standard_folds.standard_train_fold
   dev_split = DATASET.standard_folds.standard_valid_fold
   test_split = DATASET.standard_folds.standard_test_fold
   
   
   print('Train:', len(train_split))
   print('Dev:', len(dev_split))
   print('Test:', len(test_split))
    
    
   return train_split, dev_split, test_split

def return_unk():
    return UNK

def data_processing(dataset,text_field,visual_field,acoustic_field,label_field,train_split,dev_split,test_split):
    
    # we can see they are in the format of 'video_id[segment_no]', but the splits was specified with video_id only
    # we need to use regex or something to match the video IDs...
    
    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0
    
    # construct a word2id mapping that automatically takes increment when new words are encountered
    #A Python dictionary throws a KeyError if you try to get an item with a key that is not currently in the dictionary. 
    #The defaultdict in contrast will simply create any items that you try to access (provided of course they do not exist yet).
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['<unk>']
    PAD = word2id['<pad>']
    
    # place holders for the final train/dev/test dataset
    train = []
    dev = []
    test = []
    # define a regular expression to extract the video ID out of the keys
    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0 # a counter to count how many data points went into some processing issues
    
    
    for segment in dataset[label_field].keys():
        # get the video ID and the features out of the aligned dataset
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features']
        _words = dataset[text_field][segment]['features']
        _visual = dataset[visual_field][segment]['features']
        _acoustic = dataset[acoustic_field][segment]['features']
        
        # if the sequences are not same length after alignment, there must be some problem with some modalities
        # we should drop it or inspect the data again
        if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
            print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
            num_drop += 1
            continue
        
        # remove nan values
        label = np.nan_to_num(label)
        _visual = np.nan_to_num(_visual)
        _acoustic = np.nan_to_num(_acoustic)
        
        
        # remove speech pause tokens - this is in general helpful
        # we should remove speech pauses and corresponding visual/acoustic features together
        # otherwise modalities would no longer be aligned
        
        words = []
        visual = []
        acoustic = []
        
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])
            
        words = np.asarray(words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)     
       
       
        # z-normalization per instance and remove nan/infs
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
        acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
       
        if vid in train_split:
            train.append(((words, visual, acoustic), label, segment))
        elif vid in dev_split:
            dev.append(((words, visual, acoustic), label, segment))
        elif vid in test_split:
            test.append(((words, visual, acoustic), label, segment))
        else:
            print(f"Found video that doesn't belong to any splits: {vid}")
            
    print(f"Total number of {num_drop} datapoints have been dropped.")     
    print("=" * 20)
    # let's see the size of each set and shape of data
    print('Train',len(train))
    print('Dev',len(dev))
    print('Test',len(test))
    print("=" * 20)
    print(f"Total vocab size: {len(word2id)}")
    
    
    # turn off the word2id - define a named function here to allow for pickling
    word2id.default_factory = return_unk
    
    return train, dev, test, word2id
    
          
import os
import sys
import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import json
import random
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import pickle
import torch
import re


def find_longest_str_index(str_list):
    '''
    找到列表中字符串最长的位置索引
    先获取列表中每个字符串的长度，查找长度最大位置的索引值即可
    '''
    num_list = [len(one) for one in str_list]
    return max(num_list)

def pad_word2len(word, pad_num):#pad every word to len
    if word == '</bos>' or word == '<eos/>' or word == 'padding':
        word = [word]
    else:
        word = list(word)
    for i in range(pad_num - len(list(word))):
        word.append('<pad>')
    return word

def make2char_cnn(sentence, longest_num, char_dict):#return a list made of words
    char_matrix = []
    for word in sentence:
        word = pad_word2len(word, longest_num)
        for i in range(len(word)):#turn evert char to index
            if word[i] in char_dict:
                word[i] = char_dict.index(word[i])
            else:
                word[i] = char_dict.index("<UNK>")
        char_matrix.append(word)
    return char_matrix

def sentence2index(sentence_list, worddict):
    for i in range(len(sentence_list)):
        if sentence_list[i] in worddict:
            sentence_list[i] = worddict.index(sentence_list[i])
        else:
            sentence_list[i] = worddict.index("<UNK>")
    return sentence_list

class elmo_Dataset(data.Dataset):
    def __init__(self):
        with open('all_sentences.pkl', 'rb') as f:
            self.data = pickle.load(f)
        with open('words_dict.pkl', 'rb') as f:
            self.words_dict = pickle.load(f)
        with open('char_dict.pkl', 'rb') as f:
            self.char_dict = pickle.load(f)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        longest_num = find_longest_str_index(sentence)
        char_matrix = make2char_cnn(sentence, longest_num, self.char_dict)
        self.length = len(char_matrix)
        sentence = sentence2index(sentence , self.words_dict)
        return char_matrix, sentence

    def __len__(self):
        return self.length
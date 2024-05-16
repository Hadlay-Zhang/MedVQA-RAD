from __future__ import print_function
'''
Author: Hadlay Zhang
Date: 2024-04-30 06:42:45
LastEditors: Hadlay Zhang
LastEditTime: 2024-05-16 13:50:00
FilePath: /root/MedicalVQA-RAD/dataset_RAD.py
Description: Methods for loading data. Unlike using embeddings in RNNs, pure questions are directly sent to dataloader so that tokenizer can process them.
'''

"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import os
import json
import _pickle as cPickle
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from attention import BiAttention
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading, assert_eq
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
COUNTING_ONLY = False
# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['image_name'],
        'image'       : img,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_type' : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type' : data['phrase_type']}
    return entry

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError:
    return False
  return True

def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])

    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary=None, dataroot='data', question_len=12):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        assert name in ['train', 'test']
        self.dataroot = args.RAD_dir
        ans2label_path = os.path.join(self.dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(self.dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        # self.dictionary = dictionary
        self.img_id2idx = json.load(open(os.path.join(self.dataroot, 'imgid2idx.json')))
        self.entries = _load_dataset(self.dataroot, name, self.img_id2idx, self.label2ans)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the input size expected by ConvNeXt
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # self.tokenize(question_len)
        self.tensorize_answers()

        self.v_dim = args.feat_dim

    def tensorize_answers(self):
        for entry in self.entries:
            answer = entry['answer']
            if answer is not None:
                labels = torch.tensor(answer['labels'], dtype=torch.long)
                scores = torch.tensor(answer['scores'], dtype=torch.float)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry['image_name']
        image_dir = self.dataroot + 'images/'
        image_path = os.path.join(image_dir, image_id)  # Assume image_dir is specified in args
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        question = entry['question']
        answer = entry['answer']

        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return image, question, target, entry['answer_type'], entry['question_type'], entry['phrase_type']
        else:
            return image, question, entry['answer_type'], entry['question_type'], entry['phrase_type']

    def __len__(self):
        return len(self.entries)

'''
Author: Hadlay Zhang
Date: 2024-04-30 06:42:45
LastEditors: Hadlay Zhang
LastEditTime: 2024-05-16 13:34:25
FilePath: /root/MedicalVQA-RAD/text.py
Description: Text Encoder for extracting question features
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
import numpy as np

class QuestionEncoder(nn.Module):
    def __init__(self, model_name, feature_size):
        super(QuestionEncoder, self).__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

'''
Author: Hadlay Zhang
Date: 2024-04-30 06:42:45
LastEditors: Hadlay Zhang
LastEditTime: 2024-05-31 00:19:40
FilePath: /root/MedicalVQA-RAD/base_model.py
Description: General VQA model with image, text, fusion and classifier
'''

import torch
import torch.nn as nn
import numpy as np
from attention import BiAttention
from text import QuestionEncoder
from image import get_Image_Encoder
from transformers import AutoModel, AutoTokenizer
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading

class BAN_Model(nn.Module):
    def __init__(self, dataset, args):
        super(BAN_Model, self).__init__()
        self.args = args
        self.glimpse = args.gamma
        self.v_model = get_Image_Encoder(args.image)
        self.q_model = QuestionEncoder(args.text_path, args.num_hid)
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_path)

        self.v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
        self.b_net = nn.ModuleList([BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1) for _ in range(args.gamma)])
        self.q_prj = nn.ModuleList([FCNet([args.num_hid, args.num_hid], '', .2) for _ in range(args.gamma)])
        self.classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
        
        if args.use_counter:
            self.counter = Counter(objects=10)

    def forward(self, v, q_texts):
        # Image Encoder
        v_emb = self.v_model(v)
        # Text Encoder
        tokens = self.tokenizer(text=q_texts, return_tensors='pt', padding='longest', max_length=128, truncation=True, add_special_tokens = True, return_token_type_ids=True, return_attention_mask=True)
        input_ids = tokens['input_ids'].clone().detach().to(self.args.device) # torch.tensor(tokens['input_ids']).to(self.args.device)
        token_type_ids = tokens['token_type_ids'].clone().detach().to(self.args.device)
        attention_mask = tokens['attention_mask'].clone().detach().to(self.args.device)
        q_emb = self.q_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        # print(q_emb.sum(1).shape)
        return q_emb.sum(1) # torch.Size([32, 768])

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Build BAN model
def build_BAN(dataset, args):
    model = BAN_Model(dataset, args)
    return model
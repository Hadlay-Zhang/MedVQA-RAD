import torch
import torch.nn as nn
import numpy as np
from torchvision.models import convnext_large
from attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading

# 注意：这里假设dataset.v_dim、args等参数已经正确设置，并且与ConvNeXt的输出匹配

class BAN_Model(nn.Module):
    def __init__(self, dataset, args):
        super(BAN_Model, self).__init__()
        self.args = args
        self.glimpse = args.gamma

        # 使用预训练的ConvNeXt模型提取图像特征
        convnext_pretrained = convnext_large(pretrained=True)
        self.convnext_feature_extractor = nn.Sequential(*list(convnext_pretrained.children())[:-2])

        # 以下部分保持不变
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
        self.q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0,  args.rnn)
        self.v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
        self.b_net = nn.ModuleList([BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1) for _ in range(args.gamma)])
        self.q_prj = nn.ModuleList([FCNet([args.num_hid, args.num_hid], '', .2) for _ in range(args.gamma)])
        self.classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
        
        if args.use_counter:
            self.counter = Counter(objects=10)

    def forward(self, v, q):
        # 使用ConvNeXt提取图像特征
        # v = torch.stack(v).to('cuda').float()  # 假设v是一个列表，包含了多个GPU上的Tensor
        # print(v.shape)
        # v = torch.tensor(v)  # 列表转tensor
        v_emb = self.convnext_feature_extractor(v)
        v_emb = torch.flatten(v_emb, start_dim=1)
        v_emb = v_emb.unsqueeze(1)
        # print(v_emb.shape)

        # 以下部分保持不变
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)
        # print(q_emb.shape) # torch.Size([32, 12, 1024])
        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        if self.args.autoencoder:
                return q_emb.sum(1), decoder
        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Build BAN model
def build_BAN(dataset, args):
    model = BAN_Model(dataset, args)
    return model
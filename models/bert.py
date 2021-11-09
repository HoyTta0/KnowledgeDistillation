# coding: UTF-8
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERT_Model(nn.Module):

    def __init__(self, config):
        super(BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, 192)
        self.fc1 = nn.Linear(192, config.num_classes)

    def forward(self, context, mask):
        outputs = self.bert(context, attention_mask=mask)
        pooled = outputs[1]
        out = self.fc(pooled)
        out = F.relu(out)
        out = self.fc1(out)
        return out

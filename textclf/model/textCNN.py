import torch.nn as nn
import torch
import torch.nn.functional as F
from base.base_model import BaseModel
from textclf.input_layer import WordEmbedding

class TextCNN(BaseModel):
    def __init__(self, num_filters, filter_sizes, vocab_size, embeeing_dim, num_classes, pretrain=None, dropout_rt=0.5):
        super(TextCNN, self).__init__()
        self.embedding = WordEmbedding(vocab_size, embeeing_dim, pretrain)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, self.embedding.size(1))) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout_rt)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

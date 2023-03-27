import torch.nn as nn
from textclf.input_layer import WordEmbedding


class Model(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_classes, vocab_size, embeeing_dim, pretrain):
        super(Model, self).__init__()
        self.embedding = WordEmbedding(vocab_size, embeeing_dim, pretrain)
        self.lstm = nn.LSTM(self.embedding.size(1), hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

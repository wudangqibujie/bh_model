import torch.nn as nn


class WordEmbedding:
    def __init__(self, vocab_size, embeeing_dim, pretrain=None):
        super(WordEmbedding, self).__init__()
        if pretrain is None:
            self.embedding = nn.Embedding(vocab_size, embeeing_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrain)

    def forward(self, input_ids):
        out = self.embedding(input_ids)
        return out

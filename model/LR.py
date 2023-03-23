from torch import nn
import torch
from model.model import BaseModel


class LogisticRg(BaseModel):
    def __init__(self, input_dim):
        super(LogisticRg, self).__init__()
        self.lr = nn.Sequential(
            nn.Linear(input_dim, 1)
        )
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        self.reg_items.append(list(filter(lambda x: "weight" in x[0] and "bn" not in x[0], self.lr.named_parameters())))
        self.reg_items.append(self.bias)

    def forward(self, x):
        output = self.lr(x) + self.bias
        return output


import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.reg_items = []

    def cal_reg_loss(self):
        pass

    def init_weight(self):
        pass

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


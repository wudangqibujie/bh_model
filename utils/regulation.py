from abc import abstractmethod
import torch


class RegulationBase:
    def __init__(self, l1, l2):
        self.l1, self.l2 = l1, l2
        self.total_reg_loss = torch.zeros((1, ))

    @abstractmethod
    def cal_regulation_loss(self):
        raise NotImplementedError


class RegFromNamedP(RegulationBase):
    def __init__(self,reg_terms,  l1=0., l2=0.):
        super(RegFromNamedP, self).__init__(l1, l2)
        self.reg_terms = reg_terms

    def cal_regulation_loss(self):
        for weight_info in self.reg_terms:
            parameter = weight_info[1]
            if self.l1 > 0:
                self.total_reg_loss += torch.sum(self.l1 * torch.abs(parameter))
            if self.l2:
                self.total_reg_loss += torch.sum(self.l2 * torch.square(parameter))


class RegFromParam(RegulationBase):
    def __init__(self, reg_term, l1=0., l2=0.):
        super(RegFromParam, self).__init__(l1, l2)
        self.reg_term = reg_term

    def cal_regulation_loss(self):
        if self.l1 > 0:
            self.total_reg_loss += torch.sum(self.l1 * torch.abs(self.reg_term))
        if self.l2:
            self.total_reg_loss += torch.sum(self.l2 * torch.square(self.reg_term))

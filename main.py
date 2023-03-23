import argparse
import collections
import torch
import numpy as np
from parse_config import ConfigParser
from utils import prepare_device
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from trainer import Trainer
from parse_config import MyConfigParser

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    config_local = MyConfigParser.from_config('./config.json')
    logger = config_local.get_logger('train')
    data_loader = config_local.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    model = config_local.init_obj('arch', module_arch)
    logger.info(model)
    device, device_ids = prepare_device(config_local['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    criterion = getattr(module_loss, config_local['loss'])
    metrics = [getattr(module_metric, met) for met in config_local['metrics']]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config_local.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config_local.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config_local,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

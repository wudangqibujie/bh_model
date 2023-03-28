import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric
from trainer import Trainer
from config.parse_config import MyConfigParser
from utils import prepare_device

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

config = MyConfigParser.from_config('config/config.json')
logger = config.get_logger('train')
data_loader = config.init_obj('train_data_loader', module_data)
valid_data_loader = config.init_obj('valid_data_loader', module_data)
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
                  lr_scheduler=lr_scheduler,
                  len_epoch=20)
trainer.train()


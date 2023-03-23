from config.parse_config import MyConfigParser
import data_loader.data_loaders as data_module
import model.model as module_module
from utils import prepare_device
import torch

config = MyConfigParser.from_config('config/config.json')
logger = config.get_logger('train')
train_dataset = config.init_obj('train_data_loader', data_module)
valid_dataset = config.init_obj('valid_data_loader', data_module)
model = config.init_obj('arch', module_module)
logger.info(model)
device, device_ids = prepare_device(config['n_gpu'])
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
from model.LR import LogisticRg

lr = LogisticRg(128)

for i in lr.reg_items:
    print(type(i))



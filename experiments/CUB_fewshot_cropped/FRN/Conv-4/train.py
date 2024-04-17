import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
sys.path.append('../../../')
from trainers import trainer, frn_train
from datasets import dataloaders
from models.RIFCRN import FRN
args = trainer.train_parser()
'''
with open('D:\\few_shot_classification\\FRN\\config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
print(data_path)
fewshot_path = os.path.join(data_path,'mini-ImageNet')
'''
#print(fewshot_path)
fewshot_path='/opt/data_3/lyf/FRN/data/CUB_200_2011/CUB_fewshot_cropped'
pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = FRN(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)

#pretrained_model_path = '../ResNet-12_pretrain/model_ResNet-12.pth'
#pretrained_model_path = 'D:/few_shot_classification/FRN/experiments/mini-ImageNet/FRN/ResNet-12_finetune/model_Conv-4.pth'

#model.load_state_dict(torch.load(pretrained_model_path,map_location=util.get_device_map(args.gpu)),strict=False)
#model.load_state_dict(torch.load(pretrained_model_path,map_location=util.get_device_map(args.gpu)),strict=True)
train_func = partial(frn_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)


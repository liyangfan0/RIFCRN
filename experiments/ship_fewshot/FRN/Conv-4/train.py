import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
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
fewshot_path='/opt/data_3/lyf/FRN/data/ship'
#fewshot_path='/opt/data_3/lyf/FRN/data/WHU'
#fewshot_path='/opt/data_3/lyf/FRN/data/Aircraft_fewshot'
#fewshot_path='/opt/data_3/lyf/FRN/data/CUB_200_2011/CUB_fewshot_cropped'
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

train_func = partial(frn_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)

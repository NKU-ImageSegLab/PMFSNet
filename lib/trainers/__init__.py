# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 17:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from .tooth_trainer import ToothTrainer
from .mmotu_trainer import MMOTUTrainer
from .isic_2018_trainer import ISIC2018Trainer
from .isic_trainer import ISICTrainer


def get_trainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        trainer = ToothTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "MMOTU":
        trainer = MMOTUTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif "ISIC" in opt["dataset_name"]:
        trainer = ISICTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize trainer")

    return trainer

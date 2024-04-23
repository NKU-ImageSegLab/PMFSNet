# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 17:05
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import argparse

import torch
import yaml

from lib import utils, dataloaders, models, losses, metrics, trainers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/ISIC_2018.yaml', help="config file path")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset name")
    parser.add_argument("--model", type=str, default="PMFSNet", help="model name")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--dimension", type=str, default=None, help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="TINY", help="scaling version of PMFSNet")
    parser.add_argument("--epoch", type=int, default=None, help="training epoch")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="tensorboard directory")
    args = parser.parse_args()
    return args


def main():
    # analyse console arguments
    args = parse_args()
    try:
        with open(args.config, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        exit(1)

    # update the dictionary of hyperparameters used for training
    if args.dataset is not None:
        params["dataset_name"] = args.dataset
        params["dataset_path"] = os.path.join(r"./datasets", (
            "NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))

    params["dataset_path"] = args.dataset_path if args.dataset_path is not None else params["dataset_path"]
    params["model_name"] = args.model
    if args.pretrain_weight is not None:
        params["pretrain"] = args.pretrain_weight
    if args.dimension is not None:
        params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
    if args.epoch is not None:
        params["end_epoch"] = args.epoch
        params["save_epoch_freq"] = args.epoch // 4
    params["tensorboard_dir"] = args.tensorboard_dir if args.tensorboard_dir is not None else params["tensorboard_dir"]
    # launch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{params["CUDA_VISIBLE_DEVICES"]}'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # get the cuda device
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("Complete the initialization of configuration")

    # initialize the dataloader
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("Complete the initialization of dataloader")

    # initialize the model, optimizer, and lr_scheduler
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(params["model_name"],
                                                                                              params["optimizer_name"],
                                                                                              params[
                                                                                                  "lr_scheduler_name"]))

    # initialize the loss function
    loss_function = losses.get_loss_function(params)
    print("Complete the initialization of loss function")

    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # initialize the trainer
    trainer = trainers.get_trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function,
                                   metric)

    # resume or load pretrained weights
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()
    print("Complete the initialization of trainer")

    # start training
    trainer.training()


if __name__ == '__main__':
    main()

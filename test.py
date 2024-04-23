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

from lib import utils, dataloaders, models, metrics, testers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/ISIC_2018.yaml', help="config file path")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset name")
    parser.add_argument("--model", type=str, default=None, help="model name")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--dimension", type=str, default=None, help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="TINY", help="scaling version of PMFSNet")
    parser.add_argument("--result_path", type=str, default=None, help="result path")
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
    params["model_name"] = args.model if args.model is not None else params["model_name"]
    if args.pretrain_weight is None:
        print('Use the best model in the training process')
    params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension if args.dimension is not None else params["dimension"]
    params["scaling_version"] = args.scaling_version
    params["result_path"] = args.result_path if args.result_path is not None else params["result_path"]

    # launch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
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
    valid_loader = dataloaders.get_test_dataloader(params)
    print("Complete the initialization of dataloader")

    # initialize the model
    model = models.get_model(params)
    print("Complete the initialization of model:{}".format(params["model_name"]))

    # initialize the metrics
    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # initialize the tester
    tester = testers.get_tester(params, model, metric)
    print("Complete the initialization of tester")

    # load training weights
    tester.load()
    print("Complete loading training weights")

    # evaluate valid set
    tester.evaluation(valid_loader)


if __name__ == '__main__':
    main()

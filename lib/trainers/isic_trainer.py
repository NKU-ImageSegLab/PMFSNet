import os

import nni
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from lib import utils
from lib.metrics.metrics import get_binary_metrics, get_metrics, MetricsResult
from lib.tensoarboards import Writer


class ISICTrainer:
    """
    Trainer class
    """

    def __init__(self, opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):

        self.opt = opt
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.device = opt["device"]
        if self.opt["sigmoid_normalization"]:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        self.writer = Writer(opt)

        if not self.opt["optimize_params"]:
            if self.opt["resume"] is None:
                self.execute_dir = os.path.join(opt["run_dir"],
                                                opt["model_name"] + "_" + opt["dataset_name"])
            else:
                self.execute_dir = os.path.dirname(os.path.dirname(self.opt["resume"]))
            self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
            self.tensorboard_dir = os.path.join(self.execute_dir, "board")
            self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
            if self.opt["resume"] is None:
                utils.make_dirs(self.checkpoint_dir)
                utils.make_dirs(self.tensorboard_dir)
            utils.pre_write_txt("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(
                self.opt["model_name"], self.opt["optimizer_name"], self.opt["lr_scheduler_name"]), self.log_txt_path)

        self.start_epoch = self.opt["start_epoch"]
        self.end_epoch = self.opt["end_epoch"]
        self.best_metric = opt["best_metric"]
        self.terminal_show_freq = opt["terminal_show_freq"]
        self.save_epoch_freq = opt["save_epoch_freq"]

    def training(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.optimizer.zero_grad()

            tr_result, tr_loss = self.train_epoch(epoch)

            print(
                f'Epoch [{epoch + 1}'
                f'/{self.end_epoch}], Loss: {tr_loss:.4f}, \n'
                f'[Training] Acc: {tr_result.Accuracy:.4f}, '
                f'SE: {tr_result.Recall:.4f}, '
                f'SP: {tr_result.Specificity:.4f}, '
                f'PC: {tr_result.Precision:.4f}, '
                f'F1: {tr_result.F1:.4f}, '
                f'DC: {tr_result.Dice:.4f}, '
                f'MIOU: {tr_result.JaccardIndex:.4f}, '
            )

            vl_result, vl_loss = self.valid_epoch(epoch)

            print(
                f'Epoch [{epoch + 1}'
                f'/{self.end_epoch}], Loss: {vl_loss:.4f}, \n'
                f'[Validation] Acc: {vl_result.Accuracy:.4f}, '
                f'SE: {vl_result.Recall:.4f}, '
                f'SP: {vl_result.Specificity:.4f}, '
                f'PC: {vl_result.Precision:.4f}, '
                f'F1: {vl_result.F1:.4f}, '
                f'DC: {vl_result.Dice:.4f}, '
                f'MIOU: {vl_result.JaccardIndex:.4f}, '
            )

            data_list = [
                {
                    "tag": "Loss/train vs validation",
                    "value": {"Train": tr_loss, "Validation": vl_loss},
                },
                {
                    "tag": "Accuracy/train vs validation",
                    "value": {"Train": tr_result.Accuracy, "Validation": vl_result.Accuracy},
                },
                {
                    "tag": "Recall/train vs validation",
                    "value": {"Train": tr_result.Recall, "Validation": vl_result.Recall},
                },
                {
                    "tag": "Specificity/train vs validation",
                    "value": {"Train": tr_result.Specificity, "Validation": vl_result.Specificity},
                },
                {
                    "tag": "Precision/train vs validation",
                    "value": {"Train": tr_result.Precision, "Validation": vl_result.Precision},
                },
                {
                    "tag": "F1/train vs validation",
                    "value": {"Train": tr_result.F1, "Validation": vl_result.F1},
                },
                {
                    "tag": "Dice/train vs validation",
                    "value": {"Train": tr_result.Dice, "Validation": vl_result.Dice},
                },
                {
                    "tag": "Mean Intersection over Union/train vs validation",
                    "value": {"Train": tr_result.JaccardIndex, "Validation": vl_result.JaccardIndex},
                },
            ]

            for data in data_list:
                self.writer.add_scalars(
                    data["tag"],
                    data["value"],
                    epoch,
                )

            valid_JI = vl_result.JaccardIndex

            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(valid_JI)
            else:
                self.lr_scheduler.step()

            if self.opt["optimize_params"]:
                nni.report_intermediate_result(valid_JI)
        self.writer.close()
        if self.opt["optimize_params"]:
            nni.report_final_result(self.best_metric)

    def train_epoch(self, epoch):
        self.model.train()
        metrics = get_binary_metrics()
        total_loss = 0
        for batch_idx, (input_tensor, target, _) in tqdm(
                iterable=enumerate(self.train_data_loader),
                desc=f"{self.opt['dataset_name']} {self.opt['model_name']} Training [{epoch + 1}/{self.end_epoch}] Current Image",
                unit="image",
                total=len(self.train_data_loader)
        ):
            input_tensor = input_tensor.to(self.device)
            target = target.to(self.device)
            output = self.model(input_tensor)
            predict = self.normalization(output)
            # # 将预测图像进行分割
            predict = torch.argmax(predict, dim=1)
            metrics.update(predict.float(), target.int())
            dice_loss = self.loss_function(output, target)
            dice_loss.backward()
            total_loss += dice_loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
        result = metrics.compute()

        return MetricsResult(result), total_loss

    def valid_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            metrics = get_binary_metrics()
            for batch_idx, (input_tensor, target, _) in tqdm(
                    iterable=enumerate(self.valid_data_loader),
                    desc=f"{self.opt['dataset_name']} {self.opt['model_name']} Validation [{epoch + 1}/{self.end_epoch}] Current Image",
                    unit="image",
                    total=len(self.valid_data_loader)
            ):
                input_tensor = input_tensor.to(self.device)
                target = target.to(self.device)
                output = self.model(input_tensor)
                predict = self.normalization(output)
                # # 将预测图像进行分割
                predict = torch.argmax(predict, dim=1)
                metrics.update(predict.float(), target.int())
                dice_loss = self.loss_function(output, target)
                total_loss += dice_loss.item()
            result = MetricsResult(metrics.compute())
            cur_JI = result.JaccardIndex

            if cur_JI > self.best_metric:
                self.best_metric = cur_JI
                self.save(epoch, cur_JI, self.best_metric, type="best")
                print("New Best model saved at epoch: {} with JI: {:.4f}".format(epoch, cur_JI))
            return result, total_loss

    def save(self, epoch, metric, best_metric, type="normal"):
        if type == "normal":
            save_filename = "{:04d}_{}_{:.4f}.pth".format(epoch, self.opt["model_name"], metric)
        elif type == "best":
            save_filename = 'best.pth'
        else:
            save_filename = '{}_{}.pth'.format(type, self.opt["model_name"])
        save_path = os.path.join(self.checkpoint_dir, save_filename)
        torch.save(self.model.state_dict(), save_path)

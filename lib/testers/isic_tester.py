# -*- encoding: utf-8 -*-
"""
@author   :   chouheiwa
@DateTime :   2024/04/23 00:33
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm

import torch

from torchvision import transforms

from lib import utils
from lib.metrics import get_binary_metrics
from lib.metrics.metrics import MetricsResult


class ISICTester:
    """
    Tester class
    """

    def __init__(self, opt, model, metrics=None):
        self.opt = opt
        self.model = model
        self.metrics = metrics
        self.device = self.opt["device"]

        self.execute_dir = os.path.join(
            opt["run_dir"],
            opt["model_name"] + "_" + opt["dataset_name"]
        )
        self.checkpoint_dir = os.path.join(self.execute_dir.__str__(), "checkpoints")

        self.result_path = os.path.join(
            opt["result_path"],
            opt["dataset_name"],
            opt["model_name"]
        )

        # Create directories if not exist
        os.makedirs(self.result_path, exist_ok=True)

        if self.opt["sigmoid_normalization"]:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def inference(self, image_path):
        test_transforms = transforms.Compose([
            transforms.Resize(self.opt["resize_shape"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
        ])

        image_pil = Image.open(image_path)
        w, h = image_pil.size
        image = test_transforms(image_pil)
        dir_path, image_name = os.path.split(image_path)
        dot_pos = image_name.find(".")
        file_name = image_name[:dot_pos]
        segmentation_image_path = os.path.join(dir_path, file_name + "_segmentation" + ".jpg")

        self.model.eval()
        with torch.no_grad():
            image = torch.unsqueeze(image, dim=0)
            image = image.to(self.device)
            output = self.model(image)

        segmented_image = torch.argmax(output, dim=1).squeeze(0).to(dtype=torch.uint8).cpu().numpy()
        segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_AREA)
        segmented_image[segmented_image == 1] = 255
        cv2.imwrite(segmentation_image_path, segmented_image)
        print("Save segmented image to {}".format(segmentation_image_path))

    def evaluation(self, dataloader):
        self.model.eval()

        with torch.no_grad():
            metrics = get_binary_metrics()
            for input_tensor, target, image_names in tqdm(dataloader, leave=True):
                input_tensor, target = input_tensor.to(self.device), target.to(self.device)
                output = self.model(input_tensor)
                predict = self.normalization(output)
                # # 将预测图像进行分割
                predict_latest = torch.argmax(predict, dim=1)
                metrics.update(predict_latest.float(), target.int())

                for i in range(predict_latest.size(0)):
                    # predict[predict == 1] = 255
                    torchvision.utils.save_image(
                        predict_latest[i].float(),
                        os.path.join(
                            self.result_path.__str__(),
                            f'{image_names[i]}.png'
                        )
                    )


            result = MetricsResult(metrics.compute())
            try:
                params, flops = result.cal_params_flops(self.model, 256)
            except:
                params, flops = 0, 0
            result.to_result_csv(
                os.path.join(self.result_path.__str__(), "result.csv"),
                model_name=self.opt["model_name"],
                flops=flops,
                params=params
            )
        print(
            f"valid_DSC:{result.Dice:.6f}  valid_IoU:{result.JaccardIndex:.6f}  valid_ACC:{result.Accuracy:.6f}  valid_JI:{result.JaccardIndex:.6f}")
    def load(self):
        if self.opt["pretrain"] is not None:
            pretrain_state_dict = torch.load(self.opt["pretrain"])
            self.model.load_state_dict(pretrain_state_dict)
        else:
            pretrain_state_dict = torch.load(os.path.join(self.checkpoint_dir, "best.pth"))
            self.model.load_state_dict(pretrain_state_dict)
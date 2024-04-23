# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 17:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from .isic_tester import ISICTester
from .tooth_tester import ToothTester
from .mmotu_tester import MMOTUTester
from .isic_2018_tester import ISIC2018Tester


def get_tester(opt, model, metrics=None):
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        tester = ToothTester(opt, model, metrics)
    elif opt["dataset_name"] == "MMOTU":
        tester = MMOTUTester(opt, model, metrics)
    elif "ISIC" in opt["dataset_name"]:
        tester = ISICTester(opt, model, metrics)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize tester")

    return tester

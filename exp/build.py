#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import importlib
import os
import sys
sys.path.append("..")
def get_exp_by_file(exp_file):
    try:
        #sys.path.append(os.path.dirname(exp_file))
        file_name = exp_file[:-3].replace('/','.')
        current_exp = importlib.import_module(file_name)
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp(exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)


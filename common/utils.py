# ----------------------- Directory ---------------------------- #
import os


def prepare_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def exist_dir(dir_name):
    if not os.path.exists(dir_name):
        return False
    else:
        return True

## -------------------------------------- File Reader --------------------------------------------- #
import csv

def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

## -------------------------------------- Pytorch utilities --------------------------------------- ##
import torch

def is_gpu_available():
    return torch.cuda.is_available()


def load_model(model_fn, map_location=None):
    if map_location:
        return torch.load(model_fn, map_location=map_location)
    else:
        if torch.cuda.is_available():
            return torch.load(model_fn)
        else:
            return torch.load(model_fn, map_location='cpu')
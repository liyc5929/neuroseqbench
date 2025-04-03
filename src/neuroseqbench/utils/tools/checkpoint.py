import os
import torch
import shutil


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", save_path="./", epoch=1):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, "model_best.pth.tar"))

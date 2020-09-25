import os

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils.audio import Audio

class VFWSdataset():
    def __init__(self,dir_root):
        self.dir_root = dir_root
        self.load()

    def load(self):
        dir_root = self.dir_root
#        self.df_file = pd.DataFrame(columns={'file'})

    def get(self):
        return self.trainset, self.testset

class VFWStrain(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return sample

class VFWStest(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return sample





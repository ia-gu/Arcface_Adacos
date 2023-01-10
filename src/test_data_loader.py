import os
import shutil
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import glob

# dataloader for test(face recognition)
class TestDataLoader(Dataset):

    def __init__(self, dataPath, transform=None):
        super(TestDataLoader, self).__init__()
        self.transform = transform

        self.img1 = None
        self.img2 = None

        self.content = []
        with open('pairsDevTest.txt') as f:
            for line in f:
                self.content.append(line.split())

        self.data = {}
        for manPath in os.listdir(dataPath):
            datas = []
            fl = [p for p in glob.glob(dataPath+'/'+manPath+'/**', recursive=True) if os.path.isfile(p)]
            for i in fl:
                datas.append(i)
            self.data.update({manPath: datas})

    def __len__(self):
        return 1000

    def __getitem__(self, index):

        index = random.randint(1,len(self.content)-1)
        # same class
        if len(self.content[index]) == 3:
            label = 1.0
            key = self.content[index][0]
            self.img1 = Image.open(self.data[key][int(self.content[index][1])-1])
            self.img2 = Image.open(self.data[key][int(self.content[index][2])-1])

        # different class
        else:
            label = 0.0
            key1 = self.content[index][0]
            key2 = self.content[index][2]
            self.img1 = Image.open(self.data[key1][int(self.content[index][1])-1])
            self.img2 = Image.open(self.data[key2][int(self.content[index][3])-1])

        if self.transform:
            self.img1 = self.transform(self.img1)
            self.img2 = self.transform(self.img2)

        self.content.remove(self.content[index])
        return self.img1, self.img2, torch.from_numpy(np.array([label], dtype=np.float32))

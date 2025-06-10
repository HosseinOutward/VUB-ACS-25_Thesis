import numpy as np
import torch
from PIL import Image
from torchvision.datasets import SVHN


class FasterSVHN(SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data = self.data
        labels = self.labels

        self.data = []
        for img in data:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)
            self.data.append(img)

        self.labels = []
        for target in labels:
            target = int(target)
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.labels.append(target)

        self.data = np.array(self.data).transpose(0, 1, 2, 3)
        self.labels = np.array(self.labels)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]


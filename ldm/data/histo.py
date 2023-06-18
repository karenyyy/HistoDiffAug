import bisect
import os

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


CLASS2IDX = {
    'ADI': 0,
    'BACK': 1,
    'DEB': 2,
    'LYM': 3,
    'MUC': 4,
    'MUS': 5,
    'NORM': 6,
    'STR': 7,
    'TUM': 8
}


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels

        self.labels["file_path_"] = []
        self._length = 0
        for path in paths:
            if 'vgh' in path:
                self.labels["file_path_"].extend([os.path.join(path, i) for i in os.listdir(path)])
                self._length += len(os.listdir(path))
            elif 'CRC' in path:
                train_path = os.path.join(path, 'train')
                train_files = [os.path.join(os.path.join(train_path, tissue_type), i) for tissue_type in os.listdir(train_path) for i in os.listdir(os.path.join(train_path, tissue_type))]
                self.labels["file_path_"].extend(train_files)
                self._length += len(train_files)
            else:
                files = [os.path.join(os.path.join(path, patient), i) for patient in
                             os.listdir(path) for i in os.listdir(os.path.join(path, patient))]
                self.labels["file_path_"].extend(files)
                self._length += len(files)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])

        for k in self.labels:
            example[k] = self.labels[k][i]
            # example['y'] = CLASS2IDX[self.labels[k][i].split('/')[-2]]

        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image


class HistoBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class HistoTrain(HistoBase):
    def __init__(self, size):
        super().__init__()
        self.data = ImagePaths(paths=
        [
            '/data/karenyyy/vgh/class0',
            # '/data/karenyyy/CRC_Data',
            # '/data/karenyyy/PanNuke/images',
            # '/data/karenyyy/TCGA-BRCA/BLOCKS_NORM_MACENKO'
        ],
            size=size, random_crop=False)

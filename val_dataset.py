from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import torch
from utils_test import cropping, cropping_ohaze

class dehaze_val_dataset(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort()
        self.root_hazy = test_dir
        self.file_len = len(self.list_test)
        self.crop_method = crop_method
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy +'/'+ self.list_test[index])
        hazy = self.transform(hazy)

        if hazy.shape[0] == 4:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]

        hazy, vertical = cropping(hazy, self.crop_method)

        return hazy, vertical

    def __len__(self):
        return self.file_len
    




class dehaze_val_dataset_ohaze(Dataset):
    def __init__(self, test_dir, crop_method):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort()
        self.root_hazy = test_dir
        self.file_len = len(self.list_test)
        self.crop_method = crop_method
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        hazy = Image.open(self.root_hazy +'/'+ self.list_test[index])
        hazy = self.transform(hazy)
        hazy_shape = hazy.shape

        if hazy.shape[0] == 5:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]

        hazy = cropping_ohaze(hazy, index)
        
        return hazy, (index, hazy_shape)

    def __len__(self):
        return self.file_len

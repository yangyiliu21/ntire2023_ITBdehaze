from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import torch
from utils_test import cropping

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
        #hazy = cv2.imread(self.root_hazy +'/'+ self.list_test[index]) 
        hazy = self.transform(hazy)

        if hazy.shape[0] == 4:
            assert torch.equal(hazy[-1:, :, :], torch.ones(1, hazy.shape[1], hazy.shape[2])), "hazy[-1:, :, :] is not all ones"
            hazy = hazy[:3, :, :]

        hazy, vertical = cropping(hazy, self.crop_method)

        # num_row = hazy.shape[1]

        # #################### This part is for no crop, 1*6144*6144 #####################
        # if self.crop_method == 1:
        #     padding = torch.nn.ReflectionPad2d(72)
        #     hazy = padding(hazy)
        #     if num_row == 6000:
        #         vertical = True
        #         transform = transforms.Pad((1000,0))
        #     else:
        #         vertical = False
        #         transform = transforms.Pad((0,1000))
        #     hazy = transform(hazy)
        #     hazy = [hazy]

        # #################### This part is for 1*4096*4096+2*2048*2048 #####################
        # if self.crop_method == 3:
        #     padding = torch.nn.ReflectionPad2d(48)
        #     if num_row == 6000:
        #         vertical = True
        #         hazy_1 = padding(hazy[:, 0:4000, 0:4000])
        #         hazy_2 = hazy[:, 4000-48:, 0:2000+48]
        #         hazy_3 = hazy[:, 4000-48:, 2000-48:]
        #     else:
        #         vertical = False
        #         hazy_1 = padding(hazy[:, 0:4000, 0:4000])
        #         hazy_2 = hazy[:, 0:2000+48, 4000-48:]
        #         hazy_3 = hazy[:, 4000-48:, 2000-48:]
        #     hazy = [hazy_1, hazy_2, hazy_3]


        # #################### This part is for 6*2048*2048 #####################
        # # swin Transform can run only square inputs
        # # for 6k*4k image:
        # if self.crop_method == 6:
        #     if num_row == 6000:
        #         vertical = True
        #         hazy_1 = hazy[:, 0:2000+48, 0:2000+48]
        #         hazy_2 = hazy[:, 0:2000+48:, 2000-48:]
        #         hazy_3 = hazy[:, 2000-24:4000+24:, 0:2000+48]
        #         hazy_4 = hazy[:, 2000-24:4000+24:, 2000-48:]
        #         hazy_5 = hazy[:, 4000-48:, 0:2000+48]
        #         hazy_6 = hazy[:, 4000-48:, 2000-48:]
        #     else:
        #         vertical = False
        #         hazy_1 = hazy[:, 0:2000+48, 0:2000+48]
        #         hazy_2 = hazy[:, 2000-48:, 0:2000+48]
        #         hazy_3 = hazy[:, 0:2000+48, 2000-24:4000+24]
        #         hazy_4 = hazy[:, 2000-48:, 2000-24:4000+24]
        #         hazy_5 = hazy[:, 0:2000+48, 4000-48:]
        #         hazy_6 = hazy[:, 2000-48:, 4000-48:]
        #     hazy = [hazy_1, hazy_2, hazy_3, hazy_4, hazy_5, hazy_6]

        return hazy, vertical

    def __len__(self):
        return self.file_len


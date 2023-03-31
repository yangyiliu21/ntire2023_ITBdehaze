import torch
import torch.nn.functional as F
from math import log10
import cv2
import numpy as np
import torchvision
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def to_psnr(frame_out, gt):
    mse = F.mse_loss(frame_out, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def predict(gridnet, test_data_loader):

    psnr_list = []
    for batch_idx, (frame1, frame2, frame3) in enumerate(test_data_loader):
        with torch.no_grad():
            frame1 = frame1.to(torch.device('cuda'))
            frame3 = frame3.to(torch.device('cuda'))
            gt = frame2.to(torch.device('cuda'))
            # print(frame1)

            frame_out = gridnet(frame1, frame3)
            # print(frame_out)
            frame_debug = torch.cat((frame1, frame_out, gt, frame3), dim =0)
            filepath = "./image" + str(batch_idx) + '.png'
            torchvision.utils.save_image(frame_debug, filepath)
            # print(frame_out)
            # img = np.asarray(frame_out.cpu()).astype(float)
            
            # cv2.imwrite(filepath , img)



        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(frame_out, gt))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    return avr_psnr

def cropping(hazy, crop_num):

    assert hazy.shape[1]*hazy.shape[2] == 4000*6000

    num_row = hazy.shape[1]

    #################### This part is for no crop, 1*6144*6144 #####################
    if crop_num == 1:
        padding = torch.nn.ReflectionPad2d(72)
        hazy = padding(hazy)
        if num_row == 6000:
            vertical = True
            transform = transforms.Pad((1000,0))
        else:
            vertical = False
            transform = transforms.Pad((0,1000))
        hazy = transform(hazy)
        hazy = [hazy]

    #################### This part is for 1*4096*4096+2*2048*2048 #####################
    if crop_num == 3:
        padding = torch.nn.ReflectionPad2d(48)
        if num_row == 6000:
            vertical = True
            hazy_1 = padding(hazy[:, 0:4000, 0:4000])
            hazy_2 = hazy[:, 4000-48:, 0:2000+48]
            hazy_3 = hazy[:, 4000-48:, 2000-48:]
        else:
            vertical = False
            hazy_1 = padding(hazy[:, 0:4000, 0:4000])
            hazy_2 = hazy[:, 0:2000+48, 4000-48:]
            hazy_3 = hazy[:, 2000-48:, 4000-48:]
        hazy = [hazy_1, hazy_2, hazy_3]

    #################### This part is for 4 #####################
    if crop_num == 4:
        if hazy.size(1) == 6000:
            vertical = True
            hazy1 = hazy[:,0:3840,:3840]
            hazy2 = hazy[:,1080:4920,:3840]
            hazy3 = hazy[:,2160:6000,:3840]
            hazy4 = hazy[:,0:3840,160:4000]
            hazy5 = hazy[:,1080:4920,160:4000]
            hazy6 = hazy[:,2160:6000,160:4000]
            hazy = [hazy1, hazy2, hazy3, hazy4, hazy5, hazy6]
            #return [hazy1, hazy2, hazy3, hazy4, hazy5, hazy6], v
        elif hazy.size(2) == 6000:
            vertical = False
            hazy1 = hazy[:,:3840,0:3840]
            hazy2 = hazy[:,:3840,2160:6000]
            hazy3 = hazy[:,160:4000,0:3840]
            hazy4 = hazy[:,160:4000,2160:6000]
            #return [hazy1, hazy2, hazy3, hazy4], v
            hazy = [hazy1, hazy2, hazy3, hazy4]


    #################### This part is for 6*2048*2048 #####################
    # swin Transform can run only square inputs
    # for 6k*4k image:
    if crop_num == 6:
        if num_row == 6000:
            vertical = True
            hazy_1 = hazy[:, 0:2000+48, 0:2000+48]
            hazy_2 = hazy[:, 0:2000+48:, 2000-48:]
            hazy_3 = hazy[:, 2000-24:4000+24:, 0:2000+48]
            hazy_4 = hazy[:, 2000-24:4000+24:, 2000-48:]
            hazy_5 = hazy[:, 4000-48:, 0:2000+48]
            hazy_6 = hazy[:, 4000-48:, 2000-48:]
        else:
            vertical = False
            hazy_1 = hazy[:, 0:2000+48, 0:2000+48]
            hazy_2 = hazy[:, 2000-48:, 0:2000+48]
            hazy_3 = hazy[:, 0:2000+48, 2000-24:4000+24]
            hazy_4 = hazy[:, 2000-48:, 2000-24:4000+24]
            hazy_5 = hazy[:, 0:2000+48, 4000-48:]
            hazy_6 = hazy[:, 2000-48:, 4000-48:]
        hazy = [hazy_1, hazy_2, hazy_3, hazy_4, hazy_5, hazy_6]

    return hazy, vertical


def test_generate(hazy, vertical, cropping, MyEnsembleNet, device):

    #################### This part is for no crop, 1*6144*6144 #####################
    if cropping == 1:
        assert len(hazy) == 1, "cropping number not match len(hazy)"
        hazy = hazy.to(device)
        img_tensor = MyEnsembleNet(hazy)
        if vertical:
            img_tensor = img_tensor[72:6072, 1072:5072]
        else:
            img_tensor = img_tensor[1072:5072, 72:6072]


    #################### This part is for 1*4096*4096+2*2048*2048 #####################
    if cropping == 3:
        assert len(hazy) == 3, "cropping number not match len(hazy)"
        hazy_1, hazy_2, hazy_3 = hazy[0].to(device), hazy[1].to(device), hazy[2].to(device)

        out1 = MyEnsembleNet(hazy_1)
        out2 = MyEnsembleNet(hazy_2)
        out3 = MyEnsembleNet(hazy_3)

        if vertical:
            img_tensor_top = out1[:,:,48:-48, 48:-48]
            img_tensor_bot = torch.cat((out2[:,:,48:,0:2000], out3[:,:,48:,48:]), 3)
            img_tensor = torch.cat((img_tensor_top, img_tensor_bot), 2)
        else:
            img_tensor_left = out1[:,:,48:-48, 48:-48]
            img_tensor_right = torch.cat((out2[:,:,0:2000,48:], out3[:,:,48:,48:]), 2)
            img_tensor = torch.cat((img_tensor_left, img_tensor_right), 3)

    #################### This part is for 6*2048*2048 #####################
    if cropping == 6:
        assert len(hazy) == 6, "cropping number not match len(hazy)"
        hazy_1, hazy_2, hazy_3 = hazy[0].to(device), hazy[1].to(device), hazy[2].to(device)
        hazy_4, hazy_5, hazy_6 = hazy[3].to(device), hazy[4].to(device), hazy[5].to(device)
        out1 = MyEnsembleNet(hazy_1)
        out2 = MyEnsembleNet(hazy_2)
        out3 = MyEnsembleNet(hazy_3)
        out4 = MyEnsembleNet(hazy_4)
        out5 = MyEnsembleNet(hazy_5)
        out6 = MyEnsembleNet(hazy_6)

        if vertical:
            row1 = torch.cat((out1[:,:,0:2000,0:2000], out2[:,:,0:2000,48:]), 3)
            row2 = torch.cat((out3[:,:,24:2024,0:2000], out4[:,:,24:2024,48:]), 3)
            row3 = torch.cat((out5[:,:,48:,0:2000], out6[:,:,48:,48:]), 3)
            img_tensor = torch.cat((row1, row2, row3), 2)
        else:
            col1 = torch.cat((out1[:,:,0:2000,0:2000], out2[:,:,48:, 0:2000]), 2)
            col2 = torch.cat((out3[:,:,0:2000,24:2024], out4[:,:,48:, 24:2024]), 2)
            col3 = torch.cat((out5[:,:,0:2000,48:], out6[:,:,48:,48:]), 2)
            img_tensor = torch.cat((col1, col2, col3), 3)
    
    # add the fourth channel, all ones
    one_t = torch.ones_like(img_tensor[:,0,:,:])
    one_t = one_t[:, None, :, :]
    img_t = torch.cat((img_tensor, one_t) , 1)

    return img_t

# only for GPU Memory >= 40GB
def image_stick(imgs, vertical):
    out1 = imgs[0]
    out2 = imgs[1]
    out3 = imgs[2]
    out4 = imgs[3]
    if vertical:
        out5 = imgs[4]
        out6 = imgs[5]
        row1 = torch.cat((out1[:,:,0:2460,:], out2[:,:,1380:3840,:]), 2)
        row1 = torch.cat((row1[:,:,0:3540,:], out3[:,:,1380:3840,:]), 2)
        row2 = torch.cat((out4[:,:,0:2460,:], out5[:,:,1380:3840,:]), 2)
        row2 = torch.cat((row2[:,:,0:3540,:], out6[:,:,1380:3840,:]), 2)
        img_tensor = torch.cat((row1[:,:,:,0:2000], row2[:,:,:,1840:]), 3)
    else:
        row1 = torch.cat((out1[:,:,:,:3000], out2[:,:,:,840:]), 3)
        row2 = torch.cat((out3[:,:,:,:3000], out4[:,:,:,840:]), 3)
        img_tensor = torch.cat((row1[:,:,:2000,:], row2[:,:,1840:,:]), 2)
    return img_tensor


# only for GPU Memory >= 40GB
def image_stick(imgs, vertical):
    out1 = imgs[0]
    out2 = imgs[1]
    out3 = imgs[2]
    out4 = imgs[3]
    if vertical:
        out5 = imgs[4]
        out6 = imgs[5]
        row1 = torch.cat((out1[:,:,0:2460,:], out2[:,:,1380:3840,:]), 2)
        row1 = torch.cat((row1[:,:,0:3540,:], out3[:,:,1380:3840,:]), 2)
        row2 = torch.cat((out4[:,:,0:2460,:], out5[:,:,1380:3840,:]), 2)
        row2 = torch.cat((row2[:,:,0:3540,:], out6[:,:,1380:3840,:]), 2)
        img_tensor = torch.cat((row1[:,:,:,0:2000], row2[:,:,:,1840:]), 3)
    else:
        row1 = torch.cat((out1[:,:,:,:3000], out2[:,:,:,840:]), 3)
        row2 = torch.cat((out3[:,:,:,:3000], out4[:,:,:,840:]), 3)
        img_tensor = torch.cat((row1[:,:,:2000,:], row2[:,:,1840:,:]), 2)
    return img_tensor
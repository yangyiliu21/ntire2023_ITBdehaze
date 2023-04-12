import torch
import time
import argparse
from model import fusion_refine,Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset, dehaze_val_dataset_ohaze
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
from utils_test import to_psnr,to_ssim_skimage, test_generate, image_stick, image_stick_ohaze
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim

from config import get_config
from models import build_model

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_img/output_result_fromHuan_val4450_intrain2625')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
parser.add_argument('--ckpt_path', default='', type=str, help='path to model to be loaded')
parser.add_argument('--hazy_data', default='', type=str, help='apply on test data or val data')
parser.add_argument('--cropping', default='6', type=int, help='crop the 4k*6k image to # of patches for testing')

# --- SwinTransformer Parameter --- #
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )         # required
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
# easy config modification
parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# distributed training
#parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel') # required

# for acceleration
parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')           # --fused_window_process
parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')



args = parser.parse_args()

val_dataset = os.path.join(args.data_dir, args.hazy_data)        
predict_result= args.predict_result
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir=os.path.join(args.model_save_dir,'')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
# --- Define the Swin Transformer V2 model --- #
config = get_config(args)
swv2_model = build_model(config)

# --- Define the network --- #
if args.imagenet_model == 'SwinTransformerV2':
    MyEnsembleNet = fusion_refine(swv2_model, args.rcan_model)
elif args.imagenet_model == 'fromHuan':
    MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)

#SMyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

if args.hazy_data == "OHAZE_test":
    val_dataset = dehaze_val_dataset_ohaze(val_dataset, crop_method=args.cropping)
else:
    val_dataset = dehaze_val_dataset(val_dataset, crop_method=args.cropping)

val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet= torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)


# --- Load the network weight --- #
#MyEnsembleNet.load_state_dict(torch.load(args.ckpt_path)['state_dict']) torch.load('epoch7075.pkl')
MyEnsembleNet.load_state_dict(torch.load(args.ckpt_path))              
#MyEnsembleNet.load_state_dict(torch.load('epoch7075.pkl'))     
#MyEnsembleNet.load_state_dict(torch.load('./fromHuan/ckpt_huan/epoch4450.pkl'))      #'./ckpt_huan/epoch2625.pkl'
#MyEnsembleNet.module.load_state_dict(torch.load( './ckpt_huan/epoch50.pkl'))      #'./ckpt_huan/epoch2625.pkl'

# try:
#     MyEnsembleNet.load_state_dict(torch.load( 'epoch600.pkl'))
#     print('--- weight loaded ---')
# except:
#     print('--- no weight loaded ---')

# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    time_list = []
    MyEnsembleNet.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for batch_idx, (hazy,vertical) in enumerate(val_loader):   
    #for batch_idx, (hazy, vertical, hazy_1, hazy_2, hazy_3,hazy_4, hazy_5, hazy_6) in enumerate(val_loader):
        # print(len(val_loader))

        # for OHAZE
        if args.hazy_data == "OHAZE_test":
            index = vertical[0].to('cpu')
            ori_shape = vertical[1]
            img_list = []
            
            for input in hazy:
                input = input.to(device)
                img_tensor = MyEnsembleNet(input)
                img_list.append(img_tensor.cpu())

            final_image = image_stick_ohaze(img_list, index, ori_shape)
            assert final_image.shape[-1] == ori_shape[-1]
            # final_image.shape  ori_shape


            continue



        # Below for NTIRE2023
        if args.cropping == 4:
            img_list = []
            
            for input in hazy:
                input = input.to(device)
                #print(input.size())
                start = time.time()
                img_tensor = MyEnsembleNet(input)
                end = time.time()
                time_list.append((end - start))
                img_list.append(img_tensor.cpu())
        
            final_image = image_stick(img_list, vertical)
            one_t = torch.ones_like(final_image[:,0,:,:])
            one_t = one_t[:, None, :, :]
            img_t = torch.cat((final_image, one_t) , 1)
            imwrite(img_t, os.path.join(output_dir, str(batch_idx+41)+'.png'))

        else:
            start = time.time()

            img_tensor = test_generate(hazy, vertical, args.cropping, MyEnsembleNet, device)
        
            end = time.time()
            time_list.append((end - start))
            img_list.append(img_tensor)

            imwrite(img_list[batch_idx], os.path.join(imsave_dir, str(batch_idx + 41)+'.png'))
    time_cost = float(sum(time_list) / len(time_list))

    print('running time per image: ', time_cost)
                

# writer.close()









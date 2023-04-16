import torch
import time
import argparse
from model import fusion_refine,Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
from utils_test import to_psnr,to_ssim_skimage, cropping, test_generate
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim


from config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from timm.scheduler.cosine_lr import CosineLRScheduler


# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')     #path
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
parser.add_argument('--cropping', default='6', type=int, help='crop the 4k*6k image to # of patches for testing')
parser.add_argument('--generate', action='store_true', help='generate dehaze images or not during training')

parser.add_argument('--ckpt_path', default='', type=str, help='path to model to be loaded')
parser.add_argument('--finetune', action='store_true', help='finetune phase')


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


# for acceleration
parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')           # --fused_window_process
parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')


args = parser.parse_args()


# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch= args.train_epoch
train_dataset=os.path.join(args.data_dir, 'train')

# --- test --- #
test_dataset = os.path.join(args.data_dir, 'test')
val_dataset = os.path.join(args.data_dir, 'val')
predict_result= args.predict_result
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir=os.path.join(args.model_save_dir,'output_result')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# --- Define the Swin Transformer V2 model --- #
config = get_config(args)
swv2_model = build_model(config)

# --- Define the network --- #
if args.imagenet_model == 'SwinTransformerV2':
    MyEnsembleNet = fusion_refine(swv2_model, args.rcan_model)
elif args.imagenet_model == 'Res2Net':
    MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
else:
    raise Exception("Not a valid imagenet model")

print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
DNet = Discriminator()
print('# Discriminator parameters:', sum(param.numel() for param in DNet.parameters()))


# --- Build optimizer for SwinTransform--- #
G_optimizer = torch.optim.AdamW(params=MyEnsembleNet.parameters(), eps=config.TRAIN.OPTIMIZER.EPS, 
                                betas=config.TRAIN.OPTIMIZER.BETAS, lr=0.00004, weight_decay=1e-8)
scheduler_G = CosineLRScheduler(G_optimizer, t_initial=args.train_epoch, lr_min=2.5e-6)
# --- Build optimizer for Res2Net --- #
#G_optimizer = torch.optim.Adam(MyEnsembleNet.parameters(), lr=0.0001)
#scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[5000,7000,8000], gamma=0.5)

D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000,7000,8000], gamma=0.5)
# --- Load training data --- #
dataset = dehaze_train_dataset(train_dataset)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
# --- Load testing data --- #
test_dataset = dehaze_test_dataset(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

val_dataset = dehaze_val_dataset(val_dataset, crop_method=args.cropping)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet= torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)

# finetune
if args.finetune:
    MyEnsembleNet.load_state_dict(torch.load(args.ckpt_path)['state_dict']) 
    G_optimizer.load_state_dict(torch.load(args.ckpt_path)['optimizer']) 
    # check model location


DNet = DNet.to(device)
DNet= torch.nn.DataParallel(DNet, device_ids=device_ids)
writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True)
# vgg_model.load_state_dict(torch.load(os.path.join(args.vgg_model , 'vgg16.pth')))
vgg_model = vgg_model.features[:16].to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()

msssim_loss = msssim


#--- Load the network weight for finetuning--- #
if args.finetune:
	try:
	    MyEnsembleNet.load_state_dict(torch.load('epoch100000.pkl')))  # load finetune model
	    print('--- weight loaded ---')
	except:
	    print('--- no weight loaded ---')

# --- Strat training --- #
iteration = 0
best=0
msg_list = []
for epoch in range(train_epoch):
    start_time = time.time()
    scheduler_G.step(epoch=epoch)
    scheduler_D.step()
    MyEnsembleNet.train()
    DNet.train()
    print(epoch)
    for batch_idx, (hazy, clean) in enumerate(train_loader):
        # print(batch_idx)
        iteration +=1
        hazy = hazy.to(device)
        clean = clean.to(device)
        output= MyEnsembleNet(hazy)

        DNet.zero_grad()
        real_out = DNet(clean).mean()
        fake_out = DNet(output).mean()
        D_loss = 1 - real_out + fake_out
        # no more forward
        D_loss.backward(retain_graph=True)
        MyEnsembleNet.zero_grad()
        adversarial_loss = torch.mean(1 - fake_out)
        smooth_loss_l1 = F.smooth_l1_loss(output, clean)
        perceptual_loss = loss_network(output, clean)
        msssim_loss_ = -msssim_loss(output, clean, normalize=True)
        total_loss = smooth_loss_l1+0.01 * perceptual_loss+ 0.0005 * adversarial_loss+ 0.5*msssim_loss_

        total_loss.backward()
        D_optim.step()
        G_optimizer.step()

        writer.add_scalars('training', {'training total loss': total_loss.item()
                                        }, iteration)
        writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1.item(),
                                                'perceptual': perceptual_loss.item(),
                                            'msssim':msssim_loss_.item()
                                            
                                               }, iteration)
        writer.add_scalars('GAN_training', {
            'd_loss':D_loss.item(),
            'd_score':real_out.item(),
            'g_score':fake_out.item()
            }, iteration
            )


    if epoch % 25 == 0:   
        print('we are testing on epoch: ' + str(epoch))  
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            recon_psnr_list = []
            recon_ssim_list = []
            MyEnsembleNet.eval()
            for batch_idx, (hazy, clean) in enumerate(test_loader):
                clean = clean.to(device)
                hazy = hazy.to(device)
                img_tensor = MyEnsembleNet(hazy)

                psnr_list.extend(to_psnr(img_tensor, clean))
                ssim_list.extend(to_ssim_skimage(img_tensor, clean))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            print(epoch,'dehazed', avr_psnr, avr_ssim)
            frame_debug = torch.cat((img_tensor,clean), dim =0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr':avr_psnr,
                'testing ssim': avr_ssim
                                    }, epoch)

            msg = 'epoch'+ str(epoch) + ', avr_psnr:' + str(avr_psnr) + ', avr_ssim:' +str(avr_ssim)
            msg_list.append(msg)

            if (avr_psnr > best):
                best = avr_psnr
                torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir,'epoch'+ str(epoch) + '.pkl'))

            if (epoch > 2000 and epoch <= 8000):
                checkpoint = {
                    'state_dict': MyEnsembleNet.state_dict(),
                    'optimizer': G_optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(args.model_save_dir,'epoch'+ str(epoch) + '.pkl'))

        
        # generation while testing
        if args.generate:
            print('we are generating at epoch: ' + str(epoch))
            img_list = []
            time_list = []
            MyEnsembleNet.eval()
            subfolder = 'epoch' + str(epoch)
            imsave_dir = os.path.join(output_dir, subfolder)
            if not os.path.exists(imsave_dir):
                os.makedirs(imsave_dir)
            for batch_idx, (hazy,vertical) in enumerate(val_loader):   
                img_tensor = test_generate(hazy, vertical, args.cropping, MyEnsembleNet, device) 
                img_list.append(img_tensor)
                imwrite(img_list[batch_idx], os.path.join(imsave_dir, str(batch_idx + 41)+'.png'))


file = open('test_info.txt','w')
for item in msg_list:
	file.write(item+"\n")
file.close()

                

writer.close()

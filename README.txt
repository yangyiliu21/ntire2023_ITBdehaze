README for ITBdehaze

Dependencies and Installation:
python3.7
PyTorch >= 1.0
NVIDIA GPU+CUDA
numpy
matplotlib
timm 0.6.12
yaml
yacs
tensorboardX(optional)

Details of packages could be found in the file: environment.yml

Path to our best traied model: ./checkpoints/epoch5925.pkl


Command line for training:
python train.py --data_dir data_23 --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml -train_batch_size 8 --model_save_dir train_result -train_epoch 6500



Command line for testing:
python test.py --imagenet_model SwinTransformerV2 --cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --model_save_dir ./output_img/test/epoch5925_result --ckpt_path ./checkpoints/epoch5925.pkl --hazy_data NTIRE2023_Test --cropping 4

* Using this command line for generating outputs of test data, the dehazed results could be found in: ./output_img/test/epoch5925_result
* This testing command line requires GPU memory >= 40 GB to ensure best results
  If GPU memory < 40 GB, please use " --cropping 6 " instead
import os
import argparse

home_path = os.getenv("HOME")

parser = argparse.ArgumentParser(description='SAR-HEVC-Compression')

# Init
parser.add_argument("--mode", type=str, default="train", 
                    help="train or test")
parser.add_argument('--model', type=str, default="MST",
                    help='EDSR or MST or MSTpp')
parser.add_argument('--dataset', type=str, default="Sandia",
                    help="Sandia or AFRL")
parser.add_argument("--pre_train", type=bool, default=False, 
                    help="load pretrain model or not?")
parser.add_argument("--start_epoch", type=bool, default=0,
                    help="Start epoch while training.")
parser.add_argument("--which_sar", type=str, default="complex",
                    help="real or imaginary")
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_colors', type=int, default=2,
                    help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=2,
                    help='number of output color channels to use')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function[relu, prelu, leakyrelu, gelu]')
parser.add_argument('--deformable', type=bool, default=True,
                    help="Use deformable convolution in the convolution")
parser.add_argument('--multi_loss', type=int, default=0,
                    help="multi loss, 0: SAR L1 , 1: SAR L1 + AMP L1, 2: SAR_L1 + AMP L1 + PHASE L1")

# MST
parser.add_argument('--downsample_1', type=int, default=4,
                    help='downsample for first input')
parser.add_argument('--downsample_2', type=int, default=16,
                    help='downsample for second input')
parser.add_argument('--n_feats_1', type=int, default=64,
                    help='number of feature maps fro first input')
parser.add_argument('--n_feats_2', type=int, default=64,
                    help='number of feature maps for second input')

# EDSR
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--scale', type=str, default='1',
                    help='super resolution scale')

# Path
parser.add_argument('--model_path', type=str, default="/home/pmc4p/PythonDir/SAR-HEVC-Deblocking-master/ckpt",
                    help='path of model')
parser.add_argument('--train_dataset_path', type=str, default='PythonDir/dataset/SAR_dataset/nonuniform/AFRL_nonuniform_SAR_HEVC_ps256qp13_train',
                    help='dataset directory')
parser.add_argument('--validation_dataset_path', type=str, default='PythonDir/dataset/SAR_dataset/nonuniform/AFRL_nonuniform_SAR_HEVC_ps256qp13_train',
                    help='dataset directory')
# parser.add_argument('--test_dataset_path', type=str, default='PythonDir/dataset/SAR_dataset/Sandia_SAR_HEVC_qp21_test',
#                     help='dataset directory')

# parameters
parser.add_argument('--save_every', type=int, default=1,
                    help='save model per every N epoch')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')

args = parser.parse_args()

args.model_path = os.path.join(home_path, args.model_path)
if not os.path.exists(args.model_path):
    print("No model directory found.")
    if args.mode == "train":
        os.makedirs(args.model_path)
        print("New model directory created.")
    else:
        exit()
args.validation_dataset_path = os.path.join(home_path, args.validation_dataset_path)
if not os.path.exists(args.validation_dataset_path):
    print("No validation dataset found.")
args.train_dataset_path = os.path.join(home_path, args.train_dataset_path)
if not os.path.exists(args.train_dataset_path):
    print("No train dataset found.")
    exit()
if args.dataset == "Sandia":
    args.min_val = -500
    args.max_val = 500
else:
    args.min_val = -5000
    args.max_val = 5000
if (args.which_sar == "complex") & (args.n_colors == 1 | args.o_colors == 1):
    args.n_colors = 2
    args.o_colors = 2
    print("Complex data used for training. Input and Output of network changed to 2 channels.")
args.scale = list(map(lambda x: int(x), args.scale.split('+')))

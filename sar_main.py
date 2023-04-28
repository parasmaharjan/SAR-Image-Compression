# Author: Paras Maharjan
# Date: 2023-04-11
# Description: Main file for training and testing SAR images
# Cmd MSTpp: python sar_main.py --model MSTpp --multi_loss 0 
# Cmd MST without amp loss for qp 13: python sar_main.py --model MST --multi_loss 0 --train_dataset_path PythonDir/dataset/SAR_dataset/Sandia_SAR_HEVC_ps256qp13_train --validation_dataset_path PythonDir/dataset/SAR_dataset/Sandia_SAR_HEVC_ps256qp13_validation
# Cmd MST without amp loss for qp 21: python sar_main.py --model MST --multi_loss 0 --train_dataset_path PythonDir/dataset/SAR_dataset/Sandia_SAR_HEVC_ps256qp21_train --validation_dataset_path PythonDir/dataset/SAR_dataset/Sandia_SAR_HEVC_ps256qp21_validation
# Cmd MST with amp loss: python sar_main.py --model MST --multi_loss 1

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim

#from model import edsr
from mst_model.deformable_mst import EDSR
#from mst_model.deformable_mstpp import MST_Plus_Plus
from option import args
from sar_dataloader import SARDataset
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


home_path = os.getenv("HOME")
# crf = args.train_dataset_path[args.train_dataset_path.find("crf")+3:args.train_dataset_path.find("crf")+5]
qp = args.train_dataset_path[args.train_dataset_path.find("qp")+2:args.train_dataset_path.find("qp")+4]
print("***** qp: ", qp)
writer = SummaryWriter("./runs/%s_%s_ps256rb%d_qp%s_%s"%(args.model, args.dataset, args.n_resblocks, qp, args.which_sar))

device = torch.device('cuda')
nbit = 12
amp_max_val = np.sqrt(2*args.max_val**2)

def train(args):
    if args.model == "MST":
        print("***** Using MST model")
        model = EDSR(args)
    elif args.model == "EDSR":
        print("***** Using MST model")
        #model = edsr.EDSR(args)
    elif args.model == "MSTpp":
        print("***** Using MST++ model")
        #model = MST_Plus_Plus()
    if args.pre_train:
        print("***** Using pre-trained model")
        #model.load_state_dict(torch.load(os.path.join(args.model_path, "mst_model_sandia_ps128rb%d_qp%s_%s.pt"%(args.n_resblocks, qp, args.which_sar))))
        model.load_state_dict(torch.load(os.path.join(args.model_path, "mst_model_AFRL_ps256rb8_qp13_complex_best.pt")))
    model.to(device)
    #print(model)
    if args.multi_loss == 1:
        print("***** Loss = SAR L1 + amp L1")
        amp_loss = nn.L1Loss()
    sar_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # if args.pre_train:
    #     model.load_state_dict(torch.load(os.path.join(args.model_path, "mst_deformable_optimizer_sandia_ps128rb8_qp21_e23_complex.pt")))
    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 200, 250], gamma=0.1)

    train_data = SARDataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    if os.path.exists(args.validation_dataset_path):
        validation_data = SARDataset(args)
        validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    best_val_psnr = 0.0
    alpha = 0.0005
    for epoch in range(args.start_epoch, args.epochs):
        cur_lr = lr_scheduler.get_last_lr()
        print("epoch: ", epoch, "lr: ", cur_lr[0])
        writer.add_scalar("LR", cur_lr[0], epoch)
        model.train()
        total_loss = 0
        for batch, (input_img, gt_img, amp_img, pha_image) in enumerate(train_loader):
            # prep data
            # plt.figure();plt.subplot(1,2,1);plt.imshow(input_img[0,0,:,:]);plt.subplot(1,2,2);plt.imshow(gt_img[0,0,:,:]);plt.show();
            input_img = input_img.to(device)
            gt_img = gt_img.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            pred_sar_img = model(input_img)

            # Compute loss
            loss_1 = sar_loss(gt_img, pred_sar_img)
            writer.add_scalar("Loss/SAR loss", loss_1, epoch*len(train_loader)+batch)
            if args.multi_loss == 1:
                amp_img = amp_img.to(device)
                pred_sar_dequant_img = ((args.max_val - args.min_val)*pred_sar_img) + args.min_val
                pred_amp_img = torch.sqrt(pred_sar_dequant_img[:,0,:,:]**2 + pred_sar_dequant_img[:,1,:,:]**2)
                loss_2 = amp_loss(amp_img, pred_amp_img)
                writer.add_scalar("Loss/amp loss", loss_2, epoch*len(train_loader)+batch)
                
                loss = (1-alpha)*loss_1 + alpha*loss_2
            else:
                loss = loss_1
            #writer.add_scalar("Loss/Total loss", loss, epoch*len(train_loader)+batch)
            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Add to the total loss for this epoch
            total_loss += loss.item()
            writer.add_scalar("Loss/Avg loss", total_loss/(batch+1), epoch*len(train_loader)+batch)
            print("%s | epoch %d | batch %d | SAR loss %.5f | Avg loss %.5f |"%(args.which_sar, epoch, batch, loss_1.item(), total_loss/(batch+1)))
        lr_scheduler.step()
        if epoch%args.save_every == 0:
            # if epoch == 0:
            #     if os.path.exists(os.path.join(args.model_path, "%s_model_sandia_ps256rb%d_qp%s_%s.pt"%(args.model, args.n_resblocks,qp,args.which_sar))):
            #         print("***** Model name already exists.")
            #         pdb.set_trace()

            if os.path.exists(args.validation_dataset_path):
                val_psnr = validate(args, model, validation_loader, epoch*len(train_loader)+batch)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                print("***** Best model epoch: ", epoch)
                
                if args.multi_loss == 1:
                    torch.save(model.state_dict(), os.path.join(args.model_path, "%s_amploss%.4f_%s_ps256rb%d_qp%s_%s_best.pt"%(args.model, alpha, args.dataset, args.n_resblocks,qp,args.which_sar)))
                else:
                    torch.save(model.state_dict(), os.path.join(args.model_path, "%s_%s_ps256rb%d_qp%s_%s_best.pt"%(args.model, args.dataset, args.n_resblocks,qp,args.which_sar)))

            if args.multi_loss == 1:
                torch.save(model.state_dict(), os.path.join(args.model_path, "%s_amploss%.4f_%s_ps256rb%d_qp%s_%s.pt"%(args.model, alpha, args.dataset, args.n_resblocks,qp,args.which_sar)))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_path, "%s_%s_ps256rb%d_qp%s_%s.pt"%(args.model, args.dataset, args.n_resblocks,qp,args.which_sar)))

            
def validate(args, model, validate_loader, step):
    model.eval()
    psnr = 0
    mae = 0
    with torch.no_grad():
        for batch, (input_img, gt_img, amp_img, pha_image) in enumerate(validate_loader):
            # prep data
            input_img = input_img.to(device)
            gt_img = gt_img.numpy()

            # Forward pass
            pred_sar_img = model(input_img).data.cpu().numpy()
            # Compute metrics
            psnr += peak_signal_noise_ratio(gt_img, pred_sar_img)
            mae += np.abs(gt_img - pred_sar_img).mean()
        writer.add_scalar("Validation/psnr", psnr/len(validate_loader), step)
        writer.add_scalar("Validation/mae", mae/len(validate_loader), step)
        print("*****************Validation ********************")
        print(args.which_sar, "psnr: ", psnr/len(validate_loader), "mse: ", mae/len(validate_loader))
        return (psnr/len(validate_loader))
        

def test(args):
    pass

if __name__=="__main__":

    if args.mode == 'train':
        train(args)
    else:
        test(args)

    print("Done...")
# Author: Paras Maharjan
# Date: 2023-04-11
# Description: Main file for training and testing SAR images
#
# Cmd MST Sandia nonuniform quantization qp 13
# python inference_sandia_complex.py --dataset Sandia --model MST --train_dataset_path PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp13_train --validation_dataset_path PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp13_validation
#
# Cmd MST uniform quantization qp 13
# python inference_sandia_complex.py --dataset Sandia --model MST --train_dataset_path PythonDir/dataset/SAR_dataset/uniform/Sandia/Sandia_uniform_SAR_HEVC_ps256qp13_train --validation_dataset_path PythonDir/dataset/SAR_dataset/uniform/Sandia/Sandia_uniform_SAR_HEVC_ps256qp13_validation



import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

#from model import edsr
from mst_model.deformable_mst import EDSR
#from mst_model.deformable_mstpp import MST_Plus_Plus
from option import args
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from pytorch_msssim import MS_SSIM

home_path   = os.getenv("HOME")
device      = torch.device('cuda')
nbit        = 12
qp          = args.train_dataset_path[args.train_dataset_path.find("qp")+2:args.train_dataset_path.find("qp")+4]
print("***** qp: ", qp)
min_val     = -500
max_val     = 500
amp_max_val = np.sqrt(2*max_val**2)

def test(args):
    if args.model == "EDSR":
        print("***** Using EDSR model")
        #model_real = edsr.EDSR(args)
        #model_imaginary = edsr.EDSR(args)
    elif args.model == "MSTpp":
        print("***** Using MST++ model")
        if args.which_sar == "complex":
            #model = MST_Plus_Plus()
            #model.load_state_dict(torch.load(os.path.join(args.model_path, "test_model_sandia_ps256rb8_qp13_complex_best.pt")))
            #model.to(device)
            #print("total number of parameters (complex): ", sum(p.numel() for p in model.parameters()))
            #model.eval()
            pass
        else:
            print("No module found")
            exit()
    elif args.model == "MST":
        print("***** Using MST model")
        if args.which_sar == "complex":
            model = EDSR(args)
            model.load_state_dict(torch.load(os.path.join(args.model_path, "%s_%s_amploss0.00005_uniform_ps256rb8_qp%s_complex_best.pt"%(args.model, args.dataset, qp))))
            #model.load_state_dict(torch.load("/home/pmc4p/PythonDir/SAR-HEVC-Deblocking/src/ckpt/mst_deformable3_model_sandia_ps256rb8_qp21_complex.pt"))
            model.to(device)
            print("total number of parameters (complex): ", sum(p.numel() for p in model.parameters()))
            model.eval()
        else:
            model_real = EDSR(args)
            model_real.load_state_dict(torch.load(os.path.join(args.model_path, "mst_model_sandia_ps128rb%d_qp%d_real.pt"%(args.n_resblocks,qp))))
            model_imaginary = EDSR(args)
            model_imaginary.load_state_dict(torch.load(os.path.join(args.model_path, "mst_model_sandia_ps128rb%d_qp%d_imaginary.pt"%(args.n_resblocks,qp))))
            model_real.to(device)
            model_imaginary.to(device)
            print("total number of parameters: ", sum(p.numel() for p in model_real.parameters()) + sum(p.numel() for p in model_imaginary.parameters()))
            #print(model_real)
            model_real.eval()
            model_imaginary.eval()

    with torch.no_grad():
        #pdb.set_trace()
        ori_sar = np.load('/home/pmc4p/PythonDir/dataset/SAR_dataset/uniform/Sandia/Sandia_uniform_SAR_HEVC_ps256qp%s_test/gt/0.npy'%qp)
        ori_quant = (2**nbit - 1) * (ori_sar - min_val)/(max_val - min_val)
        ori_quant = np.round(ori_quant).astype(np.uint16)
        print("gt sar", ori_quant.shape, ori_quant.min(), ori_quant.max())

        rec_sar = np.load('/home/pmc4p/PythonDir/dataset/SAR_dataset/uniform/Sandia/Sandia_uniform_SAR_HEVC_ps256qp%s_test/input/0.npy'%qp)
        rec_sar = rec_sar/4095.0
	    #rec_ch1 = np.minimum(rec_ch1, 4095)
        #rec_ch1 = rec_ch1/4095
        print("recon sar", rec_sar.shape, rec_sar.min(), rec_sar.max())
        
        # ori_ch2 = cv2.imread('/home/pmc4p/PythonDir/dataset/SAR_dataset/SAR_HEVC_qp%d_test/imaginary.png'%qp, cv2.IMREAD_ANYDEPTH)
        # rec_ch2 = cv2.imread('/home/pmc4p/PythonDir/dataset/SAR_dataset/SAR_HEVC_qp%d_test/recon_imaginary.png'%qp, cv2.IMREAD_ANYDEPTH)
        # rec_ch2 = np.minimum(rec_ch2, 4095)
        # rec_ch2 = rec_ch2/4095
        # print("rec ch2", rec_ch2.shape, rec_ch2.min(), rec_ch2.max())

        amp_img = np.load('/home/pmc4p/PythonDir/dataset/SAR_dataset/uniform/Sandia/Sandia_uniform_SAR_HEVC_ps256qp%s_test/amp/0.npy'%qp)
        amp_img = amp_img/amp_max_val
        print("GT Amp", amp_img.shape, amp_img.min(), amp_img.max())

        pha_img = np.load('/home/pmc4p/PythonDir/dataset/SAR_dataset/uniform/Sandia/Sandia_uniform_SAR_HEVC_ps256qp%s_test/phase/0.npy'%qp)
        # pha_img = pha_img[:2048, :2048]
        # plt.imshow(amp_img, cmap="gray")
        # plt.show()
        
        #img_recon = np.stack((rec_ch1, rec_ch2), axis=2)
        #img_recon_dequant = np.round( ((max_val - min_val)*img_recon.astype(np.float32)/(2**nbit - 1)) + min_val )
        #print(img_recon_dequant.shape)

        # REAL
        if args.which_sar == "complex":
            input_img = torch.permute(torch.tensor(np.expand_dims(rec_sar.astype(np.float32), axis=0)), (0,3,1,2)).to(device)
            print("input image", input_img.shape, input_img.data.cpu().min(), input_img.data.cpu().max())
            pred_sar_img = torch.permute(model(input_img).data.cpu(), (0,2,3,1)).numpy().astype(np.float64)
            pred_sar_img = np.clip(pred_sar_img[0,:,:,:]*4095, 0, 4095)
            print("pred image", pred_sar_img.shape, pred_sar_img.min(), pred_sar_img.max())
            print("Complex PSNR: %f"%(peak_signal_noise_ratio(ori_quant/4095.0, pred_sar_img/4095.0)))
        else:
            input_real_img = torch.permute(torch.tensor(np.expand_dims(np.expand_dims(rec_sar[:,:,0].astype(np.float32), axis=2), axis=0)), (0,3,1,2)).to(device)
            print("input image", input_img.shape, input_img.min(), input_img.max())
            pred_real_img = torch.permute(model_real(input_real_img).data.cpu(), (0,2,3,1)).numpy().astype(np.float64)
            pred_real_img = np.clip(pred_real_img[0,:,:,0]*4095, 0, 4095)
            print("pred real", pred_real_img.shape, pred_real_img.min(), pred_real_img.max())
            print("Real PSNR: %f"%(peak_signal_noise_ratio(ori_quant[:,:,0]/4095.0, pred_real_img/4095.0)))

            # IMAGINARY
            input_imaginary_img = torch.permute(torch.tensor(np.expand_dims(np.expand_dims(rec_sar[:,:,1].astype(np.float32), axis=2), axis=0)), (0,3,1,2)).to(device)
            print("input imaginary", input_imaginary_img.shape)
            pred_imaginary_img = torch.permute(model_imaginary(input_imaginary_img).data.cpu(), (0,2,3,1)).numpy().astype(np.float64)
            pred_imaginary_img = np.clip(pred_imaginary_img[0,:,:,0]*4095, 0, 4095)
            print("Pred imaginary", pred_imaginary_img.shape, pred_imaginary_img.min(), pred_imaginary_img.max())
            print("Imaginary PSNR: %f"%(peak_signal_noise_ratio(ori_quant[:,:,1]/4095.0, pred_imaginary_img/4095.0)))

            # AMPLITUDE
            #ori_sar_img = np.stack((rec_sar[:,:,0], rec_), axis=2)*4095.0
            #ori_sar_img = np.round(((max_val - min_val)*ori_sar_img/(2**nbit - 1)) + min_val )

            pred_sar_img = np.stack((pred_real_img, pred_imaginary_img), axis=2)
            print("pred SAR", pred_sar_img.shape, pred_sar_img.min(), pred_sar_img.max())

        pred_sar_img = np.round(((max_val - min_val)*pred_sar_img/(2**nbit - 1)) + min_val )
        pred_amp_img = np.sqrt(pred_sar_img[:,:, 0]**2+pred_sar_img[:,:,1]**2)/amp_max_val
        print("pred amp", pred_amp_img.shape, pred_amp_img.min(), pred_amp_img.max())

        psnr = peak_signal_noise_ratio(pred_amp_img, amp_img)
        print("Amplitude PSNR: ", psnr, "dB")

        ms_ssim_amp_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)
        amp_msssim = ms_ssim_amp_module(torch.tensor(np.expand_dims(np.expand_dims(pred_amp_img, axis=0), axis=1)), torch.tensor(np.expand_dims(np.expand_dims(amp_img, axis=0), axis=0)))
        print("Amplitude MS-SSIM: ", amp_msssim)

        pred_pha_img = np.arctan2(pred_sar_img[:,:, 1],pred_sar_img[:,:, 0])
        print("pred pha", pred_pha_img.shape, pred_pha_img.min(), pred_pha_img.max())
        pha_mse = mean_squared_error(pred_pha_img, pha_img)
        print("Phase MSE: ", pha_mse)
        ms_ssim_pha_module = MS_SSIM(data_range=3.141592653589793, size_average=True, channel=1)
        phase_msssim = ms_ssim_pha_module(torch.tensor(np.expand_dims(np.expand_dims(pred_pha_img, axis=0), axis=1)), torch.tensor(np.expand_dims(np.expand_dims(pha_img, axis=0), axis=0)))
        print("Phase MS-SSIM: ", phase_msssim)

        # from scipy.fftpack import dct
        # dct_img = dct(dct(ori_sar_img[32:32+8,100:100+8,0], axis=0), axis=1)
        # dct_recon_img = dct(dct(pred_sar_img[32:32+8,100:100+8,0], axis=0), axis=1)
        # plt.subplot(1,2,1)
        # plt.imshow(dct_img, cmap="gray")
        # plt.subplot(1,2,2)
        # plt.imshow(dct_recon_img, cmap="gray")
        # plt.show()
        if 1:
            #plt.subplot(1,3,1)
            #plt.imshow(amp_img, cmap="gray")
            #plt.title("Original Amplitude")
            #plt.subplot(1,3,2)
            plt.imshow(pred_amp_img, cmap="gray")
            plt.title("Proposed SAR amplitude reconstruction, PSNR: %.4f dB"%(psnr))
            plt.axis("off")
            #plt.subplot(1,3,3)
            #plt.imshow(pred_amp_img, cmap="gray")
            #plt.title("Reconstructed Amplitude, PSNR: %.2f dB, BPP: 0.4097"%(psnr))
            plt.show()


if __name__=="__main__":
    test(args)

    print("Done...")

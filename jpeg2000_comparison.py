import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sarpy.io.general.nitf as nitf
import time
import torch

# from option import args
from pathlib import Path
from scipy.io import loadmat
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from pytorch_msssim import MS_SSIM

home_path        = os.getenv("HOME")
debug            = 1
uniform          = 1
dataset          = "AFRL"   # "Sandia"
mode             = "test2"
n_bits           = 12
if dataset == "AFRL":
    min_val      = -5000
    max_val      = 5000
elif dataset == "Sandia":
    min_val      = -500
    max_val      = 500
amp_max_val      = np.sqrt(min_val**2 + max_val**2)
ps               = 1024
qp               = 21
jp2k_quality     = 9
temp_dir         = "./frames_jpeg2000"

if uniform == 1:
    if dataset == "AFRL":
        ori_sar_path     = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_uniform_%s/sicd_example_2_PFA_RE32F_IM32F_HH_gt_sar.mat"%(mode))
        ori_sar_amp_path = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_uniform_%s/sicd_example_2_PFA_RE32F_IM32F_HH_amp.mat"%(mode))
        ori_sar_pha_path = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_uniform_%s/sicd_example_2_PFA_RE32F_IM32F_HH_phase.mat"%(mode))
        real_path        = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_uniform_%s/sicd_example_2_PFA_RE32F_IM32F_HH_quant_real.png"%(mode))
        imaginary_path   = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_uniform_%s/sicd_example_2_PFA_RE32F_IM32F_HH_quant_imaginary.png"%(mode))
    elif dataset == "Sandia":
        ori_sar_path     = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_test/MiniSAR20050519p0006image008_gt_sar.mat")
        ori_sar_amp_path = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_test/MiniSAR20050519p0006image008_amp.mat")
        ori_sar_pha_path = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_test/MiniSAR20050519p0006image008_phase.mat")
        real_path        = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_test/MiniSAR20050519p0006image008_quant_real.png")
        imaginary_path   = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_test/MiniSAR20050519p0006image008_quant_imaginary.png")
else:
    if dataset == "AFRL":
        ori_sar_path     = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_nonuniform_test/sicd_example_2_PFA_RE32F_IM32F_HH_gt_sar.mat")
        ori_sar_amp_path = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_nonuniform_test/sicd_example_2_PFA_RE32F_IM32F_HH_amp.mat")
        real_path        = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_nonuniform_test/sicd_example_2_PFA_RE32F_IM32F_HH_quant_real.png")
        imaginary_path   = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/AFRL_nonuniform_test/sicd_example_2_PFA_RE32F_IM32F_HH_quant_imaginary.png")
    elif dataset == "Sandia":
        ori_sar_path     = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_test/MiniSAR20050519p0006image008_gt_sar.mat")
        ori_sar_amp_path = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_test/MiniSAR20050519p0006image008_amp.mat")
        real_path        = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_test/MiniSAR20050519p0006image008_quant_real.png")
        imaginary_path   = os.path.join(home_path, "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_test/MiniSAR20050519p0006image008_quant_imaginary.png")

# Load original SAR image and SAR amplitude image
ori_sar_clipped = loadmat(ori_sar_path)['sar_image'].astype(np.float32)
ori_sar_amp     = loadmat(ori_sar_amp_path)['sar_amp'].astype(np.float32)
ori_sar_pha     = loadmat(ori_sar_pha_path)['sar_phase'].astype(np.float32)

# ENCODE
# https://stackoverflow.com/questions/66155414/convert-16bit-grayscale-png-to-hevc-x265
cmd_jp2_encode_real       = "ffmpeg -y -i %s -vcodec libopenjpeg -compression_level %d -pix_fmt gray16le %s/output_real.jp2"%(real_path, jp2k_quality, temp_dir)
cmd_jp2_encode_imaginary  = "ffmpeg -y -i %s -vcodec libopenjpeg -compression_level %d -pix_fmt gray16le %s/output_imaginary.jp2"%(imaginary_path, jp2k_quality, temp_dir)
cmd_hevc_encode_real      = "ffmpeg -y -i %s -c:v libx265 -qp %d -pix_fmt gray12le %s/output_real.mp4"%(real_path, qp, temp_dir)
cmd_hevc_encode_imaginary = "ffmpeg -y -i %s -c:v libx265 -qp %d -pix_fmt gray12le %s/output_imaginary.mp4"%(imaginary_path, qp, temp_dir)

print("\n\n------------HEVC ENCODING real-----------------------")
os.system(cmd_hevc_encode_real)
print("\n\n------------HEVC ENCODING imaginary-----------------------")
os.system(cmd_hevc_encode_imaginary)

print("\n\n------------Jpeg2000 ENCODING real-----------------------")
os.system(cmd_jp2_encode_real)
print("\n\n------------Jpeg2000 ENCODING imaginary-----------------------")
os.system(cmd_jp2_encode_imaginary)

cmd_jp2_decode_real       = "ffmpeg -y -i %s/output_real.jp2 -pix_fmt gray16be %s/recon_real_jp2.png"%(temp_dir, temp_dir)
cmd_jp2_decode_imaginary  = "ffmpeg -y -i %s/output_imaginary.jp2 -pix_fmt gray16be %s/recon_imaginary_jp2.png"%(temp_dir, temp_dir)
cmd_hevc_decode_real      = "ffmpeg -y -i %s/output_real.mp4 -pix_fmt gray16be %s/recon_real_hevc.png"%(temp_dir, temp_dir)
cmd_hevc_decode_imaginary = "ffmpeg -y -i %s/output_imaginary.mp4 -pix_fmt gray16be %s/recon_imaginary_hevc.png"%(temp_dir, temp_dir)

print("\n\n------------HEVC DECODING real-----------------------")
os.system(cmd_hevc_decode_real)

print("\n\n------------HEVC DECODING imaginary-----------------------")
os.system(cmd_hevc_decode_imaginary)

print("\n\n------------Jpeg2000 DECODING real-----------------------")
os.system(cmd_jp2_decode_real)
print("\n\n------------Jpeg2000 DECODING imaginary-----------------------")
os.system(cmd_jp2_decode_imaginary)

print("\n\n------Results nbit: %d HEVC qp: %d-------"%(n_bits, qp))
f_size_real_hevc      = Path('%s/output_real.mp4'%(temp_dir)).stat().st_size
f_size_imaginary_hevc = Path('%s/output_imaginary.mp4'%(temp_dir)).stat().st_size
bpp_hevc = (f_size_real_hevc + f_size_imaginary_hevc)*8/np.prod((ps,ps,2))
print('HEVC bitstream-size: %.2f bytes ( %.4f bpp)'%((f_size_real_hevc + f_size_imaginary_hevc), bpp_hevc))

recon_real_hevc = cv2.imread('%s/recon_real_hevc.png'%(temp_dir), cv2.IMREAD_ANYDEPTH)
recon_imaginary_hevc = cv2.imread('%s/recon_imaginary_hevc.png'%(temp_dir), cv2.IMREAD_ANYDEPTH)
recon_hevc = np.stack((np.minimum(recon_real_hevc, (2**n_bits -1)), np.minimum(recon_imaginary_hevc, (2**n_bits -1))), axis=2)
recon_dequant_hevc = np.round( ((max_val - min_val)*recon_hevc.astype(np.float32)/(2**n_bits - 1)) + min_val )

# psnr_real_hevc = peak_signal_noise_ratio(recon_hevc[:,:,0]/(2**n_bits -1), ori_sar_clipped[:,:,0]/(2**n_bits -1))
# psnr_imaginary_hevc = peak_signal_noise_ratio(recon_hevc[:,:,1]/(2**n_bits -1), ori_sar_clipped[:,:,1]/(2**n_bits -1))

hevc_recon_amp = np.sqrt(recon_dequant_hevc[:,:,0]**2 + recon_dequant_hevc[:,:,1]**2)
hevc_recon_amp_psnr = peak_signal_noise_ratio(hevc_recon_amp/amp_max_val, ori_sar_amp/amp_max_val)

# print("HEVC---> PSNR real: %.4f \t PSNR imaginary: %.4f \t PSNR amp: %.4f"%(psnr_real_hevc, psnr_imaginary_hevc, hevc_recon_amp_psnr))
print("HEVC PSNR amp: %.4f"%(hevc_recon_amp_psnr))

# HEVC AMP MS-SSIM
ms_ssim_amp_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)
amp_msssim = ms_ssim_amp_module(torch.tensor(np.expand_dims(np.expand_dims(hevc_recon_amp/amp_max_val, axis=0), axis=1)), torch.tensor(np.expand_dims(np.expand_dims(ori_sar_amp/amp_max_val, axis=0), axis=0)))
print("HEVC Amplitude MS-SSIM: ", amp_msssim)

hevc_recon_pha = np.arctan2(recon_dequant_hevc[:,:,1], recon_dequant_hevc[:,:,0])
# HEVC PHA MS-SSIM
ms_ssim_pha_module = MS_SSIM(data_range=3.141592653589793, size_average=True, channel=1)
phase_msssim = ms_ssim_pha_module(torch.tensor(np.expand_dims(np.expand_dims(hevc_recon_pha, axis=0), axis=1)), torch.tensor(np.expand_dims(np.expand_dims(ori_sar_pha, axis=0), axis=0)))
print("HEVC Phase MS-SSIM: ", phase_msssim)

print("\n\n\n------Results nbit: %d JP2K compression level: %d-------"%(n_bits, jp2k_quality))
# JP2K
f_size_real_jp2       = Path('%s/output_real.jp2'%(temp_dir)).stat().st_size
f_size_imaginary_jp2  = Path('%s/output_imaginary.jp2'%(temp_dir)).stat().st_size
bpp_jp2 = (f_size_real_jp2 + f_size_imaginary_jp2)*8/np.prod((ps,ps,2))
print('JP2K bitstream-size: %.2f bytes ( %.4f bpp)'%((f_size_real_jp2 + f_size_imaginary_jp2), bpp_jp2))
recon_real_jp2 = cv2.imread('%s/recon_real_jp2.png'%(temp_dir), cv2.IMREAD_ANYDEPTH)
recon_imaginary_jp2 = cv2.imread('%s/recon_imaginary_jp2.png'%(temp_dir), cv2.IMREAD_ANYDEPTH)
recon_jp2 = np.stack((np.minimum(recon_real_jp2, (2**n_bits -1)), np.minimum(recon_imaginary_jp2, (2**n_bits -1))), axis=2)
recon_dequant_jp2 = np.round( ((max_val - min_val)*recon_jp2.astype(np.float32)/(2**n_bits - 1)) + min_val )

# psnr_real_jp2 = peak_signal_noise_ratio(recon_jp2[:,:,0]/(2**n_bits -1), ori_sar_clipped[:,:,0]/(2**n_bits -1))
# psnr_imaginary_jp2 = peak_signal_noise_ratio(recon_jp2[:,:,1]/(2**n_bits -1), ori_sar_clipped[:,:,1]/(2**n_bits -1))

# # Recon amp
jp2_recon_amp = np.sqrt(recon_dequant_jp2[:,:,0]**2 + recon_dequant_jp2[:,:,1]**2)
jp2_recon_amp_psnr = peak_signal_noise_ratio(jp2_recon_amp/amp_max_val, ori_sar_amp/amp_max_val)

# print("JP2K---> PSNR real: %.4f \t PSNR imaginary: %.4f \t PSNR amp: %.4f"%(psnr_real_jp2, psnr_imaginary_jp2, jp2_recon_amp_psnr))
print("JP2K PSNR amp: %.4f"%(jp2_recon_amp_psnr))

# JP2K AMP MS-SSIM
ms_ssim_amp_module = MS_SSIM(data_range=1.0, size_average=True, channel=1)
amp_msssim = ms_ssim_amp_module(torch.tensor(np.expand_dims(np.expand_dims(jp2_recon_amp/amp_max_val, axis=0), axis=1)), torch.tensor(np.expand_dims(np.expand_dims(ori_sar_amp/amp_max_val, axis=0), axis=0)))
print("JP2K Amplitude MS-SSIM: ", amp_msssim)

jp2k_recon_pha = np.arctan2(recon_dequant_jp2[:,:,1], recon_dequant_jp2[:,:,0])
# JP2K PHA MS-SSIM
ms_ssim_pha_module = MS_SSIM(data_range=3.141592653589793, size_average=True, channel=1)
phase_msssim = ms_ssim_pha_module(torch.tensor(np.expand_dims(np.expand_dims(jp2k_recon_pha, axis=0), axis=1)), torch.tensor(np.expand_dims(np.expand_dims(ori_sar_pha, axis=0), axis=0)))
print("JP2k Phase MS-SSIM: ", phase_msssim)

print("Done...")

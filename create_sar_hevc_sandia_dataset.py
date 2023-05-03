import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sarpy.io.general.nitf as nitf
import scipy.io as sio
import time

from pathlib import Path
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

debug            = False
uniform          = 0
mode             = "train"
nbit             = 12
min_val          = -500
max_val          = 500
amp_max_val      = np.sqrt(min_val**2+max_val**2)
ps               = 256
qp               = 13
home_dir         = os.getenv("HOME")
if uniform == 1:
    output_file_path = "PythonDir/dataset/SAR_dataset/unifrom/Sandia_uniform_SAR_HEVC_ps256qp%d_%s/"%(qp, mode)
    temp_dir         = "./frames_uniform"
else:
    output_file_path = "PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp%d_%s/"%(qp, mode)
    temp_dir         = "./frames_nonuniform"
# make dir
if os.path.exists(temp_dir) == False:
    os.makedirs(temp_dir)
    
if mode == 'train':
    if uniform == 1:
        file_path  = "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_train/"
    else:
        file_path  = "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_train/"
    samples    = 1000
    
elif mode == 'validation':
    if uniform == 1:
        file_path  = "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_validation/"
    else:
        file_path  = "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_validation/"
    samples    = 1000

elif mode == 'test':
    if uniform == 1:
        file_path  = "PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_test/"
    else:
        file_path  = "PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_test/"
    samples    = 1
    ps         = 1024

real_list      = sorted(glob.glob(os.path.join(home_dir, file_path, "*real.png")))
imaginary_list = sorted(glob.glob(os.path.join(home_dir, file_path, "*imaginary.png")))
amp_list       = sorted(glob.glob(os.path.join(home_dir, file_path, "*amp.mat")))
pha_list       = sorted(glob.glob(os.path.join(home_dir, file_path, "*phase.mat")))
gt_sar_list    = sorted(glob.glob(os.path.join(home_dir, file_path, "*gt_sar.mat")))
if os.path.exists(os.path.join(home_dir, output_file_path)) == False:
    os.makedirs(os.path.join(home_dir, output_file_path))
with open(os.path.join(home_dir, output_file_path, "log_qp%d_%s.txt"%(qp, mode)), "w") as file:
    if uniform == 1:
        file.write("Sandia uniform quantization\ntest patch: 256:256+1024,1100:1100+1024\n")
    else: 
        file.write("Sandia non-uniform quantization mu = 2.5\ntest patch: 256:256+1024,1100:1100+1024\n")
    for i in range(len(real_list)):
        print("\n\n------------ENCODING real-----------------------") 
        cmd_encode = ['ffmpeg -y',
                    '-i %s'%real_list[i],
                    ' -c:v libx265',
                    '-qp %d'%(qp),
                    '-pix_fmt gray12le',
                    '%s/output_real_%d.mp4'%(temp_dir, i+1)]

        cmd_encode = ' '.join(cmd_encode)
        os.system(cmd_encode)

        print("\n\n------------ENCODING imaginary-----------------------")
        cmd_encode = ['ffmpeg -y',
                    '-i %s'%imaginary_list[i],
                    ' -c:v libx265', 
                    '-qp %d'%(qp),
                    '-pix_fmt gray12le',
                    '%s/output_imaginary_%d.mp4'%(temp_dir, i+1)]

        cmd_encode = ' '.join(cmd_encode)
        os.system(cmd_encode)

        # DECODE
        print("\n\n------------DECODING real-----------------------")
        cmd_decode = ['ffmpeg -y',
                    '-i %s/output_real_%d.mp4'%(temp_dir, i+1),
                    f'-pix_fmt gray16be',
                    '%s/recon_real_%d.png'%(temp_dir, i+1)]
        cmd_decode = ' '.join(cmd_decode)
        os.system(cmd_decode)

        f_size_real = Path('%s/output_real_%d.mp4'%(temp_dir, i+1)).stat().st_size

        # DECODE
        print("\n\n------------DECODING imaginary-----------------------")
        cmd_decode = ['ffmpeg -y',
                    '-i %s/output_imaginary_%d.mp4'%(temp_dir, i+1),
                    f'-pix_fmt gray16be',
                    '%s/recon_imaginary_%d.png'%(temp_dir, i+1)]
        cmd_decode = ' '.join(cmd_decode)
        os.system(cmd_decode)

        f_size_imaginary = Path('%s/output_imaginary_%d.mp4'%(temp_dir, i+1)).stat().st_size

        # load images
        rec_ch1 = cv2.imread('%s/recon_real_%d.png'%(temp_dir, i+1), cv2.IMREAD_ANYDEPTH)
        rec_ch2 = cv2.imread('%s/recon_imaginary_%d.png'%(temp_dir, i+1), cv2.IMREAD_ANYDEPTH)
        img_clipped = sio.loadmat(gt_sar_list[i])['sar_image']
        real_quant = cv2.imread(real_list[i] , cv2.IMREAD_ANYDEPTH)
        imaginary_quant = cv2.imread(imaginary_list[i] , cv2.IMREAD_ANYDEPTH)
        amp = sio.loadmat(amp_list[i])['sar_amp']
        pha = sio.loadmat(pha_list[i])['sar_phase']

        H, W, C = img_clipped.shape

        # compute bpp
        bpp = (f_size_real + f_size_imaginary)*8/np.prod(img_clipped.shape)
        print(f'bitstream-size:{f_size_real + f_size_imaginary} bytes ({bpp:.4f}) bpp')

        #img_recon = np.stack((rec_ch1, rec_ch2), axis=2)
        img_recon = np.stack((np.minimum(rec_ch1, (2**nbit -1)), np.minimum(rec_ch2, (2**nbit -1))), axis=2)
        #img_recon_dequant = np.round( ((max_val - min_val)*img_recon.astype(np.float32)/(2**nbit - 1)) + min_val )

        recon_real_psnr = peak_signal_noise_ratio(img_recon[:,:,0]/(2**nbit -1), real_quant/(2**nbit -1))
        recon_imaginary_psnr = peak_signal_noise_ratio(img_recon[:,:,1]/(2**nbit -1), imaginary_quant/(2**nbit -1))
        print("SAR psnr", recon_real_psnr, recon_imaginary_psnr)

        # Recon amp
        # recon_amp = np.sqrt(img_recon_dequant[:,:,0]**2 + img_recon_dequant[:,:,1]**2)
        # recon_psnr = peak_signal_noise_ratio(recon_amp/amp_max_val, amp/amp_max_val)
        # print("AMP psnr: ", recon_psnr)
        

        # plt.subplot(1,2,1);plt.imshow(amp, cmap="gray");plt.title("Original amplitude image (Complex SAR bpp: %.4f)"%(bpp));plt.axis('off');plt.subplot(1,2,2);plt.imshow(recon_amp, cmap='gray');plt.title("HEVC decoded amplitude SAR (QP: %d, PSNR: %.4f dB)"%(qp, recon_psnr));plt.axis('off');plt.show()
        # recon_pha = np.arctan2(img_recon_dequant[:,:,1], img_recon_dequant[:,:,0])
        # recon_pha_mse = mean_squared_error(recon_pha, pha)
        # print("Pha mse: ", recon_pha_mse)

        # make dir
        file.write(gt_sar_list[i])
        file.write(f'bitstream-size:{f_size_real + f_size_imaginary} bytes ({bpp:.4f}) bpp\n')
        file.write("SAR psnr: %f %f\n"%(recon_real_psnr, recon_imaginary_psnr))
        #file.write("AMP psnr: %f\n"%recon_psnr)
        #file.write("Pha mse: %f\n\n"%(recon_pha_mse))

        # if mode == "test":
        #     os.popen('cp ./frames/* %s'%output_file_path)
        #     exit()
                # make dir
        if not os.path.exists(os.path.join(home_dir, output_file_path, "input")):
            os.makedirs(os.path.join(home_dir, output_file_path, "input"))
            os.makedirs(os.path.join(home_dir, output_file_path, "gt"))
            os.makedirs(os.path.join(home_dir, output_file_path, "amp"))
            os.makedirs(os.path.join(home_dir, output_file_path, "phase"))
        # create intput and GT pair
        for j in range(samples):
            #print(i, j)
            if mode == "test":
                xx = 0
                yy = 0
            else:
                xx = np.random.randint(0, W - ps)
                yy = np.random.randint(0, H - ps)
            
            input_patch = img_recon[yy:yy + ps, xx:xx + ps, :]
            gt_patch = img_clipped[yy:yy + ps, xx:xx + ps, :]
            amp_patch = amp[yy:yy + ps, xx:xx + ps]
            phase_patch = pha[yy:yy + ps, xx:xx + ps]
            if 0:
                plt.figure(1)
                plt.subplot(1,2,1)
                plt.imshow(input_patch[:,:,0], cmap='gray')
                plt.title("HEVC recon real, Qp:%d, psnr:%.4f dB"%(qp, recon_real_psnr))
                plt.subplot(1,2,2)
                plt.imshow(gt_patch[:,:,0], cmap='gray')
                plt.title("GT")

                # plt.figure(2)
                # plt.subplot(1,2,1)
                # recon_amp_patch = np.sqrt(input_patch[:,:,0]**2+input_patch[:,:,1]**2)
                # plt.imshow(recon_amp_patch, cmap='gray')
                # plt.title("HEVC recon amp, Qp:%d, psnr:%.1f"%(qp, peak_signal_noise_ratio(amp_patch/amp_max_val, recon_amp_patch/amp_max_val)))
                # plt.subplot(1,2,2)
                # plt.imshow(amp_patch/amp_max_val, cmap='gray')
                # plt.title("GT amp")
                plt.show()
            np.save(os.path.join(home_dir, output_file_path, "input/%d.npy"%(j+i*samples)), input_patch)
            np.save(os.path.join(home_dir, output_file_path, "gt/%d.npy"%(j+i*samples)), gt_patch)
            np.save(os.path.join(home_dir, output_file_path, "amp/%d.npy"%(j+i*samples)), amp_patch)
            np.save(os.path.join(home_dir, output_file_path, "phase/%d.npy"%(j+i*samples)), phase_patch)
print("Done..")

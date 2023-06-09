import matplotlib.pyplot as plt
import numpy as np

def afrl_test2_plot_rd():
    # HEVC test
    hevc_uniform_bpp           = [0.5078,  0.7748,  1.0992,  1.29997]
    hevc_uniform_amp_psnr      = [20.3747, 22.8494, 25.5598, 27.0662]
    hevc_uniform_amp_msssim    = []
    hevc_uniform_pha_msssim    = []

    # hevc_gamma08_uniform_bpp   = [0.4701,  0.6793,  0.9989,  1.2005]
    # hevc_gamma08_uniform_amp   = [20.3361, 22.3321, 25.0167, 26.5621]

    # hevc_gamma07_uniform_bpp   = [0.5294,  0.7627,  0.9315,  1.1352]
    # hevc_gamma07_uniform_amp   = [20.8459, 22.9230, 24.1743, 25.5565]

    # #JPEG2000
    jp2k_bpp                   = [0.6662,  0.9995,  1.3327]
    jp2k_amp_psnr              = [20.3250, 22.5594, 24.6468]
    jp2k_amp_msssim            = []
    jp2k_pha_msssim            = []

    # Our proposed method without amp loss
    MST_bpp                    = [0.5078,  0.7748,  1.29997]
    MST_amp_psnr               = [20.9171, 23.7220, 28.2706]
    MST_amp_msssim             = []
    MST_pha_msssim             = []

    # Our proposed method
    MST_amploss_bpp            = [0.5078,  0.7748,  1.29997]
    MST_amploss_amp_psnr       = [21.8392, 24.0795, 28.5117]
    MST_amploss_amp_msssim     = []
    MST_amploss_pha_msssim     = []

    # autoencodcer test
    # autoencoder_bpp   = [ 0.4713,  0.7688,  1.0416]
    # autoencoder_amp   = [18.9312, 19.8312, 20.0118]
    # # NIC
    # nic_bpp   = [0.1982,  0.2765]
    # nic_amp   = [16.7944, 17.6355]

    # RD-curve AMP PSNR
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_amp_psnr, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_amp_psnr, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_amp_psnr, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_amp_psnr, color = '#fd625e', marker='*')

    plt.title("RD-curve for ARFL test sequence 2 amplitude image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

    # RD-curve AMP MSSSIM
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_amp_msssim, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_amp_msssim, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_amp_msssim, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_amp_msssim, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset amplitude image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("MS-SSIM")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

    # RD-curve PHASE MSSSIM
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_pha_msssim, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_pha_msssim, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_pha_msssim, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_pha_msssim, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset phase image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("MS-SSIM")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()


def afrl_test1_plot_rd():
    # HEVC test
    hevc_uniform_bpp           = [0.8552, 1.2019, 1.8223]
    hevc_uniform_amp_psnr      = [19.5955, 22.2901,26.6891]
    hevc_uniform_amp_msssim    = [0.8283, 0.9008, 0.9632]
    hevc_uniform_pha_msssim    = [0.5309, 0.6810, 0.8003]

    # JPEG2000
    jp2k_bpp                   = [0.8881,  1.3319, 1.5992]   # quality = [9, 6, 5] 
    jp2k_amp_psnr              = [17.9181, 20.5070, 22.2939]
    jp2k_amp_msssim            = [0.7871, 0.8607, 0.9093]
    jp2k_pha_msssim            = [0.4568, 0.5878, 0.6753]
    # Our proposed method without amp loss
    MST_bpp                    = [0.8552, 1.2019, 1.8223]
    MST_amp_psnr               = [20.4606, 23.3914, 28.1119]
    MST_amp_msssim             = []
    MST_pha_msssim             = []

    # Our proposed method
    MST_amploss_bpp            = [0.8552, 1.2019, 1.8223]
    MST_amploss_amp_psnr       = [20.7764, 23.5062, 28.2023]
    MST_amploss_amp_msssim     = []
    MST_amploss_pha_msssim     = []

    # # autoencodcer test
    # autoencoder_bpp            = [0.4713,  0.7688,  1.0416]
    # autoencoder_amp            = [18.9312, 19.8312, 20.0118]
    # # NIC
    # nic_bpp                    = [0.1982,  0.2765]
    # nic_amp                    = [16.7944, 17.6355]

    # RD-curve AMP PSNR 
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_amp_psnr, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_amp_psnr, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_amp_psnr, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_amp_psnr, color = '#fd625e', marker='*')
     
    plt.title("RD-curve for ARFL test sequence 1 amplitude image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()
    
    # RD-curve AMP MSSSIM
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_amp_msssim, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_amp_msssim, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_amp_msssim, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_amp_msssim, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset amplitude image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("MS-SSIM")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

    # RD-curve PHASE MSSSIM
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_pha_msssim, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_pha_msssim, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_pha_msssim, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_pha_msssim, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset phase image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("MS-SSIM")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

def sandia_plot_rd():
    # HEVC
    hevc_uniform_bpp          = [0.4131, 0.6646,  1.2229]   #QP = [21, 18, 13]
    hevc_uniform_amp_psnr     = [20.4095, 22.5332, 26.2066]
    hevc_uniform_amp_msssim   = [0.7932 , 0.8694, 0.9414]
    hevc_uniform_pha_msssim   = [0.2024, 0.2810, 0.4189]
    
    # JPEG2000
    jp2k_bpp                  = [0.5002, 0.8006, 1.1438]   # Quality = [16, 10, 7]
    jp2k_amp_psnr             = [20.1495, 22.3823, 24.2824]
    jp2k_amp_msssim           = [0.8505, 0.9056, 0.9336]
    jp2k_pha_msssim           = [0.1242, 0.1806, 0.2928]

    # Ours without amploss
    MST_bpp                   = [0.4298,  0.6633,  1.2206]
    MST_amp_psnr              = [20.5219, 22.8232, 26.7493]
    MST_amp_msssim            = [0.8195, 0.8896, 0.9519]
    MST_pha_msssim            = [0.2632, 0.3538, 0.5026]
    
    # Ours with amploss
    MST_amploss_bpp           = [0.4298,  0.6633,  1.2206]
    MST_amploss_amp_psnr      = [20.6226, 22.9218, 26.7940]
    MST_amploss_amp_msssim    = [0.8209, 0.8902, 0.9520]
    MST_amploss_pha_msssim    = [0.2665, 0.3546, 0.5040]

    # hevc_nonuniform_bpp       = [0.6362, 0.9751, 1.6623]
    # hevc_nonuniform_amp       = [21.0340, 23.2883, 27.3799]
    # hevc_nonuniform_recon_amp = [22.0904, 24.4674, 28.3770]

    # RD-curve AMP PSNR
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_amp_psnr, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_amp_psnr, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_amp_psnr, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_amp_psnr, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset amplitude image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

    # RD-curve AMP MSSSIM
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_amp_msssim, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_amp_msssim, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_amp_msssim, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_amp_msssim, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset amplitude image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("MS-SSIM")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

    # RD-curve PHASE MSSSIM
    plt.figure()
    plt.plot(hevc_uniform_bpp, hevc_uniform_pha_msssim, color = '#f2c80f', marker='.')
    plt.plot(jp2k_bpp, jp2k_pha_msssim, color = '#01b8aa', marker='^')
    plt.plot(MST_bpp, MST_pha_msssim, color = '#374649', linestyle="dashed")
    plt.plot(MST_amploss_bpp, MST_amploss_pha_msssim, color = '#fd625e', marker='*')
    
    plt.title("RD-curve for Sandia dataset phase image")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("MS-SSIM")
    plt.legend(["HEVC","JPEG2000", "Ours", "Ours with amploss"])
    plt.grid()

if __name__ == "__main__":
    #afrl_test1_plot_rd()
    #afrl_test2_plot_rd()
    sandia_plot_rd()
    plt.show()
    print("Done...")
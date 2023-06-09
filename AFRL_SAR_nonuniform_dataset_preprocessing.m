%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authur      : Paras Maharjan                                 %
% Description : Pre-process the SAR dataset.                   %
%               Read NITF, clip, non-unifrom quantize, and     %
%               save as I/Q png, GT, Amp, Phase                %
% Run on cmd  : sudo chmod -R 777 .                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% parameters
 % For debug
 % 0 = run, 
 % 1 = run + display results,
 % 2 = validate the hevc results with recon PSNR Non-unifrom (test amp psnr: 19.8670 dB)
 % 3 = validate the hevc results with recon PSNR unifrom     (test amp psnr: 19.5955 dB)
debug      = 1;       
mode       = "train";
uniform    = 1;
if uniform == 1 & debug == 2
    debug = 3;
end
sar_min    = -5000.0;
sar_max    = 5000.0;
n_bits     = 12;       % Define the number of bits used for quantization
n_levels   = 2^n_bits; % Define the number of quantization levels
mu         = 2.5;      % mu-law parameter
if uniform == 1
    output_dir = ['/home/paras/PythonDir/SAR-HEVC-Deblocking-master/AFRL_VH_uniform_' char(mode) '/'];
else
    output_dir = ['/home/paras/PythonDir/SAR-HEVC-Deblocking-master/AFRL_nonuniform_' char(mode) '/'];
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

%% Validate the output of HEVC and Network recon for non uniform
if debug==2  
    addpath('/home/paras/PythonDir/npy-matlab-master/npy-matlab')
    sar_hevc_quants = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/AFRL_nonuniform_SAR_HEVC_ps256qp21_test2/input/0.npy");
    [H, W, C]       = size(sar_hevc_quants);
    sar_hevc_quants = sar_hevc_quants(:);
    sar_gt          = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/AFRL_nonuniform_SAR_HEVC_ps256qp21_test2/gt/0.npy");
    sar_amp         = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/AFRL_nonuniform_SAR_HEVC_ps256qp21_test2/amp/0.npy");
    sar_phase       = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/AFRL_nonuniform_SAR_HEVC_ps256qp21_test2/phase/0.npy");
    
    sar_hevc_recon = (single(sar_hevc_quants) .* (sar_max - sar_min) / (n_levels - 1)) + sar_min; 

    sar_expanded = compand(sar_hevc_recon, mu, max(sar_hevc_recon), 'mu/expander');
    sar_expanded = round(sar_expanded);
        
    [h1,b1]=hist(double(sar_gt(:)),256);
    plot(b1, h1);
    hold on
    [h2,b2]=hist(double(sar_expanded),256);
    plot(b2, h2);
    
    sar_image_quants = reshape(sar_expanded, [H, W, C]);
    sar_mse = mae(sar_gt, sar_image_quants)
    
    subplot(1,2,1);imagesc(sar_image_quants(:,:,1)); colormap("gray"); title("Quantized SAR real image");

    sar_recon_amp = sqrt(sar_image_quants(:,:,1).^2 + sar_image_quants(:,:,2).^2);
    amp_psnr = psnr(sar_amp/max(sar_amp(:)), sar_recon_amp/max(sar_recon_amp(:)))
    subplot(1,2,2);imagesc(sar_recon_amp); colormap("gray"); title("Quantized SAR amplitude image (Non-Unifrom Quantization)")
    
    sar_recon_phase = atan2(sar_image_quants(:,:,2),sar_image_quants(:,:,1));
    phase_mse = mse(sar_phase, sar_recon_phase)
    display("done")
end

%% Validate the output of HEVC and Network recon for uniform
if debug==3 
    addpath('/home/paras/PythonDir/npy-matlab-master/npy-matlab')
    sar_hevc_quants = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/AFRL/AFRL_VH_uniform_SAR_HEVC_ps256qp21_test2/input/0.npy");
    sar_gt          = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/AFRL/AFRL_VH_uniform_SAR_HEVC_ps256qp21_test2/gt/0.npy");
    sar_amp         = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/AFRL/AFRL_VH_uniform_SAR_HEVC_ps256qp21_test2/amp/0.npy");
    sar_phase       = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/AFRL/AFRL_VH_uniform_SAR_HEVC_ps256qp21_test2/phase/0.npy");
    sar_image_quants = (single(sar_hevc_quants) .* (sar_max - sar_min) / (n_levels - 1)) + sar_min; 
        
    [h1,b1]=hist(double(sar_gt(:)),256);
    plot(b1, h1);
    hold on
    [h2,b2]=hist(double(sar_image_quants(:)),256);
    plot(b2, h2);
    
    sar_mse = mae(sar_gt, sar_image_quants)
    
    subplot(1,2,1);imagesc(sar_image_quants(:,:,1)); colormap("gray"); title("Quantized SAR real image");

    sar_recon_amp = sqrt(sar_image_quants(:,:,1).^2 + sar_image_quants(:,:,2).^2);
    amp_psnr = psnr(sar_amp/max(sar_amp(:)), sar_recon_amp/max(sar_recon_amp(:)))
    subplot(1,2,2);imagesc(sar_recon_amp); colormap("gray"); title("Quantized SAR amplitude image (Unifrom Quantization)")
    
    sar_recon_phase = atan2(sar_image_quants(:,:,2),sar_image_quants(:,:,1));
    phase_mse = mse(sar_phase, sar_recon_phase)
    display("done")
end

if debug == 0 | debug == 1
    %% Read SAR image
    if mode=="train"
        filename = "/home/paras/PythonDir/dataset/SAR_dataset/raw/AFRL_NGA_Products/sicd_example_1_PFA_RE32F_IM32F_VH.nitf";
    else
        filename = "/home/paras/PythonDir/dataset/SAR_dataset/raw/AFRL_NGA_Products/sicd_example_2_PFA_RE32F_IM32F_VH.nitf";
    end

    [pathstr,name,ext] = fileparts(filename);
    fprintf("File name: %s\nSAR clip range: [%d, %d])\nQuantization bits: %d\n", name, sar_min, sar_max, n_bits);

    sar_image = nitfread(filename);
    if mode == "test"
        sar_image = sar_image(2401:3424, 1201:2224, :);   % test 1
    end
    if mode == "test2"
        sar_image = sar_image(1:1024, 1:1024, :);         % test 2
    end
    [H,W,C]   = size(sar_image);
    % Clipped SAR with in some fixed range
    sar_image(sar_image < sar_min) = sar_min;
    sar_image(sar_image > sar_max) = sar_max;
    save([output_dir char(name) '_gt_sar.mat'], 'sar_image')

    % Original amplitude
    sar_amp = sqrt(sar_image(:,:,1).^2 + sar_image(:,:,2).^2);
    save([output_dir char(name) '_amp.mat'], 'sar_amp')
    if debug
        figure();
        imagesc(sar_amp);
        colormap("gray");
        title("Original SAR amplitude");
    end

    % Original Phase
    sar_phase = atan2(sar_image(:,:,2), sar_image(:,:,1));
    save([output_dir char(name) '_phase.mat'], 'sar_phase')

    %% Non uniform Quantization
    if uniform ==1
        sar_image_quants = uint16((n_levels - 1) .* (sar_image - sar_min)/(sar_max - sar_min));
    else
        sar_image = sar_image(:);
        V                = max(sar_image);
        sar_compressed   = compand(sar_image,mu,V,'mu/compressor');
        quants           = uint16((n_levels - 1) .* (sar_compressed - sar_min)/(sar_max - sar_min));
        sar_image_quants = reshape(quants, [H, W, C]);
        figure();
        [h1,b1]=hist(double(sar_image(:)),256);
        plot(b1, h1);
        hold on
        [h2,b2]=hist(double(sar_compressed(:)),256);
        plot(b2, h2);
    end

    % save image
    imwrite(sar_image_quants(:,:,1), [output_dir char(name) '_quant_real.png']);
    imwrite(sar_image_quants(:,:,2), [output_dir char(name) '_quant_imaginary.png']);

    display('Done...')
end

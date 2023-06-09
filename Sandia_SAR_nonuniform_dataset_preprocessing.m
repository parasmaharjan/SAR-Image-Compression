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

%% init
addpath("/home/paras/PythonDir/Sandia_gff_reader")

%% parameters
 % For debug
 % 0 = run, 
 % 1 = run + display results,
 % 2 = validate the hevc results with recon PSNR Non-unifrom (test amp psnr: 21.0340 dB)
 % 3 = validate the hevc results with recon PSNR unifrom     (test amp psnr: 20.4030 dB)
debug      = 1;       
mode       = "test";
uniform    = 1;
if uniform == 1 & debug == 2
    debug = 3;
end
sar_min    = -500.0;
sar_max    = 500.0;
n_bits     = 12;       % Define the number of bits used for quantization
n_levels   = 2^n_bits; % Define the number of quantization levels
mu         = 2.5;      % mu-law parameter
if uniform == 1
    output_dir = ['/home/paras/PythonDir/SAR-HEVC-Deblocking-master/Sandia_uniform_' char(mode) '/'];
else
    output_dir = ['/home/paras/PythonDir/SAR-HEVC-Deblocking-master/Sandia_nonuniform_' char(mode) '/'];
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

%% Validate the output of HEVC and Network recon for non uniform
if debug==2  
    addpath('/home/paras/PythonDir/npy-matlab-master/npy-matlab')
    sar_hevc_quants = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp21_test/input/0.npy");
    [H, W, C]       = size(sar_hevc_quants);
    sar_hevc_quants = sar_hevc_quants(:);
    sar_gt          = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp21_test/gt/0.npy");
    sar_amp         = single(readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp21_test/amp/0.npy"));
    sar_phase       = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/nonuniform/Sandia_nonuniform_SAR_HEVC_ps256qp21_test/phase/0.npy");
    
    sar_hevc_recon = (single(sar_hevc_quants) .* (sar_max - sar_min) / (n_levels - 1)) + sar_min; 

    sar_expanded = compand(sar_hevc_recon, mu, max(sar_hevc_recon), 'mu/expander');
    sar_expanded = round(sar_expanded);
     
    figure;
    [h1,b1]=hist(double(sar_gt(:)),256);
    plot(b1, h1);
    hold on
    [h2,b2]=hist(double(sar_expanded),256);
    plot(b2, h2);
    
    sar_image_quants = reshape(sar_expanded, [H, W, C]);
    sar_mse = mae(sar_gt, sar_image_quants)
    
    figure;subplot(1,2,1);imagesc(sar_image_quants(:,:,1)); colormap("gray"); title("Quantized SAR real image");

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
    sar_hevc_quants = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/Sandia_SAR_HEVC_ps256qp21_test/input/0.npy");
    sar_gt          = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/Sandia_SAR_HEVC_ps256qp21_test/gt/0.npy");
    sar_amp         = single(readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/Sandia_SAR_HEVC_ps256qp21_test/amp/0.npy"));
    sar_phase       = readNPY("/home/paras/PythonDir/dataset/SAR_dataset/uniform/Sandia_SAR_HEVC_ps256qp21_test/phase/0.npy");
    sar_image_quants = (single(sar_hevc_quants) .* (sar_max - sar_min) / (n_levels - 1)) + sar_min; 
    
    figure;
    [h1,b1]=hist(double(sar_gt(:)),256);
    plot(b1, h1);
    hold on
    [h2,b2]=hist(double(sar_image_quants(:)),256);
    plot(b2, h2);
    
    sar_mse = mae(sar_gt, sar_image_quants)
    
    figure;
    subplot(1,2,1);imagesc(sar_image_quants(:,:,1)); colormap("gray"); title("Quantized SAR real image");

    sar_recon_amp = sqrt(sar_image_quants(:,:,1).^2 + sar_image_quants(:,:,2).^2);
    amp_psnr = psnr(sar_amp/max(sar_amp(:)), sar_recon_amp/max(sar_recon_amp(:)))
    subplot(1,2,2);imagesc(sar_recon_amp); colormap("gray"); title("Quantized SAR amplitude image (Unifrom Quantization)")
    
    sar_recon_phase = atan2(sar_image_quants(:,:,2),sar_image_quants(:,:,1));
    phase_mse = mse(sar_phase, sar_recon_phase)
    
    display("done")
end

if debug == 1 | debug == 0
    %% Read SAR image
    if mode=="train"
        filedir  = '/home/paras/PythonDir/dataset/MiniSAR20050519p0009image003/train/';
        filelist = dir([filedir '*.gff']);
    else
        filedir  = '/home/paras/PythonDir/dataset/MiniSAR20050519p0009image003/test/';
        filelist = dir([filedir '*.gff']);
    end

    for i=1:length(filelist)
        fprintf("File name: %s\nSAR clip range: [%d, %d])\nQuantization bits: %d\n", filelist(i).name, sar_min, sar_max, n_bits);

        [Image, Header, fname] = load_gff_1_8b(filelist(i).name, filedir);
        real_sar = real(Image);
        imag_sar = imag(Image);
        sar_image = cat(3, real_sar, imag_sar);
        if mode == "test"
            sar_image = sar_image(256:256+1024,1100:1100+1024, :);
        end
        [H,W,C]   = size(sar_image);
        % Clipped SAR with in some fixed range
        sar_image(sar_image < sar_min) = sar_min;
        sar_image(sar_image > sar_max) = sar_max;
        save([output_dir filelist(i).name(1:end-4) '_gt_sar.mat'], 'sar_image')

        % Original amplitude
        sar_amp = sqrt(sar_image(:,:,1).^2 + sar_image(:,:,2).^2);
        save([output_dir filelist(i).name(1:end-4) '_amp.mat'], 'sar_amp')
        if debug == 1
            figure();
            imagesc(sar_amp);
            colormap("gray");
            title("Original SAR amplitude");
        end

        % Original Phase
        sar_phase = atan2(sar_image(:,:,2), sar_image(:,:,1));
        save([output_dir filelist(i).name(1:end-4) '_phase.mat'], 'sar_phase')

        %% Non uniform Quantization
        if uniform ==1
            sar_image_quants = uint16((n_levels - 1) .* (sar_image - sar_min)/(sar_max - sar_min));
        else
            sar_image = sar_image(:);
            V                = max(sar_image);
            sar_compressed   = compand(sar_image,mu,V,'mu/compressor');
            quants           = uint16((n_levels - 1) .* (sar_compressed - sar_min)/(sar_max - sar_min));
            sar_image_quants = reshape(quants, [H, W, C]);
        end
        % save image
        imwrite(sar_image_quants(:,:,1), [output_dir filelist(i).name(1:end-4) '_quant_real.png']);
        imwrite(sar_image_quants(:,:,2), [output_dir filelist(i).name(1:end-4) '_quant_imaginary.png']);
    end
    display('Done...')
end


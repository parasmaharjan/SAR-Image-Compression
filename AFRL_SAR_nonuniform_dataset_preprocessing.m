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
debug      = 1;
uniform    = 0;
sar_min    = -5000.0;
sar_max    = 5000.0;
n_bits     = 12;       % Define the number of bits used for quantization
n_levels   = 2^n_bits; % Define the number of quantization levels
mu         = 2.5;      % mu-law parameter
output_dir = '/home/paras/PythonDir/SAR-HEVC-Deblocking-master/frames_nonuniform/';
%% Read SAR image
filename = "/home/paras/PythonDir/dataset/SAR_dataset/raw/AFRL_NGA_Products/sicd_example_1_PFA_RE32F_IM32F_HH.nitf";

[pathstr,name,ext] = fileparts(filename);
fprintf("File name: %s\nSAR clip range: [%d, %d])\nQuantization bits: %d\n", name, sar_min, sar_max, n_bits);

sar_image = nitfread(filename);
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

sar_image = sar_image(:);
%% Non uniform Quantization
V                = max(sar_image);
sar_compressed   = compand(sar_image,mu,V,'mu/compressor');
quants           = uint16((n_levels - 1) .* (sar_compressed - sar_min)/(sar_max - sar_min));
sar_image_quants = reshape(quants, [H, W, C]);

% save image
imwrite(sar_image_quants(:,:,1), [output_dir char(name) '_quant_real.png']);
imwrite(sar_image_quants(:,:,2), [output_dir char(name) '_quant_imaginary.png']);

display('Done...')
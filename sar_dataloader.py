from torch.utils.data import Dataset, DataLoader
import glob
import os
import torch
import numpy as np

class SARDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, args):
        super(Dataset, self).__init__()
        self.root_dir = args.train_dataset_path
        self.input_img_paths = glob.glob(os.path.join(self.root_dir, "input/*.npy"))
        self.gt_img_paths = glob.glob(os.path.join(self.root_dir, "gt/*.npy"))
        self.amp_img_paths = glob.glob(os.path.join(self.root_dir, 'amp/*.npy'))
        self.phase_img_paths = glob.glob(os.path.join(self.root_dir, 'phase/*.npy'))
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.nbit = 12
        self.which_sar = args.which_sar
        if args.which_sar == "real":
            self.sar_idx = 0
        elif args.which_sar == "imaginary":
            self.sar_idx = 1
        else:
            self.sar_idx = [0,1]

    def __len__(self):
        return len(self.input_img_paths)

    def __getitem__(self, idx):
        input_image = np.load(self.input_img_paths[idx]).astype(np.float32)
        input_image = input_image[:,:,self.sar_idx]/(2**self.nbit - 1)
        if self.which_sar == "complex":
            input_image = torch.permute(torch.from_numpy(input_image), (2,0,1))
        else:
            input_image = torch.permute(torch.from_numpy(np.expand_dims(input_image, axis=2)), (2,0,1))
        gt_image = np.load(self.gt_img_paths[idx]).astype(np.float32)
        gt_image = (gt_image[:,:,self.sar_idx] - self.min_val)/(self.max_val - self.min_val)
        if self.which_sar == "complex":
            gt_image = torch.permute(torch.from_numpy(gt_image), (2,0,1))
        else:
            gt_image = torch.permute(torch.from_numpy(np.expand_dims(gt_image, axis=2)), (2,0,1))
        amp_image = np.load(self.amp_img_paths[idx]).astype(np.float32)
        phase_image = np.load(self.phase_img_paths[idx]).astype(np.float32)

        sample = (input_image, 
                  gt_image, 
                  torch.from_numpy(amp_image), 
                  torch.from_numpy(phase_image))

        return sample

# train_data = SARDataset("/home/paras/PythonDir/dataset/SAR_dataset/SAR_HEVC/")
# train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1)

# for step, img_batch in enumerate(train_loader):
#     print(img_batch["input"].shape)
#     print(img_batch["gt"].shape)
#     print(img_batch["amp"].shape)
#     print(img_batch["phase"].shape)

# print("done...")
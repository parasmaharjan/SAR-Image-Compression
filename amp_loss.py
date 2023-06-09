import torch
import torch.nn as nn

class AmplitudeLoss(nn.Module):
    def __init__(self, args):
        super(AmplitudeLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.min_val = args.min_val
        self.max_val = args.max_val
        self.amp_max_val = torch.sqrt(torch.tensor(self.max_val ** 2 + self.min_val ** 2))
    def forward(self, sar_output, amp_target):
        pred_sar_dequant_img = ((self.max_val - self.min_val)*sar_output) + self.min_val
        pred_amp_img = torch.sqrt(pred_sar_dequant_img[:,0,:,:]**2 + pred_sar_dequant_img[:,1,:,:]**2)
        loss = self.criterion(pred_amp_img, amp_target)
        return loss

if __name__ == "__main__":
    pass
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import archs.RRDBNet_arch as arch
import torch.nn as nn 
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

model_path1 = 'weights/RRDB_ESRGAN_FFL_x4.pth'  
model_path =  'weights/RRDB_ESRGAN_x4.pth' # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
test_img_folder = '../../autodl-nas/datasets/Urban100/image_SRF_4/*'
output_folder = '../ESRGANFFL_test_resluts/test_results_author/Urban100/'

device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# def get_bare_model(net):
#     """Get bare model, especially under wrapping with
#     DistributedDataParallel or DataParallel.
#     """
#     if isinstance(net, (DataParallel, DistributedDataParallel)):
#         net = net.module
#     return net

# # if params_ema, then use following codes
# load_net = torch.load(model_path1, map_location=lambda storage, loc: storage)
# load_net = load_net['params_ema']
# # remove unnecessary 'module.'
# # for k, v in deepcopy(load_net).items():
# #     if k.startswith('module.'):
# #         load_net[k[7:]] = v
# #         load_net.pop(k)

# model.load_state_dict(load_net, strict=False)
# model.eval()




print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    if path.endswith('_LR.png'):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(output_folder + '{:s}_SR.png'.format(base), output)

import os.path as osp
import glob
import cv2
import numpy as np
import torch
from basicsr.metrics.psnr_ssim import calculate_psnr,calculate_ssim
from basicsr.metrics.niqe import calculate_niqe

gt_img_folder = '../../autodl-nas/datasets/Urban100/image_SRF_4/*'
sr_img_folder = '../ESRGANFFL_test_resluts/test_results_author/Urban100/'

idx = 0
psnr = []
ssim = []
niqe = []

print('Urban100 SR by ESRGAN \nTesting...')

for path_gt in glob.glob(gt_img_folder):
    if path_gt.endswith('_HR.png'):
        idx += 1
        base = osp.splitext(osp.basename(path_gt))[0]
        path_sr = sr_img_folder + base[:-2] + 'LR_SR.png'
        print(idx,base[:7])
        # read images
        img_sr = cv2.imread(path_sr, cv2.IMREAD_COLOR)
        img_gt = cv2.imread(path_gt, cv2.IMREAD_COLOR)
        p = calculate_psnr(img_sr, img_gt, crop_border=4 , input_order='HWC', test_y_channel=False)
        s = calculate_ssim(img_sr, img_gt, crop_border=4 , input_order='HWC', test_y_channel=False)
        n = calculate_niqe(img_sr, crop_border=4 , input_order='HWC', convert_to='y')
        psnr.append(p)
        ssim.append(s)
        niqe.append(n)
        print("PSNR:",p,"  ssim:",s,"  niqe:",n)
print('average value:\npsnr:',np.mean(psnr),'  ssim:',np.mean(ssim),'  niqe:',np.mean(niqe))













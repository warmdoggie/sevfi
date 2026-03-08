import os
import glob 
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0

def eval_psnr_ssim(pred_dir, gt_dir, num_skip):
    step = num_skip + 1
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.png")))

    psnr_list, ssim_list = [], []
    used = 0

    for p in pred_paths:
        name = os.path.basename(p)          # e.g. 00037.png
        idx = int(os.path.splitext(name)[0])

        # 跳过关键帧（输入帧）
        if idx % step == 0:
            continue

        gt_path = os.path.join(gt_dir, name)
        if not os.path.exists(gt_path):
            continue

        pred = read_rgb(p)
        gt   = read_rgb(gt_path)

        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch {name}: pred{pred.shape} vs gt{gt.shape}")

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        used += 1

    print(f"used={used}  PSNR={np.mean(psnr_list):.4f}  SSIM={np.mean(ssim_list):.4f}")


eval_psnr_ssim(
    pred_dir="/data1/duanzhibo/riLght/sevfi/SEVFI/sample/result/SEID/insert_9/indoor_0",
    gt_dir="/data1/duanzhibo/riLght/sevfi/SEVFI/sample/dataset/SEID/indoor_0/images",
    num_skip=9
)

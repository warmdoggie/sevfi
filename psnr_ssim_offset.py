import os
import glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


def infer_pad_width(gt_dir: str) -> int:
    """根据 GT 目录里第一张图的文件名推断补零位数（如 00600 -> 5）"""
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    if not gt_paths:
        raise FileNotFoundError(f"No GT png found in {gt_dir}")
    gt0 = os.path.splitext(os.path.basename(gt_paths[0]))[0]
    return len(gt0)


def eval_psnr_ssim_with_offset(pred_dir, gt_dir, num_skip, base_offset):
    """
    pred: 00000.png ~ 00149.png
    gt  : 00600.png ~ 00750.png
    base_offset=600 即 gt_idx = pred_idx + 600
    """
    step = num_skip + 1
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    if not pred_paths:
        raise FileNotFoundError(f"No pred png found in {pred_dir}")

    pad = infer_pad_width(gt_dir)

    psnr_list, ssim_list = [], []
    used = 0
    miss_gt = 0

    for p in pred_paths:
        name = os.path.basename(p)          # e.g. 00037.png
        idx = int(os.path.splitext(name)[0])

        # 跳过关键帧（输入帧）
        if idx % step == 0:
            continue

        gt_idx = idx + base_offset
        gt_name = f"{gt_idx:0{pad}d}.png"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            miss_gt += 1
            continue

        pred = read_rgb(p)
        gt = read_rgb(gt_path)

        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch pred={name} gt={gt_name}: pred{pred.shape} vs gt{gt.shape}")

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        used += 1

    if used == 0:
        print(f"used=0 (miss_gt={miss_gt}) -> 请检查 base_offset/gt_dir/文件位数")
    else:
        print(f"used={used}  PSNR={np.mean(psnr_list):.4f}  SSIM={np.mean(ssim_list):.4f}  (miss_gt={miss_gt})")


# ====== 你的调用（按你描述 base_offset=600）======
eval_psnr_ssim_with_offset(
    pred_dir="/data1/duanzhibo/riLght/sevfi/SEVFI/sample/result/MVSEC/insert_7/indoor_flying2",
    gt_dir="/data1/duanzhibo/riLght/sevfi/SEVFI/sample/dataset/MVSEC/indoor_flying2/images",
    num_skip=7,
    base_offset=600
)
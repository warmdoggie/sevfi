import os
import sys
import cv2
import argparse
from pathlib import Path

import torch
import numpy as np

from Networks.SEVFI import SEVFI_dc_DSEC, SEVFI_dc_MVSEC, SEVFI_dc_SEID
from script.dataloader import test_DSEC_sevfi, test_MVSEC_sevfi, test_SEID_sevfi

sys.path.append('../Networks')


def build_model(dataset):
    if dataset == 'SEID':
        return SEVFI_dc_SEID()
    elif dataset == 'DSEC':
        return SEVFI_dc_DSEC()
    elif dataset == 'MVSEC':
        return SEVFI_dc_MVSEC()
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def resolve_model_path(model_path, ckpt_name=""):
    """
    支持两种传法：
    1) --model_paths ./PreTrained_ECA/run_xxx/best_train.pth
    2) --model_paths ./PreTrained_ECA/run_xxx --ckpt_name best_train.pth
    """
    p = Path(model_path)

    if p.is_file():
        return p

    if p.is_dir():
        if ckpt_name:
            ckpt_path = p / ckpt_name
            if ckpt_path.exists():
                return ckpt_path
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        for name in ['best_train.pth', 'latest.pth']:
            ckpt_path = p / name
            if ckpt_path.exists():
                return ckpt_path

        epoch_ckpts = sorted(p.glob('epoch*.pth'))
        if len(epoch_ckpts) > 0:
            return epoch_ckpts[-1]

        raise FileNotFoundError(f'No checkpoint found in directory: {p}')

    raise FileNotFoundError(f'Model path not found: {model_path}')


def build_model_tag(ckpt_path: Path):
    """
    例如：
      ./PreTrained_ECA/DSEC_s3_i3_p256_cosine/best_train.pth
    结果标签：
      DSEC_s3_i3_p256_cosine__best_train
    """
    parent_name = ckpt_path.parent.name
    stem_name = ckpt_path.stem

    if parent_name in ["", ".", "PreTrained_ECA", "PreTrained"]:
        return stem_name
    return f"{parent_name}__{stem_name}"


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def save_rgb_image(path, img_rgb):
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def build_test_dataset(dataset, test_path, num_skip, num_insert, num_bins=15):
    if dataset == 'SEID':
        return test_SEID_sevfi(
            data_path=test_path,
            num_bins=num_bins,
            num_skip=num_skip,
            num_insert=num_insert
        )
    elif dataset == 'DSEC':
        return test_DSEC_sevfi(
            data_path=test_path,
            num_bins=num_bins,
            num_skip=num_skip,
            num_insert=num_insert
        )
    elif dataset == 'MVSEC':
        return test_MVSEC_sevfi(
            data_path=test_path,
            num_bins=num_bins,
            num_skip=num_skip,
            num_insert=num_insert
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def load_model_for_test(dataset, ckpt_path, device, use_dp=False):
    net = build_model(dataset)
    state_dict = torch.load(str(ckpt_path), map_location='cpu')
    net.load_state_dict(state_dict, strict=True)

    if use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs for inference")
        net = torch.nn.DataParallel(net)

    net = net.to(device)
    net = net.eval()
    return net


def run_one_model(opt, ckpt_path, device, test_list):
    model_tag = build_model_tag(ckpt_path)

    print("=" * 70)
    print("Testing model:", str(ckpt_path))
    print("Model tag    :", model_tag)

    net = load_model_for_test(opt.dataset, ckpt_path, device, use_dp=opt.use_dp)

    data_root = os.path.join(opt.origin_path, opt.dataset)
    result_root = os.path.join(opt.save_path, opt.dataset, model_tag, f'insert_{opt.num_insert}')
    disp_root = os.path.join(opt.save_path, opt.dataset, model_tag, f'disp_{opt.num_insert}')
    mkdir(result_root)
    mkdir(disp_root)

    print(f"skip = {opt.num_skip}, insert = {opt.num_insert}")
    print("Results saved to:", result_root)
    print("Disp saved to   :", disp_root)

    for seq_idx, seq_name in enumerate(test_list):
        test_path = os.path.join(data_root, seq_name)
        testDataset = build_test_dataset(opt.dataset, test_path, opt.num_skip, opt.num_insert, num_bins=15)

        result_path = os.path.join(result_root, seq_name)
        disp_path = os.path.join(disp_root, seq_name)
        mkdir(result_path)
        mkdir(disp_path)

        print(f"[{model_tag}] {seq_name} ({seq_idx + 1}/{len(test_list)})")

        with torch.no_grad():
            for k in range(len(testDataset)):
                sample = testDataset[k]
                print(f"Processing img {k} ...")

                image_0_np = sample['image_0']       # H, W, 3
                image_1_np = sample['image_1']       # H, W, 3
                eframes_t1_np = sample['eframes_t1'] # N, C, H, W
                eframes_t0_np = sample['eframes_t0'] # N, C, H, W
                iwe_np = sample['iwe']               # N, H, W
                weight_np = sample['weight']         # N

                B = 1
                N, C, H, W = eframes_t0_np.shape

                image_0 = torch.from_numpy(image_0_np).permute(2, 0, 1).contiguous().float()
                image_1 = torch.from_numpy(image_1_np).permute(2, 0, 1).contiguous().float()

                # 和训练保持一致：若 RGB 为 0~255，则归一化到 0~1
                if image_0.max() > 1.5:
                    image_0 = image_0 / 255.0
                    image_1 = image_1 / 255.0

                image_0 = image_0.unsqueeze(0).repeat(N, 1, 1, 1)
                image_1 = image_1.unsqueeze(0).repeat(N, 1, 1, 1)

                eframes_t1 = torch.from_numpy(eframes_t1_np).reshape(B * N, C, H, W).float()
                eframes_t0 = torch.from_numpy(eframes_t0_np).reshape(B * N, C, H, W).float()
                iwe = torch.from_numpy(iwe_np).reshape(B * N, 1, H, W).float()
                weight = torch.from_numpy(weight_np).reshape(B * N).float()

                image_0 = image_0.reshape(B * N, 3, H, W).to(device)
                image_1 = image_1.reshape(B * N, 3, H, W).to(device)
                eframes_t1 = eframes_t1.to(device)
                eframes_t0 = eframes_t0.to(device)
                iwe = iwe.to(device)
                weight = weight.to(device)

                image_syn, image_fuse, image_final, disp, flowlist_t0, flowlist_t1 = net(
                    image_0, image_1, eframes_t0, eframes_t1, iwe, weight
                )

                final_t = torch.clamp(image_final, min=0, max=1)
                output = (final_t.reshape(B, N, 3, H, W) * 255.0).cpu().detach().numpy()
                output_disp = disp.reshape(B, N, 1, H, W).cpu().detach().numpy()

                # save endpoint images
                save_image_0 = image_0[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0
                name_0 = os.path.join(result_path, '{:05d}.format(int(k * (opt.num_insert + 1)))'.replace("'", ""))
                name_0 = os.path.join(result_path, '{:05d}.png'.format(int(k * (opt.num_insert + 1))))
                save_rgb_image(name_0, save_image_0)

                save_image_1 = image_1[0].permute(1, 2, 0).cpu().detach().numpy() * 255.0
                name_1 = os.path.join(result_path, '{:05d}.png'.format(int((k + 1) * (opt.num_insert + 1))))
                save_rgb_image(name_1, save_image_1)

                for i in range(opt.num_insert):
                    output_image = output[0, i].transpose(1, 2, 0)
                    out_name = os.path.join(
                        result_path,
                        '{:05d}.png'.format(int(k * (opt.num_insert + 1) + i + 1))
                    )
                    save_rgb_image(out_name, output_image)

                    out_disp = output_disp[0, i, 0]
                    disp_name = os.path.join(
                        disp_path,
                        '{:05d}.png'.format(int(k * (opt.num_insert + 1) + i + 1))
                    )
                    cv2.imwrite(disp_name, out_disp)

        torch.cuda.empty_cache()

    print(f"Finished model: {model_tag}")


def test_multi_models(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.path.join(opt.origin_path, opt.dataset)
    if opt.test_list is None or len(opt.test_list) == 0:
        test_list = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    else:
        test_list = opt.test_list

    ckpt_paths = [resolve_model_path(p, opt.ckpt_name) for p in opt.model_paths]

    print("Dataset   :", opt.dataset)
    print("Num models:", len(ckpt_paths))
    print("Sequences :", len(test_list))
    print("Device    :", device)

    for idx, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[{idx + 1}/{len(ckpt_paths)}]")
        run_one_model(opt, ckpt_path, device, test_list)

    print("\nAll model testing finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SEVFI with multiple models")

    parser.add_argument("--dataset", type=str, default="SEID", choices=["SEID", "DSEC", "MVSEC"])
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="one or more checkpoint file paths OR run directory paths")
    parser.add_argument("--ckpt_name", type=str, default="",
                        help="when model_path is a directory, choose checkpoint name, e.g. best_train.pth")
    parser.add_argument("--origin_path", type=str, default="./sample/dataset/",
                        help="path of test datasets")
    parser.add_argument("--test_list", nargs="*", default=None,
                        help="test sequence names, e.g. --test_list seq1 seq2; empty means all")
    parser.add_argument("--save_path", type=str, default="./sample/result/",
                        help="saving root path")
    parser.add_argument("--num_skip", type=int, default=5, help="num of skip frames")
    parser.add_argument("--num_insert", type=int, default=5, help="num of insert frames")
    parser.add_argument("--use_dp", action="store_true",
                        help="use DataParallel for inference when multiple GPUs are visible")

    opt = parser.parse_args()
    test_multi_models(opt)
import os
import re
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

from Networks.SEVFI import SEVFI_dc_DSEC
from script.dataloader import train_DSEC_sevfi


def parse_int_list(text: str):
    if text is None:
        return []
    text = str(text).strip()
    if text == "":
        return []
    vals = []
    for x in text.split(","):
        x = x.strip()
        if x == "":
            continue
        vals.append(int(x))
    vals = sorted(list(set(vals)))
    if len(vals) == 0:
        return []
    if any(v < 1 for v in vals):
        raise ValueError(f"All values must be positive integers, but got: {vals}")
    return vals


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_bn_eval(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        m.eval()
        if m.weight is not None:
            m.weight.requires_grad_(True)
        if m.bias is not None:
            m.bias.requires_grad_(True)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


def build_model(dataset: str):
    if dataset != "DSEC":
        raise ValueError(f"Current train.py is configured only for DSEC, but got: {dataset}")
    return SEVFI_dc_DSEC()


def resolve_root_dir(opt):
    """
    兼容两种传法：
    1) --origin_path /path/to/DSEC_mini   -> 实际训练目录 /path/to/DSEC_mini/train
    2) --origin_path /path/to/root        -> 实际训练目录 /path/to/root/DSEC/train
    """
    cand_1 = os.path.join(opt.origin_path, opt.split)
    cand_2 = os.path.join(opt.origin_path, opt.dataset, opt.split)
    cand_3 = os.path.join(opt.origin_path, opt.dataset)

    if os.path.isdir(cand_1):
        return cand_1
    if os.path.isdir(cand_2):
        return cand_2
    if os.path.isdir(cand_3):
        return cand_3

    raise RuntimeError(
        f"Cannot resolve dataset root. Tried:\n"
        f"  {cand_1}\n"
        f"  {cand_2}\n"
        f"  {cand_3}"
    )


def build_train_dataset(
    dataset: str,
    root_dir: str,
    num_bins: int,
    num_skip: int,
    num_insert: int,
    train_mode: str = "fixed",
    skip_choices=None,
    target_mode: str = "random_one",
    use_dense_start: bool = False,
):
    if dataset != "DSEC":
        raise NotImplementedError("当前 dataloader.py 只保留了 DSEC 版本，请使用 --dataset DSEC")

    seq_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if len(seq_names) == 0:
        raise RuntimeError(f"No sequences found in {root_dir}")

    print(f"Found sequences in {root_dir}:")
    for name in seq_names:
        print(f"  - {name}")

    sets = []
    for name in seq_names:
        seq_path = os.path.join(root_dir, name)
        ds = train_DSEC_sevfi(
            data_path=seq_path,
            num_bins=num_bins,
            num_skip=num_skip,
            num_insert=num_insert,
            train_mode=train_mode,
            skip_choices=skip_choices,
            target_mode=target_mode,
            use_dense_start=use_dense_start,
        )
        sets.append(ds)

    return ConcatDataset(sets)


def to_tensor(x):
    return x if torch.is_tensor(x) else torch.from_numpy(x)


def random_flip_train_tensors(
    image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t,
    disp_t=None, disp_mask=None
):
    if random.random() < 0.5:
        image_0 = torch.flip(image_0, dims=[2])
        image_1 = torch.flip(image_1, dims=[2])
        eframes_t0 = torch.flip(eframes_t0, dims=[4])
        eframes_t1 = torch.flip(eframes_t1, dims=[4])
        iwe = torch.flip(iwe, dims=[3])
        image_t = torch.flip(image_t, dims=[3])

        if disp_t is not None:
            disp_t = torch.flip(disp_t, dims=[3])
        if disp_mask is not None:
            disp_mask = torch.flip(disp_mask, dims=[3])

    return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask


def random_crop_train_tensors(
    image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t,
    disp_t=None, disp_mask=None, patch_size=224
):
    if patch_size is None or patch_size <= 0:
        return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask

    H, W = image_0.shape[1], image_0.shape[2]
    ps = min(patch_size, H, W)

    if ps == H and ps == W:
        return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask

    top = random.randint(0, H - ps)
    left = random.randint(0, W - ps)

    image_0 = image_0[:, top:top + ps, left:left + ps, :]
    image_1 = image_1[:, top:top + ps, left:left + ps, :]

    eframes_t0 = eframes_t0[:, :, :, top:top + ps, left:left + ps]
    eframes_t1 = eframes_t1[:, :, :, top:top + ps, left:left + ps]
    iwe = iwe[:, :, top:top + ps, left:left + ps]
    image_t = image_t[:, :, top:top + ps, left:left + ps, :]

    if disp_t is not None:
        disp_t = disp_t[:, :, top:top + ps, left:left + ps]
    if disp_mask is not None:
        disp_mask = disp_mask[:, :, top:top + ps, left:left + ps]

    return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask


def batch_to_cpu_patches(batch, patch_size):
    image_0 = to_tensor(batch["image_0"])
    image_1 = to_tensor(batch["image_1"])
    eframes_t0 = to_tensor(batch["eframes_t0"])
    eframes_t1 = to_tensor(batch["eframes_t1"])
    iwe = to_tensor(batch["iwe"])
    weight = to_tensor(batch["weight"])
    image_t = to_tensor(batch["image_t"])

    disp_t = to_tensor(batch["disp_t"]) if "disp_t" in batch else None
    disp_mask = to_tensor(batch["disp_mask"]) if "disp_mask" in batch else None

    image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask = \
        random_flip_train_tensors(
            image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask
        )

    image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask = \
        random_crop_train_tensors(
            image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, disp_t, disp_mask, patch_size
        )

    image_0 = image_0.permute(0, 3, 1, 2).contiguous().float()
    image_1 = image_1.permute(0, 3, 1, 2).contiguous().float()
    image_t = image_t.permute(0, 1, 4, 2, 3).contiguous().float()

    if image_0.max() > 1.5:
        image_0 = image_0 / 255.0
        image_1 = image_1 / 255.0
        image_t = image_t / 255.0

    eframes_t0 = eframes_t0.contiguous().float()
    eframes_t1 = eframes_t1.contiguous().float()
    iwe = iwe.contiguous().float()
    weight = weight.contiguous().float()

    if weight.dim() > 2:
        weight = weight.view(weight.shape[0], weight.shape[1])

    pack = {
        "image_0": image_0,
        "image_1": image_1,
        "eframes_t0": eframes_t0,
        "eframes_t1": eframes_t1,
        "iwe": iwe,
        "weight": weight,
        "image_t": image_t,
    }

    if disp_t is not None:
        pack["disp_t"] = disp_t.contiguous().float()
    if disp_mask is not None:
        pack["disp_mask"] = disp_mask.contiguous().float()

    return pack


def save_checkpoint(net, save_path):
    """
    保持原逻辑不变：只保存模型权重，避免影响你现有 test / load 流程
    """
    state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(state_dict, save_path)


def save_resume_checkpoint(net, optimizer, scheduler, scaler, epoch, best_train_loss, opt, save_path):
    """
    新增：保存完整断点，用于精确 resume
    """
    state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    ckpt = {
        "checkpoint_type": "resume_full",
        "epoch": epoch,
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_train_loss": best_train_loss,
        "args": vars(opt),
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, save_path)


def infer_start_epoch_from_path(resume_path):
    """
    兼容旧的仅权重文件：
    epoch003.pth -> 下一轮从 epoch=4 开始
    其他文件默认从 0 开始，或者让用户手动传 --resume_epoch
    """
    name = os.path.basename(resume_path)
    m = re.search(r"epoch(\d+)\.pth$", name)
    if m:
        return int(m.group(1)) + 1
    return 0


def load_model_state(net, state_dict):
    if hasattr(net, "module"):
        net.module.load_state_dict(state_dict, strict=True)
    else:
        net.load_state_dict(state_dict, strict=True)


def resume_or_load_checkpoint(net, optimizer, scheduler, scaler, resume_path, device="cpu", resume_epoch=-1):
    """
    支持两种 resume：
    1) 新格式完整断点 resume_latest.pth：精确恢复 optimizer/scheduler/scaler/epoch
    2) 旧格式纯模型权重 epochXXX.pth：只恢复模型，start_epoch 由文件名推断或 --resume_epoch 指定
    """
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)

    # 新格式完整断点
    if isinstance(ckpt, dict) and ("model" in ckpt) and ("optimizer" in ckpt) and ("epoch" in ckpt):
        load_model_state(net, ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        if "random_state" in ckpt:
            random.setstate(ckpt["random_state"])
        if "numpy_state" in ckpt:
            np.random.set_state(ckpt["numpy_state"])
        if "torch_state" in ckpt:
            torch.set_rng_state(ckpt["torch_state"])
        if torch.cuda.is_available() and ckpt.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_state"])

        start_epoch = int(ckpt["epoch"]) + 1
        best_train_loss = float(ckpt.get("best_train_loss", float("inf")))
        print(f"[Resume] Loaded FULL checkpoint: {resume_path}")
        print(f"[Resume] start_epoch={start_epoch}, best_train_loss={best_train_loss:.8f}")
        return start_epoch, best_train_loss, "full"

    # 旧格式仅权重
    if isinstance(ckpt, dict) and ("state_dict" in ckpt) and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    load_model_state(net, state_dict)

    if resume_epoch >= 0:
        start_epoch = int(resume_epoch)
    else:
        start_epoch = infer_start_epoch_from_path(resume_path)

    best_train_loss = float("inf")
    print(f"[Resume] Loaded WEIGHTS-ONLY checkpoint: {resume_path}")
    print(f"[Resume] start_epoch={start_epoch} (optimizer/scheduler/scaler not restored)")
    return start_epoch, best_train_loss, "weights_only"


def build_scheduler(optimizer, opt):
    if opt.scheduler == "none":
        return None
    elif opt.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opt.epochs,
            eta_min=opt.min_lr
        )
    elif opt.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=opt.step_size,
            gamma=opt.gamma
        )
    elif opt.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=opt.gamma,
            patience=opt.patience,
            min_lr=opt.min_lr,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {opt.scheduler}")


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def masked_smooth_l1(pred, target, mask):
    valid = mask > 0.5
    if valid.any():
        return F.smooth_l1_loss(pred[valid], target[valid], reduction="mean")
    return torch.zeros([], device=pred.device, dtype=pred.dtype)


def check_finite_tensor(name, x, epoch, it, n):
    if x is None:
        return
    if torch.is_tensor(x):
        if not torch.isfinite(x).all():
            xmin = x.min().item() if x.numel() > 0 else float("nan")
            xmax = x.max().item() if x.numel() > 0 else float("nan")
            print(f"[Bad Tensor] epoch={epoch} iter={it} n={n} {name} has NaN/Inf")
            print(f"  shape={tuple(x.shape)} min={xmin} max={xmax}")
            raise RuntimeError(f"{name} has NaN/Inf")


def main(opt):
    set_seed(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = build_model(opt.dataset)

    if opt.use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    root_dir = resolve_root_dir(opt)
    skip_choices = parse_int_list(opt.skip_choices)

    if opt.train_mode == "mixed" and len(skip_choices) == 0:
        raise ValueError("When --train_mode mixed, --skip_choices cannot be empty.")

    if opt.train_mode == "mixed" and opt.target_mode == "all" and opt.batch_size != 1:
        raise ValueError(
            "mixed + target_mode=all 时，不同 skip 会返回不同数量的目标帧，"
            "请把 --batch_size 设为 1。"
        )

    train_set = build_train_dataset(
        dataset=opt.dataset,
        root_dir=root_dir,
        num_bins=opt.num_bins,
        num_skip=opt.num_skip,
        num_insert=opt.num_insert,
        train_mode=opt.train_mode,
        skip_choices=skip_choices,
        target_mode=opt.target_mode,
        use_dense_start=opt.use_dense_start,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(opt.num_workers > 0)
    )

    if opt.loss_type == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = CharbonnierLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opt)
    scaler = GradScaler(enabled=(opt.amp and device.type == "cuda"))

    os.makedirs(opt.save_dir, exist_ok=True)

    resume_path = opt.resume.strip()

    # run_dir 逻辑：
    # 1) 指定了 run_name -> 和原来一样
    # 2) 没指定 run_name 但指定了 resume -> 直接写回 resume 所在目录
    # 3) 两者都没有 -> 自动新建 run_name
    if opt.run_name.strip():
        run_name = opt.run_name.strip()
        run_dir = os.path.join(opt.save_dir, run_name)
    elif resume_path:
        run_dir = os.path.dirname(os.path.abspath(resume_path))
        run_name = os.path.basename(run_dir)
    else:
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        if opt.train_mode == "mixed":
            skip_tag = opt.skip_choices.replace(",", "-")
            run_name = (
                f"{opt.dataset}_{opt.split}_mixed_s{skip_tag}_{opt.target_mode}"
                f"_p{opt.patch_size}_lr{opt.lr:g}_{time_tag}"
            )
        else:
            run_name = (
                f"{opt.dataset}_{opt.split}_fixed_s{opt.num_skip}_i{opt.num_insert}"
                f"_p{opt.patch_size}_lr{opt.lr:g}_{time_tag}"
            )
        run_dir = os.path.join(opt.save_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # resume / warm start
    start_epoch = 0
    best_train_loss = float("inf")
    resume_mode = "none"
    if resume_path:
        start_epoch, best_train_loss, resume_mode = resume_or_load_checkpoint(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resume_path=resume_path,
            device="cpu",
            resume_epoch=opt.resume_epoch
        )

    args_path = os.path.join(run_dir, "args.json")
    if resume_path and os.path.exists(args_path):
        resume_args_path = os.path.join(
            run_dir, f"resume_args_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(resume_args_path, "w", encoding="utf-8") as f:
            json.dump(vars(opt), f, indent=2, ensure_ascii=False)
    else:
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(opt), f, indent=2, ensure_ascii=False)

    log_path = os.path.join(run_dir, "train_log.csv")
    if (not resume_path) or (not os.path.exists(log_path)) or start_epoch == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,avg_loss,avg_rec_loss,avg_disp_loss,lr\n")
    else:
        print(f"[Resume] append log to: {log_path}")

    print("==============================================")
    print(f"train_mode      : {opt.train_mode}")
    print(f"skip_choices    : {opt.skip_choices}")
    print(f"target_mode     : {opt.target_mode}")
    print(f"use_dense_start : {opt.use_dense_start}")
    print(f"dataset         : {opt.dataset}")
    print(f"split           : {opt.split}")
    print(f"root_dir        : {root_dir}")
    print(f"run_dir         : {run_dir}")
    print(f"num_skip        : {opt.num_skip}")
    print(f"num_insert      : {opt.num_insert}")
    print(f"batch_size      : {opt.batch_size}")
    print(f"patch_size      : {opt.patch_size}")
    print(f"optimizer       : Adam")
    print(f"lr              : {opt.lr}")
    print(f"scheduler       : {opt.scheduler}")
    print(f"min_lr          : {opt.min_lr}")
    print(f"lambda_rec      : {opt.lambda_rec}")
    print(f"lambda_disp     : {opt.lambda_disp}")
    print(f"amp             : {opt.amp}")
    print(f"use_dp          : {opt.use_dp}")
    print(f"device          : {device}")
    print(f"resume          : {resume_path if resume_path else 'None'}")
    print(f"resume_mode     : {resume_mode}")
    print(f"start_epoch     : {start_epoch}")
    print(f"best_train_loss : {best_train_loss}")
    print("==============================================")

    for epoch in range(start_epoch, opt.epochs):
        net.train()
        net.apply(set_bn_eval)

        running = 0.0
        running_rec = 0.0
        running_disp = 0.0
        valid_steps = 0

        for it, batch in enumerate(train_loader):
            pack = batch_to_cpu_patches(batch, opt.patch_size)

            image_0 = pack["image_0"].to(device, non_blocking=True)
            image_1 = pack["image_1"].to(device, non_blocking=True)

            N = pack["eframes_t0"].shape[1]

            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_rec_loss = 0.0
            step_disp_loss = 0.0
            bad_step = False

            for n in range(N):
                eframes_t0 = pack["eframes_t0"][:, n].to(device, non_blocking=True)
                eframes_t1 = pack["eframes_t1"][:, n].to(device, non_blocking=True)
                iwe = pack["iwe"][:, n].unsqueeze(1).to(device, non_blocking=True)
                weight = pack["weight"][:, n].to(device, non_blocking=True)
                image_t = pack["image_t"][:, n].to(device, non_blocking=True)

                use_disp_sup = ("disp_t" in pack and "disp_mask" in pack and opt.lambda_disp > 0)
                if use_disp_sup:
                    disp_t = pack["disp_t"][:, n].unsqueeze(1).to(device, non_blocking=True)
                    disp_mask = pack["disp_mask"][:, n].unsqueeze(1).to(device, non_blocking=True)
                else:
                    disp_t = None
                    disp_mask = None

                try:
                    check_finite_tensor("image_0", image_0, epoch, it, n)
                    check_finite_tensor("image_1", image_1, epoch, it, n)
                    check_finite_tensor("eframes_t0", eframes_t0, epoch, it, n)
                    check_finite_tensor("eframes_t1", eframes_t1, epoch, it, n)
                    check_finite_tensor("iwe", iwe, epoch, it, n)
                    check_finite_tensor("weight", weight, epoch, it, n)
                    check_finite_tensor("image_t", image_t, epoch, it, n)
                    if disp_t is not None:
                        check_finite_tensor("disp_t", disp_t, epoch, it, n)
                    if disp_mask is not None:
                        check_finite_tensor("disp_mask", disp_mask, epoch, it, n)
                except RuntimeError as e:
                    print(f"[Skip Batch] epoch={epoch} iter={it} n={n} because input check failed: {e}")
                    optimizer.zero_grad(set_to_none=True)
                    bad_step = True
                    break

                with autocast(enabled=(opt.amp and device.type == "cuda")):
                    outputs = net(image_0, image_1, eframes_t0, eframes_t1, iwe, weight)

                    if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
                        raise RuntimeError("Model output format is unexpected.")

                    image_syn = outputs[0]
                    image_fuse = outputs[1]
                    image_final = outputs[2]
                    disp_pred = outputs[3] if len(outputs) > 3 else None

                    loss_rec = criterion(image_final, image_t)
                    if opt.supervise_all:
                        loss_syn = criterion(image_syn, image_t)
                        loss_fuse = criterion(image_fuse, image_t)
                        loss_rec = loss_rec + opt.w_syn * loss_syn + opt.w_fuse * loss_fuse

                    if use_disp_sup and disp_pred is not None:
                        loss_disp = masked_smooth_l1(disp_pred, disp_t, disp_mask)
                    else:
                        loss_disp = torch.zeros([], device=device)

                    loss = opt.lambda_rec * loss_rec + opt.lambda_disp * loss_disp
                    loss = loss / N

                try:
                    check_finite_tensor("image_syn", image_syn, epoch, it, n)
                    check_finite_tensor("image_fuse", image_fuse, epoch, it, n)
                    check_finite_tensor("image_final", image_final, epoch, it, n)
                    if disp_pred is not None:
                        check_finite_tensor("disp_pred", disp_pred, epoch, it, n)
                    check_finite_tensor("loss_rec", loss_rec, epoch, it, n)
                    check_finite_tensor("loss_disp", loss_disp, epoch, it, n)
                    check_finite_tensor("loss", loss, epoch, it, n)
                except RuntimeError as e:
                    print(f"[Skip Batch] epoch={epoch} iter={it} n={n} because output check failed: {e}")
                    optimizer.zero_grad(set_to_none=True)
                    bad_step = True
                    break

                if not torch.isfinite(loss):
                    print(f"[Bad Loss] epoch={epoch} iter={it} n={n} loss={loss}")
                    optimizer.zero_grad(set_to_none=True)
                    bad_step = True
                    break

                try:
                    scaler.scale(loss).backward()
                except RuntimeError as e:
                    print(f"[Skip Batch] epoch={epoch} iter={it} n={n} because backward failed: {e}")
                    optimizer.zero_grad(set_to_none=True)
                    bad_step = True
                    break

                step_loss += loss.item()
                step_rec_loss += (opt.lambda_rec * loss_rec / N).item()
                step_disp_loss += (opt.lambda_disp * loss_disp / N).item()

                del eframes_t0, eframes_t1, iwe, weight, image_t
                if disp_t is not None:
                    del disp_t, disp_mask
                del outputs, image_syn, image_fuse, image_final
                if disp_pred is not None:
                    del disp_pred

            if bad_step or (not np.isfinite(step_loss)):
                print(f"Skip optimizer step at epoch={epoch} iter={it} because step is invalid")
                del pack, image_0, image_1
                continue

            grad_norm = None
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)

            if grad_norm is not None:
                grad_norm_value = float(grad_norm)
                if (not np.isfinite(grad_norm_value)) or grad_norm_value > opt.skip_grad_norm:
                    print(f"[Bad Grad] epoch={epoch} iter={it} grad_norm={grad_norm_value}")
                    optimizer.zero_grad(set_to_none=True)
                    del pack, image_0, image_1
                    continue

            try:
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                print(f"[Skip Step] epoch={epoch} iter={it} because optimizer step failed: {e}")
                optimizer.zero_grad(set_to_none=True)
                del pack, image_0, image_1
                continue

            running += step_loss
            running_rec += step_rec_loss
            running_disp += step_disp_loss
            valid_steps += 1

            if it % opt.log_every == 0:
                if "meta_seq" in batch:
                    print(f"[Meta] seq={batch['meta_seq']} sample_index={batch['meta_index']} img0={batch['meta_img0']} img1={batch['meta_img1']}")
                if grad_norm is not None:
                    print(
                        f"[{opt.dataset}] epoch {epoch}/{opt.epochs} "
                        f"iter {it}/{len(train_loader)} "
                        f"loss {step_loss:.6f} rec {step_rec_loss:.6f} disp {step_disp_loss:.6f} "
                        f"grad_norm {float(grad_norm):.6f}"
                    )
                else:
                    print(
                        f"[{opt.dataset}] epoch {epoch}/{opt.epochs} "
                        f"iter {it}/{len(train_loader)} "
                        f"loss {step_loss:.6f} rec {step_rec_loss:.6f} disp {step_disp_loss:.6f}"
                    )

            del pack, image_0, image_1

        if valid_steps > 0:
            avg_loss = running / valid_steps
            avg_rec_loss = running_rec / valid_steps
            avg_disp_loss = running_disp / valid_steps
        else:
            avg_loss = float("nan")
            avg_rec_loss = float("nan")
            avg_disp_loss = float("nan")

        current_lr = get_current_lr(optimizer)

        print(
            f"==> epoch {epoch} "
            f"avg_loss {avg_loss:.6f} "
            f"avg_rec {avg_rec_loss:.6f} "
            f"avg_disp {avg_disp_loss:.6f} "
            f"lr {current_lr:.8f} "
            f"valid_steps {valid_steps}/{len(train_loader)}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.8f},{avg_rec_loss:.8f},{avg_disp_loss:.8f},{current_lr:.10f}\n")

        if np.isfinite(avg_loss):
            # 保持原有权重文件保存方式不变
            save_path = os.path.join(run_dir, f"epoch{epoch:03d}.pth")
            save_checkpoint(net, save_path)
            print("Saved:", save_path)

            latest_path = os.path.join(run_dir, "latest.pth")
            save_checkpoint(net, latest_path)

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_path = os.path.join(run_dir, "best_train.pth")
                save_checkpoint(net, best_path)
                print("Saved best_train:", best_path)

            # 新增完整断点文件
            resume_latest_path = os.path.join(run_dir, "resume_latest.pth")
            save_resume_checkpoint(
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_train_loss=best_train_loss,
                opt=opt,
                save_path=resume_latest_path
            )
            print("Saved resume checkpoint:", resume_latest_path)
        else:
            print(f"Skip saving epoch {epoch} because avg_loss is not finite.")

        if scheduler is not None and np.isfinite(avg_loss):
            if opt.scheduler == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SEVFI (DSEC mini stable)")
    parser.add_argument("--train_mode", type=str, default="fixed",
                        choices=["fixed", "mixed"],
                        help="fixed: 原来的固定 skip/insert 训练; mixed: 单模型混合多个 skip 和时间位置训练")

    parser.add_argument("--skip_choices", type=str, default="1,3,5",
                        help="only used when train_mode=mixed, e.g. '1,3,5'")

    parser.add_argument("--target_mode", type=str, default="random_one",
                        choices=["random_one", "all"],
                        help="random_one: 每个样本随机监督一个真实目标时刻(推荐); all: 监督当前 skip 下所有真实中间帧(batch_size需为1)")

    parser.add_argument("--use_dense_start", action="store_true",
                        help="mixed 模式下是否用每一帧都作为潜在起点；不开启时更接近你原来的按 skip+1 步长采样")
    parser.add_argument("--dataset", type=str, default="DSEC", choices=["DSEC"])
    parser.add_argument("--origin_path", type=str, default="./DSEC_mini/",
                        help="for your current setup, set this to DSEC_mini root")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="for your current setup, use train")
    parser.add_argument("--save_dir", type=str, default="./PreTrained/",
                        help="where to save checkpoints")
    parser.add_argument("--run_name", type=str, default="",
                        help="sub-folder name for this training run; empty means auto generate")

    # 新增：resume 支持
    parser.add_argument("--resume", type=str, default="",
                        help="path to resume checkpoint. "
                             "Supports both new full checkpoint (resume_latest.pth) "
                             "and old weights-only checkpoint (epochXXX.pth/latest.pth)")
    parser.add_argument("--resume_epoch", type=int, default=-1,
                        help="only used when --resume points to old weights-only checkpoint. "
                             "-1 means auto infer from filename, e.g. epoch003.pth -> start from epoch 4")

    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--num_skip", type=int, default=3)
    parser.add_argument("--num_insert", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "charb"])
    parser.add_argument("--supervise_all", action="store_true")
    parser.add_argument("--w_syn", type=float, default=0.5)
    parser.add_argument("--w_fuse", type=float, default=0.5)
    parser.add_argument("--lambda_rec", type=float, default=2.0)
    parser.add_argument("--lambda_disp", type=float, default=0.008)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--patch_size", type=int, default=224,
                        help="random square crop size for training; 0 means full image")
    parser.add_argument("--amp", action="store_true",
                        help="use mixed precision training")
    parser.add_argument("--use_dp", action="store_true",
                        help="use torch.nn.DataParallel when multiple GPUs are visible")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--skip_grad_norm", type=float, default=1000.0,
                        help="skip optimizer step if grad norm is larger than this threshold")

    parser.add_argument("--scheduler", type=str, default="step",
                        choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--step_size", type=int, default=10,
                        help="for StepLR")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="for StepLR / ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=2,
                        help="for ReduceLROnPlateau")

    opt = parser.parse_args()
    main(opt)
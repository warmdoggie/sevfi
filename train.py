import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

from Networks.SEVFI import SEVFI_dc_DSEC, SEVFI_dc_MVSEC, SEVFI_dc_SEID
from script.dataloader import train_DSEC_sevfi, train_MVSEC_sevfi, train_SEID_sevfi


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


def build_model(dataset: str):
    if dataset == "DSEC":
        return SEVFI_dc_DSEC()
    elif dataset == "MVSEC":
        return SEVFI_dc_MVSEC()
    elif dataset == "SEID":
        return SEVFI_dc_SEID()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_train_dataset(dataset: str, root_dir: str, num_bins: int, num_skip: int, num_insert: int):
    seq_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if len(seq_names) == 0:
        raise RuntimeError(f"No sequences found in {root_dir}")

    sets = []
    for name in seq_names:
        seq_path = os.path.join(root_dir, name)
        if dataset == "DSEC":
            ds = train_DSEC_sevfi(
                data_path=seq_path,
                num_bins=num_bins,
                num_skip=num_skip,
                num_insert=num_insert
            )
        elif dataset == "MVSEC":
            ds = train_MVSEC_sevfi(
                data_path=seq_path,
                num_bins=num_bins,
                num_skip=num_skip,
                num_insert=num_insert
            )
        else:
            ds = train_SEID_sevfi(
                data_path=seq_path,
                num_bins=num_bins,
                num_skip=num_skip,
                num_insert=num_insert
            )
        sets.append(ds)

    return ConcatDataset(sets)


def to_tensor(x):
    return x if torch.is_tensor(x) else torch.from_numpy(x)


def random_crop_train_tensors(image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, patch_size):
    """
    image_0:   [B, H, W, 3]
    image_1:   [B, H, W, 3]
    eframes_t0:[B, N, C, H, W]
    eframes_t1:[B, N, C, H, W]
    iwe:       [B, N, H, W]
    weight:    [B, N]
    image_t:   [B, N, H, W, 3]
    """
    if patch_size is None or patch_size <= 0:
        return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t

    H, W = image_0.shape[1], image_0.shape[2]
    ps = min(patch_size, H, W)

    if ps == H and ps == W:
        return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t

    top = random.randint(0, H - ps)
    left = random.randint(0, W - ps)

    image_0 = image_0[:, top:top + ps, left:left + ps, :]
    image_1 = image_1[:, top:top + ps, left:left + ps, :]

    eframes_t0 = eframes_t0[:, :, :, top:top + ps, left:left + ps]
    eframes_t1 = eframes_t1[:, :, :, top:top + ps, left:left + ps]
    iwe = iwe[:, :, top:top + ps, left:left + ps]
    image_t = image_t[:, :, top:top + ps, left:left + ps, :]

    return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t


def batch_to_cpu_patches(batch, patch_size):
    image_0 = to_tensor(batch["image_0"])
    image_1 = to_tensor(batch["image_1"])
    eframes_t0 = to_tensor(batch["eframes_t0"])
    eframes_t1 = to_tensor(batch["eframes_t1"])
    iwe = to_tensor(batch["iwe"])
    weight = to_tensor(batch["weight"])
    image_t = to_tensor(batch["image_t"])

    image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t = random_crop_train_tensors(
        image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t, patch_size
    )

    # [B,H,W,3] -> [B,3,H,W]
    image_0 = image_0.permute(0, 3, 1, 2).contiguous().float()
    image_1 = image_1.permute(0, 3, 1, 2).contiguous().float()

    # [B,N,H,W,3] -> [B,N,3,H,W]
    image_t = image_t.permute(0, 1, 4, 2, 3).contiguous().float()

    # 图像归一化到 0~1
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

    return {
        "image_0": image_0,
        "image_1": image_1,
        "eframes_t0": eframes_t0,
        "eframes_t1": eframes_t1,
        "iwe": iwe,
        "weight": weight,
        "image_t": image_t,
    }


def save_checkpoint(net, save_path):
    state_dict = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(state_dict, save_path)


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


def main(opt):
    set_seed(opt.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    net = build_model(opt.dataset)

    if opt.use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    # dataset
    root_dir = os.path.join(opt.origin_path, opt.dataset)
    train_set = build_train_dataset(
        opt.dataset, root_dir, opt.num_bins, opt.num_skip, opt.num_insert
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

    # loss
    if opt.loss_type == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = CharbonnierLoss()

    # optimizer: 你本来就是 Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = build_scheduler(optimizer, opt)

    scaler = GradScaler(enabled=(opt.amp and device.type == "cuda"))

    # run directory
    os.makedirs(opt.save_dir, exist_ok=True)

    if opt.run_name.strip():
        run_name = opt.run_name.strip()
    else:
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{opt.dataset}_s{opt.num_skip}_i{opt.num_insert}_p{opt.patch_size}_lr{opt.lr:g}_{time_tag}"

    run_dir = os.path.join(opt.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(opt), f, indent=2, ensure_ascii=False)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,avg_loss,lr\n")

    best_train_loss = float("inf")
    net.train()

    print("==============================================")
    print(f"dataset      : {opt.dataset}")
    print(f"root_dir     : {root_dir}")
    print(f"run_dir      : {run_dir}")
    print(f"num_skip     : {opt.num_skip}")
    print(f"num_insert   : {opt.num_insert}")
    print(f"batch_size   : {opt.batch_size}")
    print(f"patch_size   : {opt.patch_size}")
    print(f"optimizer    : Adam")
    print(f"lr           : {opt.lr}")
    print(f"scheduler    : {opt.scheduler}")
    print(f"min_lr       : {opt.min_lr}")
    print(f"amp          : {opt.amp}")
    print(f"use_dp       : {opt.use_dp}")
    print(f"device       : {device}")
    print("==============================================")

    for epoch in range(opt.epochs):
        running = 0.0

        for it, batch in enumerate(train_loader):
            pack = batch_to_cpu_patches(batch, opt.patch_size)

            image_0 = pack["image_0"].to(device, non_blocking=True)
            image_1 = pack["image_1"].to(device, non_blocking=True)

            N = pack["eframes_t0"].shape[1]

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for n in range(N):
                eframes_t0 = pack["eframes_t0"][:, n].to(device, non_blocking=True)
                eframes_t1 = pack["eframes_t1"][:, n].to(device, non_blocking=True)
                iwe = pack["iwe"][:, n].unsqueeze(1).to(device, non_blocking=True)
                weight = pack["weight"][:, n].to(device, non_blocking=True)
                image_t = pack["image_t"][:, n].to(device, non_blocking=True)

                with autocast(enabled=(opt.amp and device.type == "cuda")):
                    outputs = net(image_0, image_1, eframes_t0, eframes_t1, iwe, weight)

                    if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                        image_syn, image_fuse, image_final = outputs[0], outputs[1], outputs[2]
                    else:
                        raise RuntimeError("Model output format is unexpected.")

                    loss_final = criterion(image_final, image_t)

                    if opt.supervise_all:
                        loss_syn = criterion(image_syn, image_t)
                        loss_fuse = criterion(image_fuse, image_t)
                        loss = opt.w_syn * loss_syn + opt.w_fuse * loss_fuse + loss_final
                    else:
                        loss = loss_final

                    loss = loss / N

                    if not torch.isfinite(loss):
                        print(f"[Bad Loss] epoch={epoch} iter={it} n={n} loss={loss}")
                        optimizer.zero_grad(set_to_none=True)
                        step_loss = float("nan")
                        del eframes_t0, eframes_t1, iwe, weight, image_t
                        del outputs, image_syn, image_fuse, image_final
                        break

                scaler.scale(loss).backward()
                step_loss += loss.item()

                del eframes_t0, eframes_t1, iwe, weight, image_t
                del outputs, image_syn, image_fuse, image_final

            if not np.isfinite(step_loss):
                print(f"Skip optimizer step at epoch={epoch} iter={it} because loss is NaN/Inf")
                del pack, image_0, image_1
                continue

            grad_norm = None
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), opt.grad_clip)

            if grad_norm is not None and (not torch.isfinite(grad_norm)):
                print(f"[Bad Grad] epoch={epoch} iter={it} grad_norm={grad_norm}")
                optimizer.zero_grad(set_to_none=True)
                del pack, image_0, image_1
                continue

            scaler.step(optimizer)
            scaler.update()

            running += step_loss

            if it % opt.log_every == 0:
                if grad_norm is not None:
                    print(f"[{opt.dataset}] epoch {epoch}/{opt.epochs} iter {it}/{len(train_loader)} loss {step_loss:.6f} grad_norm {float(grad_norm):.6f}")
                else:
                    print(f"[{opt.dataset}] epoch {epoch}/{opt.epochs} iter {it}/{len(train_loader)} loss {step_loss:.6f}")

            del pack, image_0, image_1

        avg_loss = running / max(1, len(train_loader))
        current_lr = get_current_lr(optimizer)
        print(f"==> epoch {epoch} avg_loss {avg_loss:.6f} lr {current_lr:.8f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.8f},{current_lr:.10f}\n")

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

        # scheduler step
        if scheduler is not None:
            if opt.scheduler == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SEVFI (memory-optimized)")

    parser.add_argument("--dataset", type=str, default="DSEC", choices=["DSEC", "SEID", "MVSEC"])
    parser.add_argument("--origin_path", type=str, default="./sample/dataset/",
                        help="root path that contains DSEC/SEID/MVSEC")
    parser.add_argument("--save_dir", type=str, default="./PreTrained_ECA/",
                        help="where to save checkpoints")
    parser.add_argument("--run_name", type=str, default="",
                        help="sub-folder name for this training run; empty means auto generate")

    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--num_skip", type=int, default=3)
    parser.add_argument("--num_insert", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "charb"])
    parser.add_argument("--supervise_all", action="store_true")
    parser.add_argument("--w_syn", type=float, default=0.5)
    parser.add_argument("--w_fuse", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)

    # memory / stability
    parser.add_argument("--patch_size", type=int, default=256,
                        help="random square crop size for training; 0 means full image")
    parser.add_argument("--amp", action="store_true",
                        help="use mixed precision training")
    parser.add_argument("--use_dp", action="store_true",
                        help="use torch.nn.DataParallel when multiple GPUs are visible")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--step_size", type=int, default=5,
                        help="for StepLR")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="for StepLR / ReduceLROnPlateau")

    # plateau
    parser.add_argument("--patience", type=int, default=2,
                        help="for ReduceLROnPlateau")

    opt = parser.parse_args()
    main(opt)
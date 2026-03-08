import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from Networks.SEVFI import SEVFI_dc_DSEC, SEVFI_dc_MVSEC, SEVFI_dc_SEID
from script.dataloader import train_DSEC_sevfi, train_MVSEC_sevfi, train_SEID_sevfi


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    """
    root_dir 类似: ./sample/dataset/DSEC
    里面是若干序列文件夹: zurich_city_01_c, ...
    """
    seq_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if len(seq_names) == 0:
        raise RuntimeError(f"No sequences found in {root_dir}")

    sets = []
    for name in seq_names:
        seq_path = os.path.join(root_dir, name)
        if dataset == "DSEC":
            ds = train_DSEC_sevfi(data_path=seq_path, num_bins=num_bins, num_skip=num_skip, num_insert=num_insert)
        elif dataset == "MVSEC":
            ds = train_MVSEC_sevfi(data_path=seq_path, num_bins=num_bins, num_skip=num_skip, num_insert=num_insert)
        else:
            ds = train_SEID_sevfi(data_path=seq_path, num_bins=num_bins, num_skip=num_skip, num_insert=num_insert)
        sets.append(ds)

    return ConcatDataset(sets)


def batch_to_device(batch, device):
    """
    把 dataloader 返回的 batch(dict) 转成网络需要的 tensor，并展平成 [B*N, ...]
    这里复用你 test.py 的 reshape 逻辑，但支持 batch_size>1
    """
    # batch 里的 numpy/tensor 形状通常是：
    # image_0: [B, H, W, 3]
    # image_1: [B, H, W, 3]
    # eframes_t0: [B, N, C, H, W]
    # eframes_t1: [B, N, C, H, W]
    # iwe: [B, N, H, W]
    # weight: [B, N]
    # image_t(GT): [B, N, H, W, 3]
    image_0 = batch["image_0"]
    image_1 = batch["image_1"]
    eframes_t0 = batch["eframes_t0"]
    eframes_t1 = batch["eframes_t1"]
    iwe = batch["iwe"]
    weight = batch["weight"]
    image_t = batch["image_t"]

    # 确保是 torch.Tensor
    if not torch.is_tensor(image_0):
        image_0 = torch.from_numpy(image_0)
    if not torch.is_tensor(image_1):
        image_1 = torch.from_numpy(image_1)
    if not torch.is_tensor(eframes_t0):
        eframes_t0 = torch.from_numpy(eframes_t0)
    if not torch.is_tensor(eframes_t1):
        eframes_t1 = torch.from_numpy(eframes_t1)
    if not torch.is_tensor(iwe):
        iwe = torch.from_numpy(iwe)
    if not torch.is_tensor(weight):
        weight = torch.from_numpy(weight)
    if not torch.is_tensor(image_t):
        image_t = torch.from_numpy(image_t)

    B = image_0.shape[0]
    N = eframes_t0.shape[1]
    H, W = image_0.shape[1], image_0.shape[2]  # image_0: [B,H,W,3]
    C_ev = eframes_t0.shape[2]

    # image_0/1: [B,H,W,3] -> [B,3,H,W]
    image_0 = image_0.permute(0, 3, 1, 2).contiguous()
    image_1 = image_1.permute(0, 3, 1, 2).contiguous()

    # 复制到每个插帧时刻： [B,3,H,W] -> [B,N,3,H,W] -> [B*N,3,H,W]
    image_0 = image_0.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, 3, H, W)
    image_1 = image_1.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, 3, H, W)

    # eframes: [B,N,C,H,W] -> [B*N,C,H,W]
    eframes_t0 = eframes_t0.reshape(B * N, C_ev, H, W)
    eframes_t1 = eframes_t1.reshape(B * N, C_ev, H, W)

    # iwe: [B,N,H,W] -> [B*N,1,H,W]
    iwe = iwe.reshape(B * N, 1, H, W)

    # weight: [B,N] -> [B*N]
    weight = weight.reshape(B * N)

    # GT: [B,N,H,W,3] -> [B,N,3,H,W] -> [B*N,3,H,W]
    image_t = image_t.permute(0, 1, 4, 2, 3).contiguous().reshape(B * N, 3, H, W)

    # float + device
    image_0 = image_0.float().to(device)
    image_1 = image_1.float().to(device)
    eframes_t0 = eframes_t0.float().to(device)
    eframes_t1 = eframes_t1.float().to(device)
    iwe = iwe.float().to(device)
    weight = weight.float().to(device)
    image_t = image_t.float().to(device)

    return image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t


def main(opt):
    set_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    net = build_model(opt.dataset)
    #net = torch.nn.DataParallel(net).to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # dataset
    root_dir = os.path.join(opt.origin_path, opt.dataset)
    train_set = build_train_dataset(opt.dataset, root_dir, opt.num_bins, opt.num_skip, opt.num_insert)
    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # loss
    if opt.loss_type == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = CharbonnierLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    net.train()
    os.makedirs(opt.save_dir, exist_ok=True)

    for epoch in range(opt.epochs):
        running = 0.0
        for it, batch in enumerate(train_loader):
            image_0, image_1, eframes_t0, eframes_t1, iwe, weight, image_t = batch_to_device(batch, device)

            image_syn, image_fuse, image_final, disp, flowlist_t0, flowlist_t1 = \
                net(image_0, image_1, eframes_t0, eframes_t1, iwe, weight)

            # 最小可用 loss：只监督 final
            loss_final = criterion(image_final, image_t)

            # 可选：也监督 syn/fuse（更容易收敛）
            if opt.supervise_all:
                loss_syn = criterion(image_syn, image_t)
                loss_fuse = criterion(image_fuse, image_t)
                loss = opt.w_syn * loss_syn + opt.w_fuse * loss_fuse + loss_final
            else:
                loss = loss_final

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            if it % opt.log_every == 0:
                print(f"[{opt.dataset}] epoch {epoch}/{opt.epochs} iter {it} loss {loss.item():.6f}")

        avg_loss = running / max(1, len(train_loader))
        print(f"==> epoch {epoch} avg_loss {avg_loss:.6f}")

        # 保存为 test.py 能直接读的格式：save_dir/DSEC.pth
        save_path = os.path.join(opt.save_dir, f"{opt.dataset}.pth")
        torch.save(net.state_dict(), save_path)
        print("Saved:", save_path)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train SEVFI")
    parser.add_argument("--dataset", type=str, default="DSEC", choices=["DSEC", "SEID", "MVSEC"])
    parser.add_argument("--origin_path", type=str, default="./sample/dataset/", help="root path that contains DSEC/SEID/MVSEC")
    parser.add_argument("--save_dir", type=str, default="./PreTrained_ECA/", help="where to save checkpoints")
    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--num_skip", type=int, default=3)
    parser.add_argument("--num_insert", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "charb"])
    parser.add_argument("--supervise_all", action="store_true")
    parser.add_argument("--w_syn", type=float, default=0.5)
    parser.add_argument("--w_fuse", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)

    opt = parser.parse_args()
    main(opt)
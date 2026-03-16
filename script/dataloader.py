import os
import cv2
import h5py
import torch
import random
import numpy as np
from script import utils
from torch.utils.data import Dataset


# =========================
# DSEC helper functions
# =========================

def _resolve_seq_root(data_path: str):
    """
    支持两种传法：
    1) .../zurich_city_01_c
    2) .../zurich_city_01_c/data
    """
    if os.path.isdir(os.path.join(data_path, "data")):
        return os.path.join(data_path, "data")
    return data_path


def _find_existing_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _find_h5_dataset(group, keys):
    """
    在 h5 group 里按候选 key 查找 dataset
    """
    for k in keys:
        if k in group:
            return group[k]
    raise KeyError(f"Cannot find any key in {keys}. Existing keys: {list(group.keys())}")


def _load_rectify_map(rectify_file):
    """
    同时兼容：
    - rectify_map.h5
    - rectify_maps.h5
    以及里面不同的 key 名
    """
    with h5py.File(rectify_file, "r") as f:
        for key in ["rectify_map", "rectify_maps"]:
            if key in f:
                return np.asarray(f[key])

        for key in f.keys():
            arr = f[key]
            if len(arr.shape) == 3 and arr.shape[2] == 2:
                return np.asarray(arr)

        raise KeyError(
            f"Cannot find rectify map dataset in {rectify_file}. Existing keys: {list(f.keys())}"
        )


def _read_disp_png_float(path):
    """
    DSEC disparity_image:
    valid disparity = uint16 / 256.0
    0 表示无效像素
    """
    disp_u16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp_u16 is None:
        raise RuntimeError(f"Failed to read disparity png: {path}")
    if disp_u16.ndim != 2:
        raise RuntimeError(f"Disparity must be single-channel, got shape={disp_u16.shape} from {path}")
    disp = disp_u16.astype(np.float32) / 256.0
    return disp


def _nearest_index(timestamps: np.ndarray, ts: float):
    return int(np.argmin(np.abs(timestamps - ts)))


def _get_gt_indices(index_0, index_1, num_insert):
    """
    给定两端帧 index_0, index_1
    生成 num_insert 个 GT 中间帧下标
    """
    step = (index_1 - index_0) / float(num_insert + 1)
    gt_indices = []
    for i in range(num_insert):
        gt_idx = int(round(index_0 + step * (i + 1)))
        gt_indices.append(gt_idx)
    return gt_indices


def _get_all_middle_indices(index_0, index_1):
    """
    返回 index_0 和 index_1 之间所有真实中间帧下标
    """
    if index_1 <= index_0 + 1:
        return []
    return list(range(index_0 + 1, index_1))


# =========================
# DSEC dataloader
# =========================

class test_DSEC_sevfi(Dataset):
    """
    固定测试/验证版本：
    - 仍然使用固定 num_skip / num_insert
    - 但内部统一按“真实目标帧时间戳”构造 weight 和事件切分
    """

    def __init__(self, data_path, num_bins, num_skip, num_insert, delta_ts=25000):
        super().__init__()

        self.seq_root = _resolve_seq_root(data_path)

        self.images_path = os.path.join(self.seq_root, "images")
        self.events_dir = os.path.join(self.seq_root, "events")
        self.disp_path = os.path.join(self.seq_root, "disparity_image")

        self.events_file = os.path.join(self.events_dir, "events.h5")
        self.rectify_file = _find_existing_file([
            os.path.join(self.events_dir, "rectify_map.h5"),
            os.path.join(self.events_dir, "rectify_maps.h5"),
        ])

        if not os.path.isdir(self.images_path):
            raise RuntimeError(f"Images dir not found: {self.images_path}")
        if not os.path.isfile(self.events_file):
            raise RuntimeError(f"Events file not found: {self.events_file}")
        if self.rectify_file is None:
            raise RuntimeError(f"Cannot find rectify map file under {self.events_dir}")

        self.images_list = utils.get_filename(self.images_path, ".png")

        image_ts_file = _find_existing_file([
            os.path.join(self.seq_root, "image_timestamps.txt"),
            os.path.join(self.images_path, "timestamp.txt"),
        ])
        if image_ts_file is None:
            raise RuntimeError(f"Cannot find image timestamps under {self.seq_root}")
        self.image_timestamps = np.loadtxt(image_ts_file).astype(np.int64)

        disp_ts_file = os.path.join(self.seq_root, "disparity_timestamps.txt")
        if os.path.isfile(disp_ts_file):
            self.disp_timestamps = np.loadtxt(disp_ts_file).astype(np.int64)
        else:
            self.disp_timestamps = None

        if os.path.isdir(self.disp_path):
            self.disp_list = utils.get_filename(self.disp_path, ".png")
        else:
            self.disp_list = []

        self.num_bins = num_bins
        self.num_skip = num_skip
        self.num_insert = num_insert
        self.delta_ts = delta_ts

        self.target_h = 480
        self.target_w = 640

        self.rectify_map = _load_rectify_map(self.rectify_file)
        self.sensor_h, self.sensor_w = self.rectify_map.shape[:2]

        self.voxel_grid = utils.VoxelGrid(
            self.num_bins, self.target_h, self.target_w, normalize=True
        )

        self._events_h5 = None
        self._ev_x = None
        self._ev_y = None
        self._ev_p = None
        self._ev_t = None
        self._ms_to_idx = None
        self._t_offset = 0

        first_img = cv2.imread(os.path.join(self.images_path, self.images_list[0]), cv2.IMREAD_COLOR)
        if first_img is None:
            raise RuntimeError("Failed to read the first image.")
        self.orig_h, self.orig_w = first_img.shape[:2]

    def _ensure_events_open(self):
        if self._events_h5 is not None:
            return

        self._events_h5 = h5py.File(self.events_file, "r")

        if "events" not in self._events_h5:
            raise KeyError(f"'events' group not found in {self.events_file}. Keys: {list(self._events_h5.keys())}")

        g = self._events_h5["events"]
        self._ev_x = _find_h5_dataset(g, ["x"])
        self._ev_y = _find_h5_dataset(g, ["y"])
        self._ev_p = _find_h5_dataset(g, ["p"])
        self._ev_t = _find_h5_dataset(g, ["t"])

        self._ms_to_idx = np.asarray(self._events_h5["ms_to_idx"]) if "ms_to_idx" in self._events_h5 else None

        if "t_offset" in self._events_h5:
            t_offset_val = self._events_h5["t_offset"][()]
            self._t_offset = int(np.array(t_offset_val).reshape(-1)[0])
        else:
            self._t_offset = 0

    def __del__(self):
        try:
            if self._events_h5 is not None:
                self._events_h5.close()
        except Exception:
            pass

    def events_to_voxel_grid(self, x, y, p, t):
        if len(t) <= 10:
            return self.voxel_grid.voxel_grid

        t = (t - t[0]).astype(np.float32)
        if len(t) == 0 or t[-1] <= 0:
            return self.voxel_grid.voxel_grid
        t = t / t[-1]

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        p = p.astype(np.float32)

        return self.voxel_grid.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(p),
            torch.from_numpy(t)
        )

    def __len__(self):
        len_all_data = len(self.image_timestamps)
        return (len_all_data - 1) // (self.num_skip + 1)

    def _searchsorted_h5(self, dset, value, side="left"):
        lo, hi = 0, len(dset)
        while lo < hi:
            mid = (lo + hi) // 2
            mid_val = int(dset[mid])
            if side == "left":
                if mid_val < value:
                    lo = mid + 1
                else:
                    hi = mid
            else:
                if mid_val <= value:
                    lo = mid + 1
                else:
                    hi = mid
        return lo

    def _slice_events_by_time(self, t0_global, t1_global):
        """
        输入统一时钟下的时间戳（和 image_timestamps 同时钟）
        返回：
            x, y, p, t_global
        """
        self._ensure_events_open()

        t0_local = int(t0_global - self._t_offset)
        t1_local = int(t1_global - self._t_offset)

        if t1_local <= t0_local:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )

        if self._ms_to_idx is not None:
            ms0 = max(int(t0_local // 1000) - 1, 0)
            ms1 = min(int(t1_local // 1000) + 2, len(self._ms_to_idx) - 1)

            idx0 = int(self._ms_to_idx[ms0])
            idx1 = int(self._ms_to_idx[ms1])
        else:
            idx0 = self._searchsorted_h5(self._ev_t, t0_local, side="left")
            idx1 = self._searchsorted_h5(self._ev_t, t1_local, side="right")

        t = np.asarray(self._ev_t[idx0:idx1]).astype(np.int64)
        x = np.asarray(self._ev_x[idx0:idx1])
        y = np.asarray(self._ev_y[idx0:idx1])
        p = np.asarray(self._ev_p[idx0:idx1])

        if len(t) == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )

        mask = (t >= t0_local) & (t <= t1_local)
        x = x[mask]
        y = y[mask]
        p = p[mask]
        t = t[mask] + self._t_offset

        return x, y, p, t

    def _rectify_events(self, x, y, p, t):
        if len(t) == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )

        x = x.astype(np.int64)
        y = y.astype(np.int64)

        valid = (x >= 0) & (x < self.sensor_w) & (y >= 0) & (y < self.sensor_h)
        x = x[valid]
        y = y[valid]
        p = p[valid]
        t = t[valid]

        if len(t) == 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )

        coords = utils.rectify_events(self.rectify_map, x, y, self.sensor_h, self.sensor_w)
        x_r = coords[:, 0].astype(np.float32)
        y_r = coords[:, 1].astype(np.float32)

        mask = (x_r >= 0) & (x_r < self.target_w) & (y_r >= 0) & (y_r < self.target_h)
        x_r = x_r[mask]
        y_r = y_r[mask]
        p = p[mask]
        t = t[mask]

        return x_r, y_r, p, t

    def _load_image_resized(self, index):
        img = cv2.imread(os.path.join(self.images_path, self.images_list[index]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.images_list[index]}")
        img = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        return img

    def _build_sample_from_pair(self, index_0, index_1, target_indices):
        """
        核心公共函数：
        给定边界帧和若干真实目标帧下标，构造 sample
        """
        target_indices = [int(i) for i in target_indices]
        target_indices = sorted(target_indices)

        image_0 = self._load_image_resized(index_0)
        image_1 = self._load_image_resized(index_1)

        ts_0 = int(self.image_timestamps[index_0])
        ts_1 = int(self.image_timestamps[index_1])

        ts_targets = np.array([self.image_timestamps[i] for i in target_indices], dtype=np.int64)
        TS = np.concatenate([
            np.array([ts_0], dtype=np.int64),
            ts_targets,
            np.array([ts_1], dtype=np.int64)
        ], axis=0)

        t_read_0 = int(ts_0 - self.delta_ts)
        t_read_1 = int(ts_1 + self.delta_ts)

        x, y, p, t = self._slice_events_by_time(t_read_0, t_read_1)
        x, y, p, t = self._rectify_events(x, y, p, t)

        # ===== IWE around each target timestamp =====
        iwe = []
        for ts_t in ts_targets:
            t_start = int(ts_t - self.delta_ts)
            t_end = int(ts_t + self.delta_ts)

            event_iwe = utils.filter_events_by_time(x, y, p, t, t_start, t_end)

            x_img = event_iwe[0].astype(np.int32)
            y_img = event_iwe[1].astype(np.int32)
            p_img = event_iwe[2].astype(np.int8)

            if len(p_img) > 0:
                p_img[p_img == 0] = -1

            event_img = np.zeros(self.target_h * self.target_w, dtype=np.float32)
            if len(x_img) > 0:
                np.add.at(event_img, y_img * self.target_w + x_img, p_img)
            event_img = event_img.reshape([self.target_h, self.target_w])
            iwe.append(event_img)

        # ===== voxel grid =====
        eframes_t0 = []
        eframes_t1 = []
        weight = []

        dt_01 = max(float(ts_1 - ts_0), 1.0)

        for ts_t in ts_targets:
            event_0t = utils.filter_events_by_time(x, y, p, t, ts_0, int(ts_t))
            event_t1 = utils.filter_events_by_time(x, y, p, t, int(ts_t), ts_1)

            voxel_t1 = self.events_to_voxel_grid(
                event_t1[0], event_t1[1], event_t1[2], event_t1[3]
            )

            x_t0, y_t0, p_t0, t_t0 = utils.reverse_events(
                event_0t[0], event_0t[1], event_0t[2], event_0t[3]
            )
            voxel_t0 = self.events_to_voxel_grid(x_t0, y_t0, p_t0, t_t0)

            eframes_t1.append(voxel_t1.detach().numpy())
            eframes_t0.append(voxel_t0.detach().numpy())
            weight.append(float((float(ts_t) - float(ts_0)) / dt_01))

        sample = {
            "timestamps": TS.astype(np.int64),
            "target_indices": np.array(target_indices, dtype=np.int64),
            "image_0": image_0,
            "image_1": image_1,
            "eframes_t1": np.array(eframes_t1, dtype=np.float32),
            "eframes_t0": np.array(eframes_t0, dtype=np.float32),
            "weight": np.array(weight, dtype=np.float32),
            "iwe": np.array(iwe, dtype=np.float32),
        }
        return sample

    def __getitem__(self, index):
        index_0 = index * (self.num_skip + 1)
        index_1 = index_0 + (self.num_skip + 1)

        gt_indices = _get_gt_indices(index_0, index_1, self.num_insert)
        sample = self._build_sample_from_pair(index_0, index_1, gt_indices)
        return sample


class train_DSEC_sevfi(test_DSEC_sevfi):
    """
    训练版本：
    支持两种模式

    1) fixed:
       完全兼容你原来的固定 num_skip / num_insert 训练

    2) mixed:
       一个模型混合看到多个 skip，并按 target_mode 选择监督目标
       - skip_choices 例如 [1,3,5]
       - target_mode:
           "random_one" : 每个样本随机监督一个目标时刻（推荐）
           "all"        : 每个样本监督该 skip 下所有真实中间帧（batch_size 建议为 1）
    """

    def __init__(
        self,
        data_path,
        num_bins,
        num_skip,
        num_insert,
        delta_ts=25000,
        train_mode="fixed",
        skip_choices=None,
        target_mode="random_one",
        use_dense_start=False,
    ):
        base_skip = num_skip
        base_insert = num_insert

        if skip_choices is not None and len(skip_choices) > 0:
            base_skip = max(skip_choices)
            base_insert = max(skip_choices)

        super().__init__(data_path, num_bins, base_skip, base_insert, delta_ts=delta_ts)

        self.seq_name = os.path.basename(os.path.dirname(self.seq_root))

        self.train_mode = train_mode
        self.target_mode = target_mode
        self.use_dense_start = use_dense_start

        if skip_choices is None or len(skip_choices) == 0:
            self.skip_choices = [num_skip]
        else:
            self.skip_choices = sorted(list(set(int(x) for x in skip_choices)))

        self.fixed_num_skip = num_skip
        self.fixed_num_insert = num_insert

        # 已知会触发 DCN/梯度异常的坏样本（local_index 语义）
        self.bad_samples = {
            "zurich_city_02_c": {792, 892},
        }

        # 02_c 后半段不稳定，先做保守截断（local_index 语义）
        self.max_valid_index_by_seq = {
            "zurich_city_02_c": 970,
        }

        if self.train_mode not in ["fixed", "mixed"]:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")

        if self.target_mode not in ["random_one", "all"]:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        if self.train_mode == "mixed":
            self.records = self._build_mixed_records()
        else:
            self.records = None

    def _build_mixed_records(self):
        records = []
        total_imgs = len(self.image_timestamps)

        for skip in self.skip_choices:
            if self.use_dense_start:
                # 每一帧都可作为起点，更密；更贴近“按 t 采样”
                max_start = total_imgs - (skip + 2)
                for index_0 in range(max_start + 1):
                    index_1 = index_0 + (skip + 1)
                    if index_1 >= total_imgs:
                        continue
                    local_index = index_0
                    if self.seq_name in self.max_valid_index_by_seq:
                        if local_index > self.max_valid_index_by_seq[self.seq_name]:
                            continue
                    if self.seq_name in self.bad_samples:
                        if local_index in self.bad_samples[self.seq_name]:
                            continue
                    records.append({
                        "index_0": index_0,
                        "index_1": index_1,
                        "skip": skip,
                        "local_index": local_index,
                    })
            else:
                # 和你原来风格更一致：按 skip+1 的步长取样
                base_len = (total_imgs - 1) // (skip + 1)
                for local_index in range(base_len):
                    index_0 = local_index * (skip + 1)
                    index_1 = index_0 + (skip + 1)
                    if index_1 >= total_imgs:
                        continue
                    if self.seq_name in self.max_valid_index_by_seq:
                        if local_index > self.max_valid_index_by_seq[self.seq_name]:
                            continue
                    if self.seq_name in self.bad_samples:
                        if local_index in self.bad_samples[self.seq_name]:
                            continue
                    records.append({
                        "index_0": index_0,
                        "index_1": index_1,
                        "skip": skip,
                        "local_index": local_index,
                    })

        if len(records) == 0:
            raise RuntimeError(
                f"No mixed-mode training records built for seq={self.seq_name}, "
                f"skip_choices={self.skip_choices}"
            )
        return records

    def __len__(self):
        if self.train_mode == "mixed":
            return len(self.records)

        # fixed mode: 保留你原逻辑
        base_len = (len(self.image_timestamps) - 1) // (self.fixed_num_skip + 1)
        if self.seq_name in self.max_valid_index_by_seq:
            return min(base_len, self.max_valid_index_by_seq[self.seq_name] + 1)
        return base_len

    def _load_disp_resized(self, disp_index):
        if len(self.disp_list) == 0:
            raise RuntimeError(f"No disparity png found under: {self.disp_path}")

        disp_index = max(0, min(disp_index, len(self.disp_list) - 1))
        disp = _read_disp_png_float(os.path.join(self.disp_path, self.disp_list[disp_index]))

        scale_x = self.target_w / float(self.orig_w)

        disp = cv2.resize(disp, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
        disp = disp * scale_x

        return disp.astype(np.float32)

    def _get_disp_index_from_ts(self, ts, fallback_img_index):
        if self.disp_timestamps is not None and len(self.disp_list) == len(self.disp_timestamps):
            return _nearest_index(self.disp_timestamps, ts)
        return fallback_img_index

    def _select_target_indices(self, index_0, index_1, skip):
        all_middle = _get_all_middle_indices(index_0, index_1)
        if len(all_middle) == 0:
            raise RuntimeError(f"No middle frames between {index_0} and {index_1}")

        if self.train_mode == "fixed":
            return _get_gt_indices(index_0, index_1, self.fixed_num_insert)

        # mixed mode
        if self.target_mode == "all":
            return all_middle

        if self.target_mode == "random_one":
            return [random.choice(all_middle)]

        raise ValueError(f"Unknown target_mode: {self.target_mode}")

    def __getitem__(self, index):
        if self.train_mode == "mixed":
            rec = self.records[index]
            index_0 = rec["index_0"]
            index_1 = rec["index_1"]
            skip = rec["skip"]
            local_index = rec["local_index"]
        else:
            # fixed mode 兼容你原来的坏样本跳过逻辑
            local_index = index
            if self.seq_name in self.bad_samples and local_index in self.bad_samples[self.seq_name]:
                new_index = min(local_index + 1, len(self) - 1)
                print(f"[Skip Bad Sample] {self.seq_name} index={local_index} -> use index={new_index}")
                local_index = new_index

            skip = self.fixed_num_skip
            index_0 = local_index * (skip + 1)
            index_1 = index_0 + (skip + 1)

        target_indices = self._select_target_indices(index_0, index_1, skip)

        sample = self._build_sample_from_pair(index_0, index_1, target_indices)

        gt_list = []
        disp_list = []
        disp_mask_list = []

        for gi in target_indices:
            gi = max(0, min(int(gi), len(self.images_list) - 1))

            gt = self._load_image_resized(gi)
            gt_list.append(gt)

            ts_t = int(self.image_timestamps[gi])
            disp_idx = self._get_disp_index_from_ts(ts_t, gi)
            disp = self._load_disp_resized(disp_idx)
            disp_mask = (disp > 0).astype(np.float32)

            disp_list.append(disp)
            disp_mask_list.append(disp_mask)

        sample["image_t"] = np.stack(gt_list, axis=0).astype(np.uint8)
        sample["disp_t"] = np.stack(disp_list, axis=0).astype(np.float32)
        sample["disp_mask"] = np.stack(disp_mask_list, axis=0).astype(np.float32)

        sample["meta_seq"] = self.seq_name
        sample["meta_index"] = np.int64(local_index)
        sample["meta_img0"] = np.int64(index_0)
        sample["meta_img1"] = np.int64(index_1)
        sample["meta_skip"] = np.int64(skip)
        sample["meta_target_indices"] = np.array(target_indices, dtype=np.int64)
        sample["meta_target_mode"] = self.target_mode

        return sample
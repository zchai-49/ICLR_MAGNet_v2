# dataset preparation (splitting to training, validation, and testing)
# time series splicing

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json
import os as _os

DATASET_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "dataset")

REVIN_DATASETS = {10, 11, 12, 13, 14, 15, 16, 17}   # 17 = Weather (rebuttal)

HOURLY_DATASETS = {5, 6, 7, 8, 9, 10, 13, 14, 15, 16}
DAILY_DATASETS = {11, 12}

def load_dataset_info(json_path: str) -> dict:
    """Load dataset metadata from a JSON config file."""
    with open(json_path) as f:
        info = json.load(f)

    # derive convenience fields
    info['all_vars'] = info['endo'] + info['exo']
    info['n_vars']   = len(info['all_vars'])
    info['n_endo']   = len(info['endo'])
    info['n_exo']    = len(info['exo'])
    return info

def get_dataset_info(index: int):
    if index == 1:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/traffic/traffic.json')
    elif index == 2:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/weather5k/weather5k.json')
    elif index == 5:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/EPF/BE.json')
    elif index == 6:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/EPF/DE.json')  
    elif index == 7:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/EPF/FR.json')   
    elif index == 8:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/EPF/NP.json')
    elif index == 9:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/EPF/PJM.json')     
    elif index == 10:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/energy/Energy.json')
    elif index == 11:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/hydro/Colbun.json')
    elif index == 12:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/hydro/Rapel.json')     
    elif index == 13:
        DATASET_INFO = load_dataset_info(f'{DATASET_ROOT}/wind/Sdwpf.json')            
    else:
        print("Error!")
# derived for convenience
    DATASET_INFO['all_vars'] = DATASET_INFO['endo'] + DATASET_INFO['exo']
    DATASET_INFO['n_vars']   = len(DATASET_INFO['all_vars'])
    DATASET_INFO['n_endo']   = len(DATASET_INFO['endo'])
    DATASET_INFO['n_exo']    = len(DATASET_INFO['exo'])
    return DATASET_INFO

# splicing time series
def make_patches(x, patch_len):
    seq_len, n_vars = x.shape
    num_patches = seq_len // patch_len
    x = x[:num_patches * patch_len]
    return x.reshape(num_patches, patch_len, n_vars)

def make_time_features(
    start_idx    : int,
    seq_len      : int,
    patch_len    : int,
    dataset_id   : int,
    split_offset : int,
) -> np.ndarray:
    """
    Compute 4 sin/cos time features per patch, domain-specific:

    EPF (5-9)    hourly: [hour_sin, hour_cos, dow_sin,   dow_cos  ]
    Energy (10)  hourly: [hour_sin, hour_cos, month_sin, month_cos]
    Wind (13-16) hourly: [hour_sin, hour_cos, month_sin, month_cos]
    Hydro (11-12) daily: [dow_sin,  dow_cos,  month_sin, month_cos]

    Returns: [num_patches, 4]
    """
    EPF_DATASETS   = {5, 6, 7, 8, 9}
    HYDRO_DATASETS = {11, 12}

    abs_start   = split_offset + start_idx
    num_patches = seq_len // patch_len
    time_feats  = []

    for p in range(num_patches):
        t = abs_start + p * patch_len   # absolute timestep at patch start

        if dataset_id in HYDRO_DATASETS:
            # daily resolution — no sub-daily cycle
            # t = day index
            dow   = t % 7
            month = (t % 365) // 30     # approximate month (0-11)
            f1 = np.sin(2 * np.pi * dow   / 7.0)
            f2 = np.cos(2 * np.pi * dow   / 7.0)
            f3 = np.sin(2 * np.pi * month / 12.0)
            f4 = np.cos(2 * np.pi * month / 12.0)

        elif dataset_id in EPF_DATASETS:
            # hourly — daily price cycle + weekday/weekend market pattern
            hour  = t % 24
            dow   = (t // 24) % 7
            f1 = np.sin(2 * np.pi * hour / 24.0)
            f2 = np.cos(2 * np.pi * hour / 24.0)
            f3 = np.sin(2 * np.pi * dow  / 7.0)
            f4 = np.cos(2 * np.pi * dow  / 7.0)

        else:
            # hourly (Energy, Wind) — daily cycle + seasonal monsoon/solar
            hour  = t % 24
            month = (t // (24 * 30)) % 12   # approximate month from hours
            f1 = np.sin(2 * np.pi * hour  / 24.0)
            f2 = np.cos(2 * np.pi * hour  / 24.0)
            f3 = np.sin(2 * np.pi * month / 12.0)
            f4 = np.cos(2 * np.pi * month / 12.0)

        time_feats.append([f1, f2, f3, f4])

    return np.array(time_feats, dtype=np.float32)  # [num_patches, 4]

class TSDataset(Dataset):
    """
    Sliding-window dataset for a single PeMS04 sensor.
 
    Each sample returns:
        exo_patches  : [num_patches, patch_len, n_exo]   float32
        endo_patches : [num_patches, patch_len, 1]        float32
        target       : [forecast_horizon]                 float32  (raw scale)
 
    Args:
        npz_path        : path to PEMS04.npz
        sensor_id       : which sensor to use (default 0)
        seq_len         : input window length in timesteps (default 48 = 4 hrs)
        forecast_horizon: how many steps ahead to predict  (default 12 = 1 hr)
        patch_len       : length of each patch             (default 12)
        split           : 'train' | 'val' | 'test'
        train_ratio     : fraction of data for training    (default 0.6)
        val_ratio       : fraction of data for validation  (default 0.2)
    """
 
    def __init__(
        self,
        npz_path: str,
        sensor_id: int = 0,
        seq_len: int = 48,
        forecast_horizon: int = 12,
        patch_len: int = 12,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        dataset_id: int = None,
        use_global_scaler: bool = True,
        return_future_exog: bool = False,   # DAG only: true future exog as a TRAINING-time teacher signal
    ):
        super().__init__()
        # Rolling-origin validation fold B (2026-11-10): train on [0, 60%), validate on [60%, 70%). Only used to choose
        # configs for long horizons; deployed models and the test split are always fold A (70/10/20).
        if _os.environ.get("MAGNET_FOLD") == "B":
            train_ratio, val_ratio = 0.6, 0.1
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"
        assert seq_len % patch_len == 0, \
            f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})"
 
        self.seq_len          = seq_len
        self.forecast_horizon = forecast_horizon
        self.patch_len        = patch_len
        self.num_patches = seq_len // patch_len
        self.use_revin        = (dataset_id in REVIN_DATASETS)
        self.dataset_id       = dataset_id
        self.use_global_scaler = use_global_scaler
        self.return_future_exog = return_future_exog
 
        # ── load raw data for this sensor ────────────────────────────────────
        if "EPF" in npz_path:
            if "BE." in npz_path:
                data = pd.read_csv(npz_path)
                sensor_data = pd.DataFrame()
                data["idx"] = data.groupby("cols").cumcount()
                sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
                sensor_data = sensor_data[["OT", "System load forecast", "Generation forecast"]]
                DATASET_INFO = get_dataset_info(5)
            elif "DE." in npz_path:
                data = pd.read_csv(npz_path)
                sensor_data = pd.DataFrame()
                data["idx"] = data.groupby("cols").cumcount()
                sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
                sensor_data = sensor_data[["OT", "Ampirion zonal load forecast", "Wind power forecast"]]
                DATASET_INFO = get_dataset_info(6)
            elif "FR." in npz_path:
                data = pd.read_csv(npz_path)
                sensor_data = pd.DataFrame()
                data["idx"] = data.groupby("cols").cumcount()
                sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
                sensor_data = sensor_data[["OT", "System load forecast", "Generation forecast"]]
                DATASET_INFO = get_dataset_info(7)
            elif "NP." in npz_path:
                data = pd.read_csv(npz_path)
                sensor_data = pd.DataFrame()
                data["idx"] = data.groupby("cols").cumcount()
                sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
                sensor_data = sensor_data[["OT", "Grid load forecast", "Wind power forecast"]]
                DATASET_INFO = get_dataset_info(8)
            elif "PJM." in npz_path:
                data = pd.read_csv(npz_path)
                sensor_data = pd.DataFrame()
                data["idx"] = data.groupby("cols").cumcount()
                sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
                sensor_data = sensor_data[["OT", "System load forecast", "Zonal COMED load foecast"]]
                DATASET_INFO = get_dataset_info(9)
        elif "energy" in npz_path:
            data = pd.read_csv(npz_path)
            sensor_data = pd.DataFrame()
            data["idx"] = data.groupby("cols").cumcount()
            sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
            sensor_data = sensor_data[["Thermoelectric", "Battery", "Geothermal", "Hydroelectric", "Solar", "Wind"]]
            DATASET_INFO = get_dataset_info(10)
        elif "hydro" in npz_path:
            data = pd.read_csv(npz_path)
            sensor_data = pd.DataFrame()
            data["idx"] = data.groupby("cols").cumcount()
            sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
            sensor_data = sensor_data[["Water_Level_daily_avg", "Afluente (m3/seg)_daily_avg", "Precipitation_daily_sum"]]
            if "Colbun." in npz_path:
                DATASET_INFO = get_dataset_info(11)
            elif "Rapel." in npz_path:
                DATASET_INFO = get_dataset_info(12)
        elif "wind" in npz_path:
            data = pd.read_csv(npz_path)
            sensor_data = pd.DataFrame()
            data["idx"] = data.groupby("cols").cumcount()
            sensor_data = data.pivot(index='idx', columns='cols', values='data').rename_axis(None).rename_axis(None, axis=1)
            sensor_data = sensor_data[["Patv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp"]]
            if _os.environ.get("MAGNET_WIND_PHYS") == "1":
                # Wind idea 1 (2026-11-14): physics-informed exo, same 6 variables (text/graph shapes unchanged).
                #   Wspd_w -> power-curve proxy clip((v^3 - v_in^3) / (v_r^3 - v_in^3), 0, 1), cut-in 3 m/s, rated 12 m/s
                #   Wdir_w -> along-prevailing-wind component v cos(dir - theta), theta = circular mean direction of
                #             the TRAIN split (degrees wrap at 360, which a linear/attention model cannot use raw)
                sensor_data = sensor_data.copy()
                v = sensor_data["Wspd_w"].to_numpy(dtype=np.float64); dr = np.deg2rad(sensor_data["Wdir_w"].to_numpy(dtype=np.float64))
                ntr = int(len(sensor_data) * train_ratio)
                theta = np.arctan2(np.sin(dr[:ntr]).mean(), np.cos(dr[:ntr]).mean())
                sensor_data["Wspd_w"] = np.clip((v ** 3 - 27.0) / (12.0 ** 3 - 27.0), 0.0, 1.0)
                sensor_data["Wdir_w"] = v * np.cos(dr - theta)
            DATASET_INFO = get_dataset_info(13)
        elif "/generic/" in npz_path:
            # rebuttal datasets (2026-11-12): wide CSV (date + one column per variable) + <name>.json metadata
            DATASET_INFO = load_dataset_info(npz_path[:-4] + ".json")
            raw = pd.read_csv(npz_path)
            sensor_data = raw[[v["name"] for v in sorted(DATASET_INFO["all_vars"], key=lambda v: v["index"])]]
            sensor_data.columns = range(sensor_data.shape[1])
        else:
            print("Data path non-existent!")
            sensor_data = None
 
        # variable names — useful later for LLM text generation
        self.ENDO_IDX = [v['index'] for v in DATASET_INFO['endo']]
        self.EXO_IDX  = [v['index'] for v in DATASET_INFO['exo']]
        #self.ENDO_IDX = [0]
        #self.EXO_IDX = [1, 2, 3, 4]

        total_len = len(sensor_data)
 
        # ── train / val / test split (by time — no shuffling!) ───────────────
        train_end = int(total_len * train_ratio)
        val_end   = int(total_len * (train_ratio + val_ratio))
 
        if split == 'train':
            data_slice = sensor_data[:train_end]
            split_offset = 0
        elif split == 'val':
            # MAGNET_VAL_LOOKBACK=1 (standard ETT/Informer border, 2026-11-10): validation inputs may reach back
            # seq_len steps into the end of train, forecast targets stay inside [train_end, val_end). No leakage;
            # ~4x more validation windows on long horizons. Set automatically for fold B.
            lb = self.seq_len if (_os.environ.get("MAGNET_VAL_LOOKBACK") == "1" or
                                  _os.environ.get("MAGNET_FOLD") == "B") else 0
            data_slice = sensor_data[train_end - lb:val_end]
            split_offset = train_end - lb
        else:
            data_slice = sensor_data[val_end:]
            split_offset = val_end

        self.split_offset = split_offset
 
        # ── fit scaler on train, apply to all splits ──────────────────────────
        # (we always fit on train portion to avoid leakage)
        if self.use_global_scaler:
            self.scaler = StandardScaler()
            train_data  = sensor_data[:train_end]
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(data_slice).astype(np.float32)
        else:
            # skip the global fit-on-train StandardScaler entirely -- raw
            # units, relying solely on the model's own per-window instance
            # norm (use_instance_norm=True) instead of stacking both.
            self.scaler = None
            self.data = data_slice.values.astype(np.float32)

        self.use_diff = (dataset_id in {11}) and (seq_len > 100)
        if self.use_diff:
            endo_idx = self.ENDO_IDX[0]
            # store original z-scored data for last-known reconstruction
            self.orig_data = self.data.copy()
            # apply differencing to endo column only
            diff_data = self.data.copy()
            diff_data[1:, endo_idx] = np.diff(self.data[:, endo_idx])
            diff_data[0, endo_idx] = 0.0
            self.data = diff_data
        else:
            self.orig_data = self.data.copy()

        self.endo_mean =  np.float32(0.0)
        self.endo_std =  np.float32(1.0)
 
        # ── number of valid sliding windows ──────────────────────────────────
        self.n_samples = len(self.data) - self.seq_len - self.forecast_horizon + 1
 
        print(f"[{DATASET_INFO['name']} Dataset] split={split} | "
              f"sensor={sensor_id} | "
              f"windows={self.n_samples} | "
              f"seq_len={seq_len} | patch_len={patch_len} | "
              f"num_patches={self.num_patches} | "
              f"forecast_horizon={forecast_horizon}")
 
    def __len__(self) -> int:
        return self.n_samples
 
    def __getitem__(self, idx: int):
        # ── slice window ─────────────────────────────────────────────────────
        x = self.data[idx : idx + self.seq_len]                        # [seq_len, 3]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.forecast_horizon, self.ENDO_IDX[0]] # [horizon]

        if self.use_diff:
            last_known = self.orig_data[idx + self.seq_len - 1, self.ENDO_IDX[0]]
        else:
            last_known = np.float32(0.0)  # unused

        # ── split endo / exo ─────────────────────────────────────────────────
        endo = x[:, self.ENDO_IDX]   # [seq_len, 1]
        exo  = x[:, self.EXO_IDX]    # [seq_len, 2]\

        # ── patch ─────────────────────────────────────────────────────────────
        endo_patches = make_patches(endo, self.patch_len)  # [num_patches, patch_len, 1]
        exo_patches  = make_patches(exo,  self.patch_len)  # [num_patches, patch_len, 2]

        time_feats = make_time_features(
            start_idx    = idx,
            seq_len      = self.seq_len,
            patch_len    = self.patch_len,
            dataset_id   = self.dataset_id,
            split_offset = self.split_offset,
        )   # [num_patches, 4] for hourly, [num_patches, 2] for daily
 
        item = [
            torch.tensor(exo_patches),   # [num_patches, patch_len, n_exo]
            torch.tensor(endo_patches),  # [num_patches, patch_len, 1]
            torch.tensor(y),             # [forecast_horizon]
            torch.tensor(self.endo_mean),  # ← dummy, no-op denorm
            torch.tensor(self.endo_std),  # ← dummy, no-op denorm
        ]
        if self.return_future_exog:
            # Only requested by DAG's training loop (its auxiliary causality loss / teacher forcing).
            # Never requested by any other model, and never at test time.
            y_exo = self.data[idx + self.seq_len : idx + self.seq_len + self.forecast_horizon, self.EXO_IDX]
            item.append(torch.tensor(y_exo.astype(np.float32)))
        item += [torch.tensor(time_feats), torch.tensor(last_known, dtype=torch.float32)]
        return tuple(item)
 
def get_dataloaders(
    npz_path: str,
    sensor_id: int = 0,
    seq_len: int = 96,
    forecast_horizon: int = 12,
    patch_len: int = 12,
    batch_size: int = 16,
    num_workers: int = 0,
    dataset_id: int = None,
    use_global_scaler: bool = True,
    persistent_workers: bool = False,
    return_future_exog: bool = False,
):
    """
    Returns train, val, test DataLoaders for a single PeMS04 sensor.
    """
    datasets = {
        split: TSDataset(
            npz_path=npz_path,
            sensor_id=sensor_id,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            patch_len=patch_len,
            split=split,
            dataset_id=dataset_id,
            use_global_scaler=use_global_scaler,
            return_future_exog=return_future_exog,
        )
        for split in ('train', 'val', 'test')
    }
 
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
        for split, ds in datasets.items()
    }
 
    return loaders['train'], loaders['val'], loaders['test']

if __name__ == '__main__':
    NPZ_PATH = f'{DATASET_ROOT}/energy/Energy.csv'
 
    train_loader, val_loader, test_loader = get_dataloaders(
        npz_path=NPZ_PATH,
        sensor_id=0,
        seq_len=96,
        forecast_horizon=12,
        patch_len=12,
        batch_size=16,
    )
 
    exo_patches, endo_patches, target = next(iter(train_loader))
 
    print("\n── Batch shapes ──────────────────────────────")
    print(f"  exo_patches  : {exo_patches.shape}")   # [B, num_patches, patch_len, 2]
    print(f"  endo_patches : {endo_patches.shape}")  # [B, num_patches, patch_len, 1]
    print(f"  target       : {target.shape}")         # [B, forecast_horizon]
    print(f"\n── Sample values (exo_patches[0, 0]) ───────")
    print(f"  occupancy & speed patch:\n{exo_patches[0, 0]}")
    print(f"\n── Target (first sample) ────────────────────")
    print(f"  next 12 flow values: {target[0]}")
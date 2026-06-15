from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import xarray as xr
import cfgrib

PRIOR_TOKEN_X = -10.0
TARGET_TOKEN_X = -11.0
SEP_TOKEN_X = -12.0
THINKING_TOKEN_X = -13.0
# Match existing role / segment conventions
ROLE_PRIOR = 0
ROLE_EVIDENCE = 1
ROLE_THINKING = 2
ROLE_PREDICTION = 3

SEG_PRIOR = 0
SEG_TARGET = 1


def _open_era5_grib_with_elevation(grib_path: str) -> xr.Dataset:
    """
    Open a GRIB file containing ERA5 2m temperature and geopotential,
    then convert geopotential to elevation in meters.
    """
    datasets = cfgrib.open_datasets(grib_path)
    if len(datasets) == 0:
        raise RuntimeError(f"No GRIB datasets found in: {grib_path}")

    ds = xr.merge(datasets, compat="override", join="outer")

    if "expver" in ds.dims:
        ds = ds.isel(expver=0, drop=True)

    rename_map = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    temp_var = None
    for cand in ["t2m", "2t"]:
        if cand in ds.data_vars:
            temp_var = cand
            break
    if temp_var is None:
        raise KeyError(f"Could not find temperature variable in dataset. Found: {list(ds.data_vars)}")

    geo_var = None
    for cand in ["z", "geopotential"]:
        if cand in ds.data_vars:
            geo_var = cand
            break
    if geo_var is None:
        raise KeyError(f"Could not find geopotential variable in dataset. Found: {list(ds.data_vars)}")

    ds = ds[[temp_var, geo_var]].rename({
        temp_var: "temperature",
        geo_var: "geopotential",
    })

    # Convert geopotential (m^2 / s^2) to elevation (m)
    g0 = 9.80665
    ds["elevation"] = ds["geopotential"] / g0

    # Keep only 6-hour timestamps: 00, 06, 12, 18
    time_hours = pd.to_datetime(ds.time.values).hour
    keep_mask = np.isin(time_hours, [0, 6, 12, 18])
    ds = ds.isel(time=np.where(keep_mask)[0]).sortby("time")

    return ds


def _compute_train_stats(ds_train: xr.Dataset, time_origin: pd.Timestamp) -> Dict[str, Any]:
    df = ds_train[["temperature", "elevation"]].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"])
    df["time_hours"] = ((df["time"] - time_origin) / pd.Timedelta(hours=1)).astype(float)

    x_cols = ["latitude", "longitude", "time_hours", "elevation"]
    y_col = "temperature"

    x_mean = df[x_cols].mean().to_numpy(dtype=np.float64)
    x_std = df[x_cols].std(ddof=0).replace(0, 1.0).to_numpy(dtype=np.float64)

    y_mean = float(df[y_col].mean())
    y_std = float(df[y_col].std(ddof=0))
    if y_std == 0.0:
        y_std = 1.0

    return {
        "x_cols": x_cols,
        "y_col": y_col,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


class MultiTaskERA5Dataset(Dataset):
    """
    ERA5 dataset for multi-task Bayesian ICL format.

    - num_in_context_datasets in {0,1,2}
    - each in-context dataset uses the FULL 300-point [10,10,3] patch
    - target dataset also uses the FULL 300-point patch
    - loss is only applied on target rows (role = ROLE_PREDICTION)
    - prior / in-context datasets are included as observed evidence only

    Sequence per sample:
        [PRIOR_TOKEN] + (x,y)*300      repeated num_in_context_datasets times
        [TARGET_TOKEN] + (x,y)*300     single target dataset

    Returns:
        if add_task_ids == False:
            x: [L, 4], y: [L, 1], role_ids: [L]
        else:
            x: [L, 4], y: [L, 1], seg_ids: [L], task_ids: [L], role_ids: [L]
    """

    PATCH_T = 3
    PATCH_LAT = 10
    PATCH_LON = 10
    PATCH_N = PATCH_T * PATCH_LAT * PATCH_LON  # 300

    def __init__(
        self,
        grib_path: str,
        num_samples: int,
        num_in_context_datasets: int,
        split: str = "train",  # train | val | test | all
        split_strategy: str = "ood",  # iid | ood
        add_task_ids: bool = False,
        seed: int = 0,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        stats_grib_path: Optional[str] = None
    ) -> None:
        super().__init__()
        if num_in_context_datasets not in (0, 1, 2):
            raise ValueError("num_in_context_datasets must be one of {0,1,2}.")
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be one of {'train', 'val', 'test', 'all'}.")
        if split_strategy not in ("iid", "ood"):
            raise ValueError("split_strategy must be one of {'iid', 'ood'}.")

        self.grib_path = grib_path
        self.num_samples = int(num_samples)
        self.num_in_context_datasets = int(num_in_context_datasets)
        self.split = split
        self.split_strategy = split_strategy
        self.add_task_ids = bool(add_task_ids)
        self.seed = int(seed)
        self.time_start = time_start
        self.time_end = time_end
        self.stats_grib_path = stats_grib_path or grib_path

        self.x_dim = 4  # latitude, longitude, time_hours, elevation

        # self.time_origin = pd.Timestamp("2019-01-01 00:00:00")
        self.time_origin_for_stats = pd.Timestamp("2019-01-01 00:00:00")
        self.time_origin_for_dataset = pd.Timestamp(self.time_start) if self.time_start is not None else self.time_origin_for_stats

        # Load ERA5 for sample extraction. Optional time filters affect only
        # sampled/evaluated patches, not the training-distribution stats.
        ds_full = _open_era5_grib_with_elevation(self.grib_path)
        ds = ds_full
        if self.time_start is not None or self.time_end is not None: # when eval on a future year
            ds = ds.sel(time=slice(self.time_start, self.time_end))

        if self.stats_grib_path == self.grib_path:
            ds_stats_source = ds_full
        else:
            ds_stats_source = _open_era5_grib_with_elevation(self.stats_grib_path)

        if self.split_strategy == "iid":
            split_datasets = {
                "train": ds,
                "val": ds,
                "test": ds,
                "all": ds,
            }
            ds_stats = ds_stats_source
          
        elif self.split_strategy == "ood":
            split_time = np.datetime64("2019-07-01T00:00:00")
            val_start_time = split_time - np.timedelta64(14, "D")
            last_train_time = val_start_time - np.timedelta64(1, "h")
            split_datasets = {
                "train": ds.sel(time=slice(None, last_train_time)),
                "val": ds.sel(time=slice(val_start_time, split_time - np.timedelta64(1, "h"))),
                "test": ds.sel(time=slice(split_time, None)),
                "all": ds,
            }
            ds_stats = ds_stats_source.sel(time=slice(None, last_train_time))
        else:
            raise ValueError(f"Unsupported split_strategy: {self.split_strategy}")
        
        self.ds = split_datasets[self.split]

        # Standardization stats from the training distribution for this split strategy.
        self.stats = _compute_train_stats(ds_stats, self.time_origin_for_stats)
        self.x_mean_arr = self.stats["x_mean"].astype(np.float32)
        self.x_std_arr = self.stats["x_std"].astype(np.float32)
        self.y_mean_scalar = np.float32(self.stats["y_mean"])
        self.y_std_scalar = np.float32(self.stats["y_std"])

        # Cache arrays used in __getitem__ to avoid repeated xarray/pandas conversion.
        self.temperature_arr = self.ds["temperature"].values.astype(np.float32)
        self.elevation_arr = self.ds["elevation"].values.astype(np.float32)
        self.latitude_arr = self.ds["latitude"].values.astype(np.float32)
        self.longitude_arr = self.ds["longitude"].values.astype(np.float32)
        time_values = pd.to_datetime(self.ds["time"].values)
        self.time_hours_arr = (
            (time_values - self.time_origin_for_dataset) / pd.Timedelta(hours=1)
        ).to_numpy(dtype=np.float32)

        # Precompute valid starts
        self.valid_lat_starts = np.arange(0, self.ds.sizes["latitude"] - self.PATCH_LAT + 1)
        self.valid_lon_starts = np.arange(0, self.ds.sizes["longitude"] - self.PATCH_LON + 1)
        self.valid_time_starts = np.arange(0, self.ds.sizes["time"] - self.PATCH_T + 1)

        if len(self.valid_lat_starts) == 0 or len(self.valid_lon_starts) == 0 or len(self.valid_time_starts) == 0:
            raise RuntimeError("Dataset is too small for [10,10,3] patch extraction.")

    def __len__(self) -> int:
        return self.num_samples

    def _get_patch(self, t0: int, lat0: int, lon0: int) -> xr.Dataset:
        return self.ds.isel(
            time=slice(t0, t0 + self.PATCH_T),
            latitude=slice(lat0, lat0 + self.PATCH_LAT),
            longitude=slice(lon0, lon0 + self.PATCH_LON),
        )

    def _extract_standardized_patch(self, t0: int, lat0: int, lon0: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast per-sample extraction from cached arrays to improve GPU utilization.
        """
        t_slice = slice(t0, t0 + self.PATCH_T)
        lat_slice = slice(lat0, lat0 + self.PATCH_LAT)
        lon_slice = slice(lon0, lon0 + self.PATCH_LON)

        temp_patch = self.temperature_arr[t_slice, lat_slice, lon_slice]
        if self.elevation_arr.ndim == 3:
            elev_patch = self.elevation_arr[t_slice, lat_slice, lon_slice]
        else:
            elev_2d = self.elevation_arr[lat_slice, lon_slice]
            elev_patch = np.broadcast_to(elev_2d[None, :, :], temp_patch.shape)

        lat_patch = np.broadcast_to(self.latitude_arr[lat_slice][None, :, None], temp_patch.shape)
        lon_patch = np.broadcast_to(self.longitude_arr[lon_slice][None, None, :], temp_patch.shape)
        time_patch = np.broadcast_to(self.time_hours_arr[t_slice][:, None, None], temp_patch.shape)

        x_patch = np.stack([lat_patch, lon_patch, time_patch, elev_patch], axis=-1).reshape(-1, self.x_dim)
        y_patch = temp_patch.reshape(-1, 1)

        x_patch = (x_patch - self.x_mean_arr) / self.x_std_arr
        y_patch = (y_patch - self.y_mean_scalar) / self.y_std_scalar
        return (
            torch.from_numpy(np.ascontiguousarray(x_patch)),
            torch.from_numpy(np.ascontiguousarray(y_patch)),
        )

    @staticmethod
    def _windows_overlap(a: int, b: int, window: int) -> bool:
        return not (a + window <= b or b + window <= a)

    def _choose_nonoverlapping_ic_times(self, rng: np.random.Generator, t_target: int) -> List[int]:
        """
        Choose num_in_context_datasets non-overlapping time windows from the SAME spatial patch,
        excluding overlap with the target window.
        """
        candidates = list(self.valid_time_starts.copy())
        rng.shuffle(candidates)

        chosen = []
        for t in candidates:
            if self._windows_overlap(t, t_target, self.PATCH_T):
                continue
            if any(self._windows_overlap(t, c, self.PATCH_T) for c in chosen):
                continue
            chosen.append(int(t))
            if len(chosen) == self.num_in_context_datasets:
                break

        if len(chosen) != self.num_in_context_datasets:
            raise RuntimeError("Could not find enough non-overlapping in-context windows.")
        return chosen

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + int(idx))

        # Sample one spatial patch
        lat0 = int(rng.choice(self.valid_lat_starts))
        lon0 = int(rng.choice(self.valid_lon_starts))

        # Sample one target time window
        t_target = int(rng.choice(self.valid_time_starts))

        # Sample IC time windows from same spatial patch, non-overlapping in time
        if self.num_in_context_datasets == 0:
            t_ic_list = []
        else:
            t_ic_list = self._choose_nonoverlapping_ic_times(rng, t_target)

        x_chunks = []
        y_chunks = []
        role_chunks = []

        seg_chunks = []
        task_chunks = []

        # In-context / prior datasets
        for task_idx, t_ic in enumerate(t_ic_list):
            X_ic, y_ic = self._extract_standardized_patch(t_ic, lat0, lon0)

            # PRIOR token
            x_chunks.append(torch.full((1, self.x_dim), PRIOR_TOKEN_X, dtype=torch.float32))
            y_chunks.append(torch.zeros((1, 1), dtype=torch.float32))
            role_chunks.append(torch.full((1,), ROLE_PRIOR, dtype=torch.long))

            if self.add_task_ids:
                seg_chunks.append(torch.full((1,), SEG_PRIOR, dtype=torch.long))
                task_chunks.append(torch.full((1,), task_idx, dtype=torch.long))

            # Full IC patch: observed but never predicted
            x_chunks.append(X_ic)
            y_chunks.append(y_ic)
            role_chunks.append(torch.full((self.PATCH_N,), ROLE_PRIOR, dtype=torch.long))

            if self.add_task_ids:
                seg_chunks.append(torch.full((self.PATCH_N,), SEG_PRIOR, dtype=torch.long))
                task_chunks.append(torch.full((self.PATCH_N,), task_idx, dtype=torch.long))

        # Target dataset
        X_tgt, y_tgt = self._extract_standardized_patch(t_target, lat0, lon0)

        # TARGET token
        x_chunks.append(torch.full((1, self.x_dim), TARGET_TOKEN_X, dtype=torch.float32))
        y_chunks.append(torch.zeros((1, 1), dtype=torch.float32))
        role_chunks.append(torch.full((1,), ROLE_EVIDENCE, dtype=torch.long))

        if self.add_task_ids:
            seg_chunks.append(torch.full((1,), SEG_TARGET, dtype=torch.long))
            task_chunks.append(torch.full((1,), self.num_in_context_datasets, dtype=torch.long))

        # Full target patch: predict every y
        x_chunks.append(X_tgt)
        y_chunks.append(y_tgt)
        role_chunks.append(torch.full((self.PATCH_N,), ROLE_PREDICTION, dtype=torch.long))

        if self.add_task_ids:
            seg_chunks.append(torch.full((self.PATCH_N,), SEG_TARGET, dtype=torch.long))
            task_chunks.append(torch.full((self.PATCH_N,), self.num_in_context_datasets, dtype=torch.long))

        x_out = torch.cat(x_chunks, dim=0)   # [L, 4]
        y_out = torch.cat(y_chunks, dim=0)   # [L, 1]
        role_ids = torch.cat(role_chunks, dim=0)  # [L]

        if self.add_task_ids:
            seg_ids = torch.cat(seg_chunks, dim=0)
            task_ids = torch.cat(task_chunks, dim=0)
            return x_out, y_out, seg_ids, task_ids, role_ids
        else:
            return x_out, y_out, role_ids


class MultiTaskERA5DataModule(pl.LightningDataModule):
    """
    Lightning DataModule for MultiTaskERA5Dataset.
    """

    def __init__(
        self,
        grib_path: str,
        num_in_context_datasets: int,
        num_train_samples: int,
        num_val_samples: int,
        num_test_samples: Optional[int] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        add_task_ids: bool = False,
        split_strategy: str = "ood",
        seed: int = 0,
        log_test_metrics_during_fit: bool = True,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        stats_grib_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[MultiTaskERA5Dataset] = None
        self.val_dataset: Optional[MultiTaskERA5Dataset] = None
        self.test_dataset: Optional[MultiTaskERA5Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = MultiTaskERA5Dataset(
                grib_path=self.hparams.grib_path,
                num_samples=self.hparams.num_train_samples,
                num_in_context_datasets=self.hparams.num_in_context_datasets,
                split="train",
                split_strategy=self.hparams.split_strategy,
                add_task_ids=self.hparams.add_task_ids,
                seed=self.hparams.seed,
                time_start=self.hparams.time_start,
                time_end=self.hparams.time_end,
                stats_grib_path=self.hparams.stats_grib_path,
            )
            self.val_dataset = MultiTaskERA5Dataset(
                grib_path=self.hparams.grib_path,
                num_samples=self.hparams.num_val_samples,
                num_in_context_datasets=self.hparams.num_in_context_datasets,
                split="val",
                split_strategy=self.hparams.split_strategy,
                add_task_ids=self.hparams.add_task_ids,
                seed=self.hparams.seed + 20_000,
                time_start=self.hparams.time_start,
                time_end=self.hparams.time_end,
                stats_grib_path=self.hparams.stats_grib_path,
            )
            if self.hparams.log_test_metrics_during_fit:
                num_test_samples = self.hparams.num_test_samples
                if num_test_samples is None:
                    num_test_samples = self.hparams.num_val_samples
                self.test_dataset = MultiTaskERA5Dataset(
                    grib_path=self.hparams.grib_path,
                    num_samples=num_test_samples,
                    num_in_context_datasets=self.hparams.num_in_context_datasets,
                    split="test",
                    split_strategy=self.hparams.split_strategy,
                    add_task_ids=self.hparams.add_task_ids,
                    seed=self.hparams.seed + 30_000,
                    time_start=self.hparams.time_start,
                    time_end=self.hparams.time_end,
                    stats_grib_path=self.hparams.stats_grib_path,
                )
        if stage in (None, "test"):
            num_test_samples = self.hparams.num_test_samples
            if num_test_samples is None:
                num_test_samples = self.hparams.num_val_samples
            self.test_dataset = MultiTaskERA5Dataset(
                grib_path=self.hparams.grib_path,
                num_samples=num_test_samples,
                num_in_context_datasets=self.hparams.num_in_context_datasets,
                split="test",
                split_strategy=self.hparams.split_strategy,
                add_task_ids=self.hparams.add_task_ids,
                seed=self.hparams.seed + 30_000,
                time_start=self.hparams.time_start,
                time_end=self.hparams.time_end,
                stats_grib_path=self.hparams.stats_grib_path,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )
        if not self.hparams.log_test_metrics_during_fit:
            return val_loader

        assert self.test_dataset is not None
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )
        return [val_loader, test_loader]

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )
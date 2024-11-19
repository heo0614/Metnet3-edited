# Metnet3_dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import torch.nn.functional as F
import pandas as pd  # 선형 보간을 위해 추가

class WeatherDataset(Dataset):
    def __init__(self, sparse_input_path, dense_input_path, low_input_path,
                 sparse_target_path, dense_target_path, high_target_path,
                 transform=None, mode='train'):
        self.sparse_input = xr.open_dataset(sparse_input_path)
        self.dense_input = xr.open_dataset(dense_input_path)
        self.low_input = xr.open_dataset(low_input_path)
        self.sparse_target = xr.open_dataset(sparse_target_path)
        self.dense_target = xr.open_dataset(dense_target_path)
        self.high_target = xr.open_dataset(high_target_path)

        self.transform = transform
        self.mode = mode

        # 시간에 따른 슬라이딩 윈도우 생성
        self.time_indices = self.create_time_indices()

    def create_time_indices(self):
        total_times = len(self.sparse_input.time)
        time_indices = []
        for t in range(total_times - 7):  # 6시간 입력, 1시간 타겟 (t+1)
            time_indices.append(t)
        return time_indices

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        t = self.time_indices[idx]

        # 입력 데이터 로드 (6시간씩)
        sparse_input = self.sparse_input.isel(time=slice(t, t+6)).to_array().values  # [variables, times, lat, lon]
        dense_input = self.dense_input.isel(time=slice(t, t+6)).to_array().values
        low_input = self.low_input.isel(time=slice(t, t+6)).to_array().values

        # 타겟 데이터 로드 (1시간 후의 첫 번째 시간 단계)
        sparse_target = self.sparse_target.isel(time=t+6).to_array().values  # [variables, lat, lon]
        dense_target = self.dense_target.isel(time=t+6).to_array().values
        high_target = self.high_target.isel(time=t+6).to_array().values

        # NaN 값 처리: 선형 보간을 사용하여 NaN 값을 채웁니다.
        # xarray의 interpolate_na를 사용하여 lat과 lon 차원에서 선형 보간
        sparse_input_ds = self.sparse_input.isel(time=slice(t, t+6)).interpolate_na(dim=['lat', 'lon'], method='linear', fill_value="extrapolate")
        dense_input_ds = self.dense_input.isel(time=slice(t, t+6)).interpolate_na(dim=['lat', 'lon'], method='linear', fill_value="extrapolate")
        low_input_ds = self.low_input.isel(time=slice(t, t+6)).interpolate_na(dim=['lat', 'lon'], method='linear', fill_value="extrapolate")

        sparse_input = sparse_input_ds.to_array().values  # [variables, times, lat, lon]
        dense_input = dense_input_ds.to_array().values
        low_input = low_input_ds.to_array().values

        # 타겟 데이터도 선형 보간 (안전하게 처리)
        sparse_target_ds = self.sparse_target.isel(time=t+6).interpolate_na(dim=['lat', 'lon'], method='linear', fill_value="extrapolate")
        dense_target_ds = self.dense_target.isel(time=t+6).interpolate_na(dim=['lat', 'lon'], method='linear', fill_value="extrapolate")
        high_target_ds = self.high_target.isel(time=t+6).interpolate_na(dim=['lat', 'lon'], method='linear', fill_value="extrapolate")

        sparse_target = sparse_target_ds.to_array().values  # [variables, lat, lon]
        dense_target = dense_target_ds.to_array().values
        high_target = high_target_ds.to_array().values

        # 채널 수를 6배로 늘려서 시간 축을 채널 축으로 변환 (6시간 * 기존 채널 수)
        # 예를 들어, 기존 채널 수가 6개라면, 6시간 데이터는 6*6=36채널로 변환
        # 현재 sparse_input은 [variables, times, lat, lon]
        # 시간 및 변수 차원을 채널 차원으로 병합
        # 새로운 채널 수 = variables * times
        sparse_input = np.transpose(sparse_input, (1, 0, 2, 3))  # [times, variables, lat, lon]
        sparse_input = sparse_input.reshape(-1, sparse_input.shape[2], sparse_input.shape[3])  # [channels=times*variables, lat, lon]

        dense_input = np.transpose(dense_input, (1, 0, 2, 3))
        dense_input = dense_input.reshape(-1, dense_input.shape[2], dense_input.shape[3])

        low_input = np.transpose(low_input, (1, 0, 2, 3))
        low_input = low_input.reshape(-1, low_input.shape[2], low_input.shape[3])

        # 마스크 생성: NaN이 아닌 위치는 1, NaN은 0
        mask_sparse_input = ~np.isnan(sparse_input)  # [channels, lat, lon]

        # 마스크 다운샘플링 (32x32로)
        mask_sparse_input_tensor = torch.from_numpy(mask_sparse_input).float()  # [channels, lat, lon]
        mask_sparse_input_down = F.interpolate(mask_sparse_input_tensor.unsqueeze(0), size=(32,32), mode='nearest').squeeze(0)  # [channels,32,32]

        # surface 변수만 추출 (예: 첫 3 변수 * 6시간 = 18채널)
        target_channels_sparse = 3
        input_variables_sparse = 6
        num_input_steps = 6
        # 각 변수별 6시간 채널을 분리
        mask_sparse_surface = mask_sparse_input_down[:target_channels_sparse * num_input_steps].view(target_channels_sparse, num_input_steps, 32, 32)  # [3,6,32,32]

        # 채널을 유지하면서 마스크를 사용할 수 있도록 변경
        # 여기서는 타겟이 1시간 데이터이므로, 전체 마스크를 사용하는 대신 마지막 시간 단계의 마스크를 사용
        # 또는 타겟 변수에 대한 마스크를 사용
        # 예를 들어, 마지막 시간 단계의 마스크를 사용할 수 있습니다.
        # 하지만 타겟은 1시간 후이므로, 마스크는 타겟 공간의 유효성을 나타내야 합니다.
        # 따라서, 타겟 변수에 대한 마스크를 별도로 생성합니다.
        # 여기서는 타겟이 surface 변수이므로, 마지막 시간 단계의 surface 변수 마스크를 사용합니다.

        # 마지막 시간 단계의 surface 변수 마스크 추출
        mask_sparse_surface_last = mask_sparse_surface[:, -1, :, :]  # [3,32,32]

        # 필요한 전처리 적용 (예: 리사이즈)
        if self.transform:
            sparse_input = self.transform(sparse_input)
            dense_input = self.transform(dense_input)
            low_input = self.transform(low_input)
            sparse_target = self.transform(sparse_target)
            dense_target = self.transform(dense_target)
            high_target = self.transform(high_target)
            mask_sparse_surface_last = self.transform(mask_sparse_surface_last.numpy())  # numpy로 변환 후 transform

        # 리드 타임 생성
        lead_times = torch.tensor([t], dtype=torch.long)

        if np.isnan(sparse_input).any():
            raise ValueError("NaN detected in sparse_input after interpolation")
        if np.isnan(dense_input).any():
            raise ValueError("NaN detected in dense_input after interpolation")
        if np.isnan(low_input).any():
            raise ValueError("NaN detected in low_input after interpolation")
        if np.isnan(sparse_target).any():
            raise ValueError("NaN detected in sparse_target after interpolation")
        if np.isnan(dense_target).any():
            raise ValueError("NaN detected in dense_target after interpolation")
        if np.isnan(high_target).any():
            raise ValueError("NaN detected in high_target after interpolation")

        mask_sparse_input = (mask_sparse_input > 0).astype(np.bool_)

        # 마스크 변환 및 경고 해결
        if isinstance(mask_sparse_surface_last, np.ndarray):
            # Numpy 배열인 경우
            mask_sparse_surface_tensor = torch.from_numpy(mask_sparse_surface_last).float()  # [3,32,32]
        elif isinstance(mask_sparse_surface_last, torch.Tensor):
            # 이미 텐서인 경우
            mask_sparse_surface_tensor = mask_sparse_surface_last.clone().detach()  # [3,32,32]
        else:
            raise TypeError("mask_sparse_surface_last의 타입이 지원되지 않습니다.")

        return {
            'sparse_input': torch.from_numpy(sparse_input).float(),  # [channels=variables*times, lat, lon]
            'dense_input': torch.from_numpy(dense_input).float(),
            'low_input': torch.from_numpy(low_input).float(),
            'sparse_target': torch.from_numpy(sparse_target).float(),
            'dense_target': torch.from_numpy(dense_target).float(),
            'high_target': torch.from_numpy(high_target).float(),
            'mask_sparse_input': mask_sparse_surface_tensor,  # [3,32,32]
            'lead_times': lead_times
        }

def get_dataloaders(batch_size=8):
    dataset = WeatherDataset(
        sparse_input_path='E:/metnet3/weather_bench/sparse_data(40)_input/156x156_sparse_0.5_input(40del)_all.nc',
        dense_input_path='E:/metnet3/weather_bench/dense_data_input/156x156_dense_0.5_input_all.nc',
        low_input_path='E:/metnet3/weather_bench/low_data_input/156x156_low_1.0_input_all.nc',
        sparse_target_path='E:/metnet3/weather_bench/sparse_data_target/32x32_sparse_target_0.5_all.nc',
        dense_target_path='E:/metnet3/weather_bench/dense_data_target/32x32_dense_target_0.5_all.nc',
        high_target_path='E:/metnet3/weather_bench/high_data_target/64x64_high_target_0.25_all.nc',
    )

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

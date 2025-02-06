import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# (1) Dataset 정의 (사용자 코드 예시 그대로)
# ============================================================
class WeatherBenchDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # ==========================
        # (1) 입력 (Input)
        # ==========================
        self.input_sparse = np.load(os.path.join(root_dir, 'input_sparse_normalized.npy'))  # (N,30,156,156)
        self.input_stale  = np.load(os.path.join(root_dir, 'input_stale_normalized.npy'))   # (N, 6,156,156)
        self.input_dense  = np.load(os.path.join(root_dir, 'input_dense_normalized.npy'))   # (N,36,156,156)
        self.input_low    = np.load(os.path.join(root_dir, 'input_low_normalized.npy'))     # (N,12,156,156)
        self.num_samples = self.input_sparse.shape[0]

        # ==========================
        # (2) 타겟 (Target)
        # ==========================
        self.t2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_temperature.npy'))             # (N,6,32,32)
        self.d2m = np.load(os.path.join(root_dir, 'sparse_target', '2m_dewpoint_temperature.npy'))    # (N,6,32,32)
        self.precip32 = np.load(os.path.join(root_dir, 'sparse_target', 'total_precipitation.npy'))   # (N,6,32,32)

        self.dense_target_36ch = np.load(os.path.join(root_dir, 'dense_target.npy'))  # (N,36,32,32)
        self.high_precip = np.load(os.path.join(root_dir, 'high_target', 'total_precipitation.npy'))  # (N,6,128,128)
        self.hrrr_36ch   = self.dense_target_36ch  # 예시상 동일하게 사용

        self.surface_bin_names = ('temperature_2m', 'dewpoint_2m')
        self.precipitation_bin_names = ('total_precipitation',)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ---- 입력 텐서 변환 ----
        in_sparse = torch.from_numpy(self.input_sparse[idx]).float()  # (30,156,156)
        in_stale  = torch.from_numpy(self.input_stale[idx]).float()   # ( 6,156,156)
        in_dense  = torch.from_numpy(self.input_dense[idx]).float()   # (36,156,156)
        in_low    = torch.from_numpy(self.input_low[idx]).float()     # (12,156,156)

        # ---- 타겟 텐서 변환 ---- (여기서는 이미 bin index라고 가정하여 long으로 변환)
        t2m_6h    = torch.from_numpy(self.t2m[idx]).long()      # (6,32,32)
        d2m_6h    = torch.from_numpy(self.d2m[idx]).long()      # (6,32,32)
        precip_6h = torch.from_numpy(self.precip32[idx]).long() # (6,32,32)

        dense_target_36ch = torch.from_numpy(self.dense_target_36ch[idx]).long()  # (36,32,32)
        high_precip_6h    = torch.from_numpy(self.high_precip[idx]).long()        # (6,128,128)
        hrrr_36ch         = torch.from_numpy(self.hrrr_36ch[idx]).float()         # (36,32,32)

        # ---- 임의 리드타임 예시 ----
        lead_time = torch.tensor(np.random.randint(0, 722), dtype=torch.long)

        sample = {
            "lead_time": lead_time,
            "input_sparse": in_sparse,  # (30,156,156)
            "input_stale":  in_stale,   # ( 6,156,156)
            "input_dense":  in_dense,   # (36,156,156)
            "input_low":    in_low,     # (12,156,156),
            "precipitation_targets": {
                "total_precipitation": high_precip_6h[-1]   # (128,128)
            },
            "surface_targets": {
                "temperature_2m": t2m_6h[-1],  # (32,32)
                "dewpoint_2m":    d2m_6h[-1],  # (32,32)
            },
            "hrrr_target": hrrr_36ch,   # (36,32,32)
        }
        return sample

# ============================================================
# (2) DataLoader
# ============================================================
train_root = r"/projects/aiid/KIPOT_SKT/Weather/trainset"
val_root   = r"/projects/aiid/KIPOT_SKT/Weather/validationset"
test_root  = r"/projects/aiid/KIPOT_SKT/Weather/testset"

train_dataset = WeatherBenchDataset(train_root)
val_dataset   = WeatherBenchDataset(val_root)
test_dataset  = WeatherBenchDataset(test_root)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=4)

# ============================================================
# (3) MetNet3 모델 준비
# ============================================================
from metnet3_original import MetNet3

metnet3 = MetNet3(
    dim = 512,
    num_lead_times = 180,
    lead_time_embed_dim = 32,
    input_spatial_size = 156,
    attn_depth = 12,
    attn_dim_head = 8,
    attn_heads = 32,
    attn_dropout = 0.1,
    vit_window_size = 8,
    vit_mbconv_expansion_rate = 4,
    vit_mbconv_shrinkage_rate = 0.25,
    # 채널 설정
    hrrr_channels = 36,       # (dense=36채널)
    input_2496_channels = 30, # (sparse=30채널)
    input_4996_channels = 12, # (low=12채널)
    surface_and_hrrr_target_spatial_size = 32,
    precipitation_target_bins = dict(
        total_precipitation = 512,  # 512개 bin
    ),
    surface_target_bins = dict(
        temperature_2m = 256,
        dewpoint_2m    = 256,
    ),
    hrrr_loss_weight = 10,
    hrrr_norm_strategy = 'sync_batchnorm',
    hrrr_norm_statistics = None,
    crop_size_post_16km = 32,
    resnet_block_depth = 2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metnet3.to(device)

# -----------------------------
# (3-1) 다중 GPU(DataParallel)
# -----------------------------
# 4개 GPU가 있다고 가정 시:
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs: DataParallel enabled.")
    metnet3 = torch.nn.DataParallel(metnet3)  # device_ids=[0,1,2,3]로 명시 가능

optimizer = torch.optim.Adam(metnet3.parameters(), lr=1e-4)

# ============================================================
# (4) CRPS 및 CSI 계산 함수
# ============================================================
def compute_crps(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CRPS(Continuous Ranked Probability Score) 계산
    prob:   (B, C, H, W) 형태. 각 픽셀별 bin별 확률
    target: (B, H, W) 실제 bin index (0..C-1)
    return: (scalar) 미니배치 전체 CRPS 평균
    """
    device = prob.device
    B, C, H, W = prob.shape

    cdf_pred = torch.cumsum(prob, dim=1)  # (B,C,H,W)

    # target[b,h,w] = t 이면, cdf_obs[b,k,h,w] = 1(k>=t), 0(k<t)
    bin_range = torch.arange(C, device=device).view(1, C, 1, 1)
    target_4d = target.unsqueeze(1)  # (B,1,H,W)
    cdf_obs = (bin_range >= target_4d).float()

    sq_diff = (cdf_pred - cdf_obs).pow(2).sum(dim=1)  # (B,H,W)
    crps_val = sq_diff.mean()

    return crps_val

def compute_csi(prob: torch.Tensor, target: torch.Tensor,
                threshold_bin: int, prob_cut: float = 0.5) -> float:
    """
    CSI(Critical Success Index) 계산
    prob: (B, C, H, W)
    target: (B, H, W)
    threshold_bin: 이 bin 이상이면 비가 온다 라고 판단(예: bin=5 ~ 1mm/h)
    prob_cut: 누적확률이 이 값 이상이면 비 예측
    """
    # bin>=threshold_bin인 확률
    rain_prob = prob[:, threshold_bin:, :, :].sum(dim=1)  # (B,H,W)

    # 예측(비/무비) 이진화
    pred_label = (rain_prob >= prob_cut)
    true_label = (target >= threshold_bin)

    TP = (pred_label & true_label).sum().item()
    FP = (pred_label & (~true_label)).sum().item()
    FN = ((~pred_label) & true_label).sum().item()

    denom = TP + FP + FN
    csi = (TP / denom) if denom > 0 else 0.0
    return csi

# ============================================================
# (5) Training Loop + Early Stopping(최근 10 epoch 기준)
# ============================================================
save_dir = "/projects/aiid/KIPOT_SKT/Weather/test_outputs"
os.makedirs(save_dir, exist_ok=True)

results_file = os.path.join(save_dir, "train_results.txt")

best_crps = float('inf')
best_csi  = 0.0
best_model_state = None

no_improve_count = 0
early_stopping_patience = 15  # 10 epoch 연속으로 개선 없으면 stop

max_epochs = 120  # 최대 Epoch은 예시

with open(results_file, "w") as f:
    f.write("epoch\ttrain_loss\tval_crps\tval_csi\n")

for epoch in range(1, max_epochs+1):
    metnet3.train()
    epoch_loss = 0.0

    # -------------------------
    # (A) Training Phase
    # -------------------------
    for batch in train_loader:
        # 1) 입력
        lead_times = batch['lead_time'].to(device)
        in_sparse  = batch['input_sparse'].to(device)
        in_stale   = batch['input_stale'].to(device)
        in_dense   = batch['input_dense'].to(device)
        in_low     = batch['input_low'].to(device)

        # 2) 타겟
        precip_targets = {k: v.to(device) for k, v in batch['precipitation_targets'].items()}
        surface_targets = {k: v.to(device) for k, v in batch['surface_targets'].items()}
        hrrr_target = batch['hrrr_target'].to(device)

        # 3) forward & loss
        optimizer.zero_grad()
        # DataParallel 사용 시, metnet3(...) 는 내부적으로 metnet3.module(...)를 호출
        total_loss, loss_breakdown = metnet3(
            lead_times            = lead_times,
            hrrr_input_2496       = in_dense,
            hrrr_stale_state      = in_stale,
            input_2496            = in_sparse,
            input_4996            = in_low,
            precipitation_targets = precip_targets,
            surface_targets       = surface_targets,
            hrrr_target           = hrrr_target,
        )
        total_loss = total_loss.sum()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)

    # -------------------------
    # (B) Validation Phase
    # -------------------------
    metnet3.eval()
    total_crps_val = 0.0
    total_csi_val  = 0.0
    count_batches  = 0

    threshold_bin = 5
    prob_cut      = 0.5

    with torch.no_grad():
        for batch in val_loader:
            lead_times = batch['lead_time'].to(device)
            in_sparse  = batch['input_sparse'].to(device)
            in_stale   = batch['input_stale'].to(device)
            in_dense   = batch['input_dense'].to(device)
            in_low     = batch['input_low'].to(device)

            pred = metnet3(
                lead_times       = lead_times,
                hrrr_input_2496  = in_dense,
                hrrr_stale_state = in_stale,
                input_2496       = in_sparse,
                input_4996       = in_low,
            )
            precipitation_preds = pred.precipitation

            # 배치별 CRPS/CSI
            for key, logits in precipitation_preds.items():
                probs = F.softmax(logits, dim=1)  # (B, C, H, W)
                target_bin = batch['precipitation_targets'][key].to(device)

                batch_crps = compute_crps(probs, target_bin)
                total_crps_val += batch_crps.item()

                batch_csi = compute_csi(probs, target_bin, threshold_bin, prob_cut)
                total_csi_val += batch_csi

            count_batches += 1

    val_crps = total_crps_val / count_batches if count_batches > 0 else 0.0
    val_csi  = total_csi_val  / count_batches if count_batches > 0 else 0.0

    # -------------------------
    # (C) 로그 출력 및 파일 기록
    # -------------------------
    print(f"[Epoch {epoch}] Training Loss = {avg_epoch_loss:.4f}, Breakdown = {loss_breakdown}")
    print(f"[Epoch {epoch}] Validation CRPS = {val_crps:.4f} | CSI(thr=1mm/h) = {val_csi:.4f}")

    # 파일에 기록 (append)
    with open(results_file, "a") as f:
        f.write(f"{epoch}\t{avg_epoch_loss:.4f}\t{val_crps:.4f}\t{val_csi:.4f}\n")

    # -------------------------
    # (D) Early Stopping 로직 (CRPS↓, CSI↑ 중 하나라도 개선되면 best 갱신)
    # -------------------------
    improved = False

    if val_crps < best_crps:
        best_crps = val_crps
        improved = True
    if val_csi > best_csi:
        best_csi = val_csi
        improved = True

    if improved:
        if isinstance(metnet3, torch.nn.DataParallel):
            best_model_state = copy.deepcopy(metnet3.module.state_dict())
        else:
            best_model_state = copy.deepcopy(metnet3.state_dict())
        no_improve_count = 0
        print(f"  --> Improvement found (CRPS={val_crps:.4f}, CSI={val_csi:.4f}). Best model saved.")
    else:
        no_improve_count += 1
        print(f"  --> No improvement for {no_improve_count} epoch(s).")

    if no_improve_count >= early_stopping_patience:
        print(f"Early stopping triggered after {no_improve_count} epochs without improvement.")
        break

# ============================================================
# (6) 최종 모델 저장
# ============================================================
if best_model_state is not None:
    if isinstance(metnet3, torch.nn.DataParallel):
        metnet3.module.load_state_dict(best_model_state)
    else:
        metnet3.load_state_dict(best_model_state)

save_path = os.path.join(save_dir, "metnet3_final.pth")
torch.save(best_model_state, save_path)
print(f"Final model (best CRPS/CSI) saved to {save_path}")

# 최종 성능 저장
with open(results_file, "a") as f:
    f.write("\n=== Final Best Performance ===\n")
    f.write(f"Best CRPS = {best_crps:.4f}\n")
    f.write(f"Best CSI  = {best_csi:.4f}\n")

print(f"Best CRPS = {best_crps:.4f}, Best CSI = {best_csi:.4f}")

# Metnet3_test.py

import torch
from torch import nn
from Metnet3_modified import MetNet3Modified
from Metnet3_dataloader import get_dataloaders
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MetNet3Modified(
        dim=128,
        num_times=6,
        input_variables_sparse=6,
        input_variables_dense=6,
        input_variables_low=2,
        target_channels_sparse=3,
        target_channels_dense=6,
        target_channels_high=1,
        resnet_block_depth=2,
        attn_depth=12,
        attn_dim_head=64,
        attn_heads=32,
        attn_dropout=0.1,
        vit_window_size=8,
        crop_size_post=32,
        upsample_scale_factor=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, _, test_loader = get_dataloaders()

    criterion = nn.MSELoss(reduction='mean')  # 모든 위치에 대해 평균 손실 계산
    test_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
            for batch in test_loader:
                sparse_input = batch['sparse_input'].to(device)  # [batch_size, 36, 32, 32]
                dense_input = batch['dense_input'].to(device)    # [batch_size, 36, 32, 32]
                low_input = batch['low_input'].to(device)        # [batch_size, 12, 32, 32]
                lead_times = batch['lead_times'].to(device)      # [batch_size, 1]

                surface_target = batch['sparse_target'].to(device)  # [batch_size, 3, 32, 32]
                dense_target = batch['dense_target'].to(device)      # [batch_size, 6, 32, 32]
                high_target = batch['high_target'].to(device)        # [batch_size, 1, 32, 32]

                # 모델 예측
                surface_pred, dense_pred, high_pred = model(
                    sparse_input, dense_input, low_input, lead_times
                )

                # 손실 계산 (모든 위치)
                loss_surface = criterion(surface_pred, surface_target)  # [batch_size, 3, 32, 32]
                loss_dense = criterion(dense_pred, dense_target).mean()
                loss_high = criterion(high_pred, high_target).mean()

                # 전체 손실
                loss = loss_surface + loss_dense + loss_high
                test_loss += loss.item()
                pbar.update(1)

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    # 시각화
    batch = next(iter(test_loader))
    sparse_input = batch['sparse_input'].to(device)
    dense_input = batch['dense_input'].to(device)
    low_input = batch['low_input'].to(device)
    lead_times = batch['lead_times'].to(device)

    surface_target = batch['sparse_target'].to(device)
    dense_target = batch['dense_target'].to(device)
    high_target = batch['high_target'].to(device)

    # 모델 예측
    surface_pred, dense_pred, high_pred = model(
        sparse_input, dense_input, low_input, lead_times
    )

    # 예시로 첫 번째 샘플을 시각화
    idx = 0
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Surface Target')
    plt.imshow(surface_target[idx].cpu().numpy()[0], cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Surface Prediction')
    plt.imshow(surface_pred[idx].cpu().numpy()[0], cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Absolute Error')
    plt.imshow(abs(surface_pred[idx].cpu().numpy()[0] - surface_target[idx].cpu().numpy()[0]), cmap='viridis')
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    test_model('metnet3_epoch10.pth')

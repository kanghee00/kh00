#test - 그래프 데이터 수치 분석 (70/15/15 분할, Test 평균) + Early Stopping + dropout
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os

def set_all_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 시드 설정
set_all_seeds(623)

# 1. DAE 모델 정의 (Dropout 추가)
class DenoisingAutoencoder1D(nn.Module):
    def __init__(self, input_dim=1000, dropout_rate=0.2):
        super(DenoisingAutoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout 추가
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(dropout_rate)   # Dropout 추가
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout 추가
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)) 


# 2. Time-Current 데이터 생성.

def exponential_transition(t, start_val, end_val, time_constant):
    """지수적 전환 함수"""
    return start_val + (end_val - start_val) * (1 - np.exp(-t / time_constant))

time = np.linspace(0, 10, 1000)
true_signal = np.zeros_like(time)

# 0.2초에서 99% 이상 도달하도록 τ 증가
tau = 0.043  # 더 큰 시간상수

for i in range(len(time)):
    t = time[i]
    cycle_time = t % 1.0
    
    if cycle_time < 0.5:  # ON 구간
        if cycle_time < 0.2:  # 200ms 동안 상승
            true_signal[i] = exponential_transition(cycle_time, 0.0, 10.0, tau)
        else:
            true_signal[i] = 10.0
    else:  # OFF 구간
        if cycle_time < 0.7:  # 200ms 동안 하강
            fall_time = cycle_time - 0.5
            true_signal[i] = exponential_transition(fall_time, 10.0, 0.0, tau)
        else:
            true_signal[i] = 0.0
            
# 기존 true_signal 생성 후...
true_signal_base = true_signal.copy()

# 각 샘플마다 다른 true signal 생성
num_samples = 300
X_noisy = []
y_clean = []

for sample in range(num_samples):
    # True signal 변동 추가
    amplitude_variation = np.random.normal(1.0, 0.05)  # 진폭 5% 변동
    
    # 변동이 적용된 true signal
    varied_true_signal = true_signal_base * amplitude_variation
    
    # 노이즈 추가
    noisy_signal = varied_true_signal + np.abs(np.random.normal(0.0, 100.0, size=time.shape))
    
    X_noisy.append(noisy_signal)
    y_clean.append(varied_true_signal)

X_noisy = np.array(X_noisy)
y_clean = np.array(y_clean)

X_tensor = torch.tensor(X_noisy, dtype=torch.float32)
y_tensor = torch.tensor(y_clean, dtype=torch.float32)

# 3. Train/Validation/Test 분할 (70/15/15)
print("="*60)
print("데이터 분할")
print("="*60)

# 1단계: Train+Validation vs Test 분할 (85% vs 15%)
X_temp, X_test, y_temp, y_test = train_test_split(X_tensor, y_tensor, test_size=0.15, random_state=0)

# 2단계: Train vs Validation 분할 (70% vs 15%)
# 0.176 = 15/(70+15) ≈ 15/85
#X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=0)
X_train, X_val = train_test_split(X_temp, test_size=0.176, random_state=0)

print(f"Train set: {X_train.shape[0]}개 샘플")
print(f"Validation set: {X_val.shape[0]}개 샘플") 
print(f"Test set: {X_test.shape[0]}개 샘플")
print(f"총 샘플: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}개")

# 4. 모델 학습 (Early Stopping 추가)
model = DenoisingAutoencoder1D(input_dim=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = nn.MSELoss()
criterion = nn.HuberLoss(delta=1.0)


# Early Stopping 설정
best_val_loss = float('inf')
patience = 300  # 300 에포크 동안 개선이 없으면 중단
patience_counter = 0
best_model_state = None
best_epoch = 0

train_loss_list = []
val_loss_list = []

print("="*60)
print("Early Stopping 설정")
print("="*60)
print(f"Patience: {patience} epochs")
print(f"최대 에포크: 3000")
print("="*60)

print(f"\n학습 시작...")
for epoch in range(3000):
    # Training step
    model.train()
    preds = model(X_train)
    loss = criterion(preds, X_train)
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    train_loss_list.append(loss.item())
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = criterion(val_preds, X_val)
        val_loss_list.append(val_loss.item())
    
    # Early Stopping 체크
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        best_epoch = epoch
        improvement_flag = "★ NEW BEST"
    else:
        patience_counter += 1
        improvement_flag = ""
    
    # 진행상황 출력
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} - Train: {loss.item():.8f} | Val: {val_loss.item():.8f} | Patience: {patience_counter:3d}/{patience} {improvement_flag}")
    
    # Early Stopping 조건 확인
    if patience_counter >= patience:
        print(f"\n{'='*60}")
        print(f"Early Stopping 발동!")
        print(f"최고 성능 에포크: {best_epoch}")
        print(f"최고 Validation Loss: {best_val_loss:.8f}")
        print(f"총 학습 에포크: {epoch + 1}")
        print(f"{'='*60}")
        
        # 최고 성능 모델 복원
        model.load_state_dict(best_model_state)
        break

# Early Stopping이 발동하지 않고 끝까지 학습한 경우
if patience_counter < patience:
    print(f"\n{'='*60}")
    print(f"최대 에포크 도달 (Early Stopping 발동 안함)")
    print(f"최고 성능 에포크: {best_epoch}")
    print(f"최고 Validation Loss: {best_val_loss:.8f}")
    print(f"{'='*60}")
    
    # 최고 성능 모델 복원
    model.load_state_dict(best_model_state)

# 5. 학습 완료 후 Test Set으로 최종 평가
print(f"\n학습 완료! Test Set으로 최종 평가...")
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)

print(f"="*60)
print("최종 성능 평가")
print("="*60)
print(f"Best Epoch: {best_epoch}")
print(f"Final Train Loss (at best epoch): {train_loss_list[best_epoch]:.8f}")
print(f"Final Validation Loss (best): {best_val_loss:.8f}")
print(f"Final Test Loss: {test_loss:.8f}")

# 6. Test Set의 평균 신호 계산
with torch.no_grad():
    test_outputs = model(X_test)  # 모든 Test 샘플 예측
    
    # Test 샘플의 평균 계산
    avg_test_noisy = torch.mean(X_test, dim=0).numpy()
    avg_test_clean = torch.mean(y_test, dim=0).numpy()
    avg_test_denoised = torch.mean(test_outputs, dim=0).numpy()

print("="*60)
print("1. LOSS 데이터 분석 (Early Stopping 적용)")
print("="*60)

# Loss 데이터 요약 통계
actual_epochs = len(train_loss_list)
print(f"실제 학습 에포크: {actual_epochs}")
print(f"최고 성능 에포크: {best_epoch}")
print(f"Early Stopping 절약 에포크: {3000 - actual_epochs}")

print(f"\nTraining Loss:")
print(f"  시작: {train_loss_list[0]:.8f}")
print(f"  최종: {train_loss_list[-1]:.8f}")
print(f"  최고 성능시: {train_loss_list[best_epoch]:.8f}")
print(f"  최소: {min(train_loss_list):.8f}")
print(f"  평균: {np.mean(train_loss_list):.8f}")

print(f"\nValidation Loss:")
print(f"  시작: {val_loss_list[0]:.8f}")
print(f"  최종: {val_loss_list[-1]:.8f}")
print(f"  최고 성능 (best): {best_val_loss:.8f}")
print(f"  최소: {min(val_loss_list):.8f}")
print(f"  평균: {np.mean(val_loss_list):.8f}")

print(f"\nTest Loss:")
print(f"  최종: {test_loss:.8f}")

print(f"\n과적합 지표 (최고 성능 기준):")
print(f"  Train vs Val 차이: {best_val_loss - train_loss_list[best_epoch]:.8f}")
print(f"  Val vs Test 차이: {test_loss - best_val_loss:.8f}")

# Loss 값 전체 출력 (처음 10개, 마지막 10개)
print(f"\nTraining Loss 처음 10개 에포크:")
for i in range(min(10, len(train_loss_list))):
    print(f"  Epoch {i}: {train_loss_list[i]:.4f}")

print(f"\n마지막 10개 에포크 Loss 비교:")
print("Epoch\tTrain Loss\tVal Loss\tGap")
start_idx = max(0, len(train_loss_list)-10)
for i in range(start_idx, len(train_loss_list)):
    gap = val_loss_list[i] - train_loss_list[i]
    marker = " ★" if i == best_epoch else ""
    print(f"  {i}\t{train_loss_list[i]:.8f}\t{val_loss_list[i]:.8f}\t{gap:.8f}{marker}")

print(f"\n최고 성능 에포크 주변:")
print("Epoch\tTrain Loss\tVal Loss\tGap")
for i in range(max(0, best_epoch-5), min(len(train_loss_list), best_epoch+6)):
    gap = val_loss_list[i] - train_loss_list[i]
    marker = " ★ BEST" if i == best_epoch else ""
    print(f"  {i}\t{train_loss_list[i]:.8f}\t{val_loss_list[i]:.8f}\t{gap:.8f}{marker}")

print("\n" + "="*60)
print("2. 신호 복원 데이터 분석 (Test Set 평균)")
print("="*60)

print(f"데이터 포인트 수: {len(time)}")
print(f"시간 범위: {time[0]:.1f}s ~ {time[-1]:.1f}s")
print(f"사용된 Test 샘플 수: {X_test.shape[0]}개")

print(f"\nTrue Signal 통계:")
print(f"  최소값: {avg_test_clean.min():.3f}")
print(f"  최대값: {avg_test_clean.max():.3f}")
print(f"  평균값: {avg_test_clean.mean():.3f}")

print(f"\nNoisy Input 통계:")
print(f"  최소값: {avg_test_noisy.min():.3f}")
print(f"  최대값: {avg_test_noisy.max():.3f}")
print(f"  평균값: {avg_test_noisy.mean():.3f}")
print(f"  표준편차: {avg_test_noisy.std():.3f}")

print(f"\nDenoised Output 통계:")
print(f"  최소값: {avg_test_denoised.min():.3f}")
print(f"  최대값: {avg_test_denoised.max():.3f}")
print(f"  평균값: {avg_test_denoised.mean():.3f}")
print(f"  표준편차: {avg_test_denoised.std():.3f}")

# 복원 성능 지표 (평균 신호 기준)
mse_noisy = np.mean((avg_test_noisy - avg_test_clean)**2)
mse_denoised = np.mean((avg_test_denoised - avg_test_clean)**2)
improvement = ((mse_noisy - mse_denoised) / mse_noisy) * 100

print(f"\n복원 성능:")
print(f"  노이즈 신호 MSE: {mse_noisy:.4f}")
print(f"  복원 신호 MSE: {mse_denoised:.4f}")
print(f"  개선율: {improvement:.1f}%")

# 특정 시간대 데이터 샘플
print(f"\n특정 시간대 데이터 샘플 (첫 20개 포인트):")
print("Time\tTrue\tNoisy\tDenoised")
for i in range(20):
    print(f"{time[i]:.2f}\t{avg_test_clean[i]:.3f}\t{avg_test_noisy[i]:.3f}\t{avg_test_denoised[i]:.3f}")

# DataFrame으로 저장
print(f"\n데이터프레임 생성...")
df_loss = pd.DataFrame({
    'Epoch': range(len(train_loss_list)),
    'Train_Loss': train_loss_list,
    'Val_Loss': val_loss_list
})

df_signals = pd.DataFrame({
    'Time': time,
    'True_Signal': avg_test_clean,
    'Noisy_Input': avg_test_noisy,
    'Denoised_Output': avg_test_denoised
})

print("Loss 데이터프레임 샘플:")
print(df_loss.head())
print("\n신호 데이터프레임 샘플:")
print(df_signals.head())

# Denoised output을 CSV로 저장
denoised_data = pd.DataFrame({
    'Time': time,
    'Denoised_Output': avg_test_denoised
})

# CSV 파일로 저장
denoised_data.to_csv('early_stop_denoised_output_test_avg.csv', index=False)
print("Early Stopping Denoised data가 'early_stop_denoised_output_test_avg.csv' 파일로 저장되었습니다.")

# 저장된 데이터 확인
print("\n저장된 데이터 샘플:")
print(denoised_data.head(10))

# 7. 그래프 출력
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Loss 그래프 (Early Stopping 마크 포함)
axs[0].plot(train_loss_list, label="Train Loss", color='blue', alpha=0.8)
axs[0].plot(val_loss_list, label="Validation Loss", color='orange', alpha=0.8)
axs[0].axhline(y=test_loss, color='red', linestyle='--', alpha=0.7, label=f"Test Loss: {test_loss:.6f}")

# 최고 성능 지점 표시
axs[0].axvline(x=best_epoch, color='green', linestyle=':', alpha=0.8, label=f"Best Epoch: {best_epoch}")
axs[0].scatter([best_epoch], [best_val_loss], color='green', s=100, marker='*', zorder=5, label=f"Best Val Loss: {best_val_loss:.6f}")

axs[0].set_title(f"Training vs. Validation Loss (Early Stopping at Epoch {len(train_loss_list)-1})")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("MSE Loss")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Denoising 결과 (Test[0] 샘플)
with torch.no_grad():
    single_test_output = model(X_test[0:1])

single_test_noisy = X_test[0].numpy()
single_test_clean = y_test[0].numpy()  
single_test_denoised = single_test_output[0].numpy()
axs[1].plot(time, single_test_noisy, label="Noisy Input", alpha=0.5)
axs[1].plot(time, single_test_denoised, label="Denoised Output (Early Stopped)", linewidth=2)
axs[1].set_title(f"DAE Prediction on Test Sample [0] (Best Epoch: {best_epoch})")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Current [a.u.]")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. Denoised Signal 상세 분석 그래프
fig2, axs2 = plt.subplots(2, 1, figsize=(12, 10))

# 상단: Denoised vs True Signal
axs2[0].plot(time, avg_test_clean, label="True Signal", linestyle='--', linewidth=3, color='black')
axs2[0].plot(time, avg_test_denoised, label=f"Denoised Output (Early Stopped)", linewidth=2, color='blue')
axs2[0].set_title(f"Denoised Signal vs True Signal (Best Epoch: {best_epoch})", fontsize=14)
axs2[0].set_xlabel("Time [s]")
axs2[0].set_ylabel("Current [a.u.]")
axs2[0].legend()
axs2[0].grid(True, alpha=0.3)

# 하단: 복원 오차
reconstruction_error = avg_test_denoised - avg_test_clean
axs2[1].plot(time, reconstruction_error, color='red', linewidth=2)
axs2[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axs2[1].set_title("Reconstruction Error (Denoised - True)", fontsize=14)
axs2[1].set_xlabel("Time [s]")
axs2[1].set_ylabel("Error [a.u.]")
axs2[1].grid(True, alpha=0.3)

# 오차 통계 표시
error_mean = np.mean(reconstruction_error)
error_std = np.std(reconstruction_error)
axs2[1].text(0.02, 0.98, f'평균 오차: {error_mean:.4f}\n표준편차: {error_std:.4f}\n최고 에포크: {best_epoch}', 
             transform=axs2[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Reconstruction error를 CSV로 저장
reconstruction_error_data = pd.DataFrame({
    'Time': time,
    'Reconstruction_Error': reconstruction_error
})

# CSV 파일로 저장
reconstruction_error_data.to_csv('early_stop_reconstruction_error_test_avg.csv', index=False)
print("Early Stopping Reconstruction error가 'early_stop_reconstruction_error_test_avg.csv' 파일로 저장되었습니다.")

# 저장된 데이터 확인
print("\n저장된 Reconstruction Error 샘플:")
print(reconstruction_error_data.head(10))

plt.tight_layout()
plt.show()

# 9. Early Stopping 요약 정보 출력
print("\n" + "="*60)
print("EARLY STOPPING 요약")
print("="*60)
print(f"설정된 Patience: {patience} epochs")
print(f"최고 성능 달성 에포크: {best_epoch}")
print(f"실제 학습 종료 에포크: {len(train_loss_list)-1}")
print(f"절약된 에포크 수: {3000 - len(train_loss_list)}")
print(f"최고 Validation Loss: {best_val_loss:.8f}")
print(f"최종 Test Loss: {test_loss:.8f}")
print(f"과적합 방지 효과: {'성공' if best_val_loss < 0.01 else '추가 튜닝 필요'}")
print("="*60)

# ================== 새로운 플롯 추가! ==================
print("\n" + "="*60)
print("단일 노이즈 입력 신호 분석")
print("="*60)

# 단일 샘플의 실제 노이즈 범위 분석
print(f"단일 샘플 노이즈 신호 통계:")
print(f"  최소값: {single_test_noisy.min():.3f}")
print(f"  최대값: {single_test_noisy.max():.3f}")
print(f"  평균값: {single_test_noisy.mean():.3f}")
print(f"  표준편차: {single_test_noisy.std():.3f}")

# 단일 샘플 성능
single_mse_noisy = np.mean((single_test_noisy - single_test_clean)**2)
single_mse_denoised = np.mean((single_test_denoised - single_test_clean)**2)
single_improvement = ((single_mse_noisy - single_mse_denoised) / single_mse_noisy) * 100

print(f"\n단일 샘플 성능:")
print(f"  노이즈 신호 MSE: {single_mse_noisy:.4f}")
print(f"  복원 신호 MSE: {single_mse_denoised:.4f}")
print(f"  개선율: {single_improvement:.1f}%")

print(f"\n평균 vs 단일 샘플 비교:")
print(f"  평균 샘플 개선율: {improvement:.1f}%")
print(f"  단일 샘플 개선율: {single_improvement:.1f}%")
print(f"  평균화 효과: {improvement - single_improvement:.1f}%p")

# 새로운 플롯: 단일 노이즈 입력 상세 분석
fig3, ax3 = plt.subplots(1, 1, figsize=(14, 8))

ax3.plot(time, single_test_noisy, label="Single Noisy Input", alpha=0.5, color='lightcoral', linewidth=1)
ax3.plot(time, single_test_denoised, label="Single Denoised Output", linewidth=2, color='blue')
ax3.plot(time, true_signal, label="True Signal", linestyle='--', linewidth=2, color='green')

ax3.set_title(f"Single Sample: Raw Noisy Input Analysis (improvement: {single_improvement:.1f}%)", fontsize=14)
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Current [a.u.]")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("단일 노이즈 입력 분석 완료!")
print("="*60)
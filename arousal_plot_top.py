import os
import pandas as pd
import numpy as np
from glob import glob
import re
import mne
import matplotlib.pyplot as plt
from collections import defaultdict

# ======== 配置路径和文件匹配规则 ========
root_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\subject_power_multi_band\beta_alpha_ratio'
pattern = re.compile(r'^G[2467]_R\d+_task_beta_alpha_ratio\.csv$')

# ======== 匹配 CSV 文件 ========
all_files = glob(os.path.join(root_dir, '*.csv'))
matched_files = [f for f in all_files if pattern.match(os.path.basename(f))]

if not matched_files:
    print("❌ 没有找到匹配的CSV文件")
    exit()

print(f"✅ 找到 {len(matched_files)} 个文件")

# ======== 初始化通道值累加器 ========
channel_sums = defaultdict(float)
channel_counts = defaultdict(int)

# ======== 遍历所有文件并汇总值 ========
for file_path in matched_files:
    df = pd.read_csv(file_path, index_col=0)

    for ch in df.columns:
        valid_vals = df[ch].dropna().values
        if len(valid_vals) > 0:
            channel_sums[ch] += valid_vals.sum()
            channel_counts[ch] += len(valid_vals)

# ======== 计算每个通道的平均值 ========
avg_values = []
valid_chs = []

for ch in channel_sums:
    avg = channel_sums[ch] / channel_counts[ch]
    avg_values.append(avg)
    valid_chs.append(ch)

avg_values = np.array(avg_values)

# ======== 取倒数（β/α → α/β） ========
avg_values = 1 / avg_values

# ======== 构建 MNE Info 并绘制 Topomap ========
info = mne.create_info(ch_names=valid_chs, sfreq=1000, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# 过滤掉没有位置信息的通道
valid_chs_final = [ch for ch in valid_chs if ch in montage.ch_names]
keep_indices = [valid_chs.index(ch) for ch in valid_chs_final]
avg_values_final = avg_values[keep_indices]

info_final = mne.create_info(ch_names=valid_chs_final, sfreq=1000, ch_types='eeg')
info_final.set_montage(montage)

# ======== 绘图美化版 ========
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

# 使用 mne.viz.plot_topomap 绘图，并设置 contours 更丰富，增强边界感知
im, cn = mne.viz.plot_topomap(
    avg_values_final,
    info_final,
    axes=ax,
    show=False,
    contours=7,
    cmap='Spectral',
    vlim=[1, 2.1],
    outlines='head',
    sensors=True,
    sphere=0.12 # ✅ 使用 float 类型，保持布局原样，只控制缩放
)



# 获取电极位置坐标（单位球面坐标）
pos = np.array([info_final.get_montage().get_positions()['ch_pos'][ch][:2] for ch in valid_chs_final])


cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.05)
cbar.set_label('α / β Ratio', fontsize=20)
cbar.ax.tick_params(labelsize=20)
# 设置标题及美化
# ax.set_title("Topographic Map of α / β Ratio (Groups G2, G4, G6, G8)", fontsize=14)
plt.tight_layout()
plt.show()


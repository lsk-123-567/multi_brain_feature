import os
import pandas as pd
import numpy as np
from glob import glob
import re
import mne
import matplotlib.pyplot as plt
from collections import defaultdict

# ======== 配置路径 ========
root_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\isc_topomap_C123_avg5round'
pattern = re.compile(r'^group[2467]+_task_C123_weights\.csv$')

# ======== 查找所有匹配的文件 ========
all_files = glob(os.path.join(root_dir, '*.csv'))
matched_files = [f for f in all_files if pattern.match(os.path.basename(f))]

if not matched_files:
    print("❌ 没有找到匹配的CSV文件")
    exit()

print(f"✅ 找到 {len(matched_files)} 个文件")

# ======== 累加每个通道的 C1 值 ========
channel_sums = defaultdict(float)
channel_counts = defaultdict(int)

for file_path in matched_files:
    df = pd.read_csv(file_path)
    if 'Channel' not in df.columns or 'C1' not in df.columns:
        print(f"⚠️ 文件 {os.path.basename(file_path)} 缺少必要列，跳过")
        continue

    for _, row in df.iterrows():
        ch = row['Channel']
        val = row['C1']
        if pd.notna(val):
            channel_sums[ch] += val
            channel_counts[ch] += 1

# ======== 计算每个通道的平均值 ========
ch_names = []
avg_values = []

for ch in channel_sums:
    mean_val = channel_sums[ch] / channel_counts[ch]
    ch_names.append(ch)
    avg_values.append(mean_val)

avg_values = np.array(avg_values)

# ======== 创建 MNE info 对象并设置位置 ========
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# ======== 过滤掉没有位置的通道 ========
valid_chs = [ch for ch in ch_names if ch in montage.ch_names]
keep_idx = [ch_names.index(ch) for ch in valid_chs]
avg_values_final = avg_values[keep_idx]

info_final = mne.create_info(ch_names=valid_chs , sfreq=1000, ch_types='eeg')
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
    vlim=[-1, 1.1],
    outlines='head',
    sensors=True,
    sphere=0.12 # ✅ 使用 float 类型，保持布局原样，只控制缩放
)



# 获取电极位置坐标（单位球面坐标）
pos = np.array([info_final.get_montage().get_positions()['ch_pos'][ch][:2] for ch in valid_chs ])

# 手动绘制电极名称在“下方”
# for i, ch in enumerate(valid_chs ):
#     ax.text(pos[i, 0], pos[i, 1], ch,
#             fontsize=20, ha='center', va='top',  # ← 设置在电极点下方
#             color='black')

# 添加 colorbar 并标注单位
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.05)
cbar.ax.tick_params(labelsize=20)
cbar.set_label('C1 Weight',fontsize=20)
# ax.set_title("Topomap of ISC Averaged Across 1,3,5,7sc Groups")
plt.tight_layout()
plt.show()

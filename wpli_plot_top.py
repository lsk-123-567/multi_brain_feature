import os
import pandas as pd
import numpy as np
from glob import glob
import re
import mne
import matplotlib.pyplot as plt
from collections import defaultdict

# ======== 参数配置 ========
root_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\inter_subject_wpli_pairs_multi_band\alpha'
pattern = re.compile(r'^G[2467]_R\d_task_alpha_Sub\d+-\d+\.csv$')

# ======== 查找文件 ========
all_files = glob(os.path.join(root_dir, '*.csv'))
matched_files = [f for f in all_files if pattern.match(os.path.basename(f))]

if not matched_files:
    print("❌ 没有找到匹配的文件")
    exit()

print(f"✅ 共找到 {len(matched_files)} 个文件：")
for f in matched_files:
    print(" -", os.path.basename(f))

# ======== 累加每个通道的平均值 ========
sum_values = defaultdict(float)
count_values = defaultdict(int)

for file in matched_files:
    df = pd.read_csv(file, index_col=0)
    for ch in df.columns:
        avg_col_val = df[ch].mean()
        if not np.isnan(avg_col_val):
            sum_values[ch] += avg_col_val
            count_values[ch] += 1

# ======== 构造平均值并保留标准电极位置通道 ========
montage = mne.channels.make_standard_montage('standard_1020')

avg_values_final = []
valid_chs_final = []

for ch in sum_values:
    if ch in montage.ch_names and count_values[ch] > 0:
        avg_val = sum_values[ch] / count_values[ch]
        avg_values_final.append(avg_val)
        valid_chs_final.append(ch)
    else:
        print(f"⚠️ 通道 {ch} 无标准位置或无有效数据，跳过")

avg_values_final = np.array(avg_values_final)

# ======== 创建 MNE Info 并设置蒙太奇 ========
info_final = mne.create_info(ch_names=valid_chs_final, sfreq=1000, ch_types='eeg')
info_final.set_montage(montage)

# ======== 美化绘图 ========
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

im, cn = mne.viz.plot_topomap(
    avg_values_final,
    info_final,
    axes=ax,
    show=False,
    contours=7,
    cmap='Spectral', # ✅ 或 'Spectral_r' 更平滑
    vlim=[0, 0.2],    # ← 你也可以设置为 auto 或根据 np.min/max 动态调整
    outlines='head',
    sensors=True,
    sphere=0.12
)

# ======== 绘制电极标签在下方 ========
pos = np.array([info_final.get_montage().get_positions()['ch_pos'][ch][:2] for ch in valid_chs_final])
# for i, ch in enumerate(valid_chs_final):
#     ax.text(pos[i, 0], pos[i, 1], ch,
#             fontsize=20, ha='center', va='top',
#             color='black')

# ======== 添加 colorbar ========
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.05)
cbar.set_label('WPLI (α)', fontsize=20)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.show()

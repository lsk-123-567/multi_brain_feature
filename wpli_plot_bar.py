import os
import pandas as pd
import numpy as np
from glob import glob
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# ======== 基本配置 ========
base_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\inter_subject_wpli_pairs_multi_band'
frequencies = ['alpha', 'beta', 'theta']
groups = [f'G{i}' for i in range(1, 9)]
legend_elements = [
    Patch(facecolor='lightblue', label='Low_polarization_groups'),
    Patch(facecolor='lightcoral', label='High_polarization_groups')
]
# ======== 初始化数据结构 ========
freq_means = {}
for freq in frequencies:
    root_dir = os.path.join(base_dir, freq)
    group_means = {}
    for group in groups:
        pattern = re.compile(rf'^{group}_R\d+_task_{freq}_Sub\d+-\d+\.csv$')
        all_files = glob(os.path.join(root_dir, '*.csv'))
        matched_files = [f for f in all_files if pattern.match(os.path.basename(f))]

        all_values = []
        for file_path in matched_files:
            df = pd.read_csv(file_path, index_col=0)
            values = df.values.flatten()
            values = values[~np.isnan(values)]
            all_values.extend(values)

        group_means[group] = np.mean(all_values) if all_values else np.nan




# ======== 可视化设置 ========
bar_width = 0.1
x_positions = {'alpha': 0, 'beta': 1, 'theta': 2}
group_offsets = np.linspace(-0.35, 0.35, len(groups))  # 让8组在每列内平铺

# ======== 开始绘图 ========
plt.figure(figsize=(10, 6))

for i, group in enumerate(groups):
    group_color = 'lightblue' if int(group[1:]) in [1, 3, 5, 7] else 'lightcoral'
    for freq in frequencies:
        value = freq_means[freq][group]
        if not np.isnan(value):
            x = x_positions[freq] + group_offsets[i]
            plt.bar(x, value, width=bar_width, color=group_color)
            # 添加柱子上方的数值标签
            plt.text(x, value + 0.005, f'{value:.2f}', ha='center', va='bottom', fontsize=12)


# ======== 美化设置 ========
plt.xticks(list(x_positions.values()), ['Alpha', 'Beta', 'Theta'], fontsize=12)
plt.ylabel('Mean wPLI', fontsize=16)
# plt.title('Group Connectivity Comparison by Frequency Band', fontsize=14)
# ======== 美化设置 ========
plt.legend(handles=legend_elements, fontsize=17, loc='upper center')
plt.xticks(list(x_positions.values()), ['Alpha', 'Beta', 'Theta'], fontsize=12)
plt.ylabel('Mean wPLI', fontsize=16)
plt.xticks(fontsize=14)  # ← 增大横轴刻度字体
plt.yticks(fontsize=14)
# plt.title('Group Connectivity Comparison by Frequency Band', fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.5)  ← 删除或注释掉这一行
plt.tight_layout()
plt.show()


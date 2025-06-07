import os
import pandas as pd
import numpy as np
from glob import glob
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# ======== 配置路径和分组 ========
root_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\subject_power_multi_band\beta_alpha_ratio'
groups = [f'G{i}' for i in range(1, 9)]

group_means = {}
# 定义图例元素
legend_elements = [
    Patch(facecolor='#ECC97F', label='Low_polarization_groups'),  # 奇数组颜色
    Patch(facecolor='#8FC9E2', label='High_polarization_groups')   # 偶数组颜色
]

# 添加图例到图中

# ======== 遍历每个 G 组 ========
for group in groups:
    pattern = re.compile(rf'^{group}_R\d+_task_beta_alpha_ratio\.csv$')
    all_files = glob(os.path.join(root_dir, '*.csv'))
    matched_files = [f for f in all_files if pattern.match(os.path.basename(f))]

    print(f"📁 {group}: 找到 {len(matched_files)} 个文件")

    all_values = []

    for file_path in matched_files:
        df = pd.read_csv(file_path, index_col=0)
        values = df.values.flatten()
        values = values[~np.isnan(values)]
        all_values.extend(values)

    if all_values:
        group_means[group] = np.mean(all_values)
    else:
        group_means[group] = np.nan  # 标记无数据



# ======== 取倒数（β/α → α/β） ========
for k in group_means:
    if not np.isnan(group_means[k]) and group_means[k] != 0:
        group_means[k] = 1.0 / group_means[k]

# ======== 设置颜色（奇数红，偶数蓝） ========
colors = ['#ECC97F' if int(g[-1]) % 2 == 1 else '#8FC9E2' for g in groups]


# ======== 绘制柱状图 ========
plt.figure(figsize=(10, 6))
bars = plt.bar(group_means.keys(), group_means.values(), color=colors)
plt.xticks(fontsize=14)  # ← 增大横轴刻度字体
plt.yticks(fontsize=14)
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=20)


plt.ylabel('arousal (α/β)', fontsize=16)
# plt.title('Global Mean α/β Ratio by Group (Task)')

plt.legend(handles=legend_elements, fontsize=14, loc='upper right')  # 你可以根据需要改位置

plt.tight_layout()
plt.show()




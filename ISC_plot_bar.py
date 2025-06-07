import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# ======== 配置路径 ========
root_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\isc_topomap_C123_avg5round'
legend_elements = [
    Patch(facecolor='#62B197', label='Low_polarization_groups'),  # 奇数组颜色
    Patch(facecolor='#E18E6D', label='High_polarization_groups')   # 偶数组颜色
]
# ======== 初始化结果字典 ========
group_means = {}

# ======== 明确处理 group1 到 group8 ========
for i in range(1, 9):
    group_name = f'group{i}'
    filename = f'{group_name}_task_C123_weights.csv'
    file_path = os.path.join(root_dir, filename)

    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {filename}，跳过")
        group_means[group_name] = np.nan
        continue

    df = pd.read_csv(file_path)

    if 'C1' not in df.columns or 'Channel' not in df.columns:
        print(f"⚠️ 文件格式错误: {filename}，跳过")
        group_means[group_name] = np.nan
        continue

    c1_vals = df['C1'].dropna().values
    if len(c1_vals) > 0:
        group_means[group_name] = np.mean(c1_vals)
    else:
        group_means[group_name] = np.nan


# ======== 准备绘图数据 ========
groups = list(group_means.keys())
values = [group_means[g] for g in groups]

# 奇偶颜色：奇数→淡红色，偶数→蓝色
colors = ['#62B197' if int(g[-1]) % 2 == 1 else '#E18E6D' for g in groups]



# ======== 绘制柱状图 ========
plt.figure(figsize=(10, 6))
bars = plt.bar(groups, values, color=colors)
# plt.xlabel('Group', fontsize=16)
plt.ylabel(' ISC Value ', fontsize=16)
plt.xticks(fontsize=14)  # ← 增大横轴刻度字体
plt.yticks(fontsize=14)
# plt.title('Average ISC Value per Group (Group1 ~ Group8)')
plt.legend(handles=legend_elements, fontsize=14, loc='upper left')
# 添加 0 参考线
plt.axhline(0, color='gray', linewidth=1, linestyle='--')

# 添加每根柱子的数值标签（写在柱子外面）
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        if height >= 0:
            y = height + 0.002
            va = 'bottom'
        else:
            y = height - 0.002
            va = 'top'
        plt.text(bar.get_x() + bar.get_width() / 2, y,
                 f'{height:.3f}', ha='center', va=va, fontsize=18)

plt.tight_layout()
plt.show()






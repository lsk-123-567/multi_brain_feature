import os
import glob
import pandas as pd

# === 配置目录 ===
save_dir = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain\isc_topomap_C123_by_round'

for group_id in range(1, 9):
    pattern = os.path.join(save_dir, f"group{group_id}_round*_task_C123_weights.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print(f"❌ Group{group_id} 没有找到 CSV 文件")
        continue

    # 用 dict 保存每个通道的每个成分的值
    comp_values = {}  # {channel: {'C1': [...], 'C2': [...], 'C3': [...]}}

    for fp in csv_files:
        try:
            df = pd.read_csv(fp, index_col=0)
            for ch in df.index:
                for c in ['C1', 'C2', 'C3']:
                    val = df.at[ch, c]
                    comp_values.setdefault(ch, {}).setdefault(c, []).append(val)
        except Exception as e:
            print(f"⚠️ 读取失败 {fp}: {e}")

    # 平均处理
    comp_avg = {}
    for ch, comps in comp_values.items():
        comp_avg[ch] = {c: sum(vals) / len(vals) for c, vals in comps.items()}

    # 构建 DataFrame
    df_avg = pd.DataFrame.from_dict(comp_avg, orient='index')
    df_avg.index.name = 'Channel'

    # 保存
    save_path = os.path.join(save_dir, f"group{group_id}_task_C123_avg_weights.csv")
    df_avg.to_csv(save_path, float_format='%.6f')
    print(f"✅ 已保存 Group{group_id} 平均结果: {save_path}")

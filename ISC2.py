import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
import pandas as pd

# ========== 用户配置 ==========
root = r'C:\Users\27328\Desktop\multi_brain'
save_dir = os.path.join(root, 'isc_topomap_C123_perRound')
os.makedirs(save_dir, exist_ok=True)

expected_chs = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
                'C3', 'C4', 'P3', 'P4', 'T3', 'T4',
                'T5', 'T6', 'O1', 'O2']
groups = 8
subs_per_group = 5
round_ids = [2, 3, 4, 5, 7]
data_keys = ['task']  # 可选 'rest'


def compute_cov(X, Y):
    return (X @ Y.T) / X.shape[1]


for group_id in range(1, 9):
    for data_key in data_keys:
        # ==== 获取所有通道交集 ====
        all_ch_sets = []
        for round_id in round_ids:
            for sub_id in range(1, subs_per_group + 1):
                fname = f'block{group_id}_sub{sub_id}_round_{round_id}_aligned.mat'
                fpath = os.path.join(root, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    raw_chs = sio.loadmat(fpath)['channel_names'].squeeze()
                    chs = [str(x).strip() for x in raw_chs]
                    all_ch_sets.append(set(chs))
                except:
                    continue
        common_chs = sorted(set.intersection(*all_ch_sets).intersection(expected_chs))
        if len(common_chs) < 5:
            print(f"⚠️ Group{group_id} {data_key}: 共同通道太少，跳过")
            continue

        print(f"✅ Group{group_id} {data_key} 共同通道: {common_chs}")

        for round_id in round_ids:
            round_data, min_len = [], np.inf
            for sub_id in range(1, subs_per_group + 1):
                fname = f'block{group_id}_sub{sub_id}_round_{round_id}_aligned.mat'
                fpath = os.path.join(root, fname)
                try:
                    m = sio.loadmat(fpath)
                    raw_chs = [str(ch).strip() for ch in m['channel_names'].squeeze()]
                    idx_map = {ch: i for i, ch in enumerate(raw_chs)}
                    selected = [idx_map[ch] for ch in common_chs if ch in idx_map]
                    if len(selected) != len(common_chs):
                        round_data = []
                        break
                    data = m[data_key][selected, :] * 1e6
                    min_len = min(min_len, data.shape[1])
                    round_data.append(data)
                except:
                    round_data = []
                    break
            if len(round_data) != subs_per_group:
                print(f"⚠️ Group{group_id} Round{round_id} 数据不全，跳过")
                continue

            round_data = np.array([d[:, :min_len] for d in round_data])
            eeg_centered = round_data - round_data.mean(axis=2, keepdims=True)

            R_w = np.zeros((len(common_chs), len(common_chs)))
            R_b = np.zeros_like(R_w)
            cnt = 0
            for i, Xi in enumerate(eeg_centered):
                R_w += compute_cov(Xi, Xi)
                for j, Xj in enumerate(eeg_centered):
                    if i != j:
                        R_b += 0.5 * (compute_cov(Xi, Xj) + compute_cov(Xj, Xi))
                        cnt += 1
            R_w /= len(eeg_centered)
            R_b /= cnt

            eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(R_w) @ R_b)
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

            X_all = []
            for sub_id in range(1, subs_per_group + 1):
                fname = f'block{group_id}_sub{sub_id}_round_{round_id}_aligned.mat'
                fpath = os.path.join(root, fname)
                try:
                    m = sio.loadmat(fpath)
                    raw_chs = [str(ch).strip() for ch in m['channel_names'].squeeze()]
                    idx_map = {ch: i for i, ch in enumerate(raw_chs)}
                    selected = [idx_map[ch] for ch in common_chs if ch in idx_map]
                    if len(selected) != len(common_chs): continue
                    data = m[data_key][selected, :] * 1e6
                    X_all.append(data)
                except:
                    continue
            if not X_all:
                print(f"❌ 没有有效 aligned 数据: Group{group_id} Round{round_id}")
                continue
            X_all = np.concatenate(X_all, axis=1)

            montage = mne.channels.make_standard_montage('standard_1020')
            info = mne.create_info(common_chs, sfreq=1000, ch_types='eeg')
            info.set_montage(montage)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            all_im, comp_weights = [], {}

            for i in range(3):
                v_i = eigvecs[:, i:i + 1]
                component_i = (v_i.T @ X_all).flatten()
                forward_model = np.array([
                    np.corrcoef(X_all[ch], component_i)[0, 1]
                    for ch in range(X_all.shape[0])
                ])
                comp_weights[f'C{i + 1}'] = forward_model
                fake_evoked = mne.EvokedArray(forward_model[:, None], info, tmin=0)
                im, _ = mne.viz.plot_topomap(forward_model, fake_evoked.info,
                                             axes=axes[i], cmap='RdBu_r',
                                             contours=6, show=False)
                axes[i].set_title(f'C{i + 1}')
                all_im.append(im)

            cbar = fig.colorbar(all_im[-1], ax=axes, orientation='vertical',
                                fraction=0.046, pad=0.08)
            cbar.set_label("ISC Weight")
            fig.suptitle(f"Group{group_id} – Round{round_id} – {data_key.upper()} – C1~C3", fontsize=14)
            plt.tight_layout(); plt.subplots_adjust(right=0.88)

            png_path = os.path.join(save_dir,
                f"group{group_id}_round{round_id}_{data_key}_C123_topomap.png")
            plt.savefig(png_path, dpi=300); plt.close()
            print(f"✅ 保存 PNG: {png_path}")

            df = pd.DataFrame(comp_weights, index=common_chs)
            csv_path = os.path.join(save_dir,
                f"group{group_id}_round{round_id}_{data_key}_C123_weights.csv")
            df.to_csv(csv_path, index_label='Channel', float_format='%.6f')
            print(f"✅ 保存 CSV: {csv_path}")

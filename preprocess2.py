import os
import numpy as np
import scipy.io as sio
import mne

# ========= 基本参数 =========
mat_folder = r'C:\Users\27328\Desktop\multi_brain\data'
sfreq = 1000                           # 采样率 (Hz)

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
            'C3', 'C4', 'T3', 'T4', 'P3', 'P4',
            'T5', 'T6', 'O1', 'O2']

# ========= 预处理函数 =========
def preprocess_eeg(data: np.ndarray,
                   ch_names: list[str],
                   sfreq: int) -> mne.io.Raw:
    """返回滤波 + 去坏段后的 Raw，不进行重参考或ICA"""

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    # --- 1-40 Hz 滤波 ---
    raw.filter(1, 40, fir_design='firwin', verbose=False)

    # --- 坏导：σ > 5×median ---
    chan_std = np.std(raw.get_data(), axis=1)
    bads = [raw.ch_names[i] for i, s in enumerate(chan_std)
            if s > 5 * np.median(chan_std)]
    raw.info['bads'] = bads

    # --- 2 s 伪-Epoch 剔坏段 ---
    step = int(2 * sfreq)
    if raw.n_times > step:
        events = np.column_stack((
            np.arange(0, raw.n_times - step + 1, step),
            np.zeros(raw.n_times // step, int),
            np.ones(raw.n_times // step, int)
        ))
        ep = mne.Epochs(raw, events, 1, 0, 2,
                        baseline=None,
                        reject=dict(eeg=150e-6),
                        preload=True, verbose=False)
        if len(ep):
            good = ep.get_data().reshape(-1, len(ch_names)).T
            raw = mne.io.RawArray(good, info, verbose=False)
            raw.info['bads'] = bads

    return raw

# ========= 主循环 =========
for fname in os.listdir(mat_folder):
    if not fname.endswith('.mat') or not fname.startswith('block2_sub5_round_'):
        continue
    print(f'\n🧠 处理: {fname}')

    m = sio.loadmat(os.path.join(mat_folder, fname))
    rest, task = m['rest'], m['task']

    # --- 裁剪 (秒 × sfreq) ---
    if rest.shape[1] < 90 * sfreq or task.shape[1] < 180 * sfreq:
        print('⚠️  数据不足，跳过'); continue
    rest_cut = rest[:, 30*sfreq : 90*sfreq]
    task_cut = task[:, 60*sfreq : 180*sfreq]

    # --- 预处理 ---
    rest_raw = preprocess_eeg(rest_cut, ch_names, sfreq)
    task_raw = preprocess_eeg(task_cut, ch_names, sfreq)

    # --- 汇总要删除的通道 ---
    to_drop = (set(rest_raw.info['bads']) |
               set(task_raw.info['bads']))
    to_drop &= set(ch_names)  # 确保存在

    # --- 最终删除 ---
    if to_drop:
        rest_raw.drop_channels(list(to_drop))
        task_raw.drop_channels(list(to_drop))
        print('🚮 删除通道:', sorted(to_drop))

    # --- 导出 ---
    rest_clean  = rest_raw.get_data()
    task_clean  = task_raw.get_data()
    clean_chans = rest_raw.ch_names               # 两段现在通道一致

    out = dict(rest=rest_clean,
               task=task_clean,
               fs=sfreq,
               channel_names=clean_chans)
    out_name = fname.replace('.mat', '_aligned.mat')
    sio.savemat(os.path.join(mat_folder, out_name), out)
    print(f'✅ 已保存 {out_name}   ({len(clean_chans)} 通道)')

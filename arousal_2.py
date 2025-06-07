#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算每个被试每轮每频段的功率（Power），并计算 β/α 比值，输出为 CSV（允许通道缺失）
"""

import os, itertools, warnings
import numpy as np
import scipy.io as sio
import pandas as pd
import mne

# ===== 用户配置 =====
root_dir = r'C:\Users\27328\Desktop\multi_brain'
save_dir_base = os.path.join(root_dir, 'subject_power_multi_band')
os.makedirs(save_dir_base, exist_ok=True)

expected_chs = ['Fp1','Fp2','F3','F4','F7','F8','C3','C4',
                'P3','P4','T3','T4','T5','T6','O1','O2']
data_keys = ['task']
groups = 8
subs_per_group = 5
round_ids = [2, 3, 4, 5, 7]
band_ranges = {
    'theta': (4, 7),
    'alpha': (8, 13),
    'beta': (14, 30)
}

# ===== 工具函数 =====
def _compute_power(data, sf, band):
    """滤波 + Hilbert，返回各通道平均功率"""
    info = mne.create_info(data.shape[0], sf, 'eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(*band, picks='eeg', verbose=False)
    raw.set_eeg_reference('average', verbose=False)

    ep = mne.EpochsArray(raw.get_data()[None, ...], info, tmin=0, verbose=False)
    ep.apply_hilbert(envelope=False, verbose=False)

    analytic = ep.get_data()[0]
    power = np.abs(analytic) ** 2
    return power.mean(axis=1)


from mne.time_frequency import tfr_array_morlet


def _compute_power_wavelet(data, sf, band, n_cycles=5):
    """
    使用 Morlet 小波变换计算功率。
    参数:
        data: np.ndarray, shape (n_channels, n_times)
        sf: 采样率
        band: (low, high)
        n_cycles: 每个频率下的小波周期数（决定时频分辨率）
    返回:
        mean_power: shape (n_channels,), 每个通道在 band 中的平均功率
    """
    freqs = np.linspace(band[0], band[1], num=5)  # 在频段内取多个频点
    data = data[np.newaxis, ...]  # → shape: (n_epochs=1, n_channels, n_times)

    power = tfr_array_morlet(
        data,
        sfreq=sf,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power',
        decim=1,
        verbose=False
    )  # shape: (1, n_channels, n_freqs, n_times)

    power_mean = power[0].mean(axis=(0, 2))  # 对 freq 和 time 取平均 → shape (n_channels,)
    return power_mean
def _load_mat(path, chs_keep, key):
    m = sio.loadmat(path)
    ch = [str(x).strip() for x in m['channel_names'].squeeze()]
    idx = {c: i for i, c in enumerate(ch)}

    sel = [idx[c] for c in chs_keep if c in idx]
    ch_sel = [c for c in chs_keep if c in idx]
    if len(sel) < 5:
        raise ValueError(f'有效通道不足: {ch_sel}')
    data = m[key][sel, :] * 1e6
    sf = float(m.get('sampling_rate', [[1000]])[0][0])
    return data, sf, ch_sel

# ========== 主循环 ==========
for key in data_keys:
    for g in range(5, 9):  # G5 到 G8，注意 range 的右边是开区间

        for rnd in round_ids:
            result_dict = {band: [] for band in band_ranges}
            index_list = []

            for sub in range(1, subs_per_group + 1):
                try:
                    path = os.path.join(root_dir, f'block{g}_sub{sub}_round_{rnd}_aligned.mat')
                    if not os.path.isfile(path): continue

                    data, sf, ch_used = _load_mat(path, expected_chs, key)

                    for band_name, bandpass in band_ranges.items():
                        power = _compute_power_wavelet(data, sf,  bandpass)

                        result_dict[band_name].append((ch_used, power))

                    index_list.append(f'G{g}_R{rnd}_Sub{sub}')

                except Exception as e:
                    warnings.warn(f'{path} 跳过: {e}')
                    continue

            # 保存每个频段的功率 CSV
            for band_name in band_ranges:
                powers = result_dict[band_name]
                if powers:
                    all_chs = sorted(set(itertools.chain.from_iterable([chs for chs, _ in powers])))
                    rows = []
                    for chs, power in powers:
                        row = {ch: val for ch, val in zip(chs, power)}
                        row_full = [row.get(ch, np.nan) for ch in all_chs]
                        rows.append(row_full)
                    df = pd.DataFrame(rows, index=index_list, columns=all_chs)
                    band_dir = os.path.join(save_dir_base, band_name)
                    os.makedirs(band_dir, exist_ok=True)
                    fname = f'G{g}_R{rnd}_{key}_{band_name}_power.csv'
                    df.to_csv(os.path.join(band_dir, fname), float_format='%.6f')
                    print(f'✅ 保存 {fname}')

            # ==== 计算并保存 β/α 比值 ====
            if result_dict['beta'] and result_dict['alpha']:
                beta_chs_all = [chs for chs, _ in result_dict['beta']]
                alpha_chs_all = [chs for chs, _ in result_dict['alpha']]
                all_chs = sorted(set(itertools.chain.from_iterable(beta_chs_all + alpha_chs_all)))

                beta_rows, alpha_rows = [], []
                for chs_b, pow_b in result_dict['beta']:
                    row_b = {ch: val for ch, val in zip(chs_b, pow_b)}
                    beta_rows.append([row_b.get(ch, np.nan) for ch in all_chs])
                for chs_a, pow_a in result_dict['alpha']:
                    row_a = {ch: val for ch, val in zip(chs_a, pow_a)}
                    alpha_rows.append([row_a.get(ch, np.nan) for ch in all_chs])

                beta_arr = np.array(beta_rows)
                alpha_arr = np.array(alpha_rows)

                ratio = beta_arr / (alpha_arr + 1e-6)
                df_ratio = pd.DataFrame(ratio, index=index_list, columns=all_chs)

                ratio_dir = os.path.join(save_dir_base, 'beta_alpha_ratio')
                os.makedirs(ratio_dir, exist_ok=True)
                fname_ratio = f'G{g}_R{rnd}_{key}_beta_alpha_ratio.csv'
                df_ratio.to_csv(os.path.join(ratio_dir, fname_ratio), float_format='%.6f')
                print(f'📊 保存比值 {fname_ratio}')

print("\n✅ Finish: 每个被试的多频段功率和比值文件已输出")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inter-subject wPLI — 每对被试单独输出多个频段（theta, alpha, beta）
"""

import os, itertools, warnings
import numpy as np
import scipy.io as sio
import pandas as pd
import mne
import matplotlib.pyplot as plt

# ===== 用户配置 =====
root_dir = r'C:\Users\27328\Desktop\multi_brain'
save_dir_base = os.path.join(root_dir, 'inter_subject_wpli_pairs_multi_band')
os.makedirs(save_dir_base, exist_ok=True)

expected_chs = ['Fp1','Fp2','F3','F4','F7','F8','C3','C4',
                'P3','P4','T3','T4','T5','T6','O1','O2']
data_keys = ['task']
groups = 1
subs_per_group = 5
round_ids = [2, 3, 4, 5, 7]
band_ranges = {
    'theta': (4, 7),
    'alpha': (8, 13),
    'beta': (14, 30)
}
plot_figures = False  # 可根据需要开启
# ====================

def _analytic(data, sf, bandpass):
    info = mne.create_info(data.shape[0], sf, 'eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(*bandpass, picks='eeg', verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    ep = mne.EpochsArray(raw.get_data()[None, ...], info, tmin=0, verbose=False)
    ep.apply_hilbert(envelope=False, n_fft="auto", verbose=False)
    return ep.get_data()[0]

def _wpli(Zi, Zj):
    t = min(Zi.shape[1], Zj.shape[1])
    Zi, Zj = Zi[:, :t], Zj[:, :t]
    csd = Zi[:, None, :] * Zj[None, :, :].conj()
    icsd = np.imag(csd)
    num = np.abs(np.mean(np.abs(icsd)*np.sign(icsd), axis=-1))
    denom = np.mean(np.abs(icsd), axis=-1)
    denom[denom == 0] = np.finfo(float).eps
    return num / denom

def _load_mat(path, chs_keep, key):
    m = sio.loadmat(path)
    ch = [str(x).strip() for x in m['channel_names'].squeeze()]
    idx = {c: i for i, c in enumerate(ch)}
    sel = [idx[c] for c in chs_keep if c in idx]
    if len(sel) != len(chs_keep):
        raise ValueError('缺通道')
    data = m[key][sel, :] * 1e6
    sf = float(m.get('sampling_rate', [[1000]])[0][0])

    # 添加重参考步骤
    info = mne.create_info(ch_names=ch, sfreq=sf, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    data = raw.get_data()
    return data, sf

# ================= 主循环 =================
for key in data_keys:
    for g in range(1, 9):
        # 1. 求共同通道
        ch_sets = []
        for rnd, sub in itertools.product(round_ids, range(1, subs_per_group + 1)):
            p = os.path.join(root_dir, f'block{g}_sub{sub}_round_{rnd}_aligned.mat')
            if os.path.isfile(p):
                try:
                    ch = [str(x).strip() for x in sio.loadmat(p)['channel_names'].squeeze()]
                    ch_sets.append(set(ch))
                except:
                    pass
        if not ch_sets:
            continue
        common_chs = sorted(set.intersection(*ch_sets).intersection(expected_chs))
        if len(common_chs) < 5:
            continue
        print(f'\nG{g}-{key} 通道: {common_chs}')

        for rnd in round_ids:
            paths = []
            for sub in range(1, subs_per_group + 1):
                fp = os.path.join(root_dir, f'block{g}_sub{sub}_round_{rnd}_aligned.mat')
                if os.path.isfile(fp):
                    paths.append((sub, fp))
            if len(paths) < 2:
                continue

            for band_name, bandpass in band_ranges.items():
                Z_dict = {}
                sf = None
                bad_sub = []
                for sub, fp in paths:
                    try:
                        dat, sf = _load_mat(fp, common_chs, key)
                        Z_dict[sub] = _analytic(dat, sf, bandpass)
                    except Exception as e:
                        warnings.warn(f'{os.path.basename(fp)} 跳过: {e}')
                        bad_sub.append(sub)
                for b in bad_sub:
                    paths = [p for p in paths if p[0] != b]
                if len(paths) < 2:
                    continue

                for (subA, fpA), (subB, fpB) in itertools.combinations(paths, 2):
                    Z_A, Z_B = Z_dict[subA], Z_dict[subB]
                    wpli = _wpli(Z_A, Z_B)

                    band_dir = os.path.join(save_dir_base, band_name)
                    os.makedirs(band_dir, exist_ok=True)
                    csv_name = f'G{g}_R{rnd}_{key}_{band_name}_Sub{subA}-{subB}.csv'
                    pd.DataFrame(wpli, index=common_chs, columns=common_chs).to_csv(
                        os.path.join(band_dir, csv_name), float_format='%.6f')

                    print(f'  保存 {band_name} - {csv_name}')

print("\n✅ Finish: 每对被试多频段文件已输出")

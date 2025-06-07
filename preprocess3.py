import os
import numpy as np
import scipy.io as sio
import mne

# ========= 参数配置 =========
mat_folder = r'D:\桌面\HybridHarmony-master\HybridHarmony-master\mutibrain'
sfreq = 1000
group_subs = [f'block7_sub{i}' for i in range(1, 6)]
round_ids = [2, 3, 4, 5, 7]
expected_chs = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
                'C3', 'C4', 'T3', 'T4', 'P3', 'P4',
                'T5', 'T6', 'O1', 'O2']

# ========= 预处理函数 =========
def preprocess_eeg_with_mask(data: np.ndarray, ch_names: list[str], sfreq: int):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(1, 40, fir_design='firwin', verbose=False)

    chan_std = np.std(raw.get_data(), axis=1)
    bads = [raw.ch_names[i] for i, s in enumerate(chan_std) if s > 5 * np.median(chan_std)]
    raw.info['bads'] = bads

    step = int(2 * sfreq)
    mask = np.ones(raw.n_times, dtype=bool)
    if raw.n_times > step:
        events = np.column_stack((
            np.arange(0, raw.n_times - step + 1, step),
            np.zeros(raw.n_times // step, int),
            np.ones(raw.n_times // step, int)
        ))
        epochs = mne.Epochs(raw, events, 1, 0, 2,
                            baseline=None,
                            reject=dict(eeg=150e-6),
                            preload=True, verbose=False)

        for i, log in enumerate(epochs.drop_log):
            if log:
                start = i * step
                mask[start:start + step] = False

    return raw, mask

# ========= 每轮次分别处理 =========
for round_id in round_ids:
    print(f'\n🌀 正在处理 round {round_id}...')
    task_datas, rest_datas = [], []
    ch_sets = []
    min_len_task, min_len_rest = None, None

    # ==== 计算 task/rest 最短长度 ====
    for sub in group_subs:
        fpath = os.path.join(mat_folder, f'{sub}_round_{round_id}.mat')
        if not os.path.isfile(fpath):
            continue
        m = sio.loadmat(fpath)
        task = m.get('task')
        rest = m.get('rest')

        if task is not None:
            if min_len_task is None or task.shape[1] < min_len_task:
                min_len_task = task.shape[1]
        if rest is not None:
            if min_len_rest is None or rest.shape[1] < min_len_rest:
                min_len_rest = rest.shape[1]

    if min_len_task is None or min_len_rest is None:
        print(f'⚠️ round {round_id} 缺少 task 或 rest 数据，跳过')
        continue

    # ==== 被试循环 ====
    for sub in group_subs:
        fname = f'{sub}_round_{round_id}.mat'
        fpath = os.path.join(mat_folder, fname)
        if not os.path.isfile(fpath):
            print(f'❌ 缺失文件: {fname}')
            continue

        m = sio.loadmat(fpath)
        task = m.get('task')
        rest = m.get('rest')

        if task is None or rest is None:
            print(f'⚠️ {fname} 缺少 task 或 rest')
            continue

        task = task[:, :min_len_task]
        rest = rest[:, :min_len_rest]

        raw_task, mask_task = preprocess_eeg_with_mask(task, expected_chs, sfreq)
        raw_rest, mask_rest = preprocess_eeg_with_mask(rest, expected_chs, sfreq)

        good_chs_task = [ch for ch in raw_task.ch_names if ch not in raw_task.info['bads']]
        good_chs_rest = [ch for ch in raw_rest.ch_names if ch not in raw_rest.info['bads']]
        common_chs = sorted(set(good_chs_task) & set(good_chs_rest))

        if len(common_chs) < 5:
            print(f'⚠️ {sub} 通道不足，跳过')
            continue

        raw_task.pick_channels(common_chs)
        raw_rest.pick_channels(common_chs)

        task_datas.append((sub, raw_task.get_data()[:, mask_task]))
        rest_datas.append((sub, raw_rest.get_data()[:, mask_rest]))
        ch_sets.append(set(common_chs))

    # ==== 跨被试统一通道 ====
    if not task_datas:
        print(f'⚠️ round {round_id} 无有效被试，跳过')
        continue

    common_chs = sorted(set.intersection(*ch_sets))
    print(f'✅ round {round_id} 共同通道: {common_chs} ({len(common_chs)}个)')

    # ==== 保存每个被试 ====
    for (sub, task_data), (_, rest_data) in zip(task_datas, rest_datas):
        out = dict(
            task=task_data,
            rest=rest_data,
            fs=sfreq,
            channel_names=common_chs
        )
        save_path = os.path.join(mat_folder, f'{sub}_round_{round_id}_aligned.mat')
        sio.savemat(save_path, out)
        print(f'💾 已保存: {save_path} | task={task_data.shape} | rest={rest_data.shape}')

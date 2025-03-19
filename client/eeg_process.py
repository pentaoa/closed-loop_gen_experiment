import os
import numpy as np
import mne
from sklearn.utils import shuffle
import scipy
from sklearn.discriminant_analysis import _cov


def preprocessing(raw_data): 
    ch_names = [
        'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4',
        'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7',
        'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3',
        'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
        'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL',
        'VEOU', 'VEOL'
    ]
    scalings = {'eeg': 10e1}

    original_channel_names = raw_data.ch_names
    if len(original_channel_names) != len(ch_names):
        print("原始数据中的通道数量与ch_names列表中的通道数量不匹配！")
    else:
        channel_rename_map = dict(zip(original_channel_names, ch_names))
        raw_data.rename_channels(channel_rename_map)

    non_eeg_channels = ['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
    raw_data.pick_channels([ch for ch in raw_data.ch_names if ch not in non_eeg_channels])


    raw_data.resample(250)
    # powerline_frequency = 50
    # raw_data.notch_filter(freqs=powerline_frequency, picks='eeg', notch_widths=1.0, trans_bandwidth=1.0, method='spectrum_fit', filter_length='auto')
    # raw_data.filter(1, 100)

    # picks_eeg = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False)
    # ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    # ica.fit(raw_data, picks=picks_eeg)
    
    # ica.exclude = [0]
    # ica.apply(raw_data)

    scalings = {'eeg': 1e6}
    return raw_data

def npy2raw(input_path):
    # 加载.npy文件
    data = np.load(input_path)

    # 这里假设你的数据是形状为(n_channels, n_times)的数组。
    # 你还需要为通道和时间定义一些参数。
    n_channels, n_times = data.shape
    print(f'数据的形状为: {n_channels} 通道 x {n_times} 时间点')
    # 为你的数据创建一个简单的info结构
    sfreq = 1000  # 采样频率, 根据你的数据修改
    ch_names = ['EEG %03d' % i for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # 使用数据和info创建Raw对象
    raw = mne.io.RawArray(data, info)
    return raw

def save_raw(original_data_path, preprocess_data_path):
    # 读取原始数据文件
    print(f'正在处理(1): {original_data_path}')
    raw_data = npy2raw(original_data_path)
    print(f'正在处理(2): {original_data_path}')
    raw_data = preprocessing(raw_data)

    # 从 Raw 对象中提取数据
    d, times = raw_data[:, :]
    
    # 保存数据为 .npy 格式
    os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)
    np.save(preprocess_data_path, d)


def create_event_based_npy(original_data_path, preprocess_data_path, output_data_dir):
    # 读取原始数据
    raw_data = np.load(original_data_path)
    events = raw_data[64, :]  # 第65行存储event信息

    # 读取预处理后的数据
    preprocessed_data = np.load(preprocess_data_path)
    
    # 找到所有非零的event索引
    event_indices = np.where(events > 0)[0]

    # 将原始数据的索引转换为降采样后的索引
    event_indices = event_indices // 4
    for idx, event_idx in enumerate(event_indices):
        if event_idx + 25 <= preprocessed_data.shape[1]:  # 确保索引不越界
            # 取出 event后 25 个时间点的数据，对应 0.1 秒
            event_data = preprocessed_data[:64, event_idx:event_idx + 25]
            
            # 保存每个事件数据为单独的.npy文件
            event_output_path = os.path.join(output_data_dir, f'{idx+1}.npy')
            os.makedirs(os.path.dirname(event_output_path), exist_ok=True)
            np.save(event_output_path, event_data)


if __name__ == '__main__':
    # 读取原始数据文件
    original_data_path = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\original\20250117-232225.npy'
    preprocess_data_path = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\preprocessed\20250117-232225.npy'
    output_data_dir = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\event_based'

    # 保存原始数据为 .npy 格式
    save_raw(original_data_path, preprocess_data_path)

    # 创建基于事件的数据
    create_event_based_npy(original_data_path, preprocess_data_path, output_data_dir)
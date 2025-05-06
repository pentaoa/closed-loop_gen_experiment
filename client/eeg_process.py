import os
import numpy as np
import mne
from sklearn.utils import shuffle
from scipy import signal
from sklearn.discriminant_analysis import _cov

# 预先设计滤波器和参数（只需计算一次）
def prepare_filters(fs=250, new_fs=250):
    """预先计算所有需要的滤波器和参数"""
    # 设计陷波滤波器（50Hz电源干扰）
    b_notch, a_notch = signal.iirnotch(50, 30, fs)
    
    # 设计带通滤波器（1-100Hz）
    b_bp, a_bp = signal.butter(4, [1, 100], btype='bandpass', fs=fs)
    
    # 计算重采样因子
    resample_factor = new_fs/fs
    
    return {
        'notch': (b_notch, a_notch),
        'bandpass': (b_bp, a_bp),
        'resample_factor': resample_factor
    }

# def npy2raw(input_path):
#     # 加载.npy文件
#     data = np.load(input_path)

#     # 你还需要为通道和时间定义一些参数。
#     n_channels, n_times = data.shape
#     print(f'数据的形状为: {n_channels} 通道 x {n_times} 时间点')
#     # 为你的数据创建一个简单的info结构
#     sfreq = 1000  # 采样频率, 根据你的数据修改
#     ch_names = ['EEG %03d' % i for i in range(n_channels)]
#     ch_types = ['eeg'] * n_channels
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

#     # 使用数据和info创建Raw对象
#     raw = mne.io.RawArray(data, info)
#     return raw

# def save_raw(original_data_path, preprocess_data_path):
#     # 读取原始数据文件
#     print(f'正在处理(1): {original_data_path}')
#     raw_data = npy2raw(original_data_path)
#     print(f'正在处理(2): {original_data_path}')
#     raw_data = preprocessing(raw_data)

#     # 从 Raw 对象中提取数据
#     d, times = raw_data[:, :]
    
#     # 保存数据为 .npy 格式
#     os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)
#     np.save(preprocess_data_path, d)


# def preprocessing(raw_data): 
#     ch_names = [
#         'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4',
#         'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7',
#         'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3',
#         'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
#         'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL',
#         'VEOU', 'VEOL'
#     ]
#     scalings = {'eeg': 10e1}

#     original_channel_names = raw_data.ch_names
#     if len(original_channel_names) != len(ch_names):
#         print("原始数据中的通道数量与ch_names列表中的通道数量不匹配！")
#     else:
#         channel_rename_map = dict(zip(original_channel_names, ch_names))
#         raw_data.rename_channels(channel_rename_map)

#     non_eeg_channels = ['ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL']
#     raw_data.pick_channels([ch for ch in raw_data.ch_names if ch not in non_eeg_channels])


#     raw_data.resample(250)
#     powerline_frequency = 50
#     raw_data.notch_filter(freqs=powerline_frequency, picks='eeg', notch_widths=1.0, trans_bandwidth=1.0, method='spectrum_fit', filter_length='auto')
#     raw_data.filter(1, 100)

#     picks_eeg = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False)
#     ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
#     ica.fit(raw_data, picks=picks_eeg)
    
#     ica.exclude = [0]
#     ica.apply(raw_data)

#     scalings = {'eeg': 1e6}
#     return raw_data

def real_time_process(original_data, filters, apply_baseline=True):
    """高效处理实时EEG数据，包括基线校正"""
    # 应用陷波滤波器
    filtered_data = signal.filtfilt(filters['notch'][0], filters['notch'][1], original_data, axis=1)
    
    # 应用带通滤波器
    filtered_data = signal.filtfilt(filters['bandpass'][0], filters['bandpass'][1], filtered_data, axis=1)
    
    # 重采样
    if filters['resample_factor'] != 1:
        new_length = int(filtered_data.shape[1] * filters['resample_factor'])
        filtered_data = signal.resample(filtered_data, new_length, axis=1)
    
    # 应用基线校正
    if apply_baseline:
        # 每张图片显示5秒，停顿1秒的设计
        # 假设数据结构是: 停顿1秒 -> 图片呈现5秒 -> 停顿1秒 -> ...
        # 采样率为250Hz，所以1秒对应250个数据点，5秒对应1250个数据点
        
        # 确定数据长度够不够一个完整的刺激周期
        if filtered_data.shape[1] >= 1500:  # 至少需要6秒数据(停顿1秒+刺激5秒)
            # 使用停顿期的最后250ms作为基线
            baseline_window = (-250, 0)  # 刺激前250ms作为基线
            stimulus_onset = 250  # 刺激在第250个点开始(第1秒开始)
            
            # 计算基线均值(使用刺激前的250ms数据)
            baseline_start = stimulus_onset + int(baseline_window[0])
            baseline_end = stimulus_onset + int(baseline_window[1])
            baseline_mean = np.mean(filtered_data[:, baseline_start:baseline_end], axis=1, keepdims=True)
            
            # 减去基线均值
            filtered_data = filtered_data - baseline_mean
    
    return filtered_data

def real_time_processing(original_data_path, preprocess_data_path, filters, apply_baseline=True):
    """高效处理实时EEG数据，包括基线校正"""
    data = np.load(original_data_path)
    
    # 应用陷波滤波器
    filtered_data = signal.filtfilt(filters['notch'][0], filters['notch'][1], data, axis=1)
    
    # 应用带通滤波器
    filtered_data = signal.filtfilt(filters['bandpass'][0], filters['bandpass'][1], filtered_data, axis=1)
    
    # 重采样
    if filters['resample_factor'] != 1:
        new_length = int(filtered_data.shape[1] * filters['resample_factor'])
        filtered_data = signal.resample(filtered_data, new_length, axis=1)
    
    # 应用基线校正
    if apply_baseline:
        # 每张图片显示5秒，停顿1秒的设计
        # 假设数据结构是: 停顿1秒 -> 图片呈现5秒 -> 停顿1秒 -> ...
        # 采样率为250Hz，所以1秒对应250个数据点，5秒对应1250个数据点
        
        # 确定数据长度够不够一个完整的刺激周期
        if filtered_data.shape[1] >= 1500:  # 至少需要6秒数据(停顿1秒+刺激5秒)
            # 使用停顿期的最后250ms作为基线
            baseline_window = (-250, 0)  # 刺激前250ms作为基线
            stimulus_onset = 250  # 刺激在第250个点开始(第1秒开始)
            
            # 计算基线均值(使用刺激前的250ms数据)
            baseline_start = stimulus_onset + int(baseline_window[0])
            baseline_end = stimulus_onset + int(baseline_window[1])
            baseline_mean = np.mean(filtered_data[:, baseline_start:baseline_end], axis=1, keepdims=True)
            
            # 减去基线均值
            filtered_data = filtered_data - baseline_mean
    
    os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)
    np.save(preprocess_data_path, filtered_data)
    return filtered_data

def create_last_event_npy(data, count=1, fs=250, event_length=5):
    """
    从数据中提取最后count个事件的数据
    适配5秒图片呈现+1秒空白的实验设计
    
    参数:
    data: 包含EEG数据和事件标记的numpy数组
    count: 返回最后几个事件的数据
    
    返回:
    last_events_data: 最后count个事件的数据数组列表
    """
    # 提取事件通道
    event = data[64, :]  # 第65行存储event信息
    event_indices = np.where(event > 0)[0]  # 找到所有非零的event索引
    
    # 如果要求的事件数大于实际事件数，调整count
    count = min(count, len(event_indices))
    
    # 只处理最后count个事件
    last_events = event_indices[-count:] if count > 0 else []
    
    print(f"找到 {len(event_indices)} 个事件，处理最后 {count} 个")
    
    # 储存处理后的事件数据
    event_data_list = []
    
    for idx, event_idx in enumerate(last_events):
        # 检查是否有足够的前导数据作为基线
        if event_idx < fs:
            print(f"警告: 事件 {len(event_indices) - count + idx + 1} 没有足够的前导数据用于基线校正")
            continue
            
        # 新的设计: 5秒图片 = 1250个样本点 (采样率250Hz)
        if event_idx + fs*event_length <= data.shape[1]:  # 确保索引不越界（需要5秒的刺激数据）
            # 提取基线期间(事件前250ms)和事件期间的数据
            baseline_start = event_idx - 250
            baseline_end = event_idx
            event_data = data[:64, event_idx:event_idx + 1250]  # 事件后5秒数据
            
            if apply_baseline:
                # 计算基线均值
                baseline_data = data[:64, baseline_start:baseline_end]
                baseline_mean = np.mean(baseline_data, axis=1, keepdims=True)
                
                # 应用基线校正
                corrected_event_data = event_data - baseline_mean
            else:
                corrected_event_data = event_data
            
            # 添加到返回列表
            event_data_list.append(corrected_event_data)
            print(f"处理了事件 {len(event_indices) - count + idx + 1}，数据形状: {corrected_event_data.shape}")
        else:
            print(f"警告: 事件 {len(event_indices) - count + idx + 1} 没有足够的后续数据")
    
    return event_data_list

def create_event_based_npy(original_data_path, preprocess_data_path, output_data_dir, apply_baseline=True):
    """创建基于事件的数据，并应用基线校正"""
    # 读取原始数据
    raw_data = np.load(original_data_path)
    events = raw_data[64, :]  # 第65行存储event信息

    # 读取预处理后的数据
    preprocessed_data = np.load(preprocess_data_path)
    
    # 找到所有非零的event索引
    event_indices = np.where(events > 0)[0]
    
    for idx, event_idx in enumerate(event_indices):
        # 检查是否有足够的前导数据作为基线
        if event_idx < 250:
            print(f"警告: 事件 {idx+1} 没有足够的前导数据用于基线校正")
            continue
            
        # 新的设计: 5秒图片 = 1250个样本点 (采样率250Hz)
        if event_idx + 1250 <= preprocessed_data.shape[1]:  # 确保索引不越界（需要5秒的刺激数据）
            # 提取基线期间(事件前250ms)和事件期间的数据
            baseline_start = event_idx - 250
            baseline_end = event_idx
            event_data = preprocessed_data[:64, event_idx:event_idx + 1250]  # 事件后5秒数据
            
            if apply_baseline:
                # 计算基线均值
                baseline_data = preprocessed_data[:64, baseline_start:baseline_end]
                baseline_mean = np.mean(baseline_data, axis=1, keepdims=True)
                
                # 应用基线校正
                corrected_event_data = event_data - baseline_mean
            else:
                corrected_event_data = event_data
            
            # 保存每个事件数据为单独的.npy文件
            event_output_path = os.path.join(output_data_dir, f'{idx+1}.npy')
            os.makedirs(os.path.dirname(event_output_path), exist_ok=True)
            np.save(event_output_path, corrected_event_data)
            print(f"保存了事件 {idx+1} 数据，形状: {corrected_event_data.shape}")
        else:
            print(f"警告: 事件 {idx+1} 没有足够的后续数据")


def baseline_correction(eeg_data, baseline_window=(-250, 0), stimulus_onset=250):
    """
    对EEG数据进行基线校正
    
    参数:
    eeg_data: EEG数据数组或文件路径
    baseline_window: 基线窗口范围(以毫秒为单位，相对于刺激呈现时间点)
    stimulus_onset: 在数据中刺激呈现的时间点索引
    
    返回:
    baseline_corrected_data: 基线校正后的EEG数据
    """
    # 加载EEG数据，如果传入的是路径
    if isinstance(eeg_data, str):
        eeg_data = np.load(eeg_data)
    
    # 采样率为250Hz时，1ms = 0.25点，直接计算点数
    baseline_start = stimulus_onset + baseline_window[0]
    baseline_end = stimulus_onset + baseline_window[1]
    
    # 确保索引在有效范围内
    baseline_start = max(0, baseline_start)
    baseline_end = min(eeg_data.shape[1], baseline_end)
    
    # 计算基线期间的平均值
    baseline_mean = np.mean(eeg_data[:, baseline_start:baseline_end], axis=1, keepdims=True)
    
    # 减去基线均值进行校正
    baseline_corrected_data = eeg_data - baseline_mean
    
    return baseline_corrected_data

# if __name__ == '__main__':
#     # 读取原始数据文件
#     original_data_path = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\original\20250117-232225.npy'
#     preprocess_data_path = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\preprocessed\20250117-232225.npy'
#     output_data_dir = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\event_based'

#     # 保存原始数据为 .npy 格式
#     save_raw(original_data_path, preprocess_data_path)

#     # 创建基于事件的数据
#     create_event_based_npy(original_data_path, preprocess_data_path, output_data_dir)
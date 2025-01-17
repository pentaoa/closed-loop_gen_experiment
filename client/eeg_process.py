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
    powerline_frequency = 50
    raw_data.notch_filter(freqs=powerline_frequency, picks='eeg', notch_widths=1.0, trans_bandwidth=1.0, method='spectrum_fit', filter_length='auto')
    raw_data.filter(1, 100)

    picks_eeg = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False)
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_data, picks=picks_eeg)
    
    ica.exclude = [0]
    ica.apply(raw_data)

    scalings = {'eeg': 1e6}
    return raw_data


# def mvnn(epoched_tests, mvnn_dim):
#     """
#     Apply Multivariate Noise Normalization (MVNN) to EEG test data.

#     Parameters
#     ----------
#     epoched_tests : array-like, shape (trials, categories, samples, channels, time points)
#         Array containing epoched test EEG data.
#     mvnn_dim : str
#         Dimension for MVNN ('time' or 'epochs').

#     Returns
#     -------
#     whitened_tests : array-like
#         Array containing whitened test EEG data.
#     """
#     # Convert epoched_tests to a NumPy array if it's not already
#     if isinstance(epoched_tests, list):
#         epoched_tests = np.array(epoched_tests)
        
#     # Initialize the array to store whitened data
#     whitened_tests = np.zeros_like(epoched_tests)

#     # Iterate over each trial
#     for trial in range(epoched_tests.shape[0]):
#         # Flatten the categories and samples dimensions for MVNN
#         epoched_trial = epoched_tests[trial].reshape(-1, epoched_tests.shape[3], epoched_tests.shape[4])

#         # Compute covariance matrix
#         if mvnn_dim == "time":
#             sigma_cond = np.mean([_cov(epoched_trial[:, :, t].T, shrinkage='auto') for t in range(epoched_trial.shape[2])], axis=0)
#         elif mvnn_dim == "epochs":
#             sigma_cond = np.mean([_cov(epoched_trial[e, :, :].reshape(epoched_trial.shape[1], -1).T, shrinkage='auto') for e in range(epoched_trial.shape[0])], axis=0)

#         # Compute the inverse square root of the covariance matrix
#         sigma_inv = scipy.linalg.fractional_matrix_power(sigma_cond, -0.5)

#         # Whiten the data for this trial
#         for e in range(epoched_trial.shape[0]):
#             for t in range(epoched_trial.shape[2]):
#                 epoched_trial[e, :, t] = np.dot(sigma_inv, epoched_trial[e, :, t])

#         # Reshape back to original dimensions and store the result
#         whitened_tests[trial] = epoched_trial.reshape(epoched_tests.shape[1], epoched_tests.shape[2], epoched_tests.shape[3], epoched_tests.shape[4])

#     # Correct the slicing to only affect the time points dimension
#     return whitened_tests[:, :, :, :, 50:300]


# def epoching(max_rep, seed, project_dir, sfreq):
#     """
#     This function applies preprocessing steps to the raw EEG data, 
#     including channel selection, resampling, filtering, ICA, and epoching.

#     Parameters
#     ----------
#     raw_data : instance of Raw
#         The raw data.
#     tmin : float
#         Start time before event.
#     tmax : float
#         End time after event.
#     event_id : dict
#         The event ID to consider.
#     baseline : tuple or list of length 2
#         The time interval to consider as baseline.
#     max_rep : int
#         Maximum number of repetitions per condition to include.
#     seed : int
#         Random seed for reproducibility.

#     Returns
#     -------
#     epoched_data : instance of Epochs
#         The epoched data.
#     img_conditions : list of int
#         Unique image conditions of the epoched data.
#     """

#     # Preprocessing steps from the provided preprocessing function
#     chan_order = [
#         'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4',
#         'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7',
#         'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3',
#         'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
#         'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL',
#         'VEOU', 'VEOL', 'Event'
#     ]
#     chan_order_63 =  ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
# 				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
# 				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
# 				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
# 				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
# 				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
# 				  'O1', 'Oz', 'O2']
#     epoched_data_passive = []
#     img_conditions_passive = []
#     # dir_ses = os.listdir(project_dir)
#     dir_ses = ['ses2']
#     for d in dir_ses:
#         files = os.listdir(os.path.join(project_dir, d))
#         for filename in files:
#             eeg_data = np.load(os.path.join(project_dir, d, filename), allow_pickle=True)            
#             # 创建mne的info对象，注意只包含EEG通道
#             info = mne.create_info([ch for ch in chan_order], sfreq, 'eeg')
#             raw = mne.io.RawArray(eeg_data, info)
#             del eeg_data

#             # Apply band-pass filter
#             raw.filter(0.1, 100)

#             min_duration = 5 / raw.info['sfreq']
#             events = mne.find_events(raw, stim_channel='Event', shortest_event=1, min_duration=min_duration)

        
#             # 选择EEG通道，排除非EEG通道
#             raw.pick_channels([ch for ch in chan_order], ordered=True)

#             epochs = mne.Epochs(raw, events, tmin=-.2, tmax=1.0, baseline=(None, 0), preload=True)
#             del raw

#             if sfreq < 1000:
#                 epochs.resample(sfreq)

#             ch_names = epochs.info['ch_names']
#             times = epochs.times

#             # 调整通道数据
#             adjusted_data = np.zeros((len(epochs), len(chan_order_63), len(epochs.times)))
#             for i, ch_name in enumerate(chan_order_63):
#                 if ch_name in epochs.ch_names:
#                     ch_idx = epochs.ch_names.index(ch_name)
#                     adjusted_data[:, i, :] = epochs.get_data()[:, ch_idx, :]

#             # 更新Epochs对象
#             mne_info = mne.create_info(chan_order_63, sfreq, 'eeg')
#             epochs = mne.EpochsArray(adjusted_data, mne_info, tmin=epochs.tmin, baseline=epochs.baseline)

#             # Initialize variables to track the passive imagination segments
#             passive_segment = False
#             passive_data = []
#             passive_events = []
#             # print("events", events)
#             for event in events:
#                 passive_events.append(event[2])
            
                    
#             if len(passive_events)!=200:
#                 continue
#             print("adjusted_data", adjusted_data.shape)
#             # 处理收集到的被动刺激事件            
#             # print("passive_events", passive_events)
#             unique_conditions = np.unique(passive_events)
#             sorted_indices = np.argsort(passive_events)  # 根据事件 ID 排序的索引
#             passive_events = np.array(passive_events)[sorted_indices]
#             # print("unique_conditions:", len(unique_conditions))
#             sorted_data = np.zeros((len(unique_conditions), max_rep, epochs.get_data().shape[1], epochs.get_data().shape[2]))
#             # print("sorted_data", sorted_data.shape)
#             # 处理每个文件并将结果存储为独立的元素
#             file_epoched_data = []  # 存储当前文件的所有epoch数据
#             for i, cond in enumerate(unique_conditions):
#                 idx = np.where(passive_events == cond)[0]
#                 if len(idx) > max_rep:
#                     idx = shuffle(idx, random_state=seed, n_samples=max_rep)
#                 # 每个条件的数据都作为单独的元素存储
#                 file_epoched_data.append(epochs.get_data()[idx])

#             # 将当前文件的所有epoch数据追加到epoched_data_passive
#             epoched_data_passive.append(file_epoched_data)
#             img_conditions_passive.append(unique_conditions)
        
#     return epoched_data_passive, img_conditions_passive, chan_order_63, times

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

# # 创建基于事件的数据，生成三维数组，第一维是事件，第二维是通道，第三维是时间点
# def create_event_based_npy(original_data_path, preprocess_data_path, output_data_path):
#     # 读取原始数据
#     raw_data = np.load(original_data_path)
#     events = raw_data[64, :]  # 第65行存储event信息

#     # 读取预处理后的数据
#     preprocessed_data = np.load(preprocess_data_path)
    
#     event_based_data = []
#     # 找到所有非零的event索引
#     event_indices = np.where(events > 0)[0]
#     print(event_indices)

#     # 将原始数据的索引转换为降采样后的索引
#     event_indices = event_indices // 4
#     for idx in event_indices:
#         if idx + 25 <= preprocessed_data.shape[1]:  # 确保索引不越界
#             # 取出event后250个时间点的数据，对应 1 秒
#             event_data = preprocessed_data[:64, idx:idx + 25]
#             event_based_data.append(event_data)
    
#     # 将事件数据转换为NumPy数组
#     event_based_data = np.array(event_based_data)
    
#     # 保存事件数据为新的.npy文件
#     os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
#     np.save(output_data_path, event_based_data)


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
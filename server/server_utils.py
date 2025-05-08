import os
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from mne.time_frequency import psd_array_multitaper

def get_binary_labels(labels):
    """
    把标签转换为二进制。
    """
    positive_emotions = ['Amu', 'Ins', 'Ten']
    labels = np.where(np.isin(labels, positive_emotions), 1, 0)
    return labels
    

def get_selected_channel_idxes(data, fs=250, n_selected_channels=3):
    """
    挑选出 PSD 特征之间相似度最小的几个通道。
    
    :param data: EEG 数据，形状为 (n_samples, n_channels, n_timepoints)
    :param fs: 采样频率，默认 250 Hz
    :return: 最显著（相似度最小）的三个通道索引
    """
    n_channels = data.shape[1]
    psds = []

    # 计算每个通道的平均 PSD
    for channel_idx in range(n_channels):
        channel_data = data[:, channel_idx, :]  # (n_samples, n_timepoints)
        psd_sum = 0
        for sample_idx in range(channel_data.shape[0]):
            psd, _ = psd_array_multitaper(channel_data[sample_idx], fs, adaptive=True, normalization='full', verbose=0)
            psd_sum += psd
        psds.append(psd_sum / channel_data.shape[0])  # 计算通道的平均 PSD

    psds = np.array(psds)  # 转为 NumPy 数组，形状为 (n_channels, n_frequencies)

    # 计算通道之间的相似度
    similarity_matrix = cosine_similarity(psds)
    np.fill_diagonal(similarity_matrix, np.nan)  # 将对角线（自相似）设置为 NaN

    # 计算每个通道与其他通道的平均相似度
    mean_similarity = np.nanmean(similarity_matrix, axis=1)

    # 选取平均相似度最小的几个通道
    selected_channel_idxes = np.argsort(mean_similarity)[:n_selected_channels].tolist()
    
    return selected_channel_idxes


def extract_emotion_psd_features(eeg_data, labels, fs=250, selected_channel_idxes=None):
    """
    提取 EEG 数据的 PSD 特征并对应情绪标签。
    
    :param eeg_data: EEG 数据，形状为 (n_samples, n_channels, n_timepoints)
    :param labels: 每个样本对应的标签（1: positive, 2: negative）
    :param fs: 采样率，默认 250Hz
    :param selected_channel_idxes: 指定的通道索引列表。如果为 None，则使用所有通道
    :return: features (n_samples, n_features), la   axbels (n_samples,)
    """
    features = []
    valid_labels = []
    print(f"========= Extracting features from {len(eeg_data)} samples =========")
    for i in range(len(eeg_data)):
        eeg_sample = eeg_data[i]  # 处理单个样本：(n_channels, n_timepoints)
        if selected_channel_idxes:
            eeg_sample = eeg_sample[selected_channel_idxes, :]

        # 计算功率谱密度
        psd, _ = psd_array_multitaper(eeg_sample, fs, adaptive=True, normalization='full', verbose=0)
        psd_flat = psd.flatten() # 将2D的PSD矩阵(通道×频率)展平为1D向量
        print(psd_flat.mean(), psd_flat.std()) # 打印均值和标准差
        features.append(psd_flat) # 添加到特征列表
        valid_labels.append(labels[i]) # 添加对应的标签

    features = np.array(features)
    valid_labels = np.array(valid_labels)
    return features, valid_labels
 


def train_emotion_classifier(features, labels, test_size=0.2, random_state=42):
    """分类器训练，使用网格搜索优化参数"""

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, 
                                                       random_state=random_state, stratify=labels)
    
    # 创建处理管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 特征标准化
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    # 参数网格
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # 网格搜索
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # 最佳模型
    best_model = grid_search.best_estimator_
    import joblib
    joblib.dump(best_model, 'best_emotion_model.pkl')

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.5
    y_pred_custom = (y_prob >= custom_threshold).astype(int)
    report = classification_report(y_test, y_pred_custom)
    
    print(f"最佳参数: {grid_search.best_params_}")
    
    return best_model, report, y_test, y_pred_custom

def train_svc(features, labels, test_size=0.2, random_state=42):
    """
    训练SVM分类器
    
    参数:
    - features: 提取的特征数据
    - labels: 对应的标签
    - test_size: 测试集比例
    - random_state: 随机数种子
    
    返回:
    - clf: 训练好的分类器
    - report: 分类报告
    - y_test: 测试集的真实标签
    - y_pred: 测试集的预测标签
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 特征标准化（重要！SVM对特征缩放非常敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 参数网格搜索
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        SVC(probability=True, random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"SVC最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数构建和训练模型
    clf = grid_search.best_estimator_
    
    # 在测试集上评估模型
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    return clf, report, y_test, y_pred, scaler

def train_gradient_boosting_classifier(features, labels, test_size=0.2, random_state=42):
    """
    训练梯度提升分类器
    
    参数:
    - features: 提取的特征数据
    - labels: 对应的标签
    - test_size: 测试集比例
    - random_state: 随机数种子
    
    返回:
    - clf: 训练好的分类器
    - report: 分类报告
    - y_test: 测试集的真实标签
    - y_pred: 测试集的预测标签
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.ensemble import GradientBoostingClassifier
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 参数网格搜索
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=random_state),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"GBC最佳参数: {grid_search.best_params_}")
    
    # 使用最佳参数构建和训练模型
    clf = grid_search.best_estimator_
    
    # 在测试集上评估模型
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return clf, report, y_test, y_pred

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

def create_n_event_npy(data, count=1, fs=250, event_length=5):
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

    apply_baseline = True  # 是否应用基线校正
    
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
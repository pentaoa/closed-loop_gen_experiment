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
    :return: features (n_samples, n_features), labels (n_samples,)
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
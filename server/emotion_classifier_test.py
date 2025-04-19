import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

from modulation_utils import *
from emotion_devide_test_app import collect_and_save_eeg_for_all_images, extract_emotion_psd_features

def test_emotion_classifier(clf, image_set_path, test_save_path, selected_channel_idxes, fs=250, n_test_images=20):
    """
    测试情感分类器的性能
    
    参数:
    - clf: 训练好的分类器
    - image_set_path: 图像集路径
    - test_save_path: 测试数据保存路径
    - selected_channel_idxes: 选择的EEG通道索引
    - fs: 采样率
    - n_test_images: 测试图像数量
    """
    print("\n" + "="*50)
    print("开始测试情感分类器")
    print("="*50)
    
    # 创建测试保存目录
    os.makedirs(test_save_path, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_set_path) 
                  if f.endswith('.jpg') or f.endswith('.png')]
    
    # 提取标签信息
    labels = [f.split('-')[0] for f in image_files]
    unique_labels = list(set(labels))
    print(f"发现的标签类别: {unique_labels}")
    
    # 为每个类别选择测试图像
    test_images = []
    test_true_labels = []
    
    for label in unique_labels:
        # 获取该标签的所有图像
        label_images = [f for f in image_files if f.split('-')[0] == label]
        # 随机选择图像
        selected = random.sample(label_images, min(n_test_images // len(unique_labels), len(label_images)))
        
        for img in selected:
            test_images.append(os.path.join(image_set_path, img))
            # 如果标签为Amu, Ins, Ten 则为正类，否则为负类
            if (label == 'Amu' or label == 'Ins' or label == 'Ten'): 
                test_true_labels.append(1)
            else:
                test_true_labels.append(0)
    
    # 随机打乱测试集
    combined = list(zip(test_images, test_true_labels))
    random.shuffle(combined)
    test_images, test_true_labels = zip(*combined)
    
    # 获取图像文件名(不含路径)用于category_list
    category_list = [os.path.basename(img).split('.')[0] for img in test_images]
    
    print(f"选择了 {len(test_images)} 张图像进行测试")
    print(f"测试标签分布: {np.bincount(test_true_labels)}")
    
    # 收集测试图像的EEG数据
    test_eeg_path = os.path.join(test_save_path, "test_eeg")
    os.makedirs(test_eeg_path, exist_ok=True)
    
    print("正在收集测试图像的EEG数据...")
    collect_and_save_eeg_for_all_images(test_images, test_eeg_path, category_list)
    
    # 加载收集的EEG数据
    print("加载收集的EEG数据...")
    eeg_files = [f for f in os.listdir(test_eeg_path) if f.endswith('.npy')]
    eeg_data = []
    
    for idx, label in enumerate(category_list):
        filename = f"{label}_{idx+1}.npy"
        file_path = os.path.join(test_eeg_path, filename)
        if os.path.exists(file_path):
            eeg_data.append(np.load(file_path))
        else:
            print(f"警告: 找不到文件 {filename}")
    
    eeg_data = np.array(eeg_data)
    
    # 提取特征
    print("提取EEG特征...")
    test_features = extract_emotion_psd_features(eeg_data, fs, selected_channel_idxes)
    
    # 预测标签
    print("预测情感标签...")
    predicted_labels = clf.predict(test_features)
    predicted_probs = clf.predict_proba(test_features)[:, 1]  # 正类的概率
    
    # 评估性能
    accuracy = accuracy_score(test_true_labels, predicted_labels)
    conf_matrix = confusion_matrix(test_true_labels, predicted_labels)
    class_report = classification_report(test_true_labels, predicted_labels)
    
    print("\n" + "="*50)
    print(f"分类器准确率: {accuracy:.4f}")
    print("="*50)
    print("\n混淆矩阵:")
    print(conf_matrix)
    print("\n分类报告:")
    print(class_report)
    
    # 可视化结果
    plt.figure(figsize=(12, 10))
    
    # 混淆矩阵
    plt.subplot(2, 1, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('情感分类混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 预测概率分布
    plt.subplot(2, 1, 2)
    for i, label in enumerate(['Negative', 'Positive']):
        idx = np.where(np.array(test_true_labels) == i)[0]
        plt.hist(predicted_probs[idx], alpha=0.7, label=f'真实{label}')
    
    plt.axvline(x=0.5, color='red', linestyle='--', label='决策边界')
    plt.title('情感分类概率分布')
    plt.xlabel('预测为Positive的概率')
    plt.ylabel('样本数')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_save_path, 'classifier_performance.png'))
    plt.close()
    
    # 保存测试结果
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'test_true_labels': test_true_labels,
        'predicted_labels': predicted_labels,
        'predicted_probs': predicted_probs
    }
    
    np.save(os.path.join(test_save_path, 'test_results.npy'), results)
    
    print(f"测试完成。结果已保存到 {test_save_path}")
    return results

# 使用示例
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    
    # 加载已训练的分类器
    try:
        with open('emotion_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        
        # 如果没有已有的分类器，可以创建一个新的来测试代码
        if clf is None:
            print("未找到已训练的分类器，创建一个简单模型用于测试...")
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
    except:
        print("未找到已训练的分类器，创建一个简单模型用于测试...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 设置参数
    image_set_path = 'stimuli_SX'  # 图像集路径
    test_save_path = 'test_results'  # 测试结果保存路径
    selected_channel_idxes = list(range(32))  # 示例：使用所有32个通道
    fs = 250  # 采样率
    
    # 运行测试
    test_emotion_classifier(clf, image_set_path, test_save_path, selected_channel_idxes, fs)
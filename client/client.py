import os
import time
import shutil
import numpy as np
import pygame as pg
import joblib
import random
import threading

from pygame_utils import Model, View, Controller
from client_utils import *

# 定义常量
subject_id = 3
fs = 250
experiment_stage = 1
# pre_eeg_path = os.path.join('client', 'pre_eeg')
pre_eeg_path = f'client/data/sub{subject_id}/pre_eeg'
instant_eeg_path = 'server/data/instant_eeg'
image_set_path = 'stimuli_SX'

# selected_channel_idxes = [28, 37, 23, 33]
clf = None

model = Model()
view = View()
controller = Controller(model, view)

def experiment_1():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    global clf
    print("\n" + "#" * 50)
    print("情感分类器训练")
    print("#" * 50 + "\n")
    controller.start_experiment_1(image_set_path, pre_eeg_path)

    # 首先检查并加载标签文件
    labels_path = os.path.join(pre_eeg_path, 'labels.npy')
    
    # 加载标签
    labels = np.load(labels_path)
    print(f"Loaded labels: {labels}")

    # 获取所有EEG数据文件（排除labels.npy）
    eeg_files = [f for f in sorted(os.listdir(pre_eeg_path)) 
                 if f.endswith('.npy') and f != 'labels.npy']
    
    if len(eeg_files) != len(labels):
        print(f"Warning: Number of EEG files ({len(eeg_files)}) doesn't match number of labels ({len(labels)})")
    
    # 转换标签
    labels = get_binary_labels(labels)
    
    # 加载EEG数据
    eeg_file_paths = [os.path.join(pre_eeg_path, f) for f in eeg_files]
    eeg_data = np.array([np.load(file) for file in eeg_file_paths])  # (n_samples, n_channels, n_timepoints)
    
    print(f"Loaded {len(eeg_data)} EEG samples with shape {eeg_data.shape}")

    # 获取选定的通道
    # selected_channel_idxes = get_selected_channel_idxes(eeg_data, fs, 4)
    # print("Selected channels:", selected_channel_idxes)
    
    # 提取特征并训练分类器
    features, valid_labels = extract_emotion_psd_features(eeg_data, labels, fs)
    print(f"提取的特征形状: {features.shape}")
    print(f"有效标签数量: {len(valid_labels)}")
    
    # pca = PCA(n_components=0.95)
    # features = pca.fit_transform(features)
    # print(f"使用PCA降维后的特征形状: {features.shape}")
    
    # joblib.dump(pca, 'server/pca_model.pkl')

    # 训练分类器和评估
    clf, report, y_test, y_pred = train_emotion_classifier(features, valid_labels, 0.2, 42)
    print("\n分类器报告:")
    print(report)  
    global experiment_stage
    experiment_stage = 2
    run_experiments_in_thread()

def experiment_2():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    global clf

    print("\n" + "#" * 50)
    print("实时情感分类测试")
    print("#" * 50 + "\n")
    
    # 检查分类器是否为 None，如果是则加载备用模型
    if clf is None:
        best_model_path = 'server/best_emotion_model.pkl'
        # pca = joblib.load('server/pca_model.pkl')
        if os.path.exists(best_model_path):
            clf = joblib.load(best_model_path)
            print(f"成功加载分类器: {best_model_path}")

    
    # 获取所有图片    
    all_image_files = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    # 按情绪类别分组图片
    amu_images = [f for f in all_image_files if f.startswith('Amu-')]
    dis_images = [f for f in all_image_files if f.startswith('Dis-')]
    test_images = amu_images + dis_images
    print(f"找到 {len(amu_images)} 张 Amu 图片")
    print(f"找到 {len(dis_images)} 张 Dis 图片")
    print(f"一共 {len(test_images)} 张图片")
    
    # 随机所有照片
    random.shuffle(test_images)
    
    frame = 0
    frame_results = []

    while True:
        print(f"Frame {frame}")
        
        if frame >= len(test_images):
            print("所有图片已处理，结束实验")
            break

        # 获取当前图片
        current_image = test_images[frame]
        img_path = os.path.join(image_set_path, current_image)
        
        # 记录标签（假设图片名称中包含情绪类别信息）
        true_label = 1 if current_image.startswith('Amu-') else 0
        current_label = [true_label]  # 创建单个元素的标签列表
        print(f"当前标签: {true_label}")
        
        # 加载第一个找到的EEG文件
        eeg_data = collect_one_event_eeg(img_path)
        eeg_data = np.expand_dims(eeg_data, axis=0)  # 扩展为 (1, 通道数, 时间点)
        print(f"EEG 数据已加载，形状为{eeg_data.shape}")
    
        # 提取特征
        features, valid_labels = extract_emotion_psd_features(eeg_data, current_label, fs)
        # features, valid_labels = extract_emotion_psd_features(eeg_data, current_label, fs, selected_channel_idxes)
        print(f"提取的特征形状: {features.shape}")
        print(f"有效标签数量: {len(valid_labels)}")
        
        # 使用分类器进行预测
        if features.shape[0] > 0:  # 确保提取到了特征
            # 检查分类器预期的特征维度
            expected_n_features = clf.n_features_in_
            actual_n_features = features.shape[1]
            
            if actual_n_features != expected_n_features:
                print(f"警告: 特征维度不匹配! 预期 {expected_n_features}，实际 {actual_n_features}")
                
                if actual_n_features < expected_n_features:
                    # 如果特征维度小于预期，填充零
                    padding = np.zeros((features.shape[0], expected_n_features - actual_n_features))
                    features = np.hstack((features, padding))
                    print(f"已填充特征至维度: {features.shape}")
                else:
                    # 如果特征维度大于预期，截断
                    features = features[:, :expected_n_features]
                    print(f"已截断特征至维度: {features.shape}")
            
            # # 使用PCA降维
            # features = pca.transform(features)
            # print(f"使用PCA降维后的特征形状: {features.shape}")
            
            proba = clf.predict_proba(features)[0]
            predicted_label = 1 if proba[1] > proba[0] else 0
            
            score = view.rating()
            
            # 记录当前帧结果
            frame_results.append({
                'frame': frame,
                'image': current_image,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'prob_dis': proba[0],
                'prob_amu': proba[1],
                'correct': (predicted_label == true_label),
                'score': score
            })
            
            print(f"预测概率: Dis={proba[0]:.2f}, Amu={proba[1]:.2f}")
            print(f"用户评分: {score}")
            print(f"预测标签: {'Amu' if predicted_label == 1 else 'Dis'}, 实际标签: {'Amu' if true_label == 1 else 'Dis'}")
            print(f"预测结果: {'✓ 正确' if predicted_label == true_label else '✗ 错误'}\n")
            
        else:
            print("警告: 未能提取特征")
        
        frame += 1
    
    # 给出测试结果
    print("测试结果:")
    
    # 计算总体准确率
    if frame_results:
        total_frames = len(frame_results)
        correct_frames = sum(1 for r in frame_results if r['correct'])
        accuracy = correct_frames / total_frames
        
        # 按情绪类别统计
        amu_frames = sum(1 for r in frame_results if r['true_label'] == 1)
        dis_frames = total_frames - amu_frames
        
        amu_correct = sum(1 for r in frame_results if r['true_label'] == 1 and r['correct'])
        dis_correct = sum(1 for r in frame_results if r['true_label'] == 0 and r['correct'])
        
        amu_accuracy = amu_correct / amu_frames if amu_frames > 0 else 0
        dis_accuracy = dis_correct / dis_frames if dis_frames > 0 else 0
        
        print(f"总样本数: {total_frames}")
        print(f"总体准确率: {accuracy:.2f} ({correct_frames}/{total_frames})")
        print(f"Amu情绪准确率: {amu_accuracy:.2f} ({amu_correct}/{amu_frames})")
        print(f"Dis情绪准确率: {dis_accuracy:.2f} ({dis_correct}/{dis_frames})")
    
    quit

def collect_and_save_eeg(image_paths, save_path, label_list):
    """
    在客户端收集并保存EEG数据
    
    参数:
    - image_paths: 图像文件路径列表
    - save_path: 保存EEG数据的目标路径
    - label_list: 与图像对应的标签列表
    """
    # 确保目录存在并清空
    if os.path.exists(instant_eeg_path):
        shutil.rmtree(instant_eeg_path)
    os.makedirs(instant_eeg_path, exist_ok=True)
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    print("开始显示图像并收集EEG数据")
    
    # 启动实验，展示图像并收集EEG数据
    controller.start_collection(image_paths, instant_eeg_path)
    
    # 等待EEG文件出现 (这里不需要等待，因为start_collection已经完成数据收集)
    files = [f for f in os.listdir(instant_eeg_path) if f.endswith('.npy')]
    if not files:
        print("警告：未找到EEG数据文件")
        return
        
    print(f"找到 {len(files)} 个EEG数据文件")
    print("Category number:", len(label_list))

    # 遍历 label_list，寻找对应的文件
    for idx, label in enumerate(label_list):
        filename = f"{idx+1}.npy"
        file_path = os.path.join(instant_eeg_path, filename)
        if os.path.exists(file_path):
            # 创建正确的目标文件名
            new_filename = f"{label}_{idx+1}.npy"
            # 构建完整的目标路径
            dest_file_path = os.path.join(save_path, new_filename)
            
            # 如果目标文件已存在，先删除
            if os.path.exists(dest_file_path):
                os.remove(dest_file_path)
                
            # 移动文件
            shutil.move(file_path, dest_file_path)
            print(f"移动文件到 {dest_file_path}")
        else:
            print(f"文件 {filename} 未在 {instant_eeg_path} 中找到")

    # 清理临时目录
    if os.path.exists(instant_eeg_path):
        shutil.rmtree(instant_eeg_path)
        os.makedirs(instant_eeg_path, exist_ok=True)
    
    print("EEG数据收集和保存完成")

def collect_one_event_eeg(image_path):
    data = controller.collect_data(image_path)
    return data

def run_experiments_in_thread():
    """Run the experiments in a separate thread"""
    if experiment_stage == 1:
        # Run experiment 1 in a separate thread
        exp_thread = threading.Thread(target=experiment_1)
        exp_thread.daemon = True  # Make thread exit when main program exits
        exp_thread.start()
        print("Experiment 1 started in a separate thread")
    
    elif experiment_stage == 2:
        # Run experiment 2 in a separate thread
        exp_thread = threading.Thread(target=experiment_2)
        exp_thread.daemon = True  # Make thread exit when main program exits
        exp_thread.start()
        print("Experiment 2 started in a separate thread")

if __name__ == '__main__':
    # 主线程保持 Pygame 事件循环
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
        
        pg.display.update()  # 更新显示
        time.sleep(0.01)  # 短暂休眠减少 CPU 使用
    
    # Give the controller time to initialize
    time.sleep(3)
    
    # Run the appropriate experiment
    run_experiments_in_thread()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user")

import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import os
import random
import shutil
import time

from modulation_utils import *

app = Flask(__name__)
socketio = SocketIO(app)

# 路径参数
image_set_path = 'stimuli_SX'
pre_eeg_path = 'server/pre_eeg' # TODO:修改！
instant_eeg_path = 'server/instant_eeg'

# 实验参数
num_loop_random = 1
subject_id = 1 
num_loops = 10
sub = 'sub-' + (str(subject_id) if subject_id >= 10 else format(subject_id, '02')) # 如果 subject_id 大于或等于 10，直接使用其值；如果小于 10，则将其格式化为两位数字（如 01, 02）。
fs = 250

# 全局变量
selected_channel_idxes = None
target_image_path = None
target_eeg_path = None
features = None
clf = None

@socketio.on('connect')
def handle_connect(auth):
    print('Client connected')
    print('Send: experiment_1_ready')
    socketio.emit('experiment_1_ready')

@app.route('/experiment_1_eeg_upload', methods=['POST'])
def pre_experiment():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    global features
    global clf
    if 'files' not in request.files:
        return jsonify({"message": "No file part"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No selected files"}), 400

    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        if file:
            filename = file.filename
            save_path = os.path.join(pre_eeg_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)

    # 首先检查并加载标签文件
    labels_path = os.path.join(pre_eeg_path, 'labels.npy')
    if not os.path.exists(labels_path):
        return jsonify({"message": "Labels file not found"}), 400
    
    # 加载标签
    labels = np.load(labels_path)
    print(f"Loaded labels: {labels}")
    
    # 获取所有EEG数据文件（排除labels.npy）
    eeg_files = [f for f in sorted(os.listdir(pre_eeg_path)) 
                 if f.endswith('.npy') and f != 'labels.npy']
    
    if len(eeg_files) != len(labels):
        print(f"Warning: Number of EEG files ({len(eeg_files)}) doesn't match number of labels ({len(labels)})")
    
    # 加载EEG数据
    eeg_file_paths = [os.path.join(pre_eeg_path, f) for f in eeg_files]
    eeg_data = np.array([np.load(file) for file in eeg_file_paths])  # (n_samples, n_channels, n_timepoints)
    
    print(f"Loaded {len(eeg_data)} EEG samples with shape {eeg_data.shape}")

    # 获取选定的通道
    selected_channel_idxes = get_selected_channel_idxes(eeg_data, fs)
    print("Selected channels:", selected_channel_idxes)
    
    # 提取特征并训练分类器
    features = extract_emotion_psd_features(eeg_data, fs, selected_channel_idxes)
    clf, report = train_emotion_classifier(features, labels, fs, selected_channel_idxes)
    
    # 输出特征重要性
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 显示前10个最重要的特征
    for i in range(min(10, len(indices))):
        print(f"Feature {indices[i]} importance: {importances[indices[i]]}")

    # 向客户端发送信号，表示已准备好进行下一阶段的实验
    print('Send: experiment_2_ready')
    socketio.emit('experiment_2_ready') 

    return jsonify({
        "message": f"Files uploaded and processed successfully"
    }), 200
        

@app.route('/experiment_2', methods=['POST'])
def experiment():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    global features
    global clf

    print("\n" + "#" * 50)
    print("🚀 开始情感分类器测试 🚀")
    print("#" * 50 + "\n")

    time.sleep(1)
    
    # 创建测试结果保存目录
    test_save_path = f'/mnt/dataset0/xkp/closed-loop/exp_sub{subject_id}/emotion_test'
    os.makedirs(test_save_path, exist_ok=True)
    
    # 导入测试模块
    from emotion_classifier_test import test_emotion_classifier
    
    # 运行测试
    results = test_emotion_classifier(
        clf=clf,
        image_set_path=image_set_path,
        test_save_path=test_save_path,
        selected_channel_idxes=selected_channel_idxes,
        fs=fs,
        n_test_images=20  # 调整测试图像数量
    )
    
    # 保存训练好的分类器，便于以后使用
    import pickle
    with open(os.path.join(test_save_path, 'emotion_classifier.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    
    # 分析测试结果
    accuracy = results['accuracy']
    
    print(f"分类器准确率: {accuracy:.4f}")
    
    if accuracy >= 0.7:
        message = "分类器性能良好，可以用于情感调节实验"
    else:
        message = "分类器性能不佳，建议重新训练"
    
    # 向客户端发送测试完成信号
    print('Send: test_finished')
    socketio.emit('test_finished', {'message': message, 'accuracy': float(accuracy)})
    
    return jsonify({
        "message": "情感分类器测试完成",
        "accuracy": float(accuracy)
    }), 200

@app.route('/experiment_3', methods=['POST'])
def experiment_3():
    global selected_channel_idxes
    global features
    global clf
    
    print("\n" + "#" * 50)
    print("🚀 开始情感调节实验 🚀")
    print("#" * 50 + "\n")
    
    time.sleep(1)
    
    # 创建实验结果保存目录
    exp_save_path = f'/mnt/dataset0/xkp/closed-loop/exp_sub{subject_id}/emotion_regulation'
    os.makedirs(exp_save_path, exist_ok=True)
    
    # 获取所有可用图像
    all_images = []
    for root, dirs, files in os.walk(image_set_path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                # 根据文件名获取情绪标签
                emotion_label = get_emotion_label_from_path(img_path)
                all_images.append((img_path, emotion_label))
    
    # 按情绪标签分组
    positive_images = [img for img, label in all_images if label == 1]
    negative_images = [img for img, label in all_images if label == 0]
    
    print(f"可用正面情绪图像: {len(positive_images)}张")
    print(f"可用负面情绪图像: {len(negative_images)}张")
    
    # 情绪概率结果存储
    emotion_probs = []
    actual_labels = []
    block_types = []
    
    # 执行10个loop
    for loop_idx in range(num_loops):
        print(f"\n=== 开始Loop {loop_idx+1}/{num_loops} ===")
        
        # 随机选择本次loop使用positive还是negative图像
        if loop_idx % 2 == 0:  # 偶数loop使用正面情绪，奇数loop使用负面情绪
            selected_images = random.sample(positive_images, 10)
            block_type = "positive"
            expected_label = 1
        else:
            selected_images = random.sample(negative_images, 10)
            block_type = "negative"
            expected_label = 0
        
        block_types.append(block_type)
        print(f"当前block类型: {block_type}")
        
        # 创建当前loop的保存目录
        loop_save_path = os.path.join(exp_save_path, f"loop_{loop_idx+1}")
        os.makedirs(loop_save_path, exist_ok=True)
        
        # 向客户端发送图像并收集EEG数据
        print(f"向客户端发送{len(selected_images)}张图像...")
        
        # 清空临时EEG目录
        if os.path.exists(instant_eeg_path):
            shutil.rmtree(instant_eeg_path)
        os.makedirs(instant_eeg_path, exist_ok=True)
        
        # 准备图像发送
        images_base64 = []
        for img_path in selected_images:
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                images_base64.append(encoded_string)
        
        # 发送图像和当前loop信息
        socketio.emit('exp3_images', {
            'images': images_base64,
            'loop': loop_idx + 1,
            'total_loops': num_loops,
            'block_type': block_type
        })
        
        # 等待EEG数据采集完成
        while True:
            # 检查是否收到10个EEG文件
            files = [f for f in os.listdir(instant_eeg_path) if f.endswith('.npy')]
            if len(files) >= 10:
                print(f"收到{len(files)}个EEG文件，继续处理...")
                break
            time.sleep(1)
            print("等待EEG数据...")
        
        # 处理收到的EEG数据
        loop_eeg_data = []
        loop_predictions = []
        
        for i, file in enumerate(sorted(os.listdir(instant_eeg_path))):
            if file.endswith('.npy'):
                # 加载EEG数据
                eeg_path = os.path.join(instant_eeg_path, file)
                eeg_data = np.load(eeg_path)
                
                # 保存到loop目录
                dest_path = os.path.join(loop_save_path, f"image_{i+1}_{block_type}.npy")
                shutil.copy(eeg_path, dest_path)
                
                # 情绪分类
                if eeg_data.shape[0] == 64:  # 确认数据格式正确
                    # 提取单个样本的PSD特征
                    if selected_channel_idxes:
                        eeg_sample = eeg_data[selected_channel_idxes, :]
                    else:
                        eeg_sample = eeg_data
                    
                    psd, _ = psd_array_multitaper(eeg_sample, fs, adaptive=True, normalization='full', verbose=0)
                    psd_flat = psd.flatten()
                    
                    # 使用分类器预测情绪概率
                    proba = clf.predict_proba([psd_flat])[0]
                    positive_prob = proba[1]  # 正面情绪的概率
                    
                    loop_predictions.append(positive_prob)
                    emotion_probs.append(positive_prob)
                    actual_labels.append(expected_label)
                    
                    print(f"图像 {i+1} 正面情绪概率: {positive_prob:.4f}")
                else:
                    print(f"警告: 图像 {i+1} 的EEG数据形状不正确: {eeg_data.shape}")
        
        # 清空临时目录
        shutil.rmtree(instant_eeg_path)
        os.makedirs(instant_eeg_path, exist_ok=True)
        
        # 计算这个block的平均情绪概率
        avg_prob = np.mean(loop_predictions)
        print(f"Loop {loop_idx+1} 平均正面情绪概率: {avg_prob:.4f}")
        
        # 保存本loop的结果
        loop_results = {
            'block_type': block_type,
            'expected_label': expected_label,
            'emotion_probs': loop_predictions,
            'average_prob': avg_prob
        }
        np.save(os.path.join(loop_save_path, 'results.npy'), loop_results)
    
    # 实验完成后绘制情绪概率曲线
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # 创建DataFrame便于绘图
    df = pd.DataFrame({
        'Sample': range(1, len(emotion_probs) + 1),
        'Positive_Probability': emotion_probs,
        'Actual_Label': actual_labels
    })
    
    # 添加Block信息
    block_info = []
    for i, block_type in enumerate(block_types):
        block_info.extend([f"Block {i+1}: {block_type}"] * 10)
    df['Block'] = block_info
    
    # 绘制情绪概率曲线
    plt.figure(figsize=(15, 8))
    
    # 为不同block添加背景色
    for i in range(num_loops):
        plt.axvspan(i*10+1, (i+1)*10, alpha=0.2, 
                   color='green' if block_types[i] == 'positive' else 'red')
    
    # 绘制概率曲线
    plt.plot(df['Sample'], df['Positive_Probability'], 'bo-', markersize=4, label='正面情绪概率')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='决策阈值')
    
    # 标注每个block
    for i in range(num_loops):
        plt.text((i*10) + 5, 0.05, f"Block {i+1}\n{block_types[i]}", 
                horizontalalignment='center', fontsize=9)
    
    plt.title('情绪调节实验 - 正面情绪概率曲线')
    plt.xlabel('样本序号')
    plt.ylabel('正面情绪概率')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(exp_save_path, 'emotion_probability_curve.png'))
    
    # 计算每个block的平均概率
    plt.figure(figsize=(12, 6))
    block_avg = df.groupby('Block')['Positive_Probability'].mean()
    
    # 为不同类型的block使用不同颜色
    colors = ['green' if 'positive' in idx else 'red' for idx in block_avg.index]
    block_avg.plot(kind='bar', color=colors)
    
    plt.title('每个Block的平均正面情绪概率')
    plt.ylabel('平均正面情绪概率')
    plt.axhline(y=0.5, color='black', linestyle='--')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(exp_save_path, 'block_averages.png'))
    
    # 向客户端发送实验完成信号
    print('Send: experiment_3_finished')
    socketio.emit('experiment_3_finished', {
        'message': "情感调节实验完成",
        'result_path': exp_save_path
    })
    
    return jsonify({
        "message": "情感调节实验完成",
        "result_path": exp_save_path
    }), 200
    
@app.route('/instant_eeg_upload', methods=['POST'])
def process_instant_eeg():
    if 'files' not in request.files:
        return jsonify({"message": "No file part"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"message": "No selected files"}), 400

    for file in files:
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        if file:
            filename = file.filename
            save_path = os.path.join(instant_eeg_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)

    return jsonify({
        "message": f"Files uploaded successfully"
    }), 200


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def collect_and_save_eeg_for_all_images(image_paths, save_path, category_list):
    print("Sending images to client")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('images_received', {'images': images})

    os.makedirs(instant_eeg_path, exist_ok=True)

    while True:
        files = [f for f in os.listdir(instant_eeg_path) if f.endswith('.npy')]
        if files:
            break
        else:
            time.sleep(1)

    time.sleep(10)

    print("Category number:", len(category_list))

    # 遍历 category_list，寻找对应的文件
    for idx, category in enumerate(category_list):
        filename = f"{idx+1}.npy"
        file_path = os.path.join(instant_eeg_path, filename)
        if os.path.exists(file_path):
            new_filename = f"{category}_{filename}"
            dest_path = os.path.join(save_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            shutil.move(file_path, dest_path)
            print(f"Moved and renamed file to {dest_path}")
        else:
            print(f"File {filename} not found in {instant_eeg_path}")

    shutil.rmtree(instant_eeg_path)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45565)
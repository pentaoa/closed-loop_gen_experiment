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
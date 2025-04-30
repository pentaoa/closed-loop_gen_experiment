import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import shutil
import time

from server_utils import (
    get_binary_labels,
    get_selected_channel_idxes,
    extract_emotion_psd_features,
    train_emotion_classifier,
)

app = Flask(__name__)
socketio = SocketIO(app)

# 实验参数
subject_id = 3
run_id = 1 
fs = 250

# 路径参数
image_set_path = 'stimuli_SX'
pre_eeg_path = f'data/sub{subject_id}/pre_eeg' # TODO: 验证
instant_eeg_path = 'data/instant_eeg'

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
def experiment_1():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    global features
    global clf
    
    print("\n" + "#" * 50)
    print("情感分类器训练")
    print("#" * 50 + "\n")
    
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
            
    print(f"Saved {len(files)} files to {pre_eeg_path}")

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
    
    # 转换标签
    labels = get_binary_labels(labels)
    
    # 加载EEG数据
    eeg_file_paths = [os.path.join(pre_eeg_path, f) for f in eeg_files]
    eeg_data = np.array([np.load(file) for file in eeg_file_paths])  # (n_samples, n_channels, n_timepoints)
    
    print(f"Loaded {len(eeg_data)} EEG samples with shape {eeg_data.shape}")

    # 获取选定的通道
    selected_channel_idxes = get_selected_channel_idxes(eeg_data, fs, 4)
    print("Selected channels:", selected_channel_idxes)
    
    # 提取特征并训练分类器
    features, valid_labels = extract_emotion_psd_features(eeg_data, labels, fs, selected_channel_idxes)
    print(f"提取的特征形状: {features.shape}")
    print(f"有效标签数量: {len(valid_labels)}")
    
    # 训练分类器和评估
    clf, report, y_test, y_pred = train_emotion_classifier(features, valid_labels, 0.2, 42)
    print("\n分类器报告:")
    print(report)
    
    # 向客户端发送信号，表示已准备好进行下一阶段的实验
    print('Send: experiment_2_ready')
    socketio.emit('experiment_2_ready') 

    return jsonify({
        "message": f"Files uploaded and processed successfully"
    }), 200
        

@app.route('/experiment_2', methods=['POST'])
def experiment_2():
    global selected_channel_idxes
    global target_image_path
    global target_eeg_path
    global features
    global clf

    print("\n" + "#" * 50)
    print("实时情感分类测试")
    print("#" * 50 + "\n")
    
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
    
    # 初始化标签列表
    labels = []

    # 设置绘图参数
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('#f0f0f0')
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 1.5)

    # 初始值
    categories = ['Dis', 'Amu']
    colors = ['#FF6B6B', '#4ECDC4']
    probabilities = [0.5, 0.5]

    # 创建条形图
    bars = ax.bar(categories, probabilities, color=colors, width=0.5, alpha=0.8)

    # 添加概率文本标签
    prob_texts = []
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        text = ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                    f'{prob:.2f}', ha='center', va='bottom', fontweight='bold')
        prob_texts.append(text)
        
    # 设置标题和标签
    ax.set_title('Predict', fontsize=16)
    ax.set_ylabel('Probability', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def update(frame):
        print(f"Frame {frame}")
        if frame >= len(test_images):
            return list(bars) + prob_texts
        
        # 获取当前图片
        current_image = test_images[frame]
        img_path = os.path.join(image_set_path, current_image)
        
        # 将图片发送给客户端进行EEG采集
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            socketio.emit('image_for_collection', {'image': encoded_string, 'index': frame})
        
        print(f"发送图片 {frame+1}/{len(test_images)}: {current_image}")
        
        # 记录标签（假设图片名称中包含情绪类别信息）
        true_label = 1 if current_image.startswith('Amu-') else 0
        labels.append(true_label)
        
        # 等待EEG数据采集完成
        eeg_file = os.path.join(instant_eeg_path, f"{frame}.npy")
        
        # 等待一段时间，然后检查文件是否存在
        max_wait = 5  # 最多等待5秒
        wait_time = 0
        while not os.path.exists(eeg_file) and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not os.path.exists(eeg_file):
            print(f"警告: 未能接收到EEG数据文件 {eeg_file}")
            # 返回当前图表状态，不更新
            return list(bars) + prob_texts
        
        # 加载EEG数据
        eeg_data = np.load(eeg_file)
        
        # 提取特征
        features, valid_labels = extract_emotion_psd_features(eeg_data, true_label, fs, selected_channel_idxes)
        print(f"提取的特征形状: {features.shape}")
        print(f"有效标签数量: {len(valid_labels)}")
        
        # 使用分类器进行预测
        proba = clf.predict_proba(features)[0]
        
        # 更新概率值
        probabilities[0] = proba[0]  # Dis概率
        probabilities[1] = proba[1]  # Amu概率
        print(f"预测概率: Dis={probabilities[0]:.2f}, Amu={probabilities[1]:.2f}")
        
        # 更新条形高度
        for bar, prob in zip(bars, probabilities):
            bar.set_height(prob)
        
        # 更新文本标签
        for i, (text, prob) in enumerate(zip(prob_texts, probabilities)):
            text.set_text(f'{prob:.2f}')
            text.set_position((i, prob + 0.02))
    
        # 清理文件，避免重复读取
        os.remove(eeg_file)
        
        # 返回所有更新的元素
        return list(bars) + prob_texts
        
    # 创建动画
    ani = FuncAnimation(fig, update, frames=min(100, len(test_images)),
                        interval=1000, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    # 保存动画
    from matplotlib.animation import PillowWriter
    ani.save(f'sub{subject_id}_emotion_animation.gif', writer=PillowWriter(fps=1))
    print(f"动画已保存为sub{subject_id}_emotion_animation.gif")
    
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


def collect_and_save_eeg_for_all_images(image_paths, save_path, label_list):
    print("Sending images to client")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_collection', {'images': images})

    os.makedirs(instant_eeg_path, exist_ok=True)

    while True:
        files = [f for f in os.listdir(instant_eeg_path) if f.endswith('.npy')]
        if files:
            break
        else:
            time.sleep(1)

    # time.sleep(10) # 请注意，这里的时间是为了多个eeg数据全部到达

    print("Category number:", len(label_list))

    # 遍历 label_list，寻找对应的文件
    for idx, label in enumerate(label_list):
        filename = f"{idx+1}.npy"
        file_path = os.path.join(instant_eeg_path, filename)
        if os.path.exists(file_path):
            new_filename = f"{label}_{filename}"
            dest_path = os.path.join(save_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            shutil.move(file_path, dest_path)
            print(f"Moved and renamed file to {dest_path}")
        else:
            print(f"File {filename} not found in {instant_eeg_path}")

    shutil.rmtree(instant_eeg_path)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45575)

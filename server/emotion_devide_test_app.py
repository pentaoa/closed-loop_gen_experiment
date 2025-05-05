import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import joblib
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
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
pre_eeg_path = f'server/data/sub{subject_id}/pre_eeg' # TODO: 验证
instant_eeg_path = 'server/data/instant_eeg'

# 全局变量
selected_channel_idxes = []
target_image_path = None
target_eeg_path = None
clf = None

@socketio.on('connect')
def handle_connect(auth):
    print('Client connected')
    print('Send: experiment_1_ready')
    socketio.emit('experiment_2_ready')

@app.route('/experiment_1_eeg_upload', methods=['POST'])
def experiment_1():
    global selected_channel_idxes
    global target_eeg_path
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
    
    # pca = PCA(n_components=0.95)
    # features = pca.fit_transform(features)
    # print(f"使用PCA降维后的特征形状: {features.shape}")
    
    # joblib.dump(pca, 'server/pca_model.pkl')

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
    global target_eeg_path
    global clf

    print("\n" + "#" * 50)
    print("实时情感分类测试")
    print("#" * 50 + "\n")
    
    # 检查分类器是否为 None，如果是则加载备用模型
    if clf is None:
        try:
            best_model_path = 'server/best_emotion_model.pkl'
            pca = joblib.load('server/pca_model.pkl')
            if os.path.exists(best_model_path):
                clf = joblib.load(best_model_path)
                print(f"成功加载备用分类器: {best_model_path}")
            else:
                return jsonify({
                    "message": "没有可用的分类器。请先运行 experiment_1 训练分类器，或确保备用模型文件存在。",
                    "error": "classifier_not_found"
                }), 400
        except Exception as e:
            return jsonify({
                "message": f"加载备用分类器时出错: {str(e)}",
                "error": "classifier_load_failed"
            }), 500    
    
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
        
        # 为当前帧创建保存路径
        frame_save_path = os.path.join(f'server/data/sub{subject_id}/test_results', f"frame_{frame}")
        os.makedirs(frame_save_path, exist_ok=True)
        
        # 使用 collect_and_save_eeg_for_all_images 函数处理单个图像
        collect_and_save_eeg_for_all_images([img_path], frame_save_path, current_label)
        
        # time.sleep(10)
        
        # 查找保存的EEG文件
        eeg_files = [f for f in os.listdir(frame_save_path) if f.endswith('.npy')]
        
        
        # 加载第一个找到的EEG文件
        eeg_file = os.path.join(frame_save_path, eeg_files[0])
        eeg_data = np.load(eeg_file)
        eeg_data = np.expand_dims(eeg_data, axis=0)  # 扩展为 (1, 通道数, 时间点)
        print(f"EEG 数据已加载，形状为{eeg_data.shape}")
    
        # 提取特征
        features, valid_labels = extract_emotion_psd_features(eeg_data, current_label, fs, selected_channel_idxes)
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
            
            # 记录当前帧结果
            frame_results.append({
                'frame': frame,
                'image': current_image,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'prob_dis': proba[0],
                'prob_amu': proba[1],
                'correct': (predicted_label == true_label)
            })
            
            print(f"预测概率: Dis={proba[0]:.2f}, Amu={proba[1]:.2f}")
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
        
    socketio.emit('experiment_finished')
        
    return jsonify({
        "message": f"Files uploaded successfully"
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


def collect_and_save_eeg_for_all_images(image_paths, save_path, label_list):
    print("Sending images to client")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_collection', {'images': images})

    # 确保目录存在并清空
    if os.path.exists(instant_eeg_path):
        shutil.rmtree(instant_eeg_path)
    os.makedirs(instant_eeg_path, exist_ok=True)

    # 等待EEG文件出现
    while True:
        files = [f for f in os.listdir(instant_eeg_path) if f.endswith('.npy')]
        if files:
            break
        else:
            time.sleep(1)
            print("等待EEG数据文件...")

    # 给文件额外的时间完全写入
    time.sleep(1)
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


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45565)

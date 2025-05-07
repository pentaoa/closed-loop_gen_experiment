import base64
import json
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
from threading import Event

from modulation_utils import *
from modulation import fusion_image_to_images

from server_utils import (
    get_binary_labels,
    get_selected_channel_idxes,
    extract_emotion_psd_features,
    train_emotion_classifier,
)

app = Flask(__name__)
socketio = SocketIO(app)

# 实验参数
subject_id = 1
fs = 250
num_loops = 10

ratings = []

# 路径参数
image_set_path = 'stimuli_SX'
pre_eeg_path = f'server/data/sub{subject_id}/pre_eeg' # TODO: 验证
instant_eeg_path = 'server/data/instant_eeg'
target_image_path = 'stimuli_SX/Dis-07.jpg'
target_eeg_path = 'DIRTI Database/1139_body products.jpg'


# 全局变量
selected_channel_idxes = []
target_image_path = None
target_eeg_path = None
clf = None
rating_received_event = Event()

@socketio.on('connect')
def handle_connect(auth):
    print('Client connected')
    print('Send: experiment_2_ready')
    socketio.emit('experiment_2_ready')

@app.route('/experiment_2', methods=['POST'])
def experiment_2():
    global selected_channel_idxes
    global target_eeg_path
    global target_image_path
    global ratings

    print("\n" + "#" * 50)
    print("图片 rating 迭代实验")
    print("#" * 50 + "\n")
    
    seed = 10000 * time.time() % 10000
    base_save_path = f'server/data/sub{subject_id}/rating_experiment'
    os.makedirs(base_save_path, exist_ok=True)
    
    # 获取所有图片    
    # all_image_files = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
    test_images = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
    # amu_images = [f for f in all_image_files if f.startswith('Amu-')]
    # dis_images = [f for f in all_image_files if f.startswith('Dis-')]
    # test_images = amu_images + dis_images
    # print(f"找到 {len(amu_images)} 张 Amu 图片")
    # print(f"找到 {len(dis_images)} 张 Dis 图片")
    print(f"一共 {len(test_images)} 张图片")
    
    # 随机所有照片
    random.seed(seed)
    random.shuffle(test_images)
    
    test_images_path = [os.path.join(image_set_path, test_image) for test_image in test_images]
    
    # 如果存在目标图片，则从测试集中移除
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    processed_paths = set()
    
    all_chosen_ratings = []      # 记录所有选中图片的评分
    all_chosen_image_paths = []  # 记录所有选中的图片路径
    history_best_ratings = []    # 记录每一轮的最高评分
    
    for i in range(num_loops):
        print(f"第 {i+1} 轮实验")
        round_save_path = os.path.join(base_save_path, f'loop_{i+1}')
        os.makedirs(round_save_path, exist_ok=True)
        
        loop_sample_images = []  # 当前轮次的图片集合
        loop_ratings = []        # 当前轮次的评分集合
        
        if i == 0:
            # 第一轮随机抽取10张图片
            first_ten_dir = os.path.join(round_save_path, 'first_ten')
            os.makedirs(first_ten_dir, exist_ok=True)
            
            # 从未处理的图片中随机选择10张
            available_paths = [path for path in test_images_path if path not in processed_paths]
            sample_image_paths = sorted(random.sample(available_paths, min(10, len(available_paths))))
            processed_paths.update(sample_image_paths)
            
            # 发送图片并收集评分
            success = send_images_and_collect_ratings(sample_image_paths, first_ten_dir)
            if not success:
                return jsonify({"message": "评分收集失败"}), 500
            
            # 使用用户评分作为相似度
            user_ratings = ratings.copy()
            
            # 记录图片和评分
            loop_sample_images = sample_image_paths
            loop_ratings = user_ratings
            
            # 根据评分选择最好的两张图片
            chosen_indices = sorted(range(len(user_ratings)), key=lambda x: user_ratings[x], reverse=True)[:2]
            chosen_ratings = [user_ratings[idx] for idx in chosen_indices]
            chosen_image_paths = [sample_image_paths[idx] for idx in chosen_indices]
            
        else:
            # 非第一轮，使用之前选择的最佳两张图片
            chosen_ratings = all_chosen_ratings[-2:]
            chosen_image_paths = all_chosen_image_paths[-2:]
            
            # 将已选图片加入当前轮次集合
            loop_sample_images.extend(chosen_image_paths)
            loop_ratings.extend(chosen_ratings)
        
        # 基于当前两张最佳图片融合生成6张新图片
        fusion_dir = os.path.join(round_save_path, 'fusion')
        os.makedirs(fusion_dir, exist_ok=True)
        
        try:
            # 使用融合函数生成新图片
            fusion_image_to_images(chosen_image_paths, 6, fusion_dir, 256)
            
            # 获取融合生成的所有图片路径
            fusion_image_paths = []
            for image in sorted(os.listdir(fusion_dir)):
                if image.endswith('.jpg') or image.endswith('.png'):
                    fusion_image_paths.append(os.path.join(fusion_dir, image))
            
            # 发送融合图片并收集评分
            if fusion_image_paths:
                success = send_images_and_collect_ratings(fusion_image_paths, fusion_dir)
                if not success:
                    return jsonify({"message": "融合图片评分收集失败"}), 500
                
                # 获取融合图片的评分
                fusion_ratings = ratings.copy()
                
                # 将融合图片和评分添加到当前轮次集合
                loop_sample_images.extend(fusion_image_paths)
                loop_ratings.extend(fusion_ratings)
        except Exception as e:
            print(f"图片融合失败: {str(e)}")
        
        # 从未处理的图片池中随机选择2张新图片
        new_samples_dir = os.path.join(round_save_path, 'new_samples')
        os.makedirs(new_samples_dir, exist_ok=True)
        
        available_paths = [path for path in test_images_path if path not in processed_paths]
        if available_paths:
            new_sample_paths = sorted(random.sample(available_paths, min(2, len(available_paths))))
            processed_paths.update(new_sample_paths)
            
            # 发送新图片并收集评分
            if new_sample_paths:
                success = send_images_and_collect_ratings(new_sample_paths, new_samples_dir)
                if not success:
                    return jsonify({"message": "新样本评分收集失败"}), 500
                
                # 获取新图片的评分
                new_ratings = ratings.copy()
                
                # 将新图片和评分添加到当前轮次集合
                loop_sample_images.extend(new_sample_paths)
                loop_ratings.extend(new_ratings)
        
        # 保存当前轮次的所有图片和评分
        all_ratings_file = os.path.join(round_save_path, 'all_ratings.json')
        with open(all_ratings_file, 'w') as f:
            json.dump({
                "image_paths": loop_sample_images,
                "ratings": loop_ratings
            }, f, indent=4)
        
        # 计算概率分布（可以使用softmax函数）
        def softmax(x):
            exp_x = np.exp(x)
            return exp_x / exp_x.sum()
        
        # 计算选择概率
        selection_probs = softmax(loop_ratings)
        
        # 根据概率选择两张图片
        # 可以直接选择评分最高的，也可以按概率抽样
        # 这里选择直接取评分最高的两张
        if loop_ratings:
            chosen_indices = sorted(range(len(loop_ratings)), key=lambda x: loop_ratings[x], reverse=True)[:2]
            chosen_ratings = [loop_ratings[idx] for idx in chosen_indices]
            chosen_image_paths = [loop_sample_images[idx] for idx in chosen_indices]
            
            # 记录选中的图片和评分
            for rating in chosen_ratings:
                all_chosen_ratings.append(rating)
            for image_path in chosen_image_paths:
                all_chosen_image_paths.append(image_path)
            
            # 更新历史最佳评分
            if chosen_ratings:
                max_rating = max(chosen_ratings)
                if not history_best_ratings or max_rating > max(history_best_ratings):
                    history_best_ratings.append(max_rating)
                else:
                    history_best_ratings.append(history_best_ratings[-1])
        
        # 打印当前轮次信息
        print(f"当前轮次选择的图片: {[os.path.basename(p) for p in chosen_image_paths]}")
        print(f"当前轮次选择的评分: {chosen_ratings}")
        print(f"历史最佳评分: {history_best_ratings}")
        
        # 检查收敛条件
        if len(history_best_ratings) >= 2 and abs(history_best_ratings[-1] - history_best_ratings[-2]) <= 1e-4:
            print("评分已收敛，提前结束实验")
            break
    
    # 保存实验总结数据
    summary_file = os.path.join(base_save_path, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            "all_chosen_ratings": all_chosen_ratings,
            "all_chosen_image_paths": all_chosen_image_paths,
            "history_best_ratings": history_best_ratings
        }, f, indent=4)
    
    # 生成评分历史图表
    plt.figure(figsize=(10, 6))
    plt.plot(history_best_ratings, marker='o')
    plt.title('评分历史变化')
    plt.xlabel('迭代轮次')
    plt.ylabel('最佳评分')
    plt.grid(True)
    plt.savefig(os.path.join(base_save_path, 'rating_history.png'))
    
    print("实验完成")
    socketio.emit('experiment_finished')
    
    return jsonify({
        "message": "实验成功完成",
        "best_rating": history_best_ratings[-1] if history_best_ratings else 0,
        "best_image": os.path.basename(all_chosen_image_paths[-1]) if all_chosen_image_paths else ""
    }), 200
        
    
@app.route('/rating_upload', methods=['POST'])
def receive_ratings():
    global ratings
    global rating_received_event
    
    data = request.get_json()
    ratings = data.get('ratings', [])
    if len(ratings) != 10:
        return jsonify({"message": "评分数量不正确"}), 400
    
    save_path = os.path.join(instant_eeg_path, 'ratings.json')
    with open(save_path, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # 设置事件，通知等待的函数继续执行
    rating_received_event.set()
    
    return jsonify({"message": "评分接收成功"}), 200


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


def send_images_and_collect_ratings(image_paths, save_path):
    global rating_received_event
    global ratings
    
    # 重置事件状态
    rating_received_event.clear()    
    
    print("发送图片到客户端")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_rating', {'images': images})
    
    # 等待评分接收事件
    print("等待客户端评分...")
    rating_received_event.wait(timeout=300)  # 设置超时时间(秒)，避免无限等待
    
    if not rating_received_event.is_set():
        print("警告: 等待评分超时")
        return False
    
    print("已收到评分，继续执行")
    
    # 保存评分到指定路径
    ratings_file = os.path.join(save_path, 'ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4)    
    
    return True    

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

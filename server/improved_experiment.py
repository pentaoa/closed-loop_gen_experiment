"""
基于机器学习的图像-EEG-评分收集的封闭循环实验服务器
该服务器实现了一个交互式实验框架，用于收集用户对图像的评分和脑电数据，并通过优化算法生成新图像
"""

import base64
import json
import os
import random
import time
import shutil
from threading import Event

import matplotlib.pyplot as plt
import numpy as np
import open_clip  # 用于图像特征提取的CLIP模型
from PIL import Image
from scipy.special import softmax
import torch

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

# 本地应用/库导入
from model.custom_pipeline_low_level import Generator4Embeds  # 图像生成模型
from model.ATMS_retrieval import ATMS  # EEG特征提取模型
from modulation_utils import *  # 通配符导入，包含各种图像处理工具
from server_utils import *  # 通配符导入，包含服务器工具函数

# 初始化Flask应用和Socket.IO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域请求

#====================== 全局实验参数 ======================
# 实验基本设置
sub = 'sub-01'                # 受试者ID
subject_id = 1                # 数字形式的受试者ID
fs = 250                      # EEG采样频率(Hz)
num_loops = 10                # 实验循环次数
use_eeg = True                # 是否使用EEG数据
device = "cuda" if torch.cuda.is_available() else "cpu"  # 计算设备
feature_type = 'psd'      # 特征类型：'clip', 'psd', 'clip_img'
model_type = 'ViT-H-14'        # CLIP模型类型
dnn = 'alexnet'                # DNN模型类型
random.seed(40)           # 随机种子


# 数据收集容器
processed_paths = set()       # 已处理过的图像路径集合

# 实验记录列表
all_chosen_rewards = []       # 所有被选择的奖励值
all_chosen_images = []        # 所有被选择的图像
all_chosen_eegs = []          # 所有被选择的EEG数据

# 历史记录与拟合数据
history_cs = []               # 历史相似度记录
fit_images = []               # 用于拟合的图像
fit_eegs = []                 # 用于拟合的EEG数据
fit_rewards = []              # 用于拟合的奖励值
fit_losses = []               # 用于拟合的损失值

# 保存路径
save_folder = f'server/outputs/heuristic_generation'       # 基础保存文件夹
plots_save_folder = 'server/plots/Interactive_search'      # 图表保存文件夹

# 预加载的测试集嵌入
test_set_img_embeds = torch.load("/mnt/dataset1/ldy/Workspace/FLORA/data_preparing/ViT-H-14_features_test.pt")['img_features'].cpu()

#====================== 路径参数 ======================
# 图像和数据路径
# image_set_path = '/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images'  # 图像集路径
image_set_path = 'stimuli_SX'  # 图像集路径
instant_eeg_path = 'server/data/instant_eeg'                                           # 实时EEG数据存储路径
cache_path = 'server/data/cache'                                                        # 缓存路径
# target_image_path = 'stimuli_SX/Dis-07.jpg'   
target_image_path = '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00135_pie/pie_15s.jpg'
target_eeg_path = ''                                # 目标EEG数据路径
# 创建输出目录
output_save_path = f"server/outputs/heuristic_generation/{feature_type}"
shutil.rmtree(output_save_path, ignore_errors=True)  # 清除之前的输出
os.makedirs(output_save_path, exist_ok=True)

#====================== 全局变量 ======================
selected_channel_idxes = []    # 选定的EEG通道索引
target_eeg_path = None         # 目标EEG路径
target_feature = None           # 目标特征
clf = None                     # 分类器
rating_received_event = Event() # 评分接收事件(用于线程同步)
eeg_received_event = Event()   # EEG数据接收事件(用于线程同步)

# 临时存储的数据
ratings = []                   # 用户评分
eeg = None                     # EEG数据

#====================== 模型准备 ======================
# 初始化CLIP模型及其预处理函数
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
vlmodel.to(device)

# 初始化图像生成器
generator = Generator4Embeds(device=device)
pipe = generator.pipe

if use_eeg:
    # 根据特征类型加载不同的目标特征和路径
    if feature_type == 'psd':        
        # 加载基于功率谱密度的目标EEG数据
        # target_eeg_path = f'/home/ldy/Closed_loop_optimizing/tjh/eeg_encoding/results/{sub}/synthetic_eeg_data/encoding-end_to_end/dnn-alexnet/modeled_time_points-all/pretrained-False/lr-1e-05__wd-0e+00__bs-064/gene_eeg/00085_gondola_85.npy'
        # target_image_path = '/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images/00085_gondola/gondola_11s.jpg'
        pass
        
    elif feature_type == 'clip':
        # 加载基于CLIP编码的EEG嵌入
        gt_eeg_folder = f'/mnt/dataset0/kyw/closed-loop/syn_eeg_gt'
        image_gt_folder = [
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00014_bike/bike_14s.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00181_television/television_14n.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00177_t-shirt/t-shirt_13s.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00135_pie/pie_15s.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00131_pear/pear_13s.jpg'
        ]
        target_eeg_embed = "/home/ldy/Closed_loop_optimizing/data/clip_embed/open_clip/00177_t-shirt_eeg_embeds.pt"
        # target_image_path = image_gt_folder[2]
        
        
        # 加载EEG编码模型
        f_encoder = f"/mnt/dataset0/kyw/closed-loop/sub_model/{sub}/diffusion_alexnet/pretrained_True/gene_gene/ATM_S_reconstruction_scale_0_1000_40.pth"
        checkpoint = torch.load(f_encoder, map_location=device)

        eeg_model = ATMS()  # EEG特征提取模型
        eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])

    elif feature_type == 'clip_img': 
        # 加载基于CLIP的图像嵌入
        gt_eeg_folder = f'/mnt/dataset0/kyw/closed-loop/syn_eeg_gt'
        image_gt_folder = [
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00014_bike/bike_14s.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00181_television/television_14n.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00177_t-shirt/t-shirt_13s.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00135_pie/pie_15s.jpg',
            '/mnt/dataset0/ldy/datasets/THINGS_MEG/images_set/test_images/00131_pear/pear_13s.jpg'
        ]    
        # target_image_embed = "/home/ldy/Closed_loop_optimizing/data/clip_embed/open_clip/00135_pie_image_embeds.pt"
        target_image_embed = "/mnt/dataset0/xkp/closed-loop/server/target_embed/open_clip/00135_pie_eeg_embeds.pt"
        # target_image_path = "/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images/00135_pie/pie_18s.jpg"

#====================== SocketIO事件处理 ======================
@socketio.on('connect')
def handle_connect(auth=None):
    """处理客户端连接事件"""
    print('Client connected')
    print('Send: experiment_2_ready')
    socketio.emit('experiment_2_ready')  # 通知客户端实验已准备好
    # experiment_1()


#====================== Flask路由 ======================
@app.route('/experiment_1', methods=['POST'])
def experiment_1():
    """实验一：用于确定 target"""
    global selected_channel_idxes
    global target_eeg_path 
    global target_image_path
    global target_feature
    global ratings
    
    print("\n" + "#" * 50)
    print("EEG特征选择实验")
    print("#" * 50 + "\n")
    
    # 为当前实验创建保存目录
    exp_1_save_path = os.path.join(output_save_path, f'experiment_1')
    os.makedirs(exp_1_save_path, exist_ok=True)
    
    
    if (use_eeg):
        target_eeg = send_images_and_collect_ratings_and_eeg([target_image_path], exp_1_save_path, 1)
        target_eeg_path = os.path.join(exp_1_save_path, '0.npy')
        # 根据特征类型加载目标特征
        if feature_type == 'psd':
            target_feature = load_target_feature(target_eeg_path, fs, selected_channel_idxes) 
        elif feature_type == 'clip':
            target_feature = torch.load(target_eeg_embed)     
        elif feature_type == 'clip_img':
            target_feature = torch.load(target_image_embed)
    else:
        # 发送目标图像到客户端并收集评分
        success = send_images_and_collect_ratings([target_image_path], exp_1_save_path)
        if not success:
            print("获取评分失败，实验终止")
            return jsonify({"message": "获取评分失败，实验终止"}), 500
        
    print("实验一结束")
    experiment_2()
        
    

@app.route('/experiment_2', methods=['POST'])
def experiment_2():
    """主要的实验流程处理函数，通过迭代方式生成和优化图像"""
    global selected_channel_idxes
    global target_eeg_path
    global target_image_path
    global ratings

    print("\n" + "#" * 50)
    print("图片 rating 迭代实验")
    print("#" * 50 + "\n")
    
    # 初始化启发式生成器
    Generator = HeuristicGenerator(pipe, vlmodel, preprocess_train, device=device)
    
    # 获取所有图片
    test_images = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"一共 {len(test_images)} 张图片")
    
    # 构建图像路径列表
    test_images_path = [os.path.join(image_set_path, test_image) for test_image in test_images]
    
    # 如果存在目标图片，则从测试集中移除
    if target_image_path in test_images_path:
        test_images_path.remove(target_image_path)
    
    processed_paths = set()  # 记录已处理的图像路径
    
    # 实验数据记录
    all_chosen_ratings = []      # 记录所有选中图片的评分
    all_chosen_image_paths = []  # 记录所有选中的图片路径
    history_best_ratings = []    # 记录每一轮的最高评分
    
    #====================== 实验主循环 ======================
    for t in range(num_loops):
        print(f"Loop {t + 1}/{num_loops}")
        
        # 为当前循环创建保存目录
        round_save_path = os.path.join(output_save_path, f'loop{t + 1}')
        os.makedirs(round_save_path, exist_ok=True)
        
        # 当前循环的数据收集容器
        loop_sample_ten = []     # 当前循环的样本图像
        loop_reward_ten = []     # 当前循环的奖励值
        if(use_eeg):
            loop_eeg_ten = []        # 当前循环的EEG数据
        

        #====================== 第一轮：随机采样 ======================
        if t == 0:
            # 创建第一轮保存目录
            first_ten = os.path.join(round_save_path, 'first_ten')
            os.makedirs(first_ten, exist_ok=True)
    
            # 从未处理的图片中随机选择10张
            available_paths = [path for path in test_images_path if path not in processed_paths]    
            sample_image_paths = sorted(random.sample(available_paths, 10))
            
            # 或者，手动选择
            # sample_image_paths = [...]
            
            # 加载图像并准备处理
            pil_images = []
            for sample_image_path in sample_image_paths:
                pil_images.append(Image.open(sample_image_path).convert("RGB"))
            
            similarities = []
            eegs=[]
            
            if (use_eeg):
                # 发送图像到客户端并收集评分和EEG数据
                eegs = send_images_and_collect_ratings_and_eeg(sample_image_paths, first_ten, 10)
                # 计算相似度和损失
                for idx, eeg in enumerate(eegs):  
                    # 根据特征类型计算相似度和损失
                    if feature_type == 'psd':
                        cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
                    elif feature_type == 'clip':
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn)
                    elif feature_type == 'clip_img':
                        cs = reward_function_clip_embed_image(pil_images[idx], target_feature)   
                    
                    similarities.append(cs)        
            else:
                # 发送图像到客户端并收集评分
                success = send_images_and_collect_ratings(sample_image_paths, first_ten)
                if not success:
                    print("获取评分失败，实验终止")
                    return jsonify({"message": "获取评分失败，实验终止"}), 500
                # 将用户rating作为相似度（reward）
                for rating in ratings:
                    similarities.append(rating)
                # 清空评分列表
                ratings = []
                # TODO: 这里需要根据实际情况处理损失
                
            # 计算选择概率
            probabilities = softmax(similarities)
            
            if (use_eeg):
                # 根据概率选择图像
                chosen_rewards, chosen_images, chosen_eegs = select_from_image_paths(
                    probabilities, similarities, sample_image_paths, eegs, size=4
                )
                # 更新当前循环的数据
                loop_sample_ten.extend(chosen_images)
                loop_eeg_ten.extend(chosen_eegs)
                loop_reward_ten.extend(chosen_rewards)
            else:
                chosen_rewards, chosen_images = select_from_image_paths_without_eeg(
                    probabilities, similarities, sample_image_paths, size=4
                )
                # 更新当前循环的数据
                loop_sample_ten.extend(chosen_images)
                loop_reward_ten.extend(chosen_rewards)
               
        
        #====================== 后续轮次：基于之前结果优化 ======================
        else:                            
            # 将图像转换为张量并提取特征
            tensor_fit_images = [preprocess_train(i) for i in fit_images]    
            with torch.no_grad():
                img_embeds = vlmodel.encode_image(torch.stack(tensor_fit_images).to(device))    
            
            # 基于图像嵌入生成融合图像
            generated_images = fusion_image_to_images(
                Generator, img_embeds, fit_rewards, device, round_save_path, 512
            )
            
            # 为当前轮次创建保存目录
            fusion_dir = os.path.join(round_save_path, 'fusion')
            os.makedirs(fusion_dir, exist_ok=True)
            
            # 保存生成的图像
            generated_image_paths = []
            for idx, generated_image in enumerate(generated_images):
                image_path = os.path.join(fusion_dir, f'generated_{idx}.jpg')
                generated_image.save(image_path)
                generated_image_paths.append(image_path)
            
            similarities = []
            
            if use_eeg:
                # 发送融合图像到客户端并收集评分和EEG数据
                eegs = send_images_and_collect_ratings_and_eeg(generated_image_paths, fusion_dir, len(generated_images))
                
                # 计算融合图像的相似度和损失
                for idx, eeg in enumerate(eegs):  
                    if feature_type == 'psd':
                        cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
                    elif feature_type == 'clip':
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn, device)
                    elif feature_type == 'clip_img':
                        cs = reward_function_clip_embed_image(
                            generated_images[idx], target_feature, device, vlmodel, preprocess_train
                        )             
                    
                    similarities.append(cs)
                    
                # 更新当前循环的数据
                loop_sample_ten.extend(generated_images)
                loop_eeg_ten.extend(eegs)
                loop_reward_ten.extend(similarities)
            else:
                # 发送融合图像到客户端并收集评分
                success = send_images_and_collect_ratings(generated_image_paths, fusion_dir)
                if not success:
                    print("获取融合图像评分失败")
                    return jsonify({"message": "获取融合图像评分失败"}), 500
                    
                # 将用户rating作为相似度（reward）
                for rating in ratings:
                    similarities.append(rating)
                
                # 清空评分列表
                ratings = []
                
                # 更新当前循环的数据
                loop_sample_ten.extend(generated_images)
                loop_reward_ten.extend(similarities)
            
            #====================== 贪心策略：选择相似特征的图像 ======================
            greedy_images = []
            sample_image_paths = []
            TOP_K = 10  # 选择最相似的K个图像
            
            # 为每个当前图像嵌入寻找最相似的图像
            for img_embed in img_embeds:
                # 找出所有未处理的图像索引
                available_indices = []
                for i, path in enumerate(test_images_path):
                    if path not in processed_paths:
                        available_indices.append(i)
                        
                if not available_indices:  # 如果没有更多未处理的图像，则跳过
                    continue
                        
                available_features = test_set_img_embeds[available_indices]
                
                # 计算余弦相似度
                cosine_similarities = compute_embed_similarity(img_embed.to(device), available_features.to(device))    
                sorted_available_indices = np.argsort(cosine_similarities.cpu())
                
                # 获取top K的索引（相似度最高的K个）
                top_k_count = min(TOP_K, len(sorted_available_indices))
                if top_k_count <= 0:
                    continue
                    
                top_indices = sorted_available_indices[-top_k_count:]
                
                # 从top K中随机选择一个
                selected_idx = np.random.choice(top_indices)
                actual_idx = available_indices[selected_idx]  # 转换回原始索引
                greedy_path = test_images_path[actual_idx]
                greedy_image = Image.open(greedy_path).convert("RGB")
                greedy_images.append(greedy_image)
                sample_image_paths.append(greedy_path)
                    
                # 更新已处理路径
                processed_paths.add(greedy_path)
            
            # 为贪心策略创建保存目录
            greedy_dir = os.path.join(round_save_path, 'greedy')
            os.makedirs(greedy_dir, exist_ok=True)
            
            # 保存贪心选择的图像
            greedy_image_paths = []
            for idx, greedy_image in enumerate(greedy_images):
                image_path = os.path.join(greedy_dir, f'greedy_{idx}.jpg')
                greedy_image.save(image_path)
                greedy_image_paths.append(image_path)
            
            similarities = []
            
            if use_eeg and greedy_images:
                # 发送贪心选择的图像到客户端并收集评分和EEG数据
                greedy_eegs = send_images_and_collect_ratings_and_eeg(greedy_image_paths, greedy_dir, len(greedy_images))
                
                # 计算贪心图像的相似度和损失
                for idx, eeg in enumerate(greedy_eegs):  
                    if feature_type == 'psd':
                        cs = reward_function(eeg, target_feature, fs, selected_channel_idxes)
                    elif feature_type == 'clip':
                        cs, eeg_feature = reward_function_clip_embed(eeg, eeg_model, target_feature, sub, dnn, device)
                    elif feature_type == 'clip_img':
                        cs = reward_function_clip_embed_image(
                            greedy_images[idx], target_feature, device, vlmodel, preprocess_train
                        )  
                    
                    similarities.append(cs)
                
                # 更新当前循环的数据
                loop_sample_ten.extend(greedy_images)
                loop_eeg_ten.extend(greedy_eegs)
                loop_reward_ten.extend(similarities)
            elif greedy_images:  # 只有在有贪心图像且不使用EEG时才执行
                # 发送贪心选择的图像到客户端并收集评分
                success = send_images_and_collect_ratings(greedy_image_paths, greedy_dir)
                if not success:
                    print("获取贪心图像评分失败")
                    return jsonify({"message": "获取贪心图像评分失败"}), 500
                    
                # 将用户rating作为相似度（reward）
                for rating in ratings:
                    similarities.append(rating)
                
                # 清空评分列表
                ratings = []
                
                # 更新当前循环的数据
                loop_sample_ten.extend(greedy_images)
                loop_reward_ten.extend(similarities)
            
            # 基于奖励计算选择概率
            loop_probabilities = softmax(loop_reward_ten)
            
            if use_eeg:
                # 从当前循环样本中选择最佳的几个
                chosen_rewards, chosen_images, chosen_eegs = select_from_images(
                    loop_probabilities, loop_reward_ten, loop_sample_ten, loop_eeg_ten, size=4
                )
                
                # 将四个列表按照chosen_rewards的值从大到小排序
                combined = list(zip(chosen_rewards, chosen_images, chosen_eegs))
                combined.sort(reverse=True, key=lambda x: x[0])  # 按rewards降序排列

                # 解压排序后的数据
                chosen_rewards, chosen_images, chosen_eegs = zip(*combined)

                # 将结果转回列表
                chosen_rewards = list(chosen_rewards)
                chosen_images = list(chosen_images)
                chosen_eegs = list(chosen_eegs)
            else:
                # 从当前循环样本中选择最佳的几个（无EEG版本）
                chosen_rewards, chosen_images = select_from_images_without_eeg(
                    loop_probabilities, loop_reward_ten, loop_sample_ten, size=4
                )
                
                # 排序
                combined = list(zip(chosen_rewards, chosen_images))
                combined.sort(reverse=True, key=lambda x: x[0])
                
                # 解压排序后的数据
                chosen_rewards, chosen_images = zip(*combined)
                
                # 将结果转回列表
                chosen_rewards = list(chosen_rewards)
                chosen_images = list(chosen_images)

        # 更新拟合数据
        fit_images = chosen_images
        fit_rewards = chosen_rewards
        if use_eeg:
            fit_eegs = chosen_eegs
        
        # 更新全局数据记录
        all_chosen_rewards.extend(chosen_rewards)
        all_chosen_images.extend(chosen_images)
        if use_eeg:
            all_chosen_eegs.extend(chosen_eegs)
        
        # 为伪目标模型添加数据
        tensor_loop_sample_ten = [preprocess_train(i) for i in loop_sample_ten]    
        with torch.no_grad():
            tensor_loop_sample_ten_embeds = vlmodel.encode_image(torch.stack(tensor_loop_sample_ten).to(device))        
        
        # 更新伪目标模型
        Generator.pseudo_target_model.add_model_data(
            torch.tensor(tensor_loop_sample_ten_embeds).to(device), 
            (-torch.tensor(loop_reward_ten) * Generator.reward_scaling_factor).to(device)
        )
        
        # 可视化当前循环中评分最高的图像
        visualize_top_images(loop_sample_ten, loop_reward_ten, save_folder, t)

        # 记录并更新历史最佳相似度
        max_similarity = max(loop_reward_ten)
        max_index = loop_reward_ten.index(max_similarity)
        
        if len(history_cs) == 0:
            history_cs.append(max_similarity)
        else:
            max_history = max(history_cs)
            if max_similarity > max_history:
                history_cs.append(max_similarity)
            else:
                history_cs.append(max_history)

        # 收敛检查：如果连续两轮的相似度变化很小，提前终止
        if len(history_cs) >= 2:
            if history_cs[-1] != history_cs[-2]:
                diff = abs(history_cs[-1] - history_cs[-2])
                print(history_cs[-1], history_cs[-2], diff)
                if diff <= 1e-4:
                    print("The difference is within 10e-4, stopping.")
                    break
    
    # 输出实验结果统计
    print(f"chosen_rewards {len(chosen_rewards)}")
    print(f"all_chosen_images {len(all_chosen_images)}")
    if use_eeg:
        print(f"all_chosen_eegs {len(all_chosen_eegs)}")

    # 绘制相似度历史图表
    plt.figure(figsize=(10, 5))
    plt.plot(history_cs, marker='o', markersize=3, label='Similarity')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend() 
    fig_path = os.path.join(output_save_path, 'similarities.jpg')
    plt.savefig(fig_path)
    
    # 通知客户端实验完成
    socketio.emit('experiment_finished', {
        "message": "实验完成"
    })
    
    # 返回实验结果
    return jsonify({
        "message": "实验成功完成",
        "best_rating": history_best_ratings[-1] if history_best_ratings else 0,
        "best_image": os.path.basename(all_chosen_image_paths[-1]) if all_chosen_image_paths else ""
    }), 200

@app.route('/eeg_upload', methods=['POST'])
def receive_eeg():
    """处理客户端上传的EEG数据"""
    try:
        global eeg
        global eeg_received_event
        
        print("接收EEG数据...")
        
        # 验证请求包含文件
        if 'files' not in request.files:
            print("错误: 请求中没有'files'字段")
            return jsonify({"message": "没有上传文件，请确保请求包含'file'字段"}), 400
        
        file = request.files['files']
        if file.filename == '':
            print("错误: 文件名为空")
            return jsonify({"message": "文件名为空"}), 400
        
        # 确保缓存目录存在
        os.makedirs(cache_path, exist_ok=True)
        
        # 保存上传的文件
        file_path = os.path.join(cache_path, 'eeg.npy')
        file.save(file_path)
        print(f"EEG文件已保存到 {file_path}")
        
        try:
            # 读取npy文件到全局变量
            eeg = np.load(file_path)
            print(f"成功加载EEG数据，形状: {eeg.shape}")
        except Exception as e:
            print(f"EEG数据加载失败: {str(e)}")
            return jsonify({"message": f"无法加载EEG数据: {str(e)}"}), 400
        
        # 设置事件，通知等待的函数继续执行
        eeg_received_event.set()
        
        return jsonify({"message": "EEG数据接收成功"}), 200
    
    except Exception as e:
        print(f"处理EEG上传时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"处理EEG上传时出错: {str(e)}"}), 500

@app.route('/rating_upload', methods=['POST'])
def receive_ratings():
    """处理客户端上传的评分数据"""
    global ratings
    global rating_received_event
    
    # 从请求中获取评分数据
    data = request.get_json()
    ratings = data.get('ratings', [])
    
    # 确保缓存目录存在
    os.makedirs(cache_path, exist_ok=True)
    
    # 保存评分到缓存目录
    save_path = os.path.join(cache_path, 'ratings.json')
    with open(save_path, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # 设置事件，通知等待的函数继续执行
    rating_received_event.set()
    
    return jsonify({"message": "评分接收成功"}), 200

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接事件"""
    print('Client disconnected')

def send_images_and_collect_ratings(image_paths, save_path):
    """
    发送图像到客户端并收集评分
    
    Args:
        image_paths: 图像文件路径列表
        save_path: 保存评分的目录路径
        
    Returns:
        bool: 操作是否成功
    """
    global rating_received_event
    global ratings
    
    # 重置事件状态
    rating_received_event.clear()    
    
    # 编码图像并发送给客户端
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
    
    # 检查是否收到评分
    if not rating_received_event.is_set():
        print("警告: 等待评分超时")
        return False
    
    print("已收到评分，继续执行")
    
    # 打印评分
    print(f"收到评分: {ratings}")
    
    # 保存评分到指定路径
    ratings_file = os.path.join(save_path, 'ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4) 
    
    return True    

def send_images_and_collect_ratings_and_eeg(image_paths, save_dir, num_of_events):
    """
    发送图像到客户端并同时收集评分和EEG数据，并截取、切分EEG为指定的事件数
    
    Args:
        image_paths: 图像文件路径列表
        save_dir: 保存数据的目录路径
        label: 数据标签，用于生成文件名
        
    Returns:
        bool: 操作是否成功
    """
    global ratings 
    global eeg
    global eeg_received_event
    global rating_received_event
    
    # 重置事件状态
    eeg_received_event.clear()
    rating_received_event.clear()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 编码图像并发送给客户端
    print("发送图片到客户端")
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(encoded_string)
    socketio.emit('image_for_rating_and_eeg', {'images': images})
    
    # 等待评分和EEG数据
    print("等待客户端评分和EEG数据...")
    timeout = 300  # 设置超时时间(秒)
    
    # 等待两个事件
    start_time = time.time()
    while time.time() - start_time < timeout:
        if eeg_received_event.is_set() and rating_received_event.is_set():
            print("已接收评分和EEG数据")
            break
        time.sleep(0.5)  # 避免过度占用CPU
    
    # 检查是否收到所有数据
    if not (eeg_received_event.is_set() and rating_received_event.is_set()):
        print("警告: 等待评分或EEG数据超时")
        return False
    
    # 保存评分
    ratings_file = os.path.join(save_dir, f'ratings.json')
    with open(ratings_file, 'w') as f:
        json.dump(ratings, f, indent=4)
    
    # 处理EEG数据：分割、过滤和重采样
    event_data_list = create_n_event_npy(eeg, num_of_events)
    filters = prepare_filters(fs, new_fs=250)
    processed_event_data_list = []
    for idx, event_data in enumerate(event_data_list):
        data = real_time_process(event_data, filters)
        processed_event_data_list.append(data)
        eeg_file = os.path.join(save_dir, f'eeg_{idx}.npy')
        np.save(eeg_file, data)
        
    print(f"数据已保存到 {save_dir}")
    return processed_event_data_list

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=45525)
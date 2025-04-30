import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random

# 假设我们已经有了分类器和一些测试数据
# 这里可以使用你已经训练好的分类器
try:
    clf = joblib.load('best_emotion_model.pkl')
    print("加载已有模型成功")
except:
    print("未找到已有模型，将创建模拟模型...")
    time.sleep(1)
    quit

# 假设我们有特征数据用于预测（可以替换为你的实际数据）
# 这里从测试集或新数据中抽取一些样本
try:
    # 尝试加载已有数据
    features = np.load('sample_features.npy')
    print(f"加载特征数据成功，形状: {features.shape}")
except:
    print("未找到特征数据，将创建模拟数据...")
    # 创建一些模拟特征数据用于演示
    features = np.random.rand(100, 20)  # 假设特征是20维的

# 创建一个缩放器，如果需要的话
scaler = StandardScaler()
# 假设features已经被训练好的scaler处理过

# 设置绘图参数
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor('#f0f0f0')  # 设置背景色
ax.set_ylim(0, 1)  # 设置y轴范围，因为是概率所以是0-1
ax.set_xlim(-0.5, 1.5)  # 设置x轴范围，只有两个条形

# 初始值
categories = ['Dis', 'Amu']
colors = ['#FF6B6B', '#4ECDC4']  # 红色表示负面，绿松石色表示正面
probabilities = [0.5, 0.5]  # 起始概率均为0.5

# 创建条形图
bars = ax.bar(categories, probabilities, color=colors, width=0.5, alpha=0.8)

# 添加概率文本标签
prob_texts = []
for i, (bar, prob) in enumerate(zip(bars, probabilities)):
    text = ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                 f'{prob:.2f}', ha='center', va='bottom', fontweight='bold')
    prob_texts.append(text)

# 设置标题和标签
ax.set_title('Real time predict', fontsize=16, pad=20)
ax.set_ylabel('P', fontsize=14)
ax.set_xlabel('categories', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加当前情绪状态指示器
emotion_status = ax.text(0.5, 1.05, 'Now: Neu', ha='center', 
                         transform=ax.transAxes, fontsize=15, 
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# 添加时间戳
timestamp = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10)

# 用于记录历史概率的变量
history_probs = {'negative': [], 'positive': []}
counter = 0

# 创建小图显示趋势
ax_trend = fig.add_axes([0.68, 0.15, 0.25, 0.2])  # [left, bottom, width, height]
ax_trend.set_title('tendency', fontsize=10)
ax_trend.set_ylim(0, 1)
ax_trend.grid(True, alpha=0.3)
line_pos, = ax_trend.plot([], [], 'g-', label='Amuse')
line_neg, = ax_trend.plot([], [], 'r-', label='Disgust')
ax_trend.legend(fontsize=8)
ax_trend.set_xticklabels([])

# 更新函数
def update(frame):
    global counter, history_probs
    
    # 从features中随机选择一个样本用于预测
    # 或者在实际应用中，这里会接收新的EEG数据并进行实时处理
    idx = random.randint(0, len(features) - 1)
    sample = features[idx:idx+1]
    
    try:
        # 使用分类器预测概率
        # 如果模型包含了缩放器，不需要这一步
        # sample_scaled = scaler.transform(sample)
        proba = clf.predict_proba(sample)[0]
    except:
        # 如果模型没有训练或出错，生成随机概率用于演示
        proba = np.random.dirichlet(np.ones(2), size=1)[0]
    
    # 更新概率值
    probabilities[0] = proba[0]  # 负面情绪概率
    probabilities[1] = proba[1]  # 正面情绪概率
    
    # 根据概率决定情绪状态
    emotion_text = 'Now:'
    if abs(probabilities[0] - probabilities[1]) < 0.1:
        emotion_text += 'Neu'
        emotion_color = 'gray'
    elif probabilities[0] > probabilities[1]:
        emotion_text += 'Dis'
        emotion_color = '#FF6B6B'
    else:
        emotion_text += 'Amu'
        emotion_color = '#4ECDC4'
    
    # 更新条形高度
    for bar, prob in zip(bars, probabilities):
        bar.set_height(prob)
    
    # 更新文本标签
    for i, (text, prob) in enumerate(zip(prob_texts, probabilities)):
        text.set_text(f'{prob:.2f}')
        text.set_position((i, prob + 0.02))
    
    # 更新情绪状态文本
    emotion_status.set_text(emotion_text)
    emotion_status.set_bbox(dict(facecolor=emotion_color, alpha=0.2, boxstyle='round,pad=0.5'))
    
    # 更新时间戳
    timestamp.set_text(f'Update time: {time.strftime("%H:%M:%S")}')
    
    # 记录历史数据用于趋势图
    history_probs['negative'].append(probabilities[0])
    history_probs['positive'].append(probabilities[1])
    
    # 只保留最近的30个数据点
    if len(history_probs['negative']) > 30:
        history_probs['negative'] = history_probs['negative'][-30:]
        history_probs['positive'] = history_probs['positive'][-30:]
    
    # 更新趋势图
    x_data = list(range(len(history_probs['negative'])))
    line_neg.set_data(x_data, history_probs['negative'])
    line_pos.set_data(x_data, history_probs['positive'])
    ax_trend.set_xlim(0, max(29, len(x_data)))
    
    counter += 1
    return bars + prob_texts + [emotion_status, timestamp, line_neg, line_pos]

# 创建动画
ani = FuncAnimation(fig, update, frames=range(100), interval=1000, blit=True)

plt.tight_layout()
plt.show()
import os
import numpy as np
import pandas as pd


from modulation_utils import get_selected_channel_idxes, extract_emotion_psd_features, train_emotion_classifier

pre_eeg_path = 'server/pre_eeg'

fs = 250

labels_path = os.path.join(pre_eeg_path, 'labels.npy')
labels = np.load(labels_path)
print(f"Loaded labels: {labels}")

eeg_files = [f for f in os.listdir(pre_eeg_path) 
             if f.endswith('.npy') and f != 'labels.npy']

if len(eeg_files) != len(labels):
    print(f"Number of EEG files ({len(eeg_files)}) does not match number of labels ({len(labels)})")

# 加载EEG数据
eeg_file_paths = [os.path.join(pre_eeg_path, f) for f in eeg_files]
eeg_data = np.array([np.load(file) for file in eeg_file_paths])  # (n_samples, n_channels, n_timepoints)

print(f"Loaded {len(eeg_data)} EEG samples with shape {eeg_data.shape}")

# 获取选定的通道
selected_channel_idxes = get_selected_channel_idxes(eeg_data, fs)
print("Selected channels:", selected_channel_idxes)

# 提取特征并训练分类器
features, valid_labels = extract_emotion_psd_features(eeg_data, labels, fs, selected_channel_idxes)
clf, report = train_emotion_classifier(features, valid_labels, 0.2, 42)
print("Classifier report:")
print(report)

# 使用 pandas 生成打印报告并绘制图表
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 获取测试集预测
X_train, X_test, y_train, y_test = train_test_split(features, valid_labels, test_size=0.2, random_state=42)
y_pred = clf.predict(X_test)

# 1. 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
class_names = ['负面情绪', '正面情绪']
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('情绪分类混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 2. 性能指标条形图
report_dict = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_metrics = df_report.iloc[:-3]
df_metrics = df_metrics.rename(index={'0': '负面情绪', '1': '正面情绪'})

metrics = ['precision', 'recall', 'f1-score']
df_plot = df_metrics[metrics]

plt.figure(figsize=(10, 6))
df_plot.plot(kind='bar', colormap='viridis')
plt.title('情绪分类性能指标')
plt.ylabel('得分')
plt.xlabel('情绪类别')
plt.ylim(0, 1.0)
plt.legend(title='指标')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('metrics_chart.png')
print("图表已保存到当前目录")
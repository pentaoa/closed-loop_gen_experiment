import numpy as np
import os

data_type = [ 'passive_data', 'active_data']


# 创建基于事件的数据，生成三维数组，第一维是事件，第二维是通道，第三维是时间点
def create_event_based_npy(original_data_path, preprocess_data_path, output_data_path):
    # 读取原始数据
    raw_data = np.load(original_data_path)
    events = raw_data[64, :]  # 第65行存储event信息

    # 读取预处理后的数据
    preprocessed_data = np.load(preprocess_data_path)
    
    event_based_data = []
    # 找到所有非零的event索引
    event_indices = np.where(events > 0)[0]
    print(event_indices)

    # 将原始数据的索引转换为降采样后的索引
    event_indices = event_indices // 4
    for idx in event_indices:
        if idx + 25 <= preprocessed_data.shape[1]:  # 确保索引不越界
            # 取出event后250个时间点的数据，对应 1 秒
            event_data = preprocessed_data[:64, idx:idx + 25]
            event_based_data.append(event_data)
    
    # 将事件数据转换为NumPy数组
    event_based_data = np.array(event_based_data)
    
    # 保存事件数据为新的.npy文件
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    np.save(output_data_path, event_based_data)


# 拼接所有数据
def concatenate_event_data(output_dir):
    concatenated_data = []
    subjects_list = os.listdir(source_dir)
    for subject in subjects_list:
        cls_list = os.listdir(os.path.join(source_dir, subject, data_type[1]))
        for cls in cls_list:
            cls_path = os.path.join(source_dir, subject, data_type[1], cls)
            data_list = os.listdir(cls_path)
            for data in data_list:
                data_path = os.path.join(output_dir, subject, data_type[1], cls, data)
                event_data = np.load(data_path)
                concatenated_data.append(event_data)
    
    # 在第一维度（事件维度）进行拼接
    concatenated_data = np.concatenate(concatenated_data, axis=0)
    
    # 保存拼接后的数据
    save_path = os.path.join(output_dir, subject, data_type[1], cls, 'concatenated_data.npy')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, concatenated_data)



source_dir = r"C:\Users\Grada\Desktop\enriched_data"
preprocessed_dir = r"C:\Users\Grada\Desktop\preprocessed_data"
output_dir = r"C:\Users\Grada\Desktop\output_data"

# create_event_based_npy(source_dir, preprocessed_dir, output_dir)
print("Event-based data saved to", output_dir)
concatenate_event_data(output_dir)
print("Event-based data concatenated and saved to", output_dir)
"""
Author: shaolichen@neuracle.cn
Copyright (c) 2022 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/
这个测试脚本只是验证录制好的数据是否一致的，完整的测试过程还需要用到Jellyfish、串口调试助手、triggerBox、w4物理机和neuracle_api.py以及python的repl

数据一致性验证要分成两个部分：数据准备 和 数据一致性验证，以下分别说明。

【数据准备】
1.使用串口调试助手让triggerBox默认1s发一个1（尽量避免trigger频率太快，原因请参考 neuracle_api.py的main()部分说明）。
2.Jellyfish打开数据转发，开始录制
3.运行neuracle_api.py中的main()，等到其输出 start
4.通过串口助手，让TriggerBox发送 2
5.等一段时间（一定不要超过neuracle_api.py的main()里面 time.sleep() 的时间）
6.通过串口助手，让TriggerBox发送 3
7.等待neuracle_api.py的main()运行结束，确认 test.bdf文件被成功生成
8.Jellyfish停止录制，把录制好的数据转成bdf

【数据一致性验证】
1.将Jellyfish转换后的 <被试名>.bdf（假定叫 1.bdf 下面代码中要使用该文件名，注意一定要保持一致）拷贝到本文件同级目录下
2.使用这个测试脚本，分别从 test.bdf和1.bdf中读取数据，分别称为 api_data 和 Jellyfish_data
3.再从api_data和Jellyfish_data中，找到各自2和3的trigger开始的数组索引(时间*采样率)，分别截取出两段数据
4.将这两段数据相减，输出最大误差和最小误差，以此来验证数据是否有误差
5.读取两段数据，找到各自2和3以及之间所有记录的trigger，先比较这俩数量是否一致，来验证有没有丢trigger；
  数量一致的情况下，把这些trigger的时间相减，看减出来的结果是不是基本都一致，如果都一致就认为trigger接收没有误差
"""

import mne
import numpy as np
import matplotlib.pyplot as plt


def readData(bdf_file_path: str):
    """
    读取bdf数据
    :param bdf_file_path:
    :return:
    """
    # 读取数据
    bdf_data = mne.io.read_raw_bdf(bdf_file_path)
    return bdf_data


def getData(bdf_data):
    """
    得到标志trigger '2'和'3'之间的数据
    :param bdf_data:
    :return:
    """
    annotations = bdf_data.annotations
    # 找到标志为2和3的trigger对应的时间
    start_index = np.where(annotations.description == '2')
    start_time = annotations.onset[start_index][0]
    end_index = np.where(annotations.description == '3')
    end_time = annotations.onset[end_index][0]
    # 采样率
    sample_rate = bdf_data.info['sfreq']
    # 得到2和3的时间戳
    start_time_index = int(start_time * sample_rate)
    end_time_index = int(end_time * sample_rate)
    # 截取这两标志trigger之间的数据
    data = bdf_data.get_data(start=start_time_index, stop=end_time_index)
    return data


def dataDiff(api_data, jellyfish_data):
    """
    得到api_data和jellyfish_data之间的差别
    jellyfish_data需要截取那些转发的通道
    :return:
    """
    # 根据api_data找到转发的通道
    transfer_channels = api_data.ch_names
    # jellyfish_data截取部分通道数据
    jellyfish_data.pick(transfer_channels)
    # 截取两个标志trigger之间的数据
    api_data_epoch = getData(api_data)
    jellyfish_data_epoch = getData(jellyfish_data)
    # 数据的差值
    epoch_delta = api_data_epoch - jellyfish_data_epoch
    return epoch_delta


def triggerTimeDiff(api_data, jellyfish_data):
    """
    验证Trigger时间戳是否一致
    :return:
    """
    # 得到标志trigger '2'和'3'以及之间的trigger
    # 找到标志为2和3的trigger对应的时间
    api_annotations = api_data.annotations
    api_start_index = np.where(api_annotations.description == '2')[0][0]
    api_end_index = np.where(api_annotations.description == '3')[0][0]
    api_trigger = api_annotations.onset[api_start_index:api_end_index + 1]
    jellyfish_annotations = jellyfish_data.annotations
    jellyfish_start_index = np.where(jellyfish_annotations.description == '2')[0][0]
    jellyfish_end_index = np.where(jellyfish_annotations.description == '3')[0][0]
    jellyfish_trigger = jellyfish_annotations.onset[jellyfish_start_index:jellyfish_end_index + 1]
    # 先比较数量是否一致
    if len(api_trigger) != len(jellyfish_trigger):
        raise Exception("Trigger数量不一致")
    # 再比较这些trigger的差值是否一样
    trigger_time_diff = []
    for i, j in zip(api_trigger, jellyfish_trigger):
        trigger_time_diff.append(i - j)

    print("Trigger数量一致")
    return trigger_time_diff


def triggerDiffConclusion(trigger_time_diff_without_mean, sample_rate):
    """
    根据采样率输出trigger比较结果
    500采样率时误差在2ms内认为是一致的
    250采样率时误差在4ms内认为是一致的
    :param trigger_time_diff_without_mean: 去除均值后的trigger误差
    :param sample_rate: 采样率
    :return:
    """
    # 最大误差
    max_error = max(abs(trigger_time_diff_without_mean))
    if sample_rate == 500:
        if max_error < 0.002:
            print('trigger完全一致')
        else:
            print('最大误差为:', max_error, '秒')
    elif sample_rate == 250:
        if max_error < 0.004:
            print('trigger完全一致')
        else:
            print('最大误差为:', max_error, '秒')
    else:
        print('最大误差为:', max_error, '秒')


def plot_trigger_diff(trigger_time_diff):
    """
    画api_data和jellyfish_data的trigger之间的差值
    画直接的差值以及去除均值后的差值
    :param trigger_time_diff:
    :return:
    """
    # 第一个子图
    plt.subplot(2, 1, 1)
    plt.title('Before remove mean')
    plt.plot(trigger_time_diff)
    # 去除均值，方便查看
    trigger_time_diff_without_mean = np.array(trigger_time_diff) - np.mean(trigger_time_diff)
    # 第二个子图
    plt.subplot(2, 1, 2)
    plt.title('After remove mean')
    plt.plot(trigger_time_diff_without_mean)
    plt.show()


# 通过API保存的bdf文件路径名
api_data_path = './test.bdf'
# 通过API保存的bdf文件内容
api_data = readData(api_data_path)

# 通过JellyFish录制并数据格式转换后得到的bdf文件路径名
jellyfish_data_path = './1.bdf'
# 通过JellyFish录制并数据格式转换后得到的bdf文件内容
jellyfish_data = readData(jellyfish_data_path)

# 验证数据是否正确
# 采样率500时trigger不精确可能会造成截取的数据点不一样长
# d1 = getData(api_data)
# d2 = getData(jellyfish_data)
# x = d1[:, :] - d2[:, :-1]
# print('最大数据误差', x.max())
# print('最小数据误差', x.min())
data_diff = dataDiff(api_data, jellyfish_data)
print('最大数据误差', data_diff.max())
print('最小数据误差', data_diff.min())
# 验证时间戳是否正确,两段数据各个对应的时间戳应该差值几乎一样
trigger_time_diff = triggerTimeDiff(api_data, jellyfish_data)
# 画trigger差值图
plot_trigger_diff(trigger_time_diff)
# 输出trigger比较的结论，能运行到这肯定是数量一致的
trigger_time_diff_without_mean = np.array(trigger_time_diff) - np.mean(trigger_time_diff)
triggerDiffConclusion(trigger_time_diff_without_mean, api_data.info['sfreq'])

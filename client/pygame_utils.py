import random
import os
import numpy as np
import pygame as pg
import time
import gc
import threading

from neuracle_api import DataServerThread
from triggerBox import TriggerBox, PackageSensorPara
from eeg_process import prepare_filters, real_time_processing, create_event_based_npy

class Model:
    def __init__(self):
        self.sample_rate = 1000
        self.t_buffer = 100
        self.thread_data_server = DataServerThread(self.sample_rate, self.t_buffer)
        self.flagstop = False
        self.triggerbox = TriggerBox("COM3")

    def start_data_collection(self):
        notConnect = self.thread_data_server.connect(hostname='127.0.0.1', port=8712)
        if notConnect:
            raise Exception("Can't connect to JellyFish, please check the hostport.")
        else:
            while not self.thread_data_server.isReady():
                time.sleep(1)
                continue
            self.thread_data_server.start()
            print("Data collection started.")

    def trigger(self, label):
        code = int(label)  # 直接将传入的类别编号转换为整数
        print(f'Sending trigger for label {label}: {code}')
        self.triggerbox.output_event_data(code)

    def connect_to_jellyfish(self):
        notConnect = self.thread_data_server.connect(hostname='127.0.0.1', port=8712)
        if notConnect:
            raise Exception("Can't connect to JellyFish, please check the hostport.")
        else:
            while not self.thread_data_server.isReady():
                time.sleep(1)
                continue
            self.thread_data_server.start()

    def stop_data_collection(self):
        self.flagstop = True
        self.thread_data_server.stop()

    def save_pre_eeg(self, pre_eeg_path):
        original_data_path = os.path.join(pre_eeg_path, f'original\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        preprocess_data_path = os.path.join(pre_eeg_path, f'preprocessed\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(original_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)

        data = self.thread_data_server.GetBufferData()
        np.save(original_data_path, data)
        print("Pre-experiment data saved!")

        # 进行数据预处理
        filters = prepare_filters(fs = 1000, new_fs=250)
        real_time_processing(original_data_path, preprocess_data_path, filters)
        print("Pre-experiment data preprocessed!")

        # 数据 event-based 处理
        create_event_based_npy(original_data_path, preprocess_data_path, pre_eeg_path)
        
    def save_labels(self, labels, path):
        labels_path = os.path.join(path, 'labels.npy')
        # 确保目录存在
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        # 保存标签
        np.save(labels_path, labels)
        print("Labels saved!")

    def save_instant_eeg(self, instant_eeg_path):
        original_data_path = os.path.join(instant_eeg_path, f'original\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        preprocess_data_path = os.path.join(instant_eeg_path, f'preprocessed\{time.strftime("%Y%m%d-%H%M%S")}.npy')
        
        # 重新创建文件夹
        os.makedirs(os.path.dirname(original_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(preprocess_data_path), exist_ok=True)

        data = self.thread_data_server.GetBufferData()
        np.save(original_data_path, data)
        print("Pre-experiment data saved!")

        del data
        gc.collect()

        # 进行数据预处理
        filters = prepare_filters(fs = 1000, new_fs=250)
        real_time_processing(original_data_path, preprocess_data_path, filters)
        print("Pre-experiment data preprocessed!")

        # 数据 event-based 处理
        create_event_based_npy(original_data_path, preprocess_data_path, instant_eeg_path)


    def get_next_sequence(self):
        # 确保不会超出列表范围
        if self.current_sequence * self.num_per_event >= len(self.sequence_indices):
            raise Exception("All sequences have been displayed.")

        # 从打乱的索引列表中获取下一个序列的索引
        sequence_start_index = self.current_sequence * self.num_per_event
        sequence_end_index = sequence_start_index + self.num_per_event
        next_sequence_indices = self.sequence_indices[sequence_start_index:sequence_end_index]

        # 更新当前序列计数
        self.current_sequence += 1

        # 返回选中的图像和标签，即返回 20 个 images_with_labels 元组
        return [(self.images[i], i) for i in next_sequence_indices]

    def reset_sequence(self):
        self.current_sequence = 0

    def set_phase(self, phase):
        self.current_phase = phase


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        self.model.start_data_collection()
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                        
        pg.quit()
        gc.collect()
        quit()


    def start_experiment_1(self, image_set_path, pre_eeg_path):
        self.view.display_text('Ready to start experiment 1')
        time.sleep(3)
        self.model.start_data_collection()
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 打乱图片顺序
        random.shuffle(image_files)
        
        # 初始化标签列表
        labels = []
        
        time.sleep(0.50)  # 500ms 黑屏
        
        for idx, image_file in enumerate(image_files):
            # 从文件名中提取标签，格式为 <label>-<index>.jpg
            # 例如: Amu-03.jpg
            try:
                label = image_file.split('-')[0]  # 提取标签部分
                labels.append(label)  # 将标签添加到列表中
                print("label:", label)
                
                image_path = os.path.join(image_set_path, image_file)
                print("image_path:", image_path)
                
                image = pg.image.load(image_path)
                self.view.display_image(image)
                
                # 发送触发器，使用图片的索引作为触发器代码
                self.model.trigger(idx + 1)
                
                time.sleep(1)  # 显示图片 1s
                self.view.display_fixation()
                time.sleep(1)  # 显示注视点 1s
                
                if (idx + 1) % 10 == 0:
                    self.view.clear_screen()
                    time.sleep(1)  # 每十个间隔一下
            except Exception as e:
                print(f"Error processing file {image_file}: {e}")
                continue

        # 采集结束，保存数据
        # self.model.stop_data_collection()
        self.view.display_text('Pre-experiment finished')
        self.model.save_pre_eeg(pre_eeg_path)
        self.model.save_labels(labels, pre_eeg_path)
        self.view.display_text('Data saved')

    def start_collection(self, instant_image_path, instant_eeg_path):
        self.view.display_text('Ready to start')
        time.sleep(3)
        # self.model.start_data_collection()
        time.sleep(0.5)  # 500ms 黑屏
        # 获取 instant_image_path 下的所有图片
        image_files = sorted(os.listdir(instant_image_path))
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(instant_image_path, image_file)
            image = pg.image.load(image_path)  # 加载图像
            self.view.display_image(image)
            self.model.trigger(idx + 1)  # 使用图像的索引发送触发器
            time.sleep(0.1)
            self.view.display_fixation()
            time.sleep(0.1)
            if (idx + 1) % 10 == 0:
                time.sleep(1)  # 每十个间隔一下
        
        # self.model.stop_data_collection()
        self.view.display_text('Processing...')
        self.model.save_instant_eeg(instant_eeg_path)
        self.view.display_text('Data saved')

    def black_screen_post(self):
        self.view.clear_screen()
        time.sleep(0.75)  # 750ms 黑屏
        self.blink_time()

    def end_experiment(self):
        self.view.display_text('Thank you!')
        time.sleep(3)
        self.model.stop_data_collection()
        pg.quit()
        gc.collect()
        quit()


class View:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((1000, 1000))
        pg.display.set_caption('Closed Loop Experiment')
        self.font = pg.font.Font(None, 40)

    def display_text(self, text):
        self.screen.fill((0, 0, 0))
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.screen.get_width() // 2 - text_surface.get_width() // 2,
                                        self.screen.get_height() // 2 - text_surface.get_height() // 2))
        pg.display.flip()     

    def display_fixation(self):
        self.screen.fill((0, 0, 0))  # 清屏
        # 绘制红色圆
        pg.draw.circle(self.screen, (200, 0, 0), (500, 500), 30, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (325, 500), (675, 500), 5)
        pg.draw.line(self.screen, (0, 0, 0), (500, 325), (500, 675), 5)
        pg.display.flip()

    def display_image(self, image):
        # 缩放图片到屏幕大小
        image = pg.transform.scale(image, (1000, 1000))
        self.screen.blit(image, (0, 0))
        # 绘制红色圆
        pg.draw.circle(self.screen, (200, 0, 0), (500, 500), 30, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (425, 500), (575, 500), 5)
        pg.draw.line(self.screen, (0, 0, 0), (500, 425), (500, 575), 5)
        # 更新屏幕显示
        pg.display.flip()

    def clear_screen(self):
        self.screen.fill((0, 0, 0))
        pg.display.flip()

    def display_multiline_text(self, text, position, font_size, line_spacing):
        font = pg.font.Font(self.font_path, font_size)
        lines = text.splitlines()  # 分割文本为多行
        x, y = position

        for line in lines:
            line_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(line_surface, (x, y))
            y += line_surface.get_height() + line_spacing  # 更新y坐标，为下一行做准备

        pg.display.flip()  # 更新屏幕显示

# 适用于离线实验采集
if __name__ == '__main__':
    pre_eeg_path = f'client\pre_eeg'
    instant_eeg_path = f'client\instant_eeg'
    instant_image_path = f'client\instant_image'
    image_set_path = f'stimuli_SX' 
    
    model = Model()
    view = View()
    controller = Controller(model, view)

    # 连接到 JellyFish
    try:
        model.connect_to_jellyfish()
    except Exception as e:
        print(f"Error: {e}")
        pg.quit()
        quit()

    # 启动实验
    controller.run()
    controller.start_experiment_1(image_set_path, pre_eeg_path)
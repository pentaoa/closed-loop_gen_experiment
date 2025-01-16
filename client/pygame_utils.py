import random
import os
import numpy as np
import pygame as pg
import time
import gc

from neuracle_api import DataServerThread
from triggerBox import TriggerBox, PackageSensorPara
from eeg_process import save_raw, create_event_based_npy

class Model:
    def __init__(self):
        self.current_phase = 'waiting'
        self.current_sequence = 0
        self.sample_rate = 1000
        self.t_buffer = 1000
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

    def save_data(self, npy_index):
        data = self.thread_data_server.GetBufferData()
        np.save(f'JiahaoTest/{time.strftime("%Y%m%d-%H%M%S")}-data-{npy_index}.npy', data)
        print("Data saved!")

    def save_pre_eeg(self, pre_eeg_path):
        original_data_path = os.path.join(pre_eeg_path, f'original/{time.strftime("%Y%m%d-%H%M%S")}.npy')
        preprocess_data_path = os.path.join(pre_eeg_path, f'preprocessed/{time.strftime("%Y%m%d-%H%M%S")}.npy')
        data = self.thread_data_server.GetBufferData()
        np.save(original_data_path, data)
        print("Pre-experiment data saved!")

        # 进行数据预处理
        save_raw(original_data_path, preprocess_data_path)
        print("Pre-experiment data preprocessed!")

        # 数据 event-based 处理
        create_event_based_npy(original_data_path, preprocess_data_path, pre_eeg_path)
        
        
    def save_instant_eeg(self, instant_eeg_path):
        original_data_path = os.path.join(instant_eeg_path, f'original/{time.strftime("%Y%m%d-%H%M%S")}.npy')
        preprocess_data_path = os.path.join(instant_eeg_path, f'preprocessed/{time.strftime("%Y%m%d-%H%M%S")}.npy')
        data = self.thread_data_server.GetBufferData()
        np.save(original_data_path, data)
        print("Pre-experiment data saved!")

        # 进行数据预处理
        save_raw(original_data_path, preprocess_data_path)
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
        running = True
        # self.model.start_data_collection()
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                        

        # 在实验循环结束后停止数据收集并保存数据
        self.model.stop_data_collection()
        pg.quit()
        gc.collect()
        quit()


    def start_pre_experiment(self, image_set_path, pre_eeg_path):
        self.view.display_text('Ready to start')
        time.sleep(3)
        self.model.start_data_collection()
        # 获取所有文件夹
        folders = sorted(os.listdir(image_set_path))
        image_label_pairs = []
        time.sleep(0.75)  # 500ms 黑屏

        for folder in folders:
            folder_path = os.path.join(image_set_path, folder)
            if os.path.isdir(folder_path):
                label = int(folder.split('_')[0]) # 设置 label 编号
                image_files = os.listdir(folder_path)
                if image_files:
                    image_path = os.path.join(folder_path, image_files[0])
                    image_label_pairs.append((image_path, label))

        # 开始采集
        for i, (image, label) in enumerate(image_label_pairs):
            self.view.display_image(image)
            self.model.trigger(label)  # 使用图像的类别编号发送触发器
            time.sleep(0.1)
            self.view.display_fixation()
            time.sleep(0.1)
            if (i + 1) % 10 == 0:
                self.view.clear_screen()
                time.sleep(1)  # 每十个给一个长一点的间隔

        # 采集结束，分析数据
        self.model.stop_data_collection()
        self.view.display_text('Pre-experiment finished')
        self.model.save_pre_eeg(pre_eeg_path)
        self.view.display_text('Sata saved')

    def start_collection(self, instant_image_path, instant_eeg_path):
        self.view.display_text('Ready to start')
        time.sleep(3)
        self.model.start_data_collection()
        time.sleep(0.75)

        # 获取 instant_image_path 下的所有图片
        image_files = sorted(os.listdir(instant_image_path))
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(instant_image_path, image_file)
            self.view.display_image(image_path)
            self.model.trigger(i + 1)  # 使用图像的索引发送触发器
            time.sleep(0.1)
            if (i + 1) % 10 == 0:
                time.sleep(1)  # 每十个间隔一下
        
        self.view.display_text('Finished')
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
        self.screen = pg.display.set_mode((1200, 900))
        pg.display.set_caption('Closed Loop Experiment')
        self.font = pg.font.Font(None, 32)

    def display_text(self, text):
        self.screen.fill((0, 0, 0))
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.screen.get_width() // 2 - text_surface.get_width() // 2,
                                        self.screen.get_height() // 2 - text_surface.get_height() // 2))
        pg.display.flip()     

    def display_fixation(self):
        self.screen.fill((0, 0, 0))  # 清屏
        # 绘制红色圆
        pg.draw.circle(self.screen, (255, 0, 0), (600, 450), 30, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (575, 450), (625, 450), 10)
        pg.draw.line(self.screen, (0, 0, 0), (600, 425), (600, 475), 10)
        pg.display.flip()

    def display_image(self, image):
        self.screen.blit(image, (0, 0))
        pg.draw.circle(self.screen, (255, 0, 0), (600, 450), 30, 0)
        pg.draw.line(self.screen, (0, 0, 0), (575, 450), (625, 450), 10)
        pg.draw.line(self.screen, (0, 0, 0), (600, 425), (600, 475), 10)

        # 更新屏幕显示
        pg.display.flip()

    # def display_text(self, text, position):
    #     # 使用指定的中文字体渲染文本
    #     font = pg.font.Font(self.font_path, 50)
    #     text_surface = font.render(text, True, (255, 255, 255))
    #     self.screen.blit(text_surface, position)
    #     pg.display.flip()

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

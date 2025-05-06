import random
import os
import numpy as np
import pygame as pg
import time
import gc

from neuracle_api import DataServerThread
from triggerBox import TriggerBox, PackageSensorPara
from eeg_process import *

class Model:
    def __init__(self):
        self.sample_rate = 250
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
                time.sleep(0.1)
                continue
            self.thread_data_server.start()
            print("Data collection started.")

    def trigger(self, label):
        code = int(label)  # 直接将传入的类别编号转换为整数
        print(f'Sending trigger for label {label}: {code}')
        self.triggerbox.output_event_data(code)

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
        filters = prepare_filters(fs = self.sample_rate, new_fs=250)
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
        data = self.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        event_data = event_data_list[0]  # 取第一个事件数据
        filters = prepare_filters(fs = self.sample_rate, new_fs=250)
        data = real_time_process(event_data, filters)
        np.save(os.path.join(instant_eeg_path, f'{time.strftime("%Y%m%d-%H%M%S")}.npy'), data)
        print("Instant EEG data saved!")

    def get_event_data(self):
        data = self.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        return event_data_list
    
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

            # Add a small sleep to prevent high CPU usage
            time.sleep(0.01)


    def start_experiment_1(self, image_set_path, pre_eeg_path):
        print("Start experiment 1")
        self.view.display_text('Ready to start experiment 1')
        time.sleep(5)
        
        # 获取所有图片文件
        all_image_files = [f for f in os.listdir(image_set_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # 按情绪类别分组图片
        amu_images = [f for f in all_image_files if f.startswith('Amu-')]
        dis_images = [f for f in all_image_files if f.startswith('Dis-')]
        
        print(f"找到 {len(amu_images)} 张 Amu 图片")
        print(f"找到 {len(dis_images)} 张 Dis 图片")
        
        # 初始化标签列表
        labels = []
        
        time.sleep(0.50)  # 500ms 黑屏
        
        # 先展示 Amu 图片 (5次)
        for repeat in range(5):
            print(f"正在显示 Amu 图片 (第 {repeat+1}/5 轮)")
            random.shuffle(amu_images)  # 每轮随机打乱顺序
            
            for idx, image_file in enumerate(amu_images):
                label = 'Amu'
                labels.append(label)
                
                image_path = os.path.join(image_set_path, image_file)
                print(f"显示图片: {image_path}")
                
                image = pg.image.load(image_path)
                self.view.display_image(image)
                
                # 发送触发器，使用图片的索引作为触发器代码
                self.model.trigger(len(labels))  # 使用累计图片数作为触发器代码
                
                time.sleep(5)  # 显示图片 5s
                self.view.clear_screen()
                time.sleep(1)  # 显示注视点 1s
        
        # 再展示 Dis 图片 (5次)
        for repeat in range(5):
            print(f"正在显示 Dis 图片 (第 {repeat+1}/5 轮)")
            random.shuffle(dis_images)  # 每轮随机打乱顺序
            
            for idx, image_file in enumerate(dis_images):
                label = 'Dis'
                labels.append(label)
                
                image_path = os.path.join(image_set_path, image_file)
                print(f"显示图片: {image_path}")
                
                image = pg.image.load(image_path)
                self.view.display_image(image)
                
                # 发送触发器，使用图片的索引作为触发器代码
                self.model.trigger(len(labels))  # 使用累计图片数作为触发器代码
                
                time.sleep(5)  # 显示图片 5s
                self.view.clear_screen()
                time.sleep(1)  # 显示注视点 1s

        # 采集结束，保存数据
        self.view.display_text('Pre-experiment finished')
        self.model.save_pre_eeg(pre_eeg_path)
        self.model.save_labels(labels, pre_eeg_path)
        self.view.display_text('Data saved')

    def collect_data(self, image_path):
        self.view.display_text('Ready to start')
        time.sleep(0.5)

        image = pg.image.load(image_path)
        self.view.display_image(image)
        self.model.trigger(1)  # 使用图像的索引发送触发器
        time.sleep(5)
        self.view.clear_screen()
        time.sleep(1)

        self.view.display_text('Processing...')

        data = self.model.thread_data_server.GetBufferData()
        event_data_list = create_last_event_npy(data, 1)
        event_data = event_data_list[0]  # 取第一个事件数据
        filters = prepare_filters(fs = self.model.sample_rate, new_fs=250)
        data = real_time_process(event_data, filters)
        return data
    
    def rating(self):
        self.view.display_text('Please rate the image')
        time.sleep(1)
        self.view.clear_screen()
        

    def end_experiment(self):
        self.view.display_text('Thank you!')
        time.sleep(3)
        self.model.stop_data_collection()
        pg.quit()
        gc.collect()
        quit()


class View:
    def __init__(self, monitor_index=0):
        pg.init()
        
        # 获取所有显示器的信息
        monitors_info = []
        try:
            if hasattr(pg.display, 'get_desktop_sizes'):
                # Pygame 2.0.0+ 方法
                monitors_info = pg.display.get_desktop_sizes()
            else:
                # 之前版本使用 SDL 的环境变量获取显示器信息
                import os
                try:
                    from screeninfo import get_monitors
                    monitors = get_monitors()
                    for m in monitors:
                        monitors_info.append((m.width, m.height))
                except ImportError:
                    # 如果没有 screeninfo 库，则使用系统默认
                    display_info = pg.display.Info()
                    monitors_info = [(display_info.current_w, display_info.current_h)]
        except:
            # 如果无法获取多显示器信息，使用默认值
            display_info = pg.display.Info()
            monitors_info = [(display_info.current_w, display_info.current_h)]
        
        print(f"检测到 {len(monitors_info)} 个显示器")
        for i, (width, height) in enumerate(monitors_info):
            print(f"显示器 {i}: {width}x{height}")
        
        # 确保 monitor_index 在有效范围内
        if monitor_index < 0 or monitor_index >= len(monitors_info):
            print(f"警告: 显示器索引 {monitor_index} 无效，使用默认显示器 0")
            monitor_index = 0
        
        # 获取指定显示器的尺寸
        screen_width, screen_height = monitors_info[monitor_index]
        
        # 计算显示器位置（假设显示器水平排列）
        position_x = sum(w for w, _ in monitors_info[:monitor_index])
        
        # 设置窗口环境变量（告诉SDL在哪个显示器上创建窗口）
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{position_x},{0}"
        
        # 创建全屏窗口
        self.screen = pg.display.set_mode((screen_width, screen_height), pg.FULLSCREEN)
        pg.display.set_caption('Closed Loop Experiment')
        self.font = pg.font.Font(None, 40)
        
        # 存储显示器信息
        self.monitor_index = monitor_index
        self.screen_width = screen_width
        self.screen_height = screen_height
        print(f"已在显示器 {monitor_index} ({screen_width}x{screen_height}) 创建全屏窗口")

    def display_text(self, text):
        self.screen.fill((0, 0, 0))
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (self.screen.get_width() // 2 - text_surface.get_width() // 2,
                                        self.screen.get_height() // 2 - text_surface.get_height() // 2))
        pg.display.flip()     

    def display_fixation(self):
        self.screen.fill((0, 0, 0))  # 清屏
        # 绘制红色圆
        pg.draw.circle(self.screen, (200, 0, 0), (400,300), 10, 0)
        # 绘制黑色十字
        pg.draw.line(self.screen, (0, 0, 0), (425, 300), (375, 300), 3)
        pg.draw.line(self.screen, (0, 0, 0), (400, 325), (400, 275), 3)
        pg.display.flip()

    def display_image(self, image):
        # 获取当前屏幕分辨率
        screen_width, screen_height = self.screen.get_size()
        
        # 获取图片的原始尺寸
        img_width, img_height = image.get_size()
        
        # 计算宽高比
        width_ratio = screen_width / img_width
        height_ratio = screen_height / img_height
        
        # 选择较小的比例以确保图片完全显示在屏幕内
        scale_ratio = min(width_ratio, height_ratio)
        
        # 计算缩放后的尺寸
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # 缩放图片
        scaled_image = pg.transform.scale(image, (new_width, new_height))
        
        # 计算居中位置
        x_pos = (screen_width - new_width) // 2
        y_pos = (screen_height - new_height) // 2
        
        # 先填充黑色背景
        self.screen.fill((0, 0, 0))
        
        # 在居中位置绘制图片
        self.screen.blit(scaled_image, (x_pos, y_pos))
        
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
        
    def rating(self):
        """显示评分界面，让用户输入0-1之间的两位小数"""
        # 初始值
        input_text = ""
        active = True
        max_length = 4  # 最多4个字符 (0.xx)
        
        # 计算输入框位置
        screen_width = self.view.screen.get_width()
        screen_height = self.view.screen.get_height()
        input_rect = pg.Rect(screen_width//2 - 100, screen_height//2, 200, 50)
        
        # 进入输入循环
        while active:
            # 处理所有事件
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_RETURN:
                        # 确认输入，检查格式
                        try:
                            score = float(input_text)
                            if 0 <= score <= 1:
                                active = False  # 退出循环
                            else:
                                # 显示错误提示
                                self.view.display_text("请输入0-1之间的值！")
                                time.sleep(1)
                        except ValueError:
                            # 显示错误提示
                            self.view.display_text("请输入有效的小数！")
                            time.sleep(1)
                    elif event.key == pg.K_BACKSPACE:
                        # 删除最后一个字符
                        input_text = input_text[:-1]
                    elif event.key == pg.K_ESCAPE:
                        # ESC键退出
                        return None
                    else:
                        # 只接受数字和小数点
                        if event.unicode in '0123456789.' and len(input_text) < max_length:
                            # 确保只有一个小数点
                            if event.unicode == '.' and '.' in input_text:
                                continue
                            # 确保小数点后最多两位
                            if '.' in input_text and len(input_text.split('.')[1]) >= 2 and event.unicode != '.':
                                continue
                            input_text += event.unicode
            
            # 清屏
            self.view.screen.fill((0, 0, 0))
            
            # 绘制说明文本
            instruction_text = self.view.font.render("请输入是否高兴的评分 (0-1之间的两位小数):", True, (255, 255, 255))
            self.view.screen.blit(instruction_text, (screen_width//2 - instruction_text.get_width()//2, 
                                                screen_height//2 - 80))
            
            # 绘制输入框
            pg.draw.rect(self.view.screen, (255, 255, 255), input_rect, 2)
            
            # 显示当前输入的文本
            text_surface = self.view.font.render(input_text, True, (255, 255, 255))
            self.view.screen.blit(text_surface, (input_rect.x + 10, input_rect.y + 10))
            
            # 显示用法提示
            hint_text = self.view.font.render("按回车确认, ESC取消", True, (200, 200, 200))
            self.view.screen.blit(hint_text, (screen_width//2 - hint_text.get_width()//2, 
                                            screen_height//2 + 80))
            
            # 更新显示
            pg.display.flip()
            
            # 小延迟减少CPU使用
            time.sleep(0.01)
        
        # 返回有效的评分结果
        return float(input_text)
    
# 适用于离线实验采集
if __name__ == '__main__':
    pre_eeg_path = f'client\pre_eeg'
    instant_eeg_path = f'client\instant_eeg'
    instant_image_path = f'client\instant_image'
    image_set_path = f'stimuli_SX' 
    
    model = Model()
    view = View()
    controller = Controller(model, view)

    controller.start_experiment_1(image_set_path, pre_eeg_path)
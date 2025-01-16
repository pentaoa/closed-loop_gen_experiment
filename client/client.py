import base64
from io import BytesIO
import requests
import os
import numpy as np
import json
from PIL import Image
import time
import pygame as pg
from pygame_utils import Model, View, Controller
from socketIO_client import SocketIO

pre_eeg_path = f'client/pre_eeg'
instant_eeg_path = f'client/instant_eeg'
instant_image_path = f'client/instant_image'
image_set_path = '/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images'

selected_channels = []
target_image = None

def on_connect(view):
    print('Connected to server')
    view.display_text('Connected to server')
    time.sleep(1)

def on_connect_error(view):
    print('Failed to connect to server')
    view.display_text('Failed to connect to server')
    time.sleep(5)
    pg.quit()
    quit()

def on_pre_experiment_ready(controller):
    # 启动预试验
    controller.start_pre_experiment(image_set_path, pre_eeg_path)

    # 发送 pre_eeg_path 中的所有 npy 文件到服务器
    url = 'http://10.20.37.38:55565/pre_experiment_eeg_upload'

    files = []
    for filename in os.listdir(pre_eeg_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(pre_eeg_path, filename)
            files.append(('files', (filename, open(file_path, 'rb'), 'application/octet-stream')))
    
    response = requests.post(url, files=files)


def on_experiment_ready():
    time.sleep(2)
    # 向服务器发送开始实验的信号
    url = 'http://10.20.37.38:55565/experiment'
    requests.post(url)

def on_images_received(data, controller):
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # 保存图像到 client/instant_image 目录下
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')

    # 启动实验    
    controller.start_collection(instant_image_path, instant_eeg_path)
    
    # 发送 instant_eeg_path 中的所有 npy 文件到服务器
    url = 'http://10.20.37.38:55565/instant_eeg_upload'
    files = []
    for filename in os.listdir(instant_eeg_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(instant_eeg_path, filename)
            files.append(('files', (filename, open(file_path, 'rb'), 'application/octet-stream')))
    
    response = requests.post(url, files=files)

def on_experiment_finished(controller):
    controller.stop_collection()
    socketIO.disconnect()


if __name__ == '__main__':
    model = Model()
    view = View()
    controller = Controller(model, view)

    controller.run()

    # 连接到服务器，并注册事件处理程序
    socketIO = SocketIO('10.20.37.38', 55565, LoggingNamespace)
    socketIO.on('connect', lambda: on_connect(view))
    socketIO.on('connect_error', lambda: on_connect_error(view))
    socketIO.on('pre_experiment_ready', lambda: on_pre_experiment_ready(controller))
    socketIO.on('experiment_ready', lambda: on_experiment_ready())
    socketIO.on('images_received', lambda data: on_images_received(data, controller))
    socketIO.on('experiment_finished', lambda: on_experiment_finished(controller))

    # 保持 WebSocket 连接
    socketIO.wait()
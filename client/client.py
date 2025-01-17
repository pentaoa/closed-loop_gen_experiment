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
import socketio

pre_eeg_path = f'client/pre_eeg'
instant_eeg_path = f'client/instant_eeg'
instant_image_path = f'client/instant_image'
image_set_path = "\\10.20.37.22\dataset0\ldy\test"

selected_channels = []
target_image = None

sio = socketio.Client()

# @sio.event
# def connect():
#     print('Connected to server')
#     time.sleep(1)

@sio.event
def connect_error():
    print('Failed to connect to server')
    view.display_text('Failed to connect to server')
    time.sleep(3)
    pg.quit()
    quit()

@sio.event
def connect_failed():
    view.display_text('Failed to connect')

@sio.event
def pre_experiment_ready(data):
    print(data['message'])
    print("****************************************")
    controller.start_pre_experiment(image_set_path, pre_eeg_path)

    # 发送 pre_eeg_path 中的所有 npy 文件到服务器
    url = 'http://10.20.37.38:55565/pre_experiment_eeg_upload'

    files = []
    for filename in os.listdir(pre_eeg_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(pre_eeg_path, filename)
            with open(file_path, 'rb') as f:
                files.append(('files', (filename, f, 'application/octet-stream')))
    
    response = requests.post(url, files=files)

@sio.event
def experiment_ready(data):
    print(data['message'])
    time.sleep(2)
    # 向服务器发送开始实验的信号
    url = 'http://10.20.37.38:55565/experiment'
    requests.post(url)

@sio.event
def images_received(data):
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
            with open(file_path, 'rb') as f:
                files.append(('files', (filename, f, 'application/octet-stream')))
    
    response = requests.post(url, files=files)

@sio.event
def experiment_finished(data):
    print(data['message'])
    controller.stop_collection()
    # 断开连接
    sio.disconnect()

if __name__ == '__main__':
    global controller
    model = Model()
    view = View()
    controller = Controller(model, view)


    sio.connect('http://10.20.37.38:55565')

    controller.run()

    # 等待以保持连接
    try:
        sio.wait()
    except KeyboardInterrupt:
        sio.disconnect()
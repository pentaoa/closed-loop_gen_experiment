import base64
from io import BytesIO
import requests
import os
import numpy as np
import json
from PIL import Image
import time
import pygame as pg
from client.pygame_utils_n import Model, View, Controller
import socketio
import shutil

pre_eeg_path = f'client\pre_eeg'
instant_eeg_path = f'client\instant_eeg'
instant_image_path = f'client\instant_image'
image_set_path = "img_set"

selected_channels = []
target_image = None


url = 'http://10.20.37.38:45565'

sio = socketio.Client()

# @sio.event
# def connect():
#     print('Connected to server')
#     time.sleep(1)

@sio.event
def experiment_ready():
    time.sleep(1)
    # 向服务器发送开始实验的信号
    send_url = f'{url}/experiment'
    requests.post(send_url)

def send_files_to_server(pre_eeg_path, url):
    files = []
    file_objects = []

    try:
        for filename in os.listdir(pre_eeg_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(pre_eeg_path, filename)
                f = open(file_path, 'rb')
                file_objects.append(f)
                files.append(('files', (filename, f, 'application/octet-stream')))

        response = requests.post(url, files=files)
        print("Files sent successfully")
    finally:
        # 确保所有文件在请求完成后被关闭
        for f in file_objects:
            f.close()


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
    time.sleep(2)
    controller.start_pre_experiment(image_set_path, pre_eeg_path)
    print('Start data sending')
    # 发送 pre_eeg_path 中的所有 npy 文件到服务器
    send_url = f'{url}/pre_experiment_eeg_upload'

    send_files_to_server(pre_eeg_path, send_url)

@sio.event
def image_for_collection(data):
    os.makedirs(instant_image_path, exist_ok=True)
    os.makedirs(instant_eeg_path, exist_ok=True)
    # 删除 instant_image_path 和 instant_eeg_path 中的所有文件
    shutil.rmtree(instant_image_path)
    shutil.rmtree(instant_eeg_path)    
    print('Images received')
    images = data['images']
    for idx, encoded_string in enumerate(images):
        image_data = base64.b64decode(encoded_string)
        image = Image.open(BytesIO(image_data))
        # 保存图像到 client/instant_image 目录下
        image_save_path = os.path.join(instant_image_path, f'image_{idx}.png')
        os.makedirs(instant_image_path, exist_ok=True)
        image.save(image_save_path)
        print(f'Image saved to {image_save_path}')
    
    print('All images saved')

    # 启动实验
    controller.start_collection(instant_image_path, instant_eeg_path)
    
    # 发送 instant_eeg_path 中的所有 npy 文件到服务器
    send_url = f'{url}/instant_eeg_upload'
    send_files_to_server(instant_eeg_path, send_url)

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


    sio.connect(url)

    controller.run()

    # 等待以保持连接
    try:
        sio.wait()
    except KeyboardInterrupt:
        sio.disconnect()
import requests
import numpy as np
import json
import time
import pygame as pg
from pygame_utils import Model, View, Controller
from socketIO_client import SocketIO

pre_eeg_path = f'client/pre_eeg'
instant_eeg_path = f'client/instant_eeg'
image_set_path = '/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images'

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
    controller.upload_pre_eeg(pre_eeg_path)


def on_experiment_ready(controller):
    # 启动实验
    controller.start_experiment()

if __name__ == '__main__':
    model = Model()
    view = View()
    controller = Controller(model, view)

    controller.run()

    # 连接到服务器，并注册事件处理程序
    socketIO = SocketIO('10.20.37.38', 55565, LoggingNamespace)
    socketIO.on('connect', on_connect(view))
    socketIO.on('connect_error', on_connect_error)
    socketIO.on('pre_experiment_ready', on_pre_experiment_ready(controller))
    socketIO.on('experiment_ready', on_experiment_ready(controller))
    socketIO.on('message', on_message)

    # 保持 WebSocket 连接
    socketIO.wait()
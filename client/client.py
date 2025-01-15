import requests
import numpy as np
import json
import time
import pygame as pg
from pygame_utils import Model, View, Controller
from socketIO_client import SocketIO

pre_eeg_path = f'client/pre_eeg'
instant_eeg_path = f'client/instant_eeg'


def on_pre_experiment_ready(controller):
    # 启动预试验
    controller.model.set_phase('pre_experiment_waiting')
    time.sleep(100)

    response = requests.post('http://<server-ip>:55565/main_experiment_loop', json={'eeg_data': eeg_data})
    if response.status_code == 200:
        print(response.json()["message"])
    else:
        print(f"Failed to start main experiment: {response.status_code}")


def on_connect(view):
    print('Connected to server')
    view.display_text('Connected to server')


def on_connect_error(view):
    print('Failed to connect to server')
    view.display_text('Failed to connect to server')
    time.sleep(5)
    pg.quit()
    quit()


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
    socketIO.on('message', on_message)

    # 保持 WebSocket 连接
    socketIO.wait()
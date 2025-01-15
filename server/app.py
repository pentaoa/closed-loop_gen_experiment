from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import os
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# 用于存储实验状态
experiment_status = {"ready": False}

@app.route('/')
def home():
    return "Hello, World!"

def long_running_task(data):
    # 模拟长时间运行的任务
    time.sleep(20)
    # 任务完成后更新实验状态并通知客户端
    experiment_status["ready"] = True
    socketio.emit('experiment_ready', {'message': 'Experiment is ready'})

@app.route('/pre_experiment', methods=['POST'])
def pre_experiment():
    if request.method == 'POST':
        data = request.json
        # 启动一个线程来处理长时间运行的任务
        threading.Thread(target=long_running_task, args=(data,)).start()
        return jsonify({"message": "Pre-experiment started"})
    else:
        return "This is a GET request"

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    return 

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=55565)
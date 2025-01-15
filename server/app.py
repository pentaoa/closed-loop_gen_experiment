from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import os
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

pre_eeg_path = 'server/pre_eeg'


@app.route('/')
def home():
    return "Hello, World!"

def long_running_task(data):
    # 模拟长时间运行的任务
    time.sleep(20)
    # 任务完成后更新实验状态并通知客户端
    experiment_status["ready"] = True
    socketio.emit('experiment_ready', {'message': 'Experiment is ready'})

@app.route('/pre_experiment_eeg_upload', methods=['POST'])
def pre_experiment():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    if file:
        filename = file.filename
        save_path = os.path.join('server/pre_eeg', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        return jsonify({"message": f"File {filename} uploaded successfully"}), 200
        # TODO: 预处理 EEG 数据，返回显著通道和目标图片



@socketio.on('connect')
def handle_connect():
    print('Client connected')
    return 

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=55565)
    SocketIO.emit('pre_experiment_ready', {'message': 'Pre-experiment is ready'})
from neuracle_api import DataServerThread
import time
import numpy as np
import csv


def save_matrix_to_csv(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(matrix)

def main():
    # 初始化 DataServerThread 线程
    sample_rate = 1000
    t_buffer = 80
    thread_data_server = DataServerThread(sample_rate, t_buffer)
    # 建立TCP/IP连接
    notconnect = thread_data_server.connect(hostname='127.0.0.1', port=8712)
    if notconnect:
        raise TypeError("Can't connect JellyFish, Please open the hostport ")
    else:
        # meta包还没解析好就等待
        while not thread_data_server.isReady():
            time.sleep(1)
            continue
        # 启动线程
        thread_data_server.start()
        print('Data server start')
    # 在线数据获取演示：每隔一秒钟，获取数据（数据长度 = time_buffer * srate）
    # N, flagstop = 0, False
    # while not flagstop:  # get data in one second step
    #     nUpdate = thread_data_server.GetDataLenCount()
    #     # print('nUpdate', nUpdate)
    #     if nUpdate >= sample_rate:
    #         N += 1
    #         data = thread_data_server.GetBufferData()
    #         # print('data', data.shape)
    #         thread_data_server.ResetDataLenCount()
    #
    #         # print('idx', idx)
    #         # save_matrix_to_csv(np.transpose(data), f'../csv_data/raw20230926_{2}.csv')
    #         time.sleep(1)
    #     if N > 40:
    #         flagstop = True
    # 结束接收数据
    time.sleep(60)
    thread_data_server.stop()
    data = thread_data_server.GetBufferData()
    triggerChan = data[-1, :]
    idx = np.argwhere(triggerChan > 0)
    print('triggerChan', triggerChan[idx])
    print("#################")
    save_matrix_to_csv(np.transpose(data), f'../csv_data/raw20230927_{3}.csv')

    # print('triggerChan', triggerChan[idx])


if __name__ == '__main__':
    main()

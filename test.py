import numpy as np
path = '/mnt/dataset0/xkp/closed-loop/server/outputs_1/gaoyiou/psd/all_viewed_image_rewards.npy'
path = '/mnt/dataset0/xkp/closed-loop/server/outputs_1/gaoyiou/psd/fusion_image_ratings.npy'
path = '/mnt/dataset0/xkp/closed-loop/server/outputs_1/gaoyiou/rating/greedy_image_ratings.npy'

data= np.load(path, allow_pickle=True)

print(data.shape)

print(data)
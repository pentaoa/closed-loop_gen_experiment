import numpy as np

path = "/mnt/dataset0/xkp/closed-loop/server/outputs/heuristic_generation/psd/viewed_image_paths.npy"

data = np.load(path, allow_pickle=True)
print(data)

        
    
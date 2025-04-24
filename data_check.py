import numpy as np 

data_path = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\labels.npy'

data = np.load(data_path)

print(data.shape)
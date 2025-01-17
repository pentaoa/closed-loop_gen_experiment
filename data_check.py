import numpy as np 

data_path = r'C:\Users\ncclab\Documents\GitHub\closed-loop\client\pre_eeg\original\20250117-225749.npy'

data = np.load(data_path)

print(data.shape)
if np.all(data == 0):
    print("The entire array is zero.")
else:
    print("The array contains non-zero elements.")
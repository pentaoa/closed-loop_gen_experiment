import numpy as np
import os
import torch
from mne.time_frequency import psd_array_multitaper
from sklearn.metrics.pairwise import cosine_similarity

gene_path = '/mnt/dataset0/jiahua/eeg_encoding/results/sub-06/synthetic_eeg_data/encoding-end_to_end/dnn-alexnet/modeled_time_points-all/pretrained-True/lr-1e-05__wd-0e+00__bs-064/gene_eeg'

file_list = [os.path.join(gene_path, f) for f in sorted(os.listdir(gene_path))]

data = np.array([np.load(file) for file in file_list])  # (200, 17, 250)
selected_channel_idxes = [3, 4, 5]
selected_data = data[:,selected_channel_idxes, :]
fs = 250

psds = []
for i in range(selected_data.shape[0]):
    psd, _ = psd_array_multitaper(selected_data[i,:,:], fs, adaptive=True, normalization='full', verbose=0)
    psd = psd.flatten()
    psds.append(psd)
psds = np.array(psds)
psds = torch.from_numpy(psds)

similarity_matrix = cosine_similarity(psds)
np.fill_diagonal(similarity_matrix, np.nan)
mean_similarity = np.nanmean(similarity_matrix, axis=1)
lowest_indexes = np.argsort(mean_similarity)[:3]
print(lowest_indexes, mean_similarity[lowest_indexes])

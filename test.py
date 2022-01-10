from __future__ import division
import torch
import numpy as np
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import load_metr_la_rdata, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,test_error
import random
import pandas as pd
from basic_structure import D_GCN, C_GCN, K_GCN,IGNNK
import geopandas as gp
import matplotlib as mlt


n_o_n_m = 150 #sampled space dimension

h = 24 #sampled time dimension

z = 100 #hidden dimension for graph convolution

K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1

n_m = 50 #number of mask node during training

N_u = 50 #target locations, N_u locations will be deleted from the training data

Max_episode = 750 #max training episode

learning_rate = 0.0001 #the learning_rate for Adam optimizer

E_maxvalue = 80 #the max value from experience

batch_size = 4


STmodel = IGNNK(h, z, K)  # The graph neural networks


A, X = load_metr_la_rdata()

split_line1 = int(X.shape[2] * 0.7)

training_set = X[:, 0, :split_line1].transpose()

test_set = X[:, 0, split_line1:].transpose()       # split the training and test period

rand = np.random.RandomState(0) # Fixed random output, just an example when seed = 0.
unknow_set = rand.choice(list(range(0,X.shape[0])),N_u,replace=False)
unknow_set = set(unknow_set)
full_set = set(range(0,207))
know_set = full_set - unknow_set
training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                   # the adjacent matrix are based on pairwise distance,


                                                   # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix


criterion = nn.MSELoss()
optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
RMSE_list = []
MAE_list = []
MAPE_list = []
for epoch in range(Max_episode):
    for i in range(training_set.shape[0] // (h * batch_size)):  # using time_length as reference to record test_error
        t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
        know_mask = set(random.sample(range(0, training_set_s.shape[1]), n_o_n_m))  # sample n_o + n_m nodes
        feed_batch = []
        for j in range(batch_size):
            feed_batch.append(
                training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)])  # generate 8 time batches

        inputs = np.array(feed_batch)
        inputs_omask = np.ones(np.shape(inputs))
        inputs_omask[
            inputs == 0] = 0  # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
        # For other datasets, it is not necessary to mask 0 values

        missing_index = np.ones((inputs.shape))
        for j in range(batch_size):
            missing_mask = random.sample(range(0, n_o_n_m), n_m)  # Masked locations
            missing_index[j, :, missing_mask] = 0

        Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue  # normalize the value according to experience
        Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
        mask = torch.from_numpy(
            inputs_omask.astype('float32'))  # The reconstruction errors on irregular 0s are not used for training

        A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]  # Obtain the dynamic adjacent matrix
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))

        outputs = torch.from_numpy(inputs / E_maxvalue)  # The label

        optimizer.zero_grad()
        X_res = STmodel(Mf_inputs, A_q, A_h)  # Obtain the reconstruction

        loss = criterion(X_res * mask, outputs * mask)
        loss.backward()
        optimizer.step()  # Errors backward

    MAE_t, RMSE_t, MAPE_t, metr_ignnk_res = test_error(STmodel, unknow_set, test_set, A, E_maxvalue, True)
    RMSE_list.append(RMSE_t)
    MAE_list.append(MAE_t)
    MAPE_list.append(MAPE_t)
    if epoch % 50 == 0:
        print(epoch, MAE_t, RMSE_t, MAPE_t)
idx = MAE_list == min(MAE_list)
print('Best model result:', np.array(MAE_list)[idx], np.array(RMSE_list)[idx], np.array(MAPE_list)[idx])
# torch.save(STmodel.state_dict(), 'model/IGNNK.pth') # Save the model


fig,ax = plt.subplots()
ax.plot(RMSE_list,label='RMSE_on_test_set',linewidth=3.5)
ax.set_xlabel('Training Batch (x249)',fontsize=20)
ax.set_ylabel('RMSE',fontsize=20)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('fig/ignnk_learning_curve_metr-la.pdf')


print('debug')
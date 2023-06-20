import sys, os
sys.path.insert(0,os.path.abspath(os.path.join('..', 'utils')))
sys.path.insert(0,os.path.abspath(os.path.join('..', 'model')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from prepare_data import prepare_input
from class_loss import MVELoss
from class_GCN import GCN_LSTM

# Define hyperparameters
n_features = 2
n_stations = 63
hid_g = 64
hid_fc = 64
hid_l = 64
meanonly = False
homo = 0
batch_size = 32
learning_rate = 0.001
weight_decay = 0.001
dropout = 0.2
num_epochs = 10
nadj = ['euc','con','func']
dist = 'tnorm'

# Define other parameters
SAVEPATH = "../weights/model.pt"
xfile = '../data/processed/0620x.npy'
yfile = '../data/processed/0620y.npy'
adjfile = '../data/processed/adjlist.csv'
historyfile = '../data/processed/history.npy'
weatherfile = '../data/processed/weather.npy'
losfile = '../data/processed/los.npy'


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Create the network
net = GCN_LSTM(n_features,n_stations,hid_g,hid_fc,hid_l,meanonly,homo,nadj,device,dropout)
net = net.to(device)

# Define the loss function
loss_fn = MVELoss(dist)

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Read in and prepare dataset
x = np.load(xfile)
y = np.load(yfile)
adj = pd.read_csv(adjfile)
history = np.load(historyfile)
weather = np.load(weatherfile)
los = np.load(losfile)

train_loader,val_loader,test_loader,adj_torch = prepare_input(x,y,adj,nadj,history,weather,los,device,batch_size=batch_size)
print('Start training ...')
print(f"Training sample batches:{len(train_loader)}")

# Training loop
loss_history = []
for epoch in range(num_epochs):

    running_loss = 0.0
    
    # Iterate over the batches
    for i,traindata in enumerate(train_loader):

        net.train()

        batch_x, batch_y, batch_history, batch_weather, batch_los = traindata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()
        batch_history = batch_history.float()

        batch_x, batch_y, batch_history, batch_weather, batch_los = \
                batch_x.to(device), batch_y.to(device), batch_history.to(device), \
                batch_weather.to(device), batch_los.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(batch_x, adj_torch, batch_history, batch_weather, batch_los, device)
        
        # Compute the loss
        if outputs.shape[0]!=2*batch_size:
            batch_new = int((outputs.shape[0])/2)
            output_loc = outputs[:batch_new,:]
            output_scale = outputs[batch_new:,:]
        else:
            output_loc = outputs[:batch_size,:]
            output_scale = outputs[batch_size:,:]
        loss = loss_fn(output_loc, output_scale, batch_y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Save best model
    if epoch == 0:
        loss_best = running_loss
    else:
        if running_loss < loss_best:
            loss_best = running_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, SAVEPATH)

    # Print the average loss for the epoch and evaluate
    print('Ep:', epoch, 'Loss:',running_loss)

    # Evaluate every n epoches
    # if epoch%5 == 0:
    #     print('Ep:', i, 'Loss:',running_loss)
    #     net.eval()
    #     # TODO: evaluation
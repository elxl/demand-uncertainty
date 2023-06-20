import torch
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sys.path.append('../model/')
sys.path.append('../utils/')
from prepare_data import prepare_input

# Define hyperparameters
input_size = 10
hidden_size = 20
output_size = 5
batch_size = 32
learning_rate = 0.001
num_epochs = 10
nadj = ['euc','con','func']

# Define other parameters
xfile = '../data/processed/0620x.npy'
yfile = '../data/processed/0620y.npy'
adjfile = '../data/processed/adjlist.csv'
historyfile = '../data/processed/history.npy'
weatherfile = '../data/processed/weather.np'
losfile = '../data/processed/los.npy'


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Create the network
net = GCN_LSTM(input_size, hidden_size, output_size)
net = net.to(device)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Read in and prepare dataset
x = np.load(xfile)
y = np.load(yfile)
adj = pd.read_csv(adjfile)
history = np.load(historyfile)
weather = np.load(weatherfile)
los = np.load(losfile)

train_loader,val_loader,test_loader,adj = prepare_input(x,y,adj,nadj,history,weather,los,device)

# Training loop
for epoch in range(num_epochs):

    running_loss = 0.0
    
    # Iterate over the batches
    for i,traindata in enumerate(dataloader):

        inputs, targets = traindata

        net.train()

        # Forward pass
        outputs = net(inputs)
        
        # Compute the loss
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the average loss for the epoch
    if i%5 == 0:

        net.eval()
        # TODO: evaluation
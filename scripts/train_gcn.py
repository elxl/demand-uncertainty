import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('../model')
from class_GCN import GCN_LSTM

# Define hyperparameters
input_size = 10
hidden_size = 20
output_size = 5
batch_size = 32
learning_rate = 0.001
num_epochs = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Create the network
net = GCN_LSTM(input_size, hidden_size, output_size)
net = net.to(device)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Create an instance of your custom dataset
dataset = MyDataset(input_data, target_data)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
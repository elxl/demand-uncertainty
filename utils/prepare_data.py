import torch
import libpysal
import scipy
import numpy as np
import pandas as pd
import pickle as pkl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .class_dataset import NY_Data, NY_All

    
def prepare_input(x, y, adj, nadj, history, weather, los, device, train=0.8, val=0.1, test=0.1, random=False, batch_size = 32):

    #----------Input--------------#
    
    dataset = NY_Data(x, y, history, weather, los)
    num_samples = len(x)
    num_train = int(train * num_samples)
    num_val = int(val * num_samples)
    num_test = num_samples - num_train - num_val

    # Create subsets for train, validation, and test
    if random:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42)
        )
    else:
        indices = torch.arange(num_samples)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train+num_val]
        test_indices = indices[num_train+num_val:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset,val_indices)
        test_dataset = Subset(dataset,test_indices)

    # Create separate DataLoaders for each set

    if num_train!=0:
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
      train_loader = None
    if num_val!=0:
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
      val_loader = None
    if num_test!=0:
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
      test_loader = None


    #-----------Adjancent Matirx----------#
    # Prepare adjancent matrix
    w = {}
    for i in nadj:
        if i in adj.columns:
            w[i] = libpysal.weights.W.from_adjlist(adj,focal_col='start_id',neighbor_col='end_id',weight_col=i).full()[0]
        else:
            pass

    # Filter to common stations only and calculate degree matrix and adj matrix for models
    stations = x.shape[2]
    deg = {}
    adj = {}
    
    for temp in w.keys():
        w[temp] = w[temp] / np.max(w[temp])
        deg[temp] = np.sum(w[temp]+np.identity(stations), axis=0)
        adj[temp] = np.matmul(np.matmul(scipy.linalg.fractional_matrix_power(np.diag(deg[temp]), -0.5), 
            w[temp]+np.identity(len(deg[temp]))),
            scipy.linalg.fractional_matrix_power(np.diag(deg[temp]), 0.5))

    adj_torch = torch.tensor([])
    for t in nadj:
        if t in adj.keys():
            adj_torch = torch.cat((adj_torch, torch.Tensor(adj[t][:,:,np.newaxis])),dim=2)
        else:
            pass
    adj_torch = adj_torch.to(device)

    return train_loader,val_loader,test_loader,adj_torch

def prepare_input_LSTM(x, y, train=0.8, val=0.1, test=0.1, batch_size=32, random=False):
   # Split dataset
    dataset = NY_All(x, y)
    num_samples = len(x)
    num_train = int(train * num_samples)
    num_val = int(val * num_samples)
    num_test = num_samples - num_train - num_val

    # Create subsets for train, validation, and test
    if random:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(42)
        )
    else:
        indices = torch.arange(num_samples)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train+num_val]
        test_indices = indices[num_train+num_val:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset,val_indices)
        test_dataset = Subset(dataset,test_indices)

    # Create separate DataLoaders for each set

    if num_train!=0:
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
      train_loader = None
    if num_val!=0:
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
      val_loader = None
    if num_test!=0:
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
      test_loader = None

    return train_loader, val_loader, test_loader
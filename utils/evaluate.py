import torch
import torch.nn as nn
import numpy as np
from scipy.stats import poisson, norm, laplace, lognorm, nbinom
from scipy.optimize import curve_fit
from torch_geometric.nn.models.mlp import NoneType
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError


def fit_dist(data, dist):

    if dist == 'norm':
        mean = np.mean(data,axis=2)
        std = np.std(data,axis=2)
        res = [mean, std]
    elif dist == 'poisson':
        def poisson_fit(x):
            # Define the Poisson distribution function
            def poisson_dist(x, lamb):
                return poisson.pmf(x, lamb)

            # Fit the data to the Poisson distribution
            params, _ = curve_fit(poisson_dist, range(np.max(x)+1), np.bincount(x))

            # Retrieve the fitted lambda parameter
            fitted_lambda = params[0]

            return fitted_lambda
        res = np.apply_along_axis(poisson_fit,axis=2,arr=data)

    return res




def post_process_dist(dist, loc, scale, pi=None):
    
    if dist == "lognorm":
        out_predict = np.exp(loc - np.power(scale,2))
#         out_std = np.mean(np.sqrt((np.exp(scale * scale)-1)*(np.exp(2*loc+scale * scale))))
    elif dist == 'tnorm':
        out_predict = loc
#         out_std = np.mean(scale)
    elif dist == 'laplace':
        out_predict = loc
#         out_std = np.mean(scale) * np.sqrt(2)
    elif dist == 'poisson':
        out_predict = loc
#         out_std = np.sqrt(loc)
    elif dist == 'norm':
        out_predict = loc
    elif dist == 'nb':
        out_predict = loc*scale
    elif dist == 'zinb':
        pass
        
        
    return out_predict
 
    
def post_process_pi(dist, loc, scale, z):
    
    if dist == "lognorm":
        s = np.exp(loc)
        shape = scale
        lognorm_dist = lognorm(s=shape, scale=s)
        lb, ub = lognorm_dist.interval(z)
        # lb, ub = lognorm.interval(z, loc, scale)
    elif dist == 'tnorm':
        lb, ub = norm.interval(z, loc, scale)
        lb = lb * (lb>0)
    elif dist == 'laplace':
        lb, ub = laplace.interval(z, loc, scale)
    elif dist == 'poisson':
        lb,ub = poisson.interval(z, loc) 
    elif dist == 'norm':
        lb, ub = norm.interval(z, loc, scale)
    elif dist == 'nb':
        lb, ub = nbinom.interval(z, loc, scale)
    elif dist == 'zinb':
        pass

    return lb, ub

def eval_pi(output_lower, output_upper, target, stdout=False):
    lower = output_lower.flatten()
    lower[lower<0] = 0
    upper = output_upper.flatten()
    t = target.flatten()
    kh = (np.sign(upper-t) >= 0) * (np.sign(t-lower) >= 0)
    picp = np.mean(kh)
    mpiw = np.sum((upper-lower) * kh) / np.sum(kh)
    
    if stdout:
        print("MPIW: %.6f" %(mpiw))
        print("PICP: %.6f"%(picp))

    return mpiw, picp

def evaluate(net, loss_fn, adj_torch, dist, dataloader, z, device, batch_size):

    net.eval()
    eval_loss = 0
    eval_num = len(dataloader.dataset)
    for i,evaldata in enumerate(dataloader):
        batch_x, batch_y, batch_history, batch_weather, batch_los = evaldata

        batch_x = batch_x.float()
        batch_y = torch.squeeze(batch_y).float()
        batch_history = batch_history.float()

        batch_x, batch_y, batch_history, batch_weather, batch_los = \
            batch_x.to(device), batch_y.to(device), batch_history.to(device), \
            batch_weather.to(device), batch_los.to(device)

        outputs = net(batch_x, adj_torch, batch_history, batch_weather, batch_los, device)
        if outputs.shape[0]!=(batch_size + (1-net.meanonly)*batch_size + net.zinflate*batch_size):
            if net.meanonly:
                output_loc = outputs[:,:]
                output_scale = None
                output_pi = None
            elif net.zinflate == 0:
                batch_new = int((outputs.shape[0])/2)
                output_loc = outputs[:batch_new,:]
                output_scale = outputs[batch_new:,:]
                output_pi = None
            else:
                batch_new = int((outputs.shape[0])/3)
                output_loc = outputs[:batch_new,:]
                output_scale = outputs[batch_new:2*batch_new,:]
                output_pi = outputs[2*batch_new:,:]           
        else:
            if net.meanonly:
                output_loc = outputs[:,:]
                output_scale = None
                output_pi = None             
            if net.zinflate == 0:
                output_loc = outputs[:batch_size,:]
                output_scale = outputs[batch_size:,:]
                output_pi = None
            else:
                output_loc = outputs[:batch_size,:]
                output_scale = outputs[batch_size:2*batch_size,:]
                output_pi = outputs[2*batch_size:,:]
        loss = loss_fn(output_loc, output_scale, batch_y,output_pi)
        
        eval_loss += loss.item()

        # Get mean and variance
        if i == 0:
            if net.meanonly:
                test_out_mean = outputs[:,:].cpu().detach().numpy()
                test_out_var = None
                test_out_pi = None                
            elif net.zinflate == 0:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
                test_out_var = outputs[batch_size:,:].cpu().detach().numpy()
                test_out_pi = None
            else:
                test_out_mean = outputs[:batch_size,:].cpu().detach().numpy()
                test_out_var = outputs[batch_size:2*batch_size,:].cpu().detach().numpy()
                test_out_pi = outputs[2*batch_size:,:].cpu().detach().numpy()     
            y_eval = batch_y.cpu().numpy()
        else:
            if outputs.shape[0]==(batch_size + (1-net.meanonly)*batch_size + net.zinflate*batch_size):
                if net.meanonly:
                    test_out_mean = np.concatenate((test_out_mean, outputs[:,:].cpu().detach().numpy()), axis=0)
                    test_out_var = None
                    test_out_pi = None
                elif net.zinflate == 0:
                    test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)
                    test_out_var = np.concatenate((test_out_var, outputs[batch_size:,:].cpu().detach().numpy()), axis=0)
                    test_out_pi = None
                else:
                    test_out_mean = np.concatenate((test_out_mean, outputs[:batch_size,:].cpu().detach().numpy()), axis=0)
                    test_out_var = np.concatenate((test_out_var, outputs[batch_size:2*batch_size,:].cpu().detach().numpy()), axis=0)
                    test_out_pi = np.concatenate((test_out_var, outputs[2*batch_size:,:].cpu().detach().numpy()), axis=0)                   
                y_eval = np.concatenate((y_eval, batch_y.cpu().numpy()), axis=0)
            else:
                if net.meanonly:
                    test_out_mean = np.concatenate((test_out_mean, outputs[:,:].cpu().detach().numpy()), axis=0)
                    test_out_var = None
                    test_out_pi = None
                elif net.zinflate == 0:
                    batch_new = int((outputs.shape[0])/2)
                    test_out_mean = np.concatenate((test_out_mean, outputs[:batch_new,:].cpu().detach().numpy()), axis=0)
                    test_out_var = np.concatenate((test_out_var, outputs[batch_new:,:].cpu().detach().numpy()), axis=0)
                    test_out_pi = None
                else:
                    batch_new = int((outputs.shape[0])/3)
                    test_out_mean = np.concatenate((test_out_mean, outputs[:batch_new,:].cpu().detach().numpy()), axis=0)
                    test_out_var = np.concatenate((test_out_var, outputs[batch_new:2*batch_new,:].cpu().detach().numpy()), axis=0)
                    test_out_pi = np.concatenate((test_out_var, outputs[2*batch_new:,:].cpu().detach().numpy()), axis=0)
                y_eval = np.concatenate((y_eval, batch_y.cpu().numpy()), axis=0)        
    val_out_predict = post_process_dist(dist, test_out_mean, test_out_var, test_out_pi)
    # Point error
    mae = MeanAbsoluteError()
    val_mae = mae(torch.from_numpy(val_out_predict.flatten()), torch.from_numpy(y_eval.flatten()))
    val_mape = val_mae/np.mean(y_eval.flatten())

    lb, ub = post_process_pi(dist, test_out_mean, test_out_var, z)
    val_mpiw, val_picp = eval_pi(lb, ub, y_eval)
        
    return eval_loss/eval_num, val_mae.item(), val_mape.item(), val_mpiw, val_picp

def evaluate_output(out_mean, out_std, out_pi, out_true, loss_fn, dist, z):

    num = out_mean.shape[1]
    loss = loss_fn(out_mean, out_std, out_true, out_pi)
    loss = loss.item()

    out_predict = post_process_dist(dist, out_mean, out_std, out_pi)

    # convert data format
    out_mean = out_mean.cpu().numpy()
    if dist == 'norm':
        out_std = out_std.cpu().numpy()
    elif dist == 'zinb':
        out_std = out_std.cpu().numpy()
        out_pi = out_pi.cpu().numpy()
    elif dist == 'poisson':
        out_std = None
        out_pi = None
    else:
        print("Distribution not implemented!")

    out_true = out_true.cpu().numpy()
    out_predict = out_predict.cpu().numpy()

    mae = MeanAbsoluteError()
    predict_mae = mae(torch.from_numpy(out_predict.flatten()), torch.from_numpy(out_true.flatten()))
    predict_mape = predict_mae/np.mean(out_true.flatten())

    lb, ub = post_process_pi(dist, out_mean, out_std, z)
    val_mpiw, val_picp = eval_pi(lb, ub, out_true)
        
    return loss/num, predict_mae.item(), predict_mape.item(), val_mpiw, val_picp   
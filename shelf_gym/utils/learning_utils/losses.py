import numpy as np
import torch
from torch import nn 
from glob import glob 
import pdb
from torch.utils.data import Dataset, DataLoader


l2_loss = torch.nn.MSELoss(reduction = 'none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




"""
Evidential Loss is derived from this paper: Sensoy, M., Kaplan, L., & Kandemir, M. (2018). 
Evidential Deep Learning to Quantify Classification Uncertainty. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
 & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 31). 
"""

def occupancy_loss_3D(pred,gt,k = 0.2):
    # pdb.set_trace()
    alpha = pred[:,::2,:,:]
    beta = pred[:,1::2,:,:]
    S = alpha+beta
    violation1 = (alpha<0).sum() + (beta < 0).sum()   
    if(violation1 > 0):
        print('alpha beta violation')
        pdb.set_trace()
    prob = alpha/(alpha+beta)

    inv_prob = 1-prob
    violation2 = (prob.float()<0).sum() + (prob.float()>1).sum() + (gt.float()>1).sum() + (gt.float()<0).sum() + torch.isnan(prob).sum()
    if(violation2 > 0):
        print('prob violation')
        pdb.set_trace()
    # sum of square losses
    l1 = l2_loss(prob.float(),gt.float())
    l2 = torch.mean((prob*(1-prob)/(S+1)))
    l3 = torch.mean(inv_prob*(1-inv_prob)/(S+1))
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = 2)
    beta_null = torch.distributions.beta.Beta(1,1,validate_args = False)
    beta_pred = torch.distributions.beta.Beta(alpha-one_hot_gt[:,:,:,:,1],beta-one_hot_gt[:,:,:,:,0],validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)

    reg_loss = torch.distributions.kl.kl_divergence(beta_pred,beta_null).sum()/beta_pred.flatten().shape[0]
    final_loss = l1+l2+l3+k*reg_loss
    
    return final_loss



def evidential_semantic_loss_3D(alpha,gt,k = 0.2):
    # pdb.set_trace()

    S = alpha.sum(axis = 1,keepdims = True)
    prob = alpha/S
    n_classes = alpha.shape[1]
    inv_prob = 1-prob
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,3,1,2]).to(device)
    # pdb.set_trace()
    # alpha_tilde = one_hot_gt +  torch.clamp((alpha-one_hot_gt),0.01)
    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,1])
    # sum of square losses
    l1 = torch.sum(l2_loss(prob.float(),one_hot_gt.float()),dim =1)
    l2 = torch.sum(prob*(inv_prob)/(S+1),dim = 1)
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)

    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean(l1+l2+k*reg_loss)
    
    return final_loss

def evidential_occupancy_loss_3D(alpha,gt,k = 0.2,n_classes = 2):
    # pdb.set_trace()

    S = alpha.sum(axis = 1,keepdims = True)
    prob = alpha/S

    inv_prob = 1-prob
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,4,1,2,3]).to(device)
    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,4,1])
    # sum of square losses
    # pdb.set_trace()
    l1 = torch.sum(l2_loss(prob.float(),one_hot_gt.float()),dim =1)
    l2 = torch.sum(prob*(inv_prob)/(S+1),dim = 1)
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)
    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean(l1+l2+k*reg_loss)
    return final_loss


def focused_evidential_semantic_loss_3D(alpha,gt,difference,focus_factor=5,k = 0.2,n_classes = 15):
    # pdb.set_trace()
    focus_multiplier = difference*(focus_factor-1)+1
    S = alpha.sum(axis = 1,keepdims = True)
    prob = alpha/S

    inv_prob = 1-prob
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,3,1,2]).to(device)
    # pdb.set_trace()
    # alpha_tilde = one_hot_gt +  torch.clamp((alpha-one_hot_gt),0.01)
    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,1])
    # sum of square losses
    l1 = torch.sum(l2_loss(prob.float(),one_hot_gt.float()),dim =1)
    l2 = torch.sum(prob*(inv_prob)/(S+1),dim = 1)
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)

    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean((l1+l2+k*reg_loss)*focus_multiplier)
    
    return final_loss

def focused_evidential_semantic_crossentropy(alpha,gt,difference,focus_factor=1,n_classes = 15,k=0.2,focus_only=False):
    if(focus_only):
        focus_multiplier =  difference*focus_factor
    else:
        focus_multiplier = difference*(focus_factor-1) + 1
    S = alpha.sum(axis = 1,keepdims = True)

    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,3,1,2]).to(device)
    l1 = (one_hot_gt*(torch.digamma(S)-torch.digamma(alpha))).sum(axis = 1)

    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,1])
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)

    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean((l1+k*reg_loss)*focus_multiplier)
    return final_loss


def evidential_semantic_crossentropy(alpha,gt,n_classes = 2,k=0.2):
    S = alpha.sum(axis = 1,keepdims = True)

    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,3,1,2]).to(device)
    l1 = (one_hot_gt*(torch.digamma(S)-torch.digamma(alpha))).sum(axis = 1)

    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,1])
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)

    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean(l1+k*reg_loss)
    return final_loss


def focused_evidential_occupancy_crossentropy(alpha,gt,difference,focus_factor=1,n_classes = 2,k=0.2,focus_only = False):
    if(focus_only):
        focus_multiplier =  difference*focus_factor
    else:
        focus_multiplier = difference*(focus_factor-1) + 1
    S = alpha.sum(axis = 1,keepdims = True)
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,4,1,2,3]).to(device)
    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,4,1])
    l1 = (one_hot_gt*(torch.digamma(S)-torch.digamma(alpha))).sum(axis = 1)

    # sum of square losses
    # pdb.set_trace()
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)
    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean((l1+k*reg_loss)*focus_multiplier.unsqueeze(1))
    return final_loss

def evidential_occupancy_crossentropy(alpha,gt,n_classes = 2,k=0.2):

    S = alpha.sum(axis = 1,keepdims = True)
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,4,1,2,3]).to(device)
    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,4,1])
    l1 = (one_hot_gt*(torch.digamma(S)-torch.digamma(alpha))).sum(axis = 1)

    # sum of square losses
    # pdb.set_trace()
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)
    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean(l1+k*reg_loss)
    return final_loss


def focused_evidential_occupancy_loss_3D(alpha,gt,difference,focus_factor=5,k = 0.2,n_classes = 2):
    # pdb.set_trace()
    focus_multiplier =  difference*(focus_factor-1)+1

    S = alpha.sum(axis = 1,keepdims = True)
    prob = alpha/S

    inv_prob = 1-prob
    one_hot_gt = torch.nn.functional.one_hot(gt.long(),num_classes = n_classes).permute([0,4,1,2,3]).to(device)
    alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
    alpha_tilde = alpha_tilde.permute([0,2,3,4,1])
    # sum of square losses
    # pdb.set_trace()
    l1 = torch.sum(l2_loss(prob.float(),one_hot_gt.float()),dim =1)
    l2 = torch.sum(prob*(inv_prob)/(S+1),dim = 1)
    dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
    dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
    # beta_pred_2 = torch.distributions.beta.Beta(alpha,beta,validate_args = True)
    reg_loss = torch.distributions.kl.kl_divergence(dir_pred,dir_null)
    final_loss = torch.mean((l1+l2+k*reg_loss)*focus_multiplier.unsqueeze(1))
    return final_loss
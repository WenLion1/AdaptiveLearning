#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:27:27 2022

@author: ningmei
"""
import os,torch,torchvision,gc,warnings

import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from PIL import Image as pil_image
from itertools import product
from time import sleep

from torch.utils.data import Dataset,DataLoader
from torch import nn
from torchvision import models as Tmodels
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.utils import shuffle

from joblib import Parallel,delayed

from matplotlib import pyplot as plt

from typing import List, Callable, Union, Any, TypeVar, Tuple
###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################
torch.manual_seed(12345)
np.random.seed(12345)

##############################################################################
def noise_func(x:Tensor,noise_level:float = 0.):
    """
    add guassian noise to the images during agumentation procedures
    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, level of noise, between 0 and 1
    """
    
    generator   = torch.distributions.Normal(x.mean(),x.std())
    noise       = generator.sample(x.shape)
    new_x       = x * (1 - noise_level) + noise * noise_level
    new_x       = torch.clamp(new_x,x.min(),x.max(),)
    return new_x

def concatenate_transform_steps(image_resize:int = 128,
                                num_output_channels:int = 3,
                                noise_level:float = 0.,
                                flip:bool = False,
                                rotate:float = 0.,
                                fill_empty_space:int = 255,
                                grayscale:bool = True,
                                center_crop:bool = False,
                                center_crop_size:Tuple = (1200,1200),
                                ):
    """
    from image to tensors

    Parameters
    ----------
    image_resize : int, optional
        DESCRIPTION. The default is 128.
    num_output_channels : int, optional
        DESCRIPTION. The default is 3.
    noise_level : float, optional
        DESCRIPTION. The default is 0..
    flip : bool, optional
        DESCRIPTION. The default is False.
    rotate : float, optional
        DESCRIPTION. The default is 0.,
    fill_empty_space : int, optional
        DESCRIPTION. The defaultis 130.
    grayscale: bool, optional
        DESCRIPTION. The default is True.
    center_crop : bool, optional
        DESCRIPTION. The default is False.
    center_crop_size : Tuple, optional
        DESCRIPTION. The default is (1200, 1200)

    Returns
    -------
    transformer_steps : TYPE
        DESCRIPTION.

    """
    transformer_steps = []
    # crop the image - for grid like layout
    if center_crop:
        transformer_steps.append(transforms.CenterCrop(center_crop_size))
    # resize
    transformer_steps.append(transforms.Resize((image_resize,image_resize)))
    # flip
    if flip:
        transformer_steps.append(transforms.RandomHorizontalFlip(p = .5))
        transformer_steps.append(transforms.RandomVerticalFlip(p = .5))
    # rotation
    if rotate > 0.:
        transformer_steps.append(transforms.RandomRotation(degrees = rotate,
                                                           fill = fill_empty_space,
                                                           ))
    # grayscale
    if grayscale:
        transformer_steps.append(# it needs to be 3 if we want to use pretrained CV models
                                transforms.Grayscale(num_output_channels = num_output_channels)
                                )
    # rescale to [0,1] from int8
    transformer_steps.append(transforms.ToTensor())
    # add noise
    if noise_level > 0:
        transformer_steps.append(transforms.Lambda(lambda x:noise_func(x,noise_level)))
    # normalization
    transformer_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                             )
    transformer_steps = transforms.Compose(transformer_steps)
    return transformer_steps

def append_to_dict_list(df:pd.core.frame.DataFrame,
                        attribute,
                        variable,
                        ):
    try:
        idx_column = int(attribute[-1]) - 1
    except:
        idx_column = 0
    temp = variable.detach().cpu().numpy()[:,idx_column]
    [df[attribute].append(item) for item in temp]
    return df



#candidate models
def candidates(model_name:str,weights:str = 'IMAGENET1K_V1') -> nn.Module:
    """
    A simple loader for the CNN backbone models
    Parameters
    ----------
    model_name : str
        DESCRIPTION.
    weights : str
        DESCRIPTION.
    Returns
    -------
    nn.Module
        A pretrained CNN model.
    """
    picked_models = dict(
            # resnet18        = Tmodels.resnet18(weights              = "IMAGENET1K_V1",
            #                                    progress             = False,),
            alexnet         = Tmodels.alexnet(weights               = weights,
                                              progress              = False,),
            # squeezenet      = Tmodels.squeezenet1_1(weights              = "IMAGENET1K_V1",
            #                                        progress         = False,),
            vgg19           = Tmodels.vgg19_bn(weights              = weights,
                                              progress              = False,),
            vgg16           = Tmodels.vgg16(weights                 = weights,
                                               progress             = False,),
            vgg11           = Tmodels.vgg11_bn(weights              = weights,
                                               progress             = False,),
            densenet169     = Tmodels.densenet169(weights           = weights,
                                                  progress          = False,),
            # inception       = Tmodels.inception_v3(weights              = "IMAGENET1K_V1",
            #                                       progress          = False,),
            # googlenet       = Tmodels.googlenet(weights              = "IMAGENET1K_V1",
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(weights              = "IMAGENET1K_V1",
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(weights          = weights,
                                                  progress          = False,),
            # mobilenet_v3_l  = Tmodels.mobilenet_v3_large(weights              = "IMAGENET1K_V1",
            #                                              progress   = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(weights              = "IMAGENET1K_V1",
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(weights              = weights,
                                              progress              = False,),
            resnext101      = Tmodels.resnext101_32x8d(weights      = weights,
                                                       progress     = False,),
            )
    return picked_models[model_name]

def define_type(model_name:str) -> str:
    """
    We define the type of the pretrained CNN models for easier transfer learning
    Parameters
    ----------
    model_name : str
        DESCRIPTION.
    Returns
    -------
    str
        DESCRIPTION.
    """
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            vgg16       = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            resnext101  = 'resnet',
            )
    return model_type[model_name]

def hidden_activation_functions(activation_func_name:str,num_parameters:int=3) -> nn.Module:
    """
    A simple loader for some of the nonlinear activation functions
    Parameters
    Parameters
    ----------
    activation_func_name : str
        DESCRIPTION.
    num_parameters : int
        I don't know how to use this yet.
    Returns
    -------
    nn.Module
        The activation function.
    """
    funcs = dict(relu       = nn.ReLU(),
                 selu       = nn.SELU(),
                 elu        = nn.ELU(),
                 celu       = nn.CELU(),
                 gelu       = nn.GELU(),
                 silu       = nn.SiLU(),
                 sigmoid    = nn.Sigmoid(),
                 tanh       = nn.Tanh(),
                 linear     = None,
                 leaky_relu = nn.LeakyReLU(),
                 hardshrink = nn.Hardshrink(lambd = .1),
                 softshrink = nn.Softshrink(lambd = .1),
                 tanhshrink = nn.Tanhshrink(),
                 # weight decay should not be used when learning aa for good performance.
                 prelu      = nn.PReLU(num_parameters=num_parameters,),
                 )
    return funcs[activation_func_name]

def compute_image_loss(image_loss_func:Callable,
                       image_category:Tensor,
                       labels:Tensor,
                       device:str,
                       n_noise:int      = 0,
                       num_classes:int  = 2,
                       ) -> Tensor:
    """
    Compute the loss of predicting the image categories
    Parameters
    ----------
    image_loss_func : Callable
        DESCRIPTION.
    image_category : Tensor
        DESCRIPTION.
    labels : Tensor
        DESCRIPTION.
    device : str
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    num_classes : int, optional
        DESCRIPTION. The default is 10.
    Returns
    -------
    image_loss: Tensor
        DESCRIPTION.
    """
    if "Binary Cross Entropy" in image_loss_func.__doc__:
        labels = labels.float()
        if n_noise > 0:
            noisy_labels    = torch.ones(labels.shape) * (1/num_classes)
            noisy_labels    = noisy_labels[:n_noise]
            labels          = torch.cat([labels.to(device),noisy_labels.to(device)])
        image_loss = image_loss_func(image_category.to(device),
                                     labels.view(image_category.shape).to(device)
                                     )
    elif "negative log likelihood loss" in image_loss_func.__doc__:
        idx                 = labels[:,0] != 0.5
        labels              = labels.argmax(1).long()
        if n_noise > 0:
            image_category  = image_category[:-n_noise]
            # print('should not use this loss function if use noise examples')
        image_loss          = image_loss_func(torch.log(image_category[idx]).to(device),
                                              labels[idx].to(device))
    elif "Kullback-Leibler divergence loss" in image_loss_func.__doc__:
        image_loss_func.reduction  = 'batchmean'
        image_loss_func.log_target = True
        if n_noise > 0:
            noisy_labels    = torch.ones(labels.shape) * (1/num_classes)
            noisy_labels    = noisy_labels[:n_noise]
            labels          = torch.cat([labels.to(device),noisy_labels.to(device)])
        image_loss = image_loss_func(torch.log(image_category).to(device),
                                     labels.to(device))
    return image_loss

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

def add_noise_instance_for_training(batch_features:Tensor,
                                    n_noise:int = 1,
                                    clip_output:bool = False,
                                    ) -> Tensor:
    """
    

    Parameters
    ----------
    batch_features : Tensor
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 1.
    clip_output : bool, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    batch_features : Tensor
        DESCRIPTION.

    """
    if n_noise > 0:
        noise_generator         = torch.distributions.normal.Normal(batch_features.mean(),
                                                            batch_features.std(),)
        noise_features          = noise_generator.sample(batch_features.shape)[:n_noise]
        if clip_output:
            temp                = invTrans(batch_features[:n_noise])
            idx_pixels          = torch.where(temp == 1)
            temp                = invTrans(noise_features)
            temp[idx_pixels]    = 1
            noise_features      = normalizer(temp)
        batch_features          = torch.cat([batch_features,noise_features])
    else:
        pass
    return batch_features

def train_valid_cnn_classifier(net:nn.Module,
                               dataloader:torch.utils.data.dataloader.DataLoader,
                               optimizer:torch.optim,
                               classification_loss:nn.Module,
                               idx_epoch:int    = 0,
                               device           = 'cpu',
                               train:bool       = True,
                               verbose          = 0,
                               n_noise:int      = 0,
                               sleep_time:int   = 0,
                               ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    classification_loss : nn.Module
        DESCRIPTION.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : string or torch.device, optional
        DESCRIPTION. The default is 'cpu'.
    train : bool, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
     n_noise : int, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if train:
        net.train(True)
    else:
        net.eval()
    loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    for idx_batch,(batch_features,batch_labels) in iterator:
        batch_features = add_noise_instance_for_training(batch_features,
                                                         n_noise,
                                                         clip_output = True,)
        # zero grad
        optimizer.zero_grad()
        # forward pass
        if train:
            (batch_extract_features,
             batch_hidden_representation,
             batch_prediction) = net(batch_features.to(device))
        else:
            with torch.no_grad():
                (batch_extract_features,
                 batch_hidden_representation,
                 batch_prediction) = net(batch_features.to(device))
        # compute loss
        batch_loss = compute_image_loss(classification_loss,
                                        batch_prediction,
                                        batch_labels,
                                        device,
                                        n_noise = n_noise,
                                        )
        if train:
            # backprop
            batch_loss.backward()
            # modify weights
            optimizer.step()
        # record the loss of a mini-batch
        loss += batch_loss.item()
        if verbose > 0:
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{idx_batch + 1:4.0f}/{100*(idx_batch+1)/len(dataloader):2.3f}%,loss = {loss/(idx_batch+1):2.6f}')
    if sleep_time > 0:
        sleep(sleep_time)
    return net,loss/(idx_batch+1)

def _concatenate_image_representations(batch_hidden_representation1:Tensor,
                                       batch_hidden_representation2:Tensor,
                                       batch_prediction1:Tensor,
                                       batch_prediction2:Tensor,
                                       signal_source:str = 'hidden',
                                       ):
    if signal_source == 'hidden':
        detection_input = torch.cat((batch_hidden_representation1,
                                   batch_hidden_representation2,
                                   ), 1)
    elif signal_source == 'prediction':
        detection_input = torch.cat((batch_prediction1,
                                   batch_prediction2,
                                   ), 1)
    elif signal_source == 'both':
        detection_input = torch.cat((batch_hidden_representation1,
                                   batch_hidden_representation2,
                                   batch_prediction1,
                                   batch_prediction2,
                                   ), 1)
    else:
        raise NotImplementedError
    return detection_input

def train_valid_detection_network(classification_net:nn.Module,
                                  detection_net:nn.Module,
                                  dataloader:torch.utils.data.dataloader.DataLoader,
                                  optimizer:torch.optim,
                                  classification_loss:nn.Module,
                                  detection_loss:nn.Module,
                                  idx_epoch:int    = 0,
                                  device           = 'cpu',
                                  train:bool       = True,
                                  verbose          = 0,
                                  sleep_time:int   = 0,
                                  signal_source:str= 'hidden',
                                  n_noise:int      = 0,
                                  idx_cnn          = None,
                                  noise_cnn:float  = 0,
                                  noise_type:str   = None,
                                  ):
    """
    

    Parameters
    ----------
    classification_net : nn.Module
        DESCRIPTION.
    detection_net : nn.Module
        DESCRIPTION.
    dataloader : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    classification_loss : nn.Module
        DESCRIPTION.
    detection_loss : nn.Module
        DESCRIPTION.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    train : bool, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.
    signal_source : str, optional
        DESCRIPTION. The default is 'hidden'.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    idx_cnn : TYPE, optional
        1. positive number: the indices of the CNN layer
        2. negative two: hidden layer
        3. string, 'adaptive_pooling', add noise to the pooled outputs
        The default is None.
    noise_cnn : float, optional
        DESCRIPTION. The default is 0.
    noise_type : str, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if train:
        classification_net.train(True)
        detection_net.train(True)
    else:
        classification_net.eval()
        detection_net.eval()
    loss        = 0.
    iterator    = tqdm(enumerate(dataloader))
    y_true1     = []
    y_true2     = []
    y_pred1     = []
    y_pred2     = []
    for idx_batch,(batch_image1,batch_label1,
                   batch_image2,batch_label2,
                   batch_correct_bet) in iterator:
        # zero grad
        optimizer.zero_grad()
        # forward pass
        with torch.no_grad():
            (batch_features1,
             batch_hidden_representation1,
             batch_prediction1) = classification_net(batch_image1.to(device),
                                                     idx_layer = idx_cnn,
                                                     noise_level = noise_cnn,
                                                     noise_type = noise_type,)
            (batch_features2,
             batch_hidden_representation2,
             batch_prediction2) = classification_net(batch_image2.to(device),
                                                     idx_layer = idx_cnn,
                                                     noise_level = noise_cnn,
                                                     noise_type = noise_type,)
            batch_detection_input = _concatenate_image_representations(
                                        batch_hidden_representation1,
                                        batch_hidden_representation2,
                                        batch_prediction1,
                                        batch_prediction2,
                                        signal_source = signal_source,
                                        )
            y_true1.append(batch_label1.detach().cpu().numpy())
            y_true2.append(batch_label2.detach().cpu().numpy())
            y_pred1.append(batch_prediction1.detach().cpu().numpy())
            y_pred2.append(batch_prediction2.detach().cpu().numpy())
        batch_detection_input = add_noise_instance_for_training(batch_detection_input,
                                                                n_noise,
                                                                clip_output = False,
                                                                )
        
        if train:
            batch_detection_output = detection_net(batch_detection_input)
        else:
            with torch.no_grad():
                batch_detection_output = detection_net(batch_detection_input)
        # compute loss
        batch_loss = compute_image_loss(
                                    detection_loss,
                                    batch_detection_output,
                                    batch_correct_bet,
                                    device,
                                    n_noise = n_noise,
                                    )
        if train:
            # backprop
            batch_loss.backward()
            # modify weights
            optimizer.step()
        # record the loss of a mini-batch
        loss += batch_loss.item()
        if verbose > 0:
            try:
                score1 = roc_auc_score(np.concatenate(y_true1).argmax(1),
                                       np.concatenate(y_pred1)[:,-1])
                score2 = roc_auc_score(np.concatenate(y_true2).argmax(1),
                                       np.concatenate(y_pred2)[:,-1])
            except:
                score1 = np.nan
                score2 = np.nan
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{idx_batch + 1:4.0f}/{100*(idx_batch+1)/len(dataloader):2.3f}%,loss = {loss/(idx_batch+1):2.6f},score1 = {score1:.4f}, score2 = {score2:.4f}')
    if sleep_time > 0:
        sleep(sleep_time)
    return (classification_net,detection_net),loss/(idx_batch+1)

def train_valid_both_network(classification_net:nn.Module,
                             detection_net:nn.Module,
                             dataloader:torch.utils.data.dataloader.DataLoader,
                             optimizer:torch.optim,
                             classification_loss:nn.Module,
                             detection_loss:nn.Module,
                             idx_epoch:int    = 0,
                             device           = 'cpu',
                             train:bool       = True,
                             verbose          = 0,
                             sleep_time:int   = 0,
                             signal_source:str= 'hidden',
                             n_noise:int      = 0,
                             ):
    """
    

    Parameters
    ----------
    classification_net : nn.Module
        DESCRIPTION.
    detection_net : nn.Module
        DESCRIPTION.
    dataloader : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    classification_loss : nn.Module
        DESCRIPTION.
    detection_loss : nn.Module
        DESCRIPTION.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    train : bool, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.
    signal_source : str, optional
        DESCRIPTION. The default is 'hidden'.
    n_noise : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if train:
        classification_net.train(True)
        detection_net.train(True)
    else:
        classification_net.eval()
        detection_net.eval()
    loss        = 0.
    iterator    = tqdm(enumerate(dataloader))
    y_true1     = []
    y_true2     = []
    y_pred1     = []
    y_pred2     = []
    y_cor       = []
    y_bet       = []
    for idx_batch,(batch_image1,batch_label1,
                   batch_image2,batch_label2,
                   batch_correct_bet) in iterator:
        batch_image1 = add_noise_instance_for_training(batch_image1,n_noise)
        batch_image2 = add_noise_instance_for_training(batch_image2,n_noise)
        # zero grad
        optimizer.zero_grad()
        # forward pass
        (batch_features1,
         batch_hidden_representation1,
         batch_prediction1) = classification_net(batch_image1.to(device),
                                              )
        (batch_features2,
         batch_hidden_representation2,
         batch_prediction2) = classification_net(batch_image2.to(device),
                                              )
        image_loss = compute_image_loss(
                                    classification_loss,
                                    batch_prediction1,
                                    batch_label1,
                                    device,
                                    n_noise = n_noise,
                                    )
        image_loss += compute_image_loss(
                                    classification_loss,
                                    batch_prediction2,
                                    batch_label2,
                                    device,
                                    n_noise = n_noise,
                                    )
        image_loss = image_loss / 2.0
        batch_detection_input = _concatenate_image_representations(
                                    batch_hidden_representation1,
                                    batch_hidden_representation2,
                                    batch_prediction1,
                                    batch_prediction2,
                                    signal_source = signal_source,
                                    )
        batch_detection_output = detection_net(batch_detection_input)
        # compute loss
        bet_loss = compute_image_loss(
                                    detection_loss,
                                    batch_detection_output,
                                    batch_correct_bet,
                                    device,
                                    n_noise = n_noise,
                                    )
        # sum up the losses
        batch_loss = image_loss + bet_loss
        y_true1.append(batch_label1.detach().cpu().numpy())
        y_true2.append(batch_label2.detach().cpu().numpy())
        y_cor.append(batch_correct_bet.detach().cpu().numpy())
        if n_noise > 0:
            y_pred1.append(batch_prediction1.detach().cpu().numpy()[:-n_noise])
            y_pred2.append(batch_prediction2.detach().cpu().numpy()[:-n_noise])
            y_bet.append(batch_detection_output.detach().cpu().numpy()[:-n_noise])
        else:
            y_pred1.append(batch_prediction1.detach().cpu().numpy())
            y_pred2.append(batch_prediction2.detach().cpu().numpy())
            y_bet.append(batch_detection_output.detach().cpu().numpy())
        if train:
            # backprop
            batch_loss.backward()
            # modify weights
            optimizer.step()
        # record the loss of a mini-batch
        loss += batch_loss.item()
        if verbose > 0:
            try:
                score1 = roc_auc_score(np.concatenate(y_true1).argmax(1),
                                       np.concatenate(y_pred1).argmax(1)#[:,-1]
                                       )
                score2 = roc_auc_score(np.concatenate(y_true2).argmax(1),
                                       np.concatenate(y_pred2).argmax(1)#[:,-1]
                                       )
                ss = roc_auc_score(np.concatenate(y_cor).argmax(1),
                                   np.concatenate(y_bet).argmax(1)
                                   )
            except Exception as e:
                print(e)
                score1 = np.nan
                score2 = np.nan
                ss = np.nan
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{idx_batch + 1:4.0f}/{100*(idx_batch+1)/len(dataloader):2.3f}%, loss = {loss/(idx_batch+1):2.6f} ({image_loss.item():2.3f}+{bet_loss.item():2.3f}), score1 = {score1:.4f},score2 = {score2:.4f},meta-score = {ss:.4f}')
    if sleep_time > 0:
        sleep(sleep_time)
    return (classification_net,detection_net),loss/(idx_batch+1)


def determine_training_stops(classification_net,
                             detection_net,
                             idx_epoch:int,
                             warmup_epochs:int,
                             valid_loss:Tensor,
                             model_stage:str    = 'cnn',
                             counts: int        = 0,
                             device             = 'cpu',
                             best_valid_loss    = np.inf,
                             tol:float          = 1e-4,
                             cnn_name:str       = 'temp_cnn.h5',
                             bet_name:str       = 'temp_bet.h5'
                             ) -> Tuple[Tensor,int]:
    """
    

    Parameters
    ----------
    classification_net : TYPE
        DESCRIPTION.
    detection_net : TYPE
        DESCRIPTION.
    idx_epoch : int
        DESCRIPTION.
    warmup_epochs : int
        DESCRIPTION.
    valid_loss : Tensor
        DESCRIPTION.
    model_stage : str, optional
        DESCRIPTION. The default is 'cnn'.
    counts : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    best_valid_loss : TYPE, optional
        DESCRIPTION. The default is np.inf.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    cnn_name : str, optional
        DESCRIPTION. The default is 'temp_cnn.h5'.
    bet_name : str, optional
        DESCRIPTION. The default is 'temp_bet.h5'.

    Returns
    -------
    Tuple[Tensor,int]
        DESCRIPTION.

    """
    if model_stage == 'cnn':
        if idx_epoch >= warmup_epochs: # warming up
            temp = valid_loss
            if np.logical_and(temp < best_valid_loss,np.abs(best_valid_loss - temp) >= tol):
                best_valid_loss = valid_loss
                torch.save(classification_net.state_dict(),cnn_name)# why do i need state_dict()?
                counts = 0
            else:
                counts += 1
    elif model_stage == 'detection':
        if idx_epoch >= warmup_epochs: # warming up
            temp = valid_loss
            if np.logical_and(temp < best_valid_loss,np.abs(best_valid_loss - temp) >= tol):
                best_valid_loss = valid_loss
                torch.save(detection_net.state_dict(),bet_name)# why do i need state_dict()?
                counts = 0
            else:
                counts += 1
    elif model_stage == 'simultaneously':
        if idx_epoch >= warmup_epochs:
            temp = valid_loss
            if np.logical_and(temp < best_valid_loss,np.abs(best_valid_loss - temp) >= tol):
                best_valid_loss = valid_loss
                torch.save(classification_net.state_dict(),cnn_name)
                torch.save(detection_net.state_dict(),bet_name)
                counts = 0
            else:
                counts += 1
    return best_valid_loss,counts

def train_valid_loop(classification_net:nn.Module,
                     dataloader_train:torch.utils.data.dataloader.DataLoader,
                     dataloader_valid:torch.utils.data.dataloader.DataLoader,
                     optimizer:torch.optim,
                     detection_net:nn.Module = None,
                     classification_loss= nn.BCELoss(),
                     detection_loss       = nn.BCELoss(),
                     scheduler          = None,
                     device             = 'cpu',
                     verbose            = 0,
                     n_epochs:int       = 1000,
                     warmup_epochs:int  = 5,
                     patience:int       = 5,
                     tol:float          = 1e-4,
                     cnn_name:str       = 'temp_cnn.h5',
                     bet_name:str       = 'temp_bet.h5',
                     model_stage:str    = 'cnn',
                     n_noise:int        = 0,
                     sleep_time:int     = 0,
                     signal_source:str  = 'hidden',
                     ):
    """
    

    Parameters
    ----------
    classification_net : nn.Module
        DESCRIPTION.
    dataloader_train : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    dataloader_valid : torch.utils.data.dataloader.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    detection_net : nn.Module, optional
        DESCRIPTION. The default is None.
    classification_loss : TYPE, optional
        DESCRIPTION. The default is nn.BCELoss().
    detection_loss : TYPE, optional
        DESCRIPTION. The default is nn.BCELoss().
    scheduler : TYPE, optional
        DESCRIPTION. The default is None.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    verbose : TYPE, optional
        DESCRIPTION. The default is 0.
    n_epochs : int, optional
        DESCRIPTION. The default is 1000.
    warmup_epochs : int, optional
        DESCRIPTION. The default is 5.
    patience : int, optional
        DESCRIPTION. The default is 5.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    f_name : str, optional
        DESCRIPTION. The default is 'temp.h5'.
    model_stage : str, optional
        DESCRIPTION. The default is 'cnn'.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    sleep_time : int, optional
        DESCRIPTION. The default is 0.
    signal_source : str, optional
        DESCRIPTION. The default is 'hidden'.

    Returns
    -------
    classification_net : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    losses : TYPE
        DESCRIPTION.

    """
    best_valid_loss         = np.inf
    losses                  = []
    counts                  = 0
    for idx_epoch in range(n_epochs):
        if model_stage == 'cnn':
            print('\ntraining...')
            classification_net,train_loss = train_valid_cnn_classifier(classification_net,
                                                        dataloader_train,
                                                        optimizer,
                                                        classification_loss,
                                                        idx_epoch   = idx_epoch,
                                                        device      = device,
                                                        train       = True,
                                                        verbose     = verbose,
                                                        n_noise     = n_noise,
                                                        sleep_time  = sleep_time,
                                                        )
            print('\nvalidating...')
            classification_net,valid_loss = train_valid_cnn_classifier(classification_net,
                                                        dataloader_valid,
                                                        optimizer,
                                                        classification_loss,
                                                        idx_epoch   = idx_epoch,
                                                        device      = device,
                                                        train       = False,
                                                        verbose     = verbose,
                                                        sleep_time  = sleep_time,
                                                        )
        elif model_stage == 'detection':
            print('\ntraining...')
            (classification_net,detection_net),train_loss = train_valid_detection_network(
                                                         classification_net,
                                                         detection_net,
                                                         dataloader_train,
                                                         optimizer,
                                                         classification_loss,
                                                         detection_loss,
                                                         idx_epoch     = idx_epoch,
                                                         device        = device,
                                                         train         = True,
                                                         verbose       = verbose,
                                                         sleep_time    = sleep_time,
                                                         n_noise       = n_noise,
                                                         signal_source = signal_source,
                                                         )
            print('\nvalidating...')
            (classification_net,detection_net),valid_loss = train_valid_detection_network(
                                                         classification_net,
                                                         detection_net,
                                                         dataloader_valid,
                                                         optimizer,
                                                         classification_loss,
                                                         detection_loss,
                                                         idx_epoch     = idx_epoch,
                                                         device        = device,
                                                         train         = False,
                                                         verbose       = verbose,
                                                         sleep_time    = sleep_time,
                                                         signal_source = signal_source,
                                                         )
        elif model_stage == 'simultaneously':
            print('\ntraining...')
            (classification_net,detection_net),train_loss = train_valid_both_network(
                                                         classification_net,
                                                         detection_net,
                                                         dataloader_train,
                                                         optimizer,
                                                         classification_loss,
                                                         detection_loss,
                                                         idx_epoch     = idx_epoch,
                                                         device        = device,
                                                         train         = True,
                                                         verbose       = verbose,
                                                         sleep_time    = sleep_time,
                                                         n_noise       = n_noise,
                                                         signal_source = signal_source,
                                                         )
            print('\nvalidating...')
            (classification_net,detection_net),valid_loss = train_valid_both_network(
                                                         classification_net,
                                                         detection_net,
                                                         dataloader_valid,
                                                         optimizer,
                                                         classification_loss,
                                                         detection_loss,
                                                         idx_epoch     = idx_epoch,
                                                         device        = device,
                                                         train         = False,
                                                         verbose       = verbose,
                                                         sleep_time    = sleep_time,
                                                         signal_source = signal_source,
                                                         )
        if scheduler != None and idx_epoch >= warmup_epochs:
            scheduler.step(valid_loss)
        best_valid_loss,counts = determine_training_stops(classification_net,
                                                          detection_net,
                                                          idx_epoch,
                                                          warmup_epochs,
                                                          valid_loss,
                                                          model_stage       = model_stage,
                                                          counts            = counts,
                                                          device            = device,
                                                          best_valid_loss   = best_valid_loss,
                                                          tol               = tol,
                                                          cnn_name          = cnn_name,
                                                          bet_name          = bet_name,
                                                          )
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
        else:
            print(f'\nepoch {idx_epoch + 1}, best valid loss = {best_valid_loss:.8f},count = {counts}')
        losses.append(valid_loss)
    return (classification_net,detection_net),losses

def __psychometric_curve(x,a,b,c,d):
    return a / (1. + np.exp(-c * (x - d))) + b
def _psychometric_curve(x,alpha,beta,gamma,lambd):
    return gamma + (1 - gamma - lambd) / (1 + np.exp(-beta * (x - alpha)))
def psychometric_curve(x,alpha,beta):
    return 1. / (1 + np.exp( -(x - alpha) / beta ))

# for plotting psychometric curve
def plot_psychometric_curve(df,):
    from scipy import optimize
    df['response'] = np.array(df['y_pred_1'].values > 0.5,dtype = int)
    df['answer'] = np.array(df['y_true_1'].values > 0.5,dtype = int)
    # fit psychometric curve
    df_temp = dict(n_living = [],
                   true_prob = [],
                   pred_prob = [],
                   )
    for (n_living),df_sub in df.groupby('n_living'):
        df_temp['n_living'].append(n_living)
        df_temp['true_prob'].append(df_sub['answer'].mean())
        df_temp['pred_prob'].append(df_sub['response'].mean())
    df_temp = pd.DataFrame(df_temp)
    
    df_curve = dict(x=[],y=[])
    popt, pcov = optimize.curve_fit(psychometric_curve, 
                                    df_temp['n_living'].values - df_temp['n_living'].min(), 
                                    df_temp['pred_prob'].values, 
                                    method="lm")
    x = np.linspace(0,
                    df_temp['n_living'].max() - df_temp['n_living'].min(),
                    1000)
    y = psychometric_curve(x,*popt)
    [df_curve['x'].append(item) for item in x]
    [df_curve['y'].append(item) for item in y]
    df_curve = pd.DataFrame(df_curve)
    return df_curve,df

def A(y_true:np.ndarray,y_pred:np.ndarray) -> float:
    """
    

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_pred : np.ndaray
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    fpr,tpr,thres = roc_curve(y_true, y_pred)
    fpr = fpr[1]
    tpr = tpr[1]
    if  fpr > tpr:
        A = 1/2 + ((fpr - tpr)*(1 + fpr - tpr))/((4 * fpr)*(1 - tpr))
    elif fpr <= tpr:
        A = 1/2 + ((tpr - fpr)*(1 + tpr - fpr))/((4 * tpr)*(1 - fpr))
    return A

def compute_A(h:float,f:float) -> float:
    """
    

    Parameters
    ----------
    h : float
        DESCRIPTION.
    f : float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    if (.5 >= f) and (h >= .5):
        a = .75 + (h - f) / 4 - f * (1 - h)
    elif (h >= f) and (.5 >= h):
        a = .75 + (h - f) / 4 - f / (4 * h)
    else:
        a = .75 + (h - f) / 4 - (1 - h) / (4 * (1 - f))
    return a
def check_nan(temp):
    if np.isnan(temp[1]):
        return 0
    else:
        return temp[1]
def binary_response_score_func(y_true:np.ndarray, y_pred:np.ndarray):
    """
    

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_pred : np.ndarray
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    fpr,tpr,thresholds = roc_curve(y_true, y_pred)
    tpr = check_nan(tpr)
    fpr = check_nan(fpr)
    a = compute_A(tpr,fpr)
    return a

def collect_data(dataloader_test:pd.core.frame.DataFrame,
                 discrimination_network:nn.Module,
                 detection_network:nn.Module,
                 model_control,
                 idx_layer:int = 0,
                 noise_level:float = 0,
                 noise_type:str = 'reduction',
                 model_part:str = 'discrimination',
                 n_total:int = 15,
                 ) -> Union[pd.core.frame.DataFrame,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray]:
    """
    

    Parameters
    ----------
    dataloader_test : pd.core.frame.DataFrame
        DESCRIPTION.
    discrimination_network : nn.Module
        DESCRIPTION.
    detection_network : nn.Module
        DESCRIPTION.
    model_control : TYPE
        DESCRIPTION.
    idx_layer : int, optional
        DESCRIPTION. The default is 0.
    noise_level : float, optional
        DESCRIPTION. The default is 0.
    noise_type : str, optional
        DESCRIPTION. The default is 'reduction'.
    model_part : str, optional
        DESCRIPTION. The default is 'discrimination'.
    n_total : int, optional
        DESCRIPTION. Tje defailt is 15.

    Returns
    -------
    df_score : pd.core.frame.DataFrame
        DESCRIPTION.
    features_discrimination : np.ndarray
        DESCRIPTION.
    features_detection : np.ndarray
        DESCRIPTION.
    labels_detection : np.ndarray
        DESCRIPTION.

    """
    
    title = f'add noise to {model_part} network layer {idx_layer} with {noise_type} noise level {noise_level:.4f}' if noise_level > 0 else 'no noise'
    y_true,y_pred,choose,order = [],[],[],[]
    features_discrimination = []
    features_detection = []
    labels_detection = []
    with torch.no_grad():
        for (batch_image1,batch_label1,
             batch_image2,batch_label2,
             batch_correct_bet) in tqdm(dataloader_test,
                                        desc = title):
            
            with torch.no_grad():
                # classify first image
                if model_part == 'discrimination':
                    (batch_features1,
                     batch_hidden_representation1,
                     batch_prediction1) = discrimination_network(
                                             batch_image1.to(model_control.device),
                                             idx_layer = idx_layer,
                                             noise_level = noise_level,
                                             noise_type = noise_type,)
                else:
                    (batch_features1,
                     batch_hidden_representation1,
                     batch_prediction1) = discrimination_network(
                                             batch_image1.to(model_control.device),)
                # classify second image
                if model_part == 'discrimination':
                    (batch_features2,
                     batch_hidden_representation2,
                     batch_prediction2) = discrimination_network(
                                             batch_image2.to(model_control.device),
                                             idx_layer = idx_layer,
                                             noise_level = noise_level,
                                             noise_type = noise_type,)
                else:
                    (batch_features2,
                     batch_hidden_representation2,
                     batch_prediction2) = discrimination_network(
                                             batch_image2.to(model_control.device),)
                # detection
                batch_betting_input = _concatenate_image_representations(
                                            batch_hidden_representation1,
                                            batch_hidden_representation2,
                                            batch_prediction1,
                                            batch_prediction2,
                                            signal_source = model_control.signal_source,
                                            )
                if model_part == 'detection':
                    batch_betting_output = detection_network(
                                            batch_betting_input,
                                            idx_layer = idx_layer,
                                            noise_level = noise_level,
                                            noise_type = noise_type,
                                            )
                else:
                    batch_betting_output = detection_network(batch_betting_input,)
                
                # store outputs
                idx1 = batch_label1[:,0] != 0.5 # pick informative image
                y_true.append(batch_label1[idx1].detach().cpu().numpy())
                y_pred.append(batch_prediction1[idx1].detach().cpu().numpy())
                choose.append(batch_betting_output[idx1].detach().cpu().numpy())
                order.append([0] * np.sum(idx1.numpy()))# this is the label for detection for image1
                features_discrimination.append(batch_hidden_representation1[idx1].detach().cpu().numpy())
                
                idx2 = batch_label2[:,0] != 0.5 # pick informative image
                y_true.append(batch_label2[idx2].detach().cpu().numpy())
                y_pred.append(batch_prediction2[idx2].detach().cpu().numpy())
                choose.append(batch_betting_output[idx2].detach().cpu().numpy())
                order.append([1] * np.sum(idx2.numpy()))# this is the label for detection for image2
                features_discrimination.append(batch_hidden_representation2[idx2].detach().cpu().numpy())
                
                features_detection.append(batch_betting_input.detach().cpu().numpy())
                ratio1 = batch_label1.detach().cpu().numpy().min(1) / batch_label1.detach().cpu().numpy().max(1)
                ratio2 = batch_label2.detach().cpu().numpy().min(1) / batch_label2.detach().cpu().numpy().max(1)
                temp = np.zeros((batch_betting_input.shape[0],2))
                temp[:,0] = ratio1
                temp[:,1] = ratio2
                labels_detection.append(temp.argmin(1))
    y_true = np.concatenate(y_true)
    # for some reasons, the labels are flipped
    y_true = np.fliplr(y_true)
    y_pred = np.concatenate(y_pred)
    choose = np.concatenate(choose)
    order = np.concatenate(order)
    features_discrimination = np.concatenate(features_discrimination)
    features_detection = np.concatenate(features_detection)
    labels_detection = np.concatenate(labels_detection)
    
    df_score = pd.DataFrame(dict(y_true_0 = y_true[:,0],
                                 y_true_1 = y_true[:,1],
                                 y_pred_0 = y_pred[:,0],
                                 y_pred_1 = y_pred[:,1],
                                 choose = choose[:,-1],
                                 order = order,
                                 )
                            )
    temp = np.sort(y_true,axis = 1)
    temp = temp[:,0] / temp[:,1]
    df_score['ratio'] = temp
    df_score['n_living'] = df_score['y_true_0'] * n_total
    
    df_score['pretrained_model_name'] = model_control.pretrained_model_name
    df_score['hidden_activation'] = model_control.hidden_activation_name
    df_score['hidden_size'] = model_control.hidden_layer_size
    df_score['hidden_dropout'] = model_control.hidden_dropout
    df_score['idx_layer_noise'] = idx_layer
    df_score['noise_level'] = noise_level
    df_score['noise_type'] = noise_type
    df_score['model_part'] = model_part
    
    df_score['n_class0'] = n_total / (df_score['ratio'] + 1)
    df_score['n_class1'] = n_total - df_score['n_class0']
    df_score['n_class0'] = df_score['n_class0'].apply(round).values.astype(int)
    df_score['n_class1'] = df_score['n_class1'].apply(round).values.astype(int)
    df_score['condition'] = [f'{int(a)}-{int(b)}' for a,b in df_score[['n_class0','n_class1']].values]
    
    return df_score,features_discrimination,features_detection,labels_detection

def decode_from_hidden_layer_representations(
                                  df_score:pd.core.frame.DataFrame,
                                  features_discrimination:np.ndarray,
                                  features_detection:np.ndarray,
                                  labels_detection:np.ndarray,
                                  model_control,
                                  idx_layer:int = 0,
                                  noise_level:float = 0,
                                  noise_type:str = 'reduction',
                                  model_part:str = 'discrimination',
                                  ) -> pd.core.frame.DataFrame:
    """
    

    Parameters
    ----------
    df_score : pd.core.frame.DataFrame
        DESCRIPTION.
    features_discrimination : np.ndarray
        DESCRIPTION.
    features_detection : np.ndarray
        DESCRIPTION.
    labels_detection : np.ndarray
        DESCRIPTION.
    model_control : TYPE
        DESCRIPTION.
    idx_layer : int, optional
        DESCRIPTION. The default is 0.
    noise_level : float, optional
        DESCRIPTION. The default is 0.
    noise_type : str, optional
        DESCRIPTION. The default is 'reduction'.
    model_part : str, optional
        DESCRIPTION. The default is 'discrimination'.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    df_decode : pd.core.frame.DataFrame
        DESCRIPTION.

    """
    # decode from the hidden layer
    df_decode = dict(condition = [],
                     discrimination = [],
                     detection = [],
                     label_discrimination = [],
                     label_detection = [],
                     )
    for condition,df_sub in df_score.groupby('condition'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # decode discrimination labels (living vs nonliving)
            gc.collect()
            features = features_discrimination[df_sub.index]
            targets = df_sub[['y_true_0','y_true_1']].values.argmax(1)
            pipeline = make_pipeline(StandardScaler(),
                                     LinearSVC(max_iter = int(3e3),
                                               random_state = 12345,))
            cv = StratifiedShuffleSplit(n_splits = model_control.n_splits,test_size = .2,random_state = 12345)
            scores_disc = cross_val_score(pipeline,
                                          features,
                                          targets,
                                          scoring = 'roc_auc',
                                          cv = cv,
                                          n_jobs = 8,
                                          verbose = 0,
                                          )
            y_disc = [targets[test] for _,test in cv.split(features,targets)]
            print(f'decode discrimination from hidden layer: {features.shape}\n{condition},noise level:{noise_level:.4f},add to layer {idx_layer}')
            print(f'score = {scores_disc.mean():.4f} +/- {scores_disc.std():.4f}')
            del features,targets
            gc.collect()
            features = features_detection[df_sub.index]
            targets = labels_detection[df_sub.index]
            pipeline = make_pipeline(StandardScaler(),
                                     LinearSVC(max_iter = int(3e3),
                                               random_state = 12345,))
            cv = StratifiedShuffleSplit(n_splits = model_control.n_splits,test_size = .2,random_state = 12345)
            scores_dete = cross_val_score(pipeline,
                                          features,
                                          targets,
                                          scoring = 'roc_auc',
                                          cv = cv,
                                          n_jobs = 8,
                                          verbose = 0,
                                          )
            y_dete = [targets[test] for _,test in cv.split(features,targets)]
            print(f'decode detection from hidden layer: {features.shape}\n{condition},noise level:{noise_level:.4f},add to layer {idx_layer}')
            print(f'score = {scores_dete.mean():.4f} +/- {scores_dete.std():.4f}')
            del features,targets
            
            for _scores_disc,_scores_dete,_y_disc,_y_dete in zip(scores_disc,scores_dete,y_disc,y_dete):
                df_decode['condition'].append(condition)
                df_decode['discrimination'].append(_scores_disc)
                df_decode['detection'].append(_scores_dete)
                df_decode['label_discrimination'].append('-'.join(_y_disc.astype(str)))
                df_decode['label_detection'].append('-'.join(_y_dete.astype(str)))
    df_decode = pd.DataFrame(df_decode)
    df_decode['pretrained_model_name'] = model_control.pretrained_model_name
    df_decode['hidden_activation'] = model_control.hidden_activation_name
    df_decode['hidden_size'] = model_control.hidden_layer_size
    df_decode['hidden_dropout'] = model_control.hidden_dropout
    df_decode['idx_layer_noise'] = idx_layer
    df_decode['noise_level'] = noise_level
    df_decode['noise_type'] = noise_type
    df_decode['model_part'] = model_part
    return df_decode

def easy_empirical_chance_level_estimate_for_binary(features,
                                                    labels,
                                                    cv,):
    y_tests = np.vstack([labels[test] for _,test in cv.split(features,labels)])
    y_preds = np.random.choice([0,1],size = y_tests.shape,replace = True,)
    chance_scores = np.array([roc_auc_score(a,b) for a,b in zip(y_tests,y_preds)])
    return chance_scores

def performance_per_ratio(df_sub,model_control,n_jobs = -1,verbose = 0):
    score_detection = np.sum(df_sub['picked'] == 'Confidence chosen') / df_sub.shape[0]
    score_discrimination = np.sum(np.array(df_sub['y_true_0'] >= 0.5,dtype = np.int32,) ==\
                    np.array(df_sub['y_pred_1'] >= 0.5,dtype = np.int32)) / df_sub.shape[0]
    def _proc(df_sub):
        df_temp = df_sub.sample(frac = 1.,replace = False)
        return np.sum(df_temp['picked'] == 'Confidence chosen') / df_temp.shape[0]
    temp = np.array(Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(_proc)(*[df_sub])) for _ in range(model_control.n_permutations))
    p_detection = (np.sum(temp > score_detection) + 1) / (model_control.n_permutations + 1)
    
    def _proc(df_sub):
        df_temp = df_sub.sample(frac = 1.,replace = False)
        return np.sum(np.array(df_temp['y_true_0'] >= 0.5,dtype = np.int32,) ==\
                      np.array(df_temp['y_pred_1'] >= 0.5,dtype = np.int32)) / df_temp.shape[0]
    temp = np.array(Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(_proc)(*[df_sub])) for _ in range(model_control.n_permutations))
    p_discrimination = (np.sum(temp > score_discrimination) + 1) / (model_control.n_permutations + 1)
    
    return score_detection,p_detection,score_discrimination,p_discrimination

def shuffle_roc_auc_score(y_true,y_pred,condition = 'discrimination'):
    y_chance = shuffle(y_pred)
    if condition == 'discrimination':
        return roc_auc_score(y_true.argmax(1),y_chance[:,-1])
    elif condition == 'detection':
        return roc_auc_score(y_true,y_chance)
    
def criteria_discrimination(df_sub,model_control,n_jobs = -1,verbose = 0,compute_p = True,):
    y_true = df_sub[['y_true_0','y_true_1']].values
    y_pred = df_sub[['y_pred_1','y_pred_0']].values
    score_discrimination = roc_auc_score(y_true.argmax(1),
                                         y_pred[:,-1],
                                         )
    if compute_p:
        scores_chance = np.array(Parallel(n_jobs = n_jobs,
                                          verbose = verbose)(delayed(shuffle_roc_auc_score)(*[
                                    y_true,y_pred,'discrimination']) for _ in range(
                                            model_control.n_permutations))
                                 )
        pval_discrimination = (np.sum(scores_chance >= score_discrimination) + 1) / (model_control.n_permutations + 1)
        return score_discrimination,pval_discrimination
    else:
        return score_discrimination

def criteria_detection(df_sub,model_control,n_jobs = -1,verbose = 0,compute_p = True):
    # y_true,y_pred = np.zeros((df_sub.shape[0],2)),np.zeros((df_sub.shape[0],2))
    # for idx_row,row in df_sub.reset_index(drop=True).iterrows():
    #     y_true[idx_row,int(row['order'])] = 1
    #     y_pred[idx_row,int(row['order'])] = row['choose']
    #     y_pred[idx_row,int(1 - row['order'])] = 1 - row['choose']
    y_true = df_sub['order'].values
    y_pred = df_sub['choose'].values
    score_detection = roc_auc_score(y_true,y_pred)
    if compute_p:
        scores_chance = np.array(Parallel(n_jobs = n_jobs,
                                          verbose = verbose)(delayed(shuffle_roc_auc_score)(*[
                                    y_true,y_pred,'detection']) for _ in range(
                                            model_control.n_permutations))
                                 )
        pval_detection = (np.sum(scores_chance >= score_detection) + 1) / (model_control.n_permutations + 1)
        return score_detection,pval_detection
    else:
        return score_detection

def collect_data_on_test(y_true:np.ndarray,y_pred:np.ndarray,
                         grid_size:int,) -> pd.core.frame.DataFrame:
    """
    

    Parameters
    ----------
    y_true : np.ndarray
        DESCRIPTION.
    y_pred : np.ndarray
        DESCRIPTION.
    grid_size : int
        DESCRIPTION.

    Returns
    -------
    df_temp : TYPE
        DESCRIPTION.

    """
    groups = ['{}-{}'.format(*np.sort(row)) for row in y_true]
    df_temp = pd.DataFrame(y_true,columns = ['class_0','class_1'])
    df_temp['prob_0'] = y_pred[:,0]
    df_temp['prob_1'] = y_pred[:,1]
    df_temp['correct answer'] = np.array(df_temp['class_1'].values >= 0.5,dtype = np.int64)
    df_temp['response'] = np.array(df_temp['prob_1'].values >= 0.5,dtype = np.int64)
    df_temp['group'] = groups
    df_temp['grid_size'] = grid_size
    return df_temp

###############################################################################
# old functions for loading old benckmarks ####################################
###############################################################################
class CustomImageDataset(Dataset):
    """
    
    """
    def __init__(self,
                 img_dir:str,
                 transform:torchvision.transforms.transforms.Compose        = None,
                 sparse_target:bool                                         = True):
        self.img_dir            = img_dir
        self.transform          = transform
        self.sparse_target      = sparse_target
        
        self.images = glob(os.path.join(img_dir,'*','*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path    = self.images[idx]
        image,label = lock_and_load(img_path,
                                    self.transform,
                                    self.sparse_target,
                                    )
        return image, label

def lock_and_load(img_path:str,
                  transformer_steps:torchvision.transforms.transforms.Compose,
                  sparse_target:bool):
    """
    

    Parameters
    ----------
    img_path : str
        DESCRIPTION.
    transformer_steps : torchvision.transforms.transforms.Compose
        DESCRIPTION.
    sparse_target : bool
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    """
    image       = pil_image.open(img_path)
    label       = img_path.split('/')[-2]
    label       = torch.tensor([int(label.split('-')[0]),int(label.split('-')[1])])
    label       = label / label.sum() # hard max
    if sparse_target and label.detach().numpy()[0] != 0.5:
        label   = torch.vstack([1 - label.argmax(),label.argmax()]).T
    
    image = transformer_steps(image)
    return image,label

class detection_network_dataloader(Dataset):
    def __init__(self, 
                 dataframe:pd.core.frame.DataFrame,
                 transformer_steps = None,
                 noise_level:float = 0.,
                 sparse_target:bool = False,
                 ):
        """
        

        Parameters
        ----------
        dataframe : pd.core.frame.DataFrame
            DESCRIPTION.
        transformer_steps : TYPE, optional
            DESCRIPTION. The default is None.
        noise_level : float, optional
            DESCRIPTION. The default is 0..
        sparse_target : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.dataframe = dataframe
        if transformer_steps == None:
            self.transformer_steps = concatenate_transform_steps(
                                    128,
                                    noise_level = noise_level,)
        else:
            self.transformer_steps = transformer_steps
        self.noise_level = noise_level,
        self.sparse_target = sparse_target
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image1,label1 = lock_and_load(row['image1'],
                                      self.transformer_steps,
                                      self.sparse_target,)
        image2,label2 = lock_and_load(row['image2'],
                                      self.transformer_steps,
                                      self.sparse_target,)
        if self.sparse_target:
            correct_bet = torch.tensor([row['sparse_label'],1-row['sparse_label']])
        else:
            correct_bet = torch.tensor([row['correct_bet1'],
                                        row['correct_bet2']])
        return image1,label1,image2,label2,correct_bet

def append_to_list(df:pd.core.frame.DataFrame,
                   image1,
                   image2):
    """
    

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        DESCRIPTION.
    image1 : TYPE
        DESCRIPTION.
    image2 : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df['image1'].append(image1)
    df['image2'].append(image2)
    return df

def build_betting_dataloader(img_dir:str,
                             transformer_steps  = None,
                             batch_size:int     = 16,
                             shuffle:bool       = True,
                             num_workers:int    = 2,
                             sparse_target:bool = False,
                             noise_level:float  = 0.,
                             memory_samples:int = 10,
                             random_state       = None,
                             ):
    """
    

    Parameters
    ----------
    img_dir : str
        DESCRIPTION.
    transformer_steps : TYPE, optional
        DESCRIPTION. The default is None.
    batch_size : int, optional
        DESCRIPTION. The default is 16.
    shuffle : bool, optional
        DESCRIPTION. The default is True.
    num_workers : int, optional
        DESCRIPTION. The default is 2.
    sparse_target : bool, optional
        DESCRIPTION. The default is False.
    noise_level : float, optional
        DESCRIPTION. The default is 0..
    memory_samples : int, optional
        DESCRIPTION. The default is 100.
    random_state : None or int, optional
        DESCRIPTION. The default is None

    Returns
    -------
    dataloader : TYPE
        DESCRIPTION.

    """
    
    all_images = glob(os.path.join(img_dir,'*','*.jpg'))
    df_images = pd.DataFrame(all_images,columns = ['image_path'])
    df_images['group'] = df_images['image_path'].apply(lambda x:x.split('/')[-2])
    df_images['group'] = df_images['group'].apply(compute_ratio_from_group)
    df = dict(image1=[],image2=[])
    for group1,df_sub1 in df_images.groupby('group'):
        for group2,df_sub2 in df_images.groupby('group'):
            if group1 != group2:
                if memory_samples != None:
                    df_sub1 = df_sub1.sample(memory_samples,replace = False,random_state = random_state)
                    df_sub2 = df_sub2.sample(memory_samples,replace = False,random_state = random_state)
                pairs = product(df_sub1['image_path'],df_sub2['image_path'])
                [append_to_list(df, image1, image2) for image1,image2 in pairs]
                
    df = pd.DataFrame(df)
    df['group1'] = df['image1'].apply(lambda x:x.split('/')[-2])
    df['group2'] = df['image2'].apply(lambda x:x.split('/')[-2])
    
    df['image1_ratio1'] = df['group1'].apply(lambda x:float(x.split('-')[0]))
    df['image1_ratio2'] = df['group1'].apply(lambda x:float(x.split('-')[1]))
    df['image2_ratio1'] = df['group2'].apply(lambda x:float(x.split('-')[0]))
    df['image2_ratio2'] = df['group2'].apply(lambda x:float(x.split('-')[1]))
    
    group1 = np.sort(df[['image1_ratio1', 'image1_ratio2']].values,axis = 1)
    group1 = group1[:,0] / group1[:,1]
    group2 = np.sort(df[['image2_ratio1', 'image2_ratio2']].values,axis = 1)
    group2 = group2[:,0] / group2[:,1]
    
    df['difficulty1'] = group1
    df['difficulty2'] = group2
    
    df['sparse_label'] = np.array(df['difficulty1'].values > df['difficulty2'].values,
                           dtype = np.float64
                           )
    
    temp = df[['difficulty1','difficulty2']].values
    temp = temp / temp.sum(1).reshape(-1,1)
    temp = 1 - temp
    df['correct_bet1'] = temp[:,0]
    df['correct_bet2'] = temp[:,1]
    
    dataset             = detection_network_dataloader(
                                            dataframe = df,
                                            transformer_steps = transformer_steps,
                                            noise_level = noise_level,
                                            sparse_target = sparse_target,
                                            )
    dataloader          = DataLoader(dataset,
                                     batch_size         = batch_size,
                                     shuffle            = shuffle,
                                     num_workers        = num_workers,
                                     )
    return dataloader

def compute_ratio_from_group(x):
    """
    This function can only apply to dataframes

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x = np.sort([float(x.split('-')[0]), float(x.split('-')[1])])
    return x[0] / x[1]

def build_dataloader(img_dir:str,
                     transformer_steps  = None,
                     batch_size:int     = 16,
                     shuffle:bool       = True,
                     num_workers:int    = 2,
                     sparse_target:bool = False,
                     ):
    """
    build a dataloader for batch feeding

    Parameters
    ----------
    img_dir : str
        DESCRIPTION.
    transformer_steps : TYPE, optional
        DESCRIPTION. The default is None.
    batch_size : int, optional
        DESCRIPTION. The default is 16.
    shuffle : bool, optional
        DESCRIPTION. The default is True.
    num_workers : int, optional
        DESCRIPTION. The default is 2.
    sparse_target : bool, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    dataloader : TYPE
        DESCRIPTION.

    """
    dataset             = CustomImageDataset(img_dir       = img_dir,
                                             transform     = transformer_steps,
                                             sparse_target = sparse_target,
                                             )
    dataloader          = DataLoader(dataset,
                                     batch_size         = batch_size,
                                     shuffle            = shuffle,
                                     num_workers        = num_workers,
                                     )
    return dataloader

##############################################################################

if __name__ == "__main__":
    pass
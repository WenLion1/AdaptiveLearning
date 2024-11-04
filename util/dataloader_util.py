#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:59:04 2024

@author: MeiNing_01
"""
import os,torch
import math
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from util.utils_deep import (concatenate_transform_steps,invTrans)


np.random.seed(12345)
torch.manual_seed(12345)

def draw_image(ball_angle:float,
               ball_radius:int = 10,
               background_size:int = 256,
               background_color:tuple = (255,255,255),
               circle_radius:int = 100,
               ):
    """
    draw a dot on a circle

    Parameters
    ----------
    ball_angle : float
        DESCRIPTION.
    ball_radius : int, optional
        DESCRIPTION. The default is 10.
    background_size : int, optional
        DESCRIPTION. The default is 256.
    background_color : tuple, optional
        DESCRIPTION. The default is (255,255,255).
    circle_radius : int, optional
        DESCRIPTION. The default is 100.
        DESCRIPTION.

    Returns
    -------
    background : PIL object
        DESCRIPTION.

    """
    # create background
    background = Image.new('RGB',
                           size = (background_size,background_size),
                           color = background_color)
    center_x,center_y = int(background_size/2),int(background_size/2)
    # draw a circle
    draw = ImageDraw.Draw(background,mode = 'RGB')
    draw.ellipse((int(center_x - circle_radius),
                  int(center_y - circle_radius),
                  int(center_x + circle_radius),
                  int(center_y + circle_radius),),
                 fill = background_color,
                 outline = 'gray',
                 width = 10,)
    # calculate the dot position in radians
    angle_radians = math.radians(ball_angle)
    dot_x = int(center_x + circle_radius * math.cos(angle_radians))
    dot_y = int(center_y + circle_radius * math.sin(angle_radians))
    draw.ellipse((dot_x - ball_radius,
                  dot_y - ball_radius,
                  dot_x + ball_radius,
                  dot_y + ball_radius,),
                 fill = 'green',
                 outline = 'green',
                 )
    return background

def generate_switch_sequence(sequence_length:int,
                             trial_variance:tuple = (5,10),
                             current_value:int = 0 # start with 0
                             ):
    """
    generate sequence of switches (could be both change point or odd ball condition)

    Parameters
    ----------
    sequence_length : int
        DESCRIPTION.
    trial_variance : tuple, optional
        DESCRIPTION. The default is (5,10).
    current_value : int, optional
        DESCRIPTION. The default is 0 # start with 0.

    Returns
    -------
    np.ndarray
        DESCRIPTION.

    """
    sequence = []
    position = 0
    while position < sequence_length:
        # determine the switching point randomly between a and b
        switch_point = np.random.randint(*trial_variance)
        # append the current value multiple times up to the switch point or until reaching the length
        append_count = min(switch_point,sequence_length - position)
        sequence.extend([current_value] * append_count)
        # update the position and switch the current value
        position += append_count
        current_value = 1 - current_value
    return np.array(sequence[:sequence_length])

def get_right_angle(angle:float,):
    """
    account for fill circle

    Parameters
    ----------
    angle : float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    return angle % 360

# generate sequence specifically for change point condition
def generate_change_point_sequence(n_trials:int = 100,
                                   trial_variance:tuple = (5,10),
                                   small_variance:tuple = (1,5),
                                   large_variance:tuple = (45,135),
                                   ) -> pd.core.frame.DataFrame:
    condition = 'ChangePoint'
    initial_angle = get_right_angle(np.random.uniform(0,360))
    # initialize
    df = dict(angle = [],
              condition = [],
              change = [],
              idx_trial = [],
              )
    angle = initial_angle
    # generate a sequence of 0s and 1s
    sequence = generate_switch_sequence(n_trials,
                                        trial_variance,
                                        np.random.choice([0,1]),
                                        )
    for idx_trial,seq in enumerate(sequence):
        df['idx_trial'].append(idx_trial)
        if idx_trial == 0:
            df['angle'].append(angle)
            df['condition'].append(condition)
            df['change'].append(0)
            #
            previous_seq = seq
        else:
            df['condition'].append(condition)
            if seq == previous_seq: # if the last trial the same as the current trial
                new_angle = get_right_angle(angle + np.random.choice([-1,1]) * np.random.randint(*small_variance))
                df['change'].append(0)
            else:# change point
                new_angle = get_right_angle(angle + np.random.choice([-1,1]) * np.random.randint(*large_variance))
                df['change'].append(1)
            previous_seq = seq
            angle = new_angle
            df['angle'].append(angle)
    df = pd.DataFrame(df)
    return df

# generate sequence specifically for odd ball condition
def generate_odd_ball_sequence(n_trials:int = 100,
                               trial_variance:tuple = (5,10),
                               small_variance:tuple = (1,5),
                               large_variance:tuple = (45,135),
                               ) -> pd.core.frame.DataFrame:
    condition = 'OddBall'
    initial_angle = get_right_angle(np.random.uniform(90,135))
    # generate a sequence of 0s and 1s
    sequence = generate_switch_sequence(n_trials,
                                        trial_variance,
                                        np.random.choice([0,1]),
                                        )
    # initialize
    df = dict(angle = [],
              condition = [],
              change = [],
              idx_trial = [],
              )
    angle = initial_angle
    for idx_trial,seq in enumerate(sequence):
        df['idx_trial'].append(idx_trial)
        if idx_trial == 0:
            df['angle'].append(angle)
            df['condition'].append(condition)
            df['change'].append(0)
            #
            previous_seq = seq
        else:
            df['condition'].append(condition)
            if seq == previous_seq: # if the last trial the same as the current trial
                new_angle = get_right_angle(angle + np.random.choice([-1,1]) * np.random.randint(*small_variance))
                df['change'].append(0)
                previous_seq = seq
                angle = new_angle
                df['angle'].append(new_angle)
            else:# change point
                new_angle = get_right_angle(angle + np.random.choice([-1,1]) * np.random.randint(*large_variance))
                df['change'].append(1)
                previous_seq = seq
                # angle = new_angle # don't update angle for the next
                df['angle'].append(new_angle)
    df = pd.DataFrame(df)
    return df

class build_image_dataset(Dataset):
    """
    generate the image at the spot we need them
    """
    def __init__(self,
                 condition:str = 'ChangePoint',
                 n_trials:int = 100,
                 trial_variance:tuple = (5,10),
                 small_variance:tuple = (1,5),
                 large_variance:tuple = (45,135),
                 #
                 fill_empty_space = 255,
                 batch_size:int = 32,
                 #
                 image_size:int = 256,
                 ball_radius:int = 10,
                 background_color:tuple = (255,255,255,),
                 circle_radius:int = 100,
                 transform_steps = None,
                 ):
        # specify the augmentation steps
        if transform_steps == None:
            self.transform_steps = concatenate_transform_steps(image_resize = image_size,
                                                               fill_empty_space = fill_empty_space,
                                                               grayscale = False,
                                                               )
        else:
            self.transform_steps = transform_steps
        
        # preallocate
        self.image_size = image_size
        self.condition = condition
        self.n_trials = n_trials
        
        self.ball_radius = ball_radius
        self.background_color = background_color
        self.circle_radius = circle_radius
        
        dfs = []
        for idx in range(batch_size):
            if condition == 'ChangePoint':
                df = generate_change_point_sequence(n_trials = n_trials,
                                                    trial_variance = trial_variance,
                                                    small_variance = small_variance,
                                                    large_variance = large_variance,)
            elif condition == 'OddBall':
                df = generate_odd_ball_sequence(n_trials = n_trials,
                                                trial_variance = trial_variance,
                                                small_variance = small_variance,
                                                large_variance = large_variance,)
            df['universe'] = idx
            dfs.append(df)
        dfs = pd.concat(dfs)
        dfs = dfs.sort_values(['idx_trial','universe'])
        self.dataframe = dfs
    
    def __len__(self,):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        # pick the line
        row = self.dataframe.iloc[index]
        # make labels
        label = torch.tensor([[row['angle']],[row['condition'] == 'ChangePoint']])
        # angle
        image = draw_image(ball_angle = row['angle'],
                           ball_radius = self.ball_radius,
                           background_size = self.image_size,
                           background_color = self.background_color,
                           circle_radius = self.circle_radius,
                           )
        # turn to tensor
        image = self.transform_steps(image)
        return image,label

def build_image_dataloader(condition:str = 'ChangePoint',
                           n_trials:int = 100,
                           trial_variance:tuple = (5,10),
                           small_variance:tuple = (1,5),
                           large_variance:tuple = (45,135),
                           #
                           fill_empty_space = 255,
                           batch_size:int = 32,
                           #
                           image_size:int = 256,
                           ball_radius:int = 10,
                           background_color:tuple = (255,255,255,),
                           circle_radius:int = 100,
                           transform_steps = None,
                           ) -> torch.utils.data.dataloader.DataLoader:
    dataset = build_image_dataset(condition = condition,
                                  n_trials = n_trials,
                                  trial_variance = trial_variance,
                                  small_variance = small_variance,
                                  large_variance = large_variance,
                                  fill_empty_space = fill_empty_space,
                                  batch_size = batch_size,
                                  image_size = image_size,
                                  ball_radius = ball_radius,
                                  background_color = background_color,
                                  circle_radius = circle_radius,
                                  transform_steps = transform_steps,
                                  )
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 2,
                            )
    return dataloader

if __name__ == "__main__":
    # batch_size = 8
    # dataloader_CP = build_image_dataloader(condition = 'ChangePoint',batch_size = batch_size)
    # dataloader_OB = build_image_dataloader(condition = 'OddBall',batch_size = batch_size)

    image = draw_image(20)  # Draw a dot at 180 degrees
    image.show()  # Display the image
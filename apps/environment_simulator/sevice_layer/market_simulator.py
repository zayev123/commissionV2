import gymnasium as gym
import ray
from gymnasium import spaces
from ray.rllib.algorithms import ppo
import numpy as np


class M(gym.Env):
    def __init__(self, env_config=None):
       # There are two actions, first will get reward of 1, second reward of -1. 
        if env_config:
            self.form = env_config.get("form", None)       
        self.action_space = spaces.Box(low=np.array([0,0,-2,0,1]),
                               high=np.array([1,1,2,1,20]),
                               dtype=np.float32)     #<gym.Space>
        self.observation_space = spaces.Discrete(2) #<gym.Space>
        
    
    def reset(self):
        state = 0
        #return <obs>
        return state
                           
    def step(self, action):

        # if we took an action, we were in state 1
        state = 1
    
        sum = action[0] + action[1] + action[2] + action[3] + action[4]
        if sum == 5:
            reward = 1
        else:
            reward = -1
            
        # regardless of the action, game is done after a single step
        done = True
        info = {}
        # return <obs>, <reward: float>, <done: bool>, <info: dict>

        return state, reward, done, info


class MyEnv(gym.Env):
    def __init__(self, env_config=None):
       # There are two actions, first will get reward of 1, second reward of -1. 
        self.action_space = spaces.Box(low=np.array([0,0,-2,0,1]),
                               high=np.array([1,1,2,1,20]),
                               dtype=np.float32)     #<gym.Space>
        self.observation_space = spaces.Discrete(2) #<gym.Space>
    
    def reset(self):
        state = 0
        #return <obs>
        return state
                           
    def step(self, action):

        # if we took an action, we were in state 1
        state = 1
    
        sum = action[0] + action[1] + action[2] + action[3] + action[4]
        if sum == 5:
            reward = 1
        else:
            reward = -1
            
        # regardless of the action, game is done after a single step
        done = True
        info = {}
        # return <obs>, <reward: float>, <done: bool>, <info: dict>

        return state, reward, done, info  
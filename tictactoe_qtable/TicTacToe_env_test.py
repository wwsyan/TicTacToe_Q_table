# -*- coding: utf-8 -*-
import gym
import random
import time
from TicTacToe_env import TicTacToeEnv
 

def randomAction(env_, mark): # 随机选择未占位的格子动作
    action_space = []
    for i, row in enumerate(env_.state):
        for j, one in enumerate(row):
            if one == 0: action_space.append((i,j))  
    action_pos = random.choice(action_space)
    action = {'mark':mark, 'pos':action_pos}
    return action

def randomFirst():
    if random.random() > 0.5: # 随机先后手
        first_, second_ = 'blue', 'red'
    else: 
        first_, second_ = 'red', 'blue'
    return first_, second_

env = TicTacToeEnv()
env.reset()
first, second = randomFirst()
while True:
    # 先手行动
    action = randomAction(env, first)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.5)
    if done: 
        env.reset()
        env.render()
        first, second = randomFirst()
        time.sleep(0.5)
        continue
    # 后手行动
    action = randomAction(env, second)
    state, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.5)
    if done: 
        env.reset()
        env.render()
        first, second = randomFirst()
        time.sleep(0.5)
        continue
        
        
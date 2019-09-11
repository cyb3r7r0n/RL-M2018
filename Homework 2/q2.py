#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


A_coordinate = np.array([0, 1])
A_prime_coordinate = np.array([4, 1])
B_coordinate = np.array([0, 3])
B_prime_coordinate = np.array([2, 3])


# In[3]:


def perform_action(s, a, GRID_SIZE, A_reward = 10.0, B_reward = 5.0):
    next_state = None
    r = None
    if s[0] == A_coordinate[0] and s[1] == A_coordinate[1]:
        next_state, r = A_prime_coordinate, A_reward
    elif s[0] == B_coordinate[0] and s[1] == B_coordinate[1]:
        next_state, r = B_prime_coordinate, B_reward
    else:
        potential_next_state = s + a
        if potential_next_state[0] < GRID_SIZE and potential_next_state[0] >= 0 and potential_next_state[1] < GRID_SIZE and potential_next_state[1] >= 0:
            next_state = potential_next_state
            r = 0.0
        else:
            next_state = s
            r = -1.0
    return next_state, r


# In[4]:


GRID_SIZE = 5
gamma = 0.9
THRESHOLD = 1e-5
actions = np.array([
    [0, -1], 
    [1, 0], 
    [0, 1], 
    [-1, 0]
])
pi_action = [0.25 for i in range(4)]


# In[11]:


COEFFICIENTS = np.zeros((GRID_SIZE**2, GRID_SIZE**2))
CONSTANTS = np.zeros((GRID_SIZE**2))
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        s = np.array([i, j])
        COEFFICIENTS[i*GRID_SIZE+j, i*GRID_SIZE+j] += -1
        for k in range(actions.shape[0]):
            a = actions[k]
            s_prime, r = perform_action(s, a, GRID_SIZE)
            COEFFICIENTS[i*GRID_SIZE+j, s_prime[0]*GRID_SIZE+s_prime[1]] += pi_action[k] * gamma
            CONSTANTS[i*GRID_SIZE+j] += -pi_action[k] * r
v = np.matmul(np.linalg.inv(COEFFICIENTS), CONSTANTS).reshape((GRID_SIZE, GRID_SIZE))
v = np.around(v, decimals=2)


# In[12]:


print(v)


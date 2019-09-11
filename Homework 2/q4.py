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
THRESHOLD = 1e-7
actions = np.array([
    [0, -1], 
    [1, 0], 
    [0, 1], 
    [-1, 0]
])
actions_name = ["L", "D", "R", "U"]


# In[5]:


v = np.zeros((GRID_SIZE, GRID_SIZE))
policy = [[ np.arange(4) for i in range(GRID_SIZE) ] for j in range(GRID_SIZE)]

def policy_evaluation(policy):
    COEFFICIENTS = np.zeros((GRID_SIZE**2, GRID_SIZE**2))
    CONSTANTS = np.zeros((GRID_SIZE**2))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = np.array([i, j])
            COEFFICIENTS[i*GRID_SIZE+j, i*GRID_SIZE+j] += -1
            for k in policy[i][j]:
                a = actions[k]
                s_prime, r = perform_action(s, a, GRID_SIZE)
                COEFFICIENTS[i*GRID_SIZE+j, s_prime[0]*GRID_SIZE+s_prime[1]] += (1/len(policy[i][j])) * gamma
                CONSTANTS[i*GRID_SIZE+j] += -(1/len(policy[i][j])) * r
    return np.matmul(np.linalg.inv(COEFFICIENTS), CONSTANTS).reshape((GRID_SIZE, GRID_SIZE))

def policy_improvement(v):
    policy_ = [[ np.arange(4) for i in range(GRID_SIZE) ] for j in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = np.array([i, j])
            values = []
            for k in range(len(actions)):
                a = actions[k]
                s_prime, r = perform_action(s, a, GRID_SIZE)
                values.append(r+gamma*v[s_prime[0], s_prime[1]])
            values = np.around(np.array(values), decimals=5)
            policy_[i][j] = np.where(values == np.max(values))[0]
    return policy_
         
iteration = 0
while True:
    v_ = policy_evaluation(policy)
    policy_ = policy_improvement(v_)
    print(iteration)
    
    if np.abs(v-v_).max()<THRESHOLD:
        policy = policy_
        v = v_
        break
    policy = policy_
    v = v_
    iteration += 1


# In[6]:


print(np.around(v, decimals=2))


# In[7]:


decision = []
for i in range(GRID_SIZE):
    decision.append([])
    for j in range(GRID_SIZE):
        string = ""
        for k in policy[i][j]:
            string += actions_name[k]
        decision[i].append(string)
decision = np.array(decision)
print(decision)


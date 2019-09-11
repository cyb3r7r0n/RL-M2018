#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def perform_action(s, a, GRID_SIZE):
    next_state = None
    r = None
    if (s[0] == 0 and s[1] == 0) or (s[0] == GRID_SIZE-1 and s[1] == GRID_SIZE-1):
        next_state = s
        r = 0
    else:
        potential_next_state = s + a
        if potential_next_state[0] < GRID_SIZE and potential_next_state[0] >= 0 and potential_next_state[1] < GRID_SIZE and potential_next_state[1] >= 0:
            next_state = potential_next_state
            r = -1
        else:
            next_state = s
            r = -1
    return next_state, r


# In[3]:


GRID_SIZE = 4
gamma = 1
THRESHOLD = 1e-4
actions = np.array([
    [0, -1], 
    [1, 0], 
    [0, 1], 
    [-1, 0]
])
actions_name = ["L", "D", "R", "U"]


# In[4]:


v = np.zeros((GRID_SIZE, GRID_SIZE))
policy = [[ np.arange(4) for i in range(GRID_SIZE) ] for j in range(GRID_SIZE)]

def policy_evaluation(policy, v):
    v_ = np.zeros_like(v)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            values = []
            s = np.array([i, j])
            for k in policy[i][j]:
                a = actions[k]
                s_prime, r = perform_action(s, a, GRID_SIZE)
                values.append((1/len(policy[i][j])) * (r + gamma * v[s_prime[0], s_prime[1]]))
            v_[s[0], s[1]] = np.sum(values)
    return v_

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

iteration_1 = 0
while True:
    print("Iteration 1: ", iteration_1)
    iteration_2 = 0
    while True:
        print("Iteration 2 ", iteration_2)
        v_ = policy_evaluation(policy, v)
        if np.abs(v-v_).max() < THRESHOLD:
            v = v_
            break
        v = v_
        iteration_2+=1
    policy_ = policy_improvement(v_)
    
    if np.abs(v-v_).max()<THRESHOLD:
        policy = policy_
        v = v_
        break
    policy = policy_
    v = v_
    iteration_1 += 1


# In[5]:


print(np.around(v, decimals=2))


# In[6]:


decision = []
for i in range(GRID_SIZE):
    decision.append([])
    for j in range(GRID_SIZE):
        string = ""
        if not ((i==0 and j==0) or (i==GRID_SIZE-1 and j==GRID_SIZE-1)):
            for k in policy[i][j]:
                string += actions_name[k]
        decision[i].append(string)
decision = np.array(decision)
print(decision)


# In[7]:


GRID_SIZE = 4
gamma = 1
THRESHOLD = 1e-4
actions = np.array([
    [0, -1], 
    [1, 0], 
    [0, 1], 
    [-1, 0]
])
pi_action = [0.25 for i in range(4)]
actions_name = ["L", "D", "R", "U"]
# interval = 20


# In[8]:


v = np.zeros((GRID_SIZE, GRID_SIZE))

def value_iteration(v):
    v_ = np.copy(v)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            values = []
            s = np.array([i, j])
            for k in range(len(actions)):
                a = actions[k]
                s_prime, r = perform_action(s, a, GRID_SIZE)
                values.append(pi_action[k] * (r + gamma * v[s_prime[0], s_prime[1]]))
            v_[s[0], s[1]] = np.max(values)
    return v_

def policy_calculate(v):
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

iteration_1 = 0
while True:
    print("Iteration 1: ", iteration_1)
    print(v)
    print()
    v_ = value_iteration(v)
    if np.abs(v-v_).max()<THRESHOLD:
        v = v_
        break
    v = v_
    iteration_1 += 1
policy = policy_calculate(v)


# In[9]:


print(v)


# In[10]:


decision = []
for i in range(GRID_SIZE):
    decision.append([])
    for j in range(GRID_SIZE):
        string = ""
        if not ((i==0 and j==0) or (i==GRID_SIZE-1 and j==GRID_SIZE-1)):
            for k in policy[i][j]:
                string += actions_name[k]
        decision[i].append(string)
decision = np.array(decision)
print(decision)


# In[ ]:





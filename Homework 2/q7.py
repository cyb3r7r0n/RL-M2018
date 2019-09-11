#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


max_cars_loc_1 = 20
max_cars_loc_2 = 20
mean_rental_requests_loc_1 = 3
mean_rental_requests_loc_2 = 4
rent_reward_per_car = 10
mean_cars_returned_loc_1 = 3
mean_cars_returned_loc_2 = 2
gamma = 0.9
move_cost_per_car = 3
max_cars_move_per_day = 5
THRESHOLD = 1e-4
parking_cost = 4
max_parking_loc_1 = 10
max_parking_loc_2 = 10


# In[3]:


poisson_memo = {}
def poisson_probability(x, LAMBDA):
    key = str(x) + "_" + str(LAMBDA)
    if key not in poisson_memo:
        poisson_memo[key] = poisson.pmf(x, LAMBDA)
    return poisson_memo[key]
    


# In[4]:


def perform_action(s, a, v):
    Gt = 0.0
    num_cars_loc_1_after_moving = min(s[0]-a, max_cars_loc_1)    
    num_cars_loc_2_after_moving = min(s[1]+a, max_cars_loc_2)
    
    for num_rental_requests_loc_1 in range(11):
        for num_rental_requests_loc_2 in range(11):  
            prob_rental_requests = poisson_probability(num_rental_requests_loc_1, mean_rental_requests_loc_1) * poisson_probability(num_rental_requests_loc_2, mean_rental_requests_loc_2)
            
            num_cars_presently_at_loc_1 = num_cars_loc_1_after_moving
            num_cars_presently_at_loc_2 = num_cars_loc_2_after_moving
            
            valid_rental_requests_loc_1 = min(num_cars_loc_1_after_moving, num_rental_requests_loc_1)
            valid_rental_requests_loc_2 = min(num_cars_loc_2_after_moving, num_rental_requests_loc_2)
            
            r = (valid_rental_requests_loc_1 + valid_rental_requests_loc_2) * rent_reward_per_car
            
            num_cars_presently_at_loc_1 = num_cars_presently_at_loc_1 - valid_rental_requests_loc_1            
            num_cars_presently_at_loc_2 = num_cars_presently_at_loc_2- valid_rental_requests_loc_2
            
            for num_cars_returned_loc_1 in range(10):
                for num_cars_returned_loc_2 in range(10):
                    r_ = r
                    prob_car_returns = poisson_probability(num_cars_returned_loc_1, mean_cars_returned_loc_1) * poisson_probability(num_cars_returned_loc_2, mean_cars_returned_loc_2)
                    num_cars_presently_at_loc_1 = min(num_cars_presently_at_loc_1 + num_cars_returned_loc_1, max_cars_loc_1)
                    num_cars_presently_at_loc_2 = min(num_cars_presently_at_loc_2 + num_cars_returned_loc_2, max_cars_loc_2)
                    if num_cars_presently_at_loc_1 > max_parking_loc_1:
                        r_ -= parking_cost
                    if num_cars_presently_at_loc_2 > max_parking_loc_2:
                        r_ -= parking_cost
                    p = prob_rental_requests * prob_car_returns
                    Gt += p * (r_ + gamma * v[num_cars_presently_at_loc_1, num_cars_presently_at_loc_2])
            
    if a > 0:
        Gt -= move_cost_per_car * (a-1)
    else:
        Gt -= move_cost_per_car * (-a)
    return Gt             
                    


# In[5]:


v = np.zeros((max_cars_loc_1 + 1, max_cars_loc_2 + 1))
policy = np.zeros(v.shape, dtype=np.int)
actions = np.arange(-max_cars_move_per_day, max_cars_move_per_day + 1)

iteration_policies = []

iteration_1 = 0

def policy_evaluation(policy, v):
    v_ = v.copy()
    for i in range(max_cars_loc_1 + 1):
        for j in range(max_cars_loc_2 + 1):
            s = np.array([i, j])
            a = policy[i, j]
            v_[i, j] = perform_action(s, a, v_)
    return v_

def policy_improvement(policy, v):
    policy_ = policy.copy()
    for i in range(max_cars_loc_1 + 1):
        for j in range(max_cars_loc_2 + 1):
            s = np.array([i, j])
            values_per_action = []
            for a in actions:
                if -j <= a <= i:
                    values_per_action.append(perform_action(s, a, v))
                else:
                    values_per_action.append(-np.inf)
            policy_[i, j] = actions[np.argmax(values_per_action)]
    return policy_
            
while True:
    iteration_2 = 0
    while True:
        v_ = policy_evaluation(policy, v)
        difference = np.abs(v_ - v).max()
        print("Iteration: ", iteration_2, " Difference :", difference)
        if difference < THRESHOLD:
            v = v_
            break
        v = v_
        iteration_2 += 1

    stable = True
    policy_ = policy_improvement(policy, v)
            
    if stable and np.any(policy_ != policy):
        stable = False
    print("Iteration: ", iteration_1, ' policy stable {}'.format(stable))
    iteration_policies.append(policy)
    iteration_1 += 1
    policy = policy_
    if stable:
        break


# In[6]:


_, axes = plt.subplots(len(iteration_policies), 1, figsize=(8, (len(iteration_policies)+1) * 5))

for i in range(len(iteration_policies)):
    axes[i].set_title("Iteration "+str(i+1))
    sns.heatmap(np.flipud(iteration_policies[i]), cmap="YlGnBu", ax=axes[i])
    axes[i].set_xlabel("cars at second location")
    axes[i].set_ylabel("cars at first location")
plt.show()
plt.close("all")


# In[7]:


print(np.flipud(policy))


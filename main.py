import numpy as np
import matplotlib.pyplot as plt

def performKArmedBandit(K, MAX_TIME, epsilon, alpha = None, c = 0, initial_value = 0, non_stationary = False):

    q_star_std = 1
    q_star = np.random.normal(loc = 0, scale = q_star_std, size = K)

    Qt = [initial_value for i in range(K)]
    numTimesActionDone = np.zeros((K))
    averageReward = 0
    averageRewardArray = []
    optimal_action = []

    for t in range(MAX_TIME):
        actionToBeDone = None
        if np.random.rand() >= epsilon:
            # exploit
            actionToBeDone = np.argmax(Qt + c * np.sqrt(np.log(t+1)/(numTimesActionDone + 1e-10)))
        else:
            # explore
            actionToBeDone = np.random.randint(0, K)
        rewardObtained = np.random.normal(loc = q_star[actionToBeDone], scale = q_star_std)
        numTimesActionDone[actionToBeDone] += 1
        if alpha == None:
            Qt[actionToBeDone] += (1/numTimesActionDone[actionToBeDone]) * (rewardObtained - Qt[actionToBeDone])
        else:
            Qt[actionToBeDone] += alpha * (rewardObtained - Qt[actionToBeDone])
        # averageReward = (averageReward*t + rewardObtained)/(t+1)
        averageRewardArray.append(rewardObtained)
        if actionToBeDone == np.argmax(q_star):
            optimal_action.append(1)
        else:
            optimal_action.append(0)
        if non_stationary:
            q_star += np.random.normal(loc = 0, scale = 0.01, size = K)
    return averageRewardArray, optimal_action


def run_test_bed(N, K, MAX_TIME, epsilon, alpha = None, c = 0, initial_value = 0, non_stationary = False):
    average_rewards = []
    optimal_action = []
    for i in range(N):
        averageReward, optimalAction = performKArmedBandit(K, MAX_TIME, epsilon, alpha, c, initial_value, non_stationary)
        average_rewards.append(averageReward)
        optimal_action.append(optimalAction)

    average_rewards = np.array(average_rewards)
    average_rewards = average_rewards.mean(axis=0)

    optimal_action = np.array(optimal_action)
    optimal_action = optimal_action.mean(axis=0)

    return average_rewards, optimal_action * 100

def q1_1():
    average_rewards_1, optimal_action_1 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = None, initial_value = 0, non_stationary = True)
    average_rewards_2, optimal_action_2 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = 0.1, initial_value = 0, non_stationary = True)

    f, axarr = plt.subplots(1, 2, figsize=(12,5))

    axarr[0].title.set_text("Average Rewards Non-Stationary")
    axarr[0].plot(average_rewards_1, label="mean alpha")
    axarr[0].plot(average_rewards_2, label="constant alpha")
    axarr[0].legend()
    axarr[0].set_xlabel("t")
    axarr[0].set_ylabel("Average Reward")

    axarr[1].title.set_text("Optimal Action Non-Stationary")
    axarr[1].plot(optimal_action_1, label="mean alpha")
    axarr[1].plot(optimal_action_2, label="constant alpha")
    axarr[1].legend()
    axarr[1].set_xlabel("t")
    axarr[1].set_ylabel("Optimal Action")

    plt.savefig("q1_1.png")
    # plt.show()
    plt.close("all")

def q1_2():
    average_rewards_1, optimal_action_1 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = None, initial_value = 0, non_stationary = False)
    average_rewards_2, optimal_action_2 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = 0.1, initial_value = 0, non_stationary = False)

    f, axarr = plt.subplots(1, 2, figsize=(12,5))

    axarr[0].title.set_text("Average Rewards Stationary")
    axarr[0].plot(average_rewards_1, label="mean alpha")
    axarr[0].plot(average_rewards_2, label="constant alpha")
    axarr[0].legend()
    axarr[0].set_xlabel("t")
    axarr[0].set_ylabel("Average Reward")

    axarr[1].title.set_text("Optimal Action Stationary")
    axarr[1].plot(optimal_action_1, label="mean alpha")
    axarr[1].plot(optimal_action_2, label="constant alpha")
    axarr[1].legend()
    axarr[1].set_xlabel("t")
    axarr[1].set_ylabel("Optimal Action")

    plt.savefig("q1_2.png")
    # plt.show()
    plt.close("all")

def q2():
    average_rewards_1, optimal_action_1 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0, alpha = 0.1, initial_value = 5, non_stationary = False)
    average_rewards_2, optimal_action_2 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = 0.1, initial_value = 0, non_stationary = False)

    average_rewards_3, optimal_action_3 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0, alpha = 0.1, initial_value = 5, non_stationary = True)
    average_rewards_4, optimal_action_4 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = 0.1, initial_value = 0, non_stationary = True)

    f, axarr = plt.subplots(1, 2, figsize=(12,5))

    axarr[0].title.set_text("Optimal Action Stationary")
    axarr[0].plot(optimal_action_1, label="optimistic")
    axarr[0].plot(optimal_action_2, label="E-Greedy")
    axarr[0].legend()
    axarr[0].set_xlabel("t")
    axarr[0].set_ylabel("Optimal Action")

    axarr[1].title.set_text("Optimal Action Non-Stationary")
    axarr[1].plot(optimal_action_3, label="optimistic")
    axarr[1].plot(optimal_action_4, label="E-Greedy")
    axarr[1].legend()
    axarr[1].set_xlabel("t")
    axarr[1].set_ylabel("Optimal Action")

    plt.savefig("q2.png")
    # plt.show()
    plt.close("all")

def q4_1():
    average_rewards_1, optimal_action_1 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0, alpha = 0.1, c = 2, initial_value = 0, non_stationary = True)
    average_rewards_2, optimal_action_2 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = 0.1, c = 0, initial_value = 0, non_stationary = True)
    average_rewards_3, optimal_action_3 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0, alpha = 0.1, c = 0, initial_value = 5, non_stationary = True)

    f, axarr = plt.subplots(1, 2, figsize=(12,5))

    axarr[0].title.set_text("Average Rewards Non-Stationary")
    axarr[0].plot(average_rewards_1, label="UCB")
    axarr[0].plot(average_rewards_2, label="E-Greedy")
    axarr[0].plot(average_rewards_3, label="Optimistic")
    axarr[0].legend()
    axarr[0].set_xlabel("t")
    axarr[0].set_ylabel("Optimal Action")

    axarr[1].title.set_text("Optimal Action Non-Stationary")
    axarr[1].plot(optimal_action_1, label="UCB")
    axarr[1].plot(optimal_action_2, label="E-Greedy")
    axarr[1].plot(optimal_action_3, label="Optimistic")
    axarr[1].legend()
    axarr[1].set_xlabel("t")
    axarr[1].set_ylabel("Optimal Action")

    plt.savefig("q4_1.png")
    # plt.show()
    plt.close("all")

def q4_2():
    average_rewards_1, optimal_action_1 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0, alpha = 0.1, c = 2, initial_value = 0, non_stationary = False)
    average_rewards_2, optimal_action_2 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0.1, alpha = 0.1, c = 0, initial_value = 0, non_stationary = False)
    average_rewards_3, optimal_action_3 = run_test_bed(N = 2000, K = 10, MAX_TIME = 1000, epsilon = 0, alpha = 0.1, c = 0, initial_value = 5, non_stationary = False)

    f, axarr = plt.subplots(1, 2, figsize=(12,5))

    axarr[0].title.set_text("Average Rewards Stationary")
    axarr[0].plot(average_rewards_1, label="UCB")
    axarr[0].plot(average_rewards_2, label="E-Greedy")
    axarr[0].plot(average_rewards_3, label="Optimistic")
    axarr[0].legend()
    axarr[0].set_xlabel("t")
    axarr[0].set_ylabel("Optimal Action")

    axarr[1].title.set_text("Optimal Action Stationary")
    axarr[1].plot(optimal_action_1, label="UCB")
    axarr[1].plot(optimal_action_2, label="E-Greedy")
    axarr[1].plot(optimal_action_3, label="Optimistic")
    axarr[1].legend()
    axarr[1].set_xlabel("t")
    axarr[1].set_ylabel("Optimal Action")

    plt.savefig("q4_2.png")
    # plt.show()
    plt.close("all")

# q1_1()
# print("q1_1 completed")
# q1_2()
# print("q1_2 completed")
# q2()
# print("q2 completed")
# q4_1()
# print("q4_1 completed")
# q4_2()
# print("q4_2 completed")

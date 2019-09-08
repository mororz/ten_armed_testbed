import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict


class Bandit:
    def __init__(self, k: int):
        self.__k = k

        self.__values = np.random.randn(k)
        self.__optimal_arm = np.argmax(self.__values)

    def get_n_arms(self):
        return self.__k

    def get_values(self):
        return self.__values

    def get_optimal_arm(self):
        return self.__optimal_arm

    def reset_values(self):
        self.__values = np.random.randn(self.__k)

    def get_reward(self, arm: int):
        reward = np.random.normal(self.__values[arm])

        return reward

    def visualize_bandit(self):
        samples = [np.random.normal(loc=x, size=10000) for x in self.__values]
        columns_names = ['arm_' + str(i) for i in range(self.__k)]
        df = pd.DataFrame(dict(zip(columns_names, samples)))

        sns.violinplot(data=df.melt(), x='variable', y='value')

        plt.xlabel("Arms")
        plt.ylabel("Reward distribution")
        plt.axhline(max(self.__values))
        plt.axhline(0, ls='--', c='grey')
        plt.show()


def greedy(initial_value: float):
    def inner(bandit: Bandit,
              n_steps: int = 1000) -> List[float]:

        n_arms = bandit.get_n_arms()
        Q = [initial_value] * n_arms
        N = [0] * n_arms
        rewards = []

        for _ in range(n_steps):
            A = np.argmax(Q)
            R = bandit.get_reward(A)
            N[A] += 1
            Q[A] = Q[A] + (R - Q[A]) / N[A]

            rewards.append(R)
        return rewards

    return inner


def epsilon_greedy(initial_value: float, epsilon: float):
    def inner(bandit: Bandit,
              n_steps: int = 1000) -> List[float]:

        n_arms = bandit.get_n_arms()
        Q = [initial_value] * n_arms
        N = [0] * n_arms
        rewards = []

        for _ in range(n_steps):
            coin = np.random.random()
            if coin < epsilon:
                A = np.random.randint(n_arms)
            else:
                A = np.argmax(Q)
            R = bandit.get_reward(A)
            N[A] += 1
            Q[A] = Q[A] + (R - Q[A]) / N[A]

            rewards.append(R)
        return rewards

    return inner


def run_experiment(methods: dict,
                   n_arms: int = 10,
                   n_steps: int = 1000,
                   n_runs: int = 2000) -> None:

    rewards = defaultdict(list)

    for _ in range(n_runs):
        bandit = Bandit(n_arms)
        for func, params in methods.items():
            for param_set in params:
                function = eval(func + str(param_set))

                rewards[func + '_' + str(param_set)].append(
                    function(bandit, n_steps))

    rewards = {k: np.array(v) for k, v in rewards.items()}
    mean_rewards = {k: np.mean(v, axis=0) for k, v in rewards.items()}

    for k, v in mean_rewards.items():
        plt.plot(v, label=k)

    plt.xlabel('Step')
    plt.ylabel('Mean reward')
    plt.title('Algorithms comparison')
    plt.legend()
    plt.savefig('models_comparison.pdf')


if __name__ == '__main__':
    algos = {'greedy': [], 'epsilon_greedy': []}
    enough = True

    n_arms = 10
    n_steps = 1000
    n_runs = 2000

    change_settings = input("Do you want to change settings for\n"
                            "experinment? (10 arms, 1000 steps, 2000 runs) "
                            "[y/n]: ")

    if change_settings == 'y':
        n_arms = int(input("Choose number of arms: "))
        n_steps = int(input("Choose number of steps per run: "))
        n_runs = int(input("Choose number of runs: "))

    while enough:

        print("Choose number of algorithm to add to analysis:")
        print("1. greedy")
        print("2. epsilon-greedy")
        print()

        choice = input("Number of algorithm: ")

        if choice == '1':
            init = input("Choose initial value for greedy algorithm (float): ")

            algos['greedy'].append((float(init),))
        elif choice == '2':
            init = input("Choose initial value for "
                         "epsilon greedy algorithm (float): ")
            epsilon = input("Choose epsilon for "
                            "epsilon greedy algprithm (float): ")

            algos['epsilon_greedy'].append((float(init), float(epsilon)))

        print()
        proceed = input("Add more algos for analysis? [y/n]: ")

        if proceed == 'y':
            continue
        else:
            break

    run_experiment(algos, n_arms, n_steps, n_runs)

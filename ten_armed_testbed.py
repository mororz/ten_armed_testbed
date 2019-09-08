import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


class Bandit:
    def __init__(self, k):
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

    def get_reward(self, arm):
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


def sample_average_method(choice_func, initial_value, bandit, n_steps):
    n_arms = bandit.get_n_arms()
    Q = [initial_value] * n_arms
    N = [0] * n_arms
    rewards = []

    for t in range(n_steps):
        A = choice_func(Q, N, t)
        R = bandit.get_reward(A)
        N[A] += 1
        Q[A] = Q[A] + (R - Q[A]) / N[A]

        rewards.append(R)
    return rewards


def greedy_choice(Q, N, t):
    return np.argmax(Q)


def epsilon_greedy_choice(epsilon):
    def inner(Q, N, t):
        coin = np.random.random()
        if coin < epsilon:
            n_arms = len(Q)
            A = np.random.randint(n_arms)
        else:
            A = np.argmax(Q)

        return A

    return inner


def ucb_choice(c):
    def inner(Q, N, t):
        return np.argmax(Q + c * np.sqrt(np.log(t) / N))

    return inner


def greedy(initial_value):
    def inner(bandit, n_steps):
        return sample_average_method(greedy_choice, initial_value,
                                     bandit, n_steps)

    return inner


def epsilon_greedy(initial_value, epsilon):
    def inner(bandit, n_steps):
        return sample_average_method(epsilon_greedy_choice(epsilon),
                                     initial_value, bandit, n_steps)

    return inner


def ucb(initial_value, c):
    def inner(bandit, n_steps):
        return sample_average_method(ucb_choice(c),
                                     initial_value, bandit, n_steps)

    return inner


def run_experiment(methods,
                   n_arms=10,
                   n_steps=1000,
                   n_runs=2000):

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
    algos = {'greedy': [], 'epsilon_greedy': [], 'ucb': []}
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
        print("3. UCB")
        print()

        choice = input("Number of algorithm: ")

        if choice == '1':
            init = input("Choose initial value for greedy algorithm (float): ")

            algos['greedy'].append((float(init),))
        elif choice == '2':
            init = input("Choose initial value for "
                         "epsilon greedy algorithm (float): ")
            epsilon = input("Choose epsilon for "
                            "epsilon greedy algorithm (float): ")

            algos['epsilon_greedy'].append((float(init), float(epsilon)))
        elif choice == '3':
            init = input("Choose initial value for "
                         "UCB algorithm (float): ")
            epsilon = input("Choose c for "
                            "UCB algorithm (float): ")

            algos['ucb'].append((float(init), float(epsilon)))

        print()
        proceed = input("Add more algos for analysis? [y/n]: ")

        if proceed == 'y':
            continue
        else:
            break

    run_experiment(algos, n_arms, n_steps, n_runs)

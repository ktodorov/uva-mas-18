from gridworld import GridWorld
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import heatmap, annotate_heatmap

def get_random_start_pos(gridworld):
    width, height = len(gridworld.grid), len(gridworld.grid[0])
    
    while True:
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)

        if(gridworld.grid[x][y] == ' '):
            return (x,y)

def eps_greedy(state, Q, eps):
    rnd = random.random()
    if(rnd < eps):
        return random.randint(0, 3)
    else:
        qs = [Q[(state, 0)], Q[(state, 1)], Q[(state, 2)], Q[(state, 3)]]
        return np.argmax(qs)

def qDictToMatrix(Q, gridworld):
    width, height = len(gridworld.grid), len(gridworld.grid[0])
    m = np.empty((width, height))

    for x in range(width):
        for y in range(height):
            state = (x,y)
            qs = [Q[((state), 0)], Q[(state, 1)], Q[(state, 2)], Q[(state, 3)]]
            m[x][y] = np.max(qs)

            r = gridworld.get_state_reward(state)
            if(r != gridworld.move_value):
                m[x][y] = r

            if(gridworld.grid[x][y] == '#'):
                m[x][y] = None

    return m

def plotQ(Q, gridworld, title=''):
    m = qDictToMatrix(Q, gridworld)

    plt.title(title)
    im, _ = heatmap(m.transpose())
    annotate_heatmap(im)
    plt.show()

def SARSA(gridworld, episodes = 100, eps = 0.1, lr = 0.1, g = 1):
    """
    Runs episodes with epislon-greedy policy and updates q-values with SARSA.

    Arguments:
        gridworld   : The MPD to sample from of type GridWorld.
        episodes    : Number of episodes to simulate.
        eps         : Chance to sample random action from epsilon-greedy policy.
        lr          : Learning rate to control how fast SARSA converges.
        g           : Discount factor for future rewards

    Returns:
        Q           : Dictionary of q-values after completing run.
    """
    Q = defaultdict(float)

    for _ in range(episodes):
        start = get_random_start_pos(gridworld)
        terminal = False
        state = start

        last_sar = (state, -1, 0)
        while not terminal:
            # sample next action from policy
            action = eps_greedy(state, Q, eps)
            # take step and get result
            new_state, reward, terminal = gridworld.move_dir(state, action)

            if last_sar[1] != -1:
                s, a, r = last_sar
                # SARSA update
                Q[(s,a)] = Q[(s,a)] + lr * (r + g * Q[(state, action)] - Q[(s,a)])

            last_sar = (state, action, reward)
            state = new_state
        
        # terminal state update
        s, a, r = last_sar
        Q[(s,a)] = Q[(s,a)] + lr * (r - Q[(s,a)])
    
    return Q

def QLearning(gridworld, episodes = 100, eps = 0.2, lr = 0.1, g = 1):
    """
    Runs episodes with epislon-greedy policy and updates q-values with Q-Learning.

    Arguments:
        gridworld   : The MPD to sample from of type GridWorld.
        episodes    : Number of episodes to simulate.
        eps         : Chance to sample random action from epsilon-greedy policy.
        lr          : Learning rate to control how fast SARSA converges.
        g           : Discount factor for future rewards

    Returns:
        Q           : Dictionary of q-values after completing run.
    """
    Q = defaultdict(float)

    for _ in range(episodes):
        start = get_random_start_pos(gridworld)
        terminal = False
        state = start

        while not terminal:
            # sample next action from policy
            action = eps_greedy(state, Q, eps)
            # take step and get result
            new_state, reward, terminal = gridworld.move_dir(state, action)

            # Q-Learning update
            s, a, r, s2 = (state, action, reward, new_state)
            max_Qs2 = np.max([Q[((s2), 0)], Q[(s2, 1)], Q[(s2, 2)], Q[(s2, 3)]])
            Q[(s,a)] = Q[(s,a)] + lr * (r + g * max_Qs2 - Q[(s,a)])

            state = new_state

    return Q

def main():
    grid = ''
    with open("grid.lay","r") as file:
        grid = file.read()

    eps = 0.2
    episodes = 100

    random.seed(1)
    gw = GridWorld(grid)
    Q = SARSA(gw, episodes=episodes, eps=eps)
    plotQ(Q, gw, f'SARSA after {episodes} episodes')

    random.seed(1)
    Q = QLearning(gw, episodes=episodes, eps=eps)
    plotQ(Q, gw, f'Q-Learning after {episodes} episodes')

if __name__ == '__main__':
    main()
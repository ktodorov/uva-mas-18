from gridworld import GridWorld
import random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from plot_utils import heatmap, annotate_heatmap
import math

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

def qDictToPolicy(Q, gridworld):
    width, height = len(gridworld.grid), len(gridworld.grid[0])
    m = np.empty((width, height))

    for x in range(width):
        for y in range(height):
            state = (x,y)
            qs = [Q[((state), 0)], Q[(state, 1)], Q[(state, 2)], Q[(state, 3)]]
            m[x][y] = np.argmax(qs)

            r = gridworld.get_state_reward(state)
            if(r != gridworld.move_value):
                m[x][y] = None

            if(gridworld.grid[x][y] == '#'):
                m[x][y] = None

    return m

def plotQ(Q, gridworld, title=''):
    m = qDictToMatrix(Q, gridworld)

    plt.title(title)
    im, _ = heatmap(m.transpose())
    annotate_heatmap(im)
    plt.show()

def plotArrow(x, y, action):
    color = 'black'

    dx = 0
    dy = 0
    arrowWidth = 0.15
    arrowLength = 0.3
    arrowHeadWidth = 0.35
    arrowHeadLength = 0.25

    x = x 
    y = y + 1

    if action == 0:
        dx = 0
        dy = arrowLength
        x = x - (arrowLength + arrowHeadLength) / 2
    elif action == 1:
        dx = arrowLength
        dy = 0
        arrowWidth = arrowWidth
        arrowHeadWidth = arrowHeadWidth / 1.2
        y = y - (arrowWidth + arrowHeadWidth) / 2
    elif action == 2:
        dx = 0
        dy = -arrowLength
        x = x + (arrowLength + arrowHeadLength) / 2
    elif action == 3:
        dx = -arrowLength
        dy = 0
        arrowWidth = arrowWidth
        arrowHeadWidth = arrowHeadWidth / 1.2
        y = y + (arrowWidth + arrowHeadWidth) / 2
    
    plt.arrow(y, x, dx, dy, head_width=arrowHeadWidth, head_length=arrowHeadLength, width=arrowWidth, color=color)

def plotPolicy(Q, gridworld, title = ""):
    m = qDictToPolicy(Q, gridworld)
    height, width = len(m), len(m[0])

    for x in range(width):
        for y in range(height):
            if(not math.isnan(m[x][y])):
                plotArrow(height - y, x, m[x][y])
            if(gridworld.grid[x][y] == 'G' or gridworld.grid[x][y] == 'P'):
                plt.text(x + 0.75, height - y - 0.15, gridworld.grid[x][y], fontsize = 25)

    xTicks = np.arange(0, width + 2, 1)
    subXTicks = np.arange(0.5, width + 1.5, 1)
    yTicks = np.arange(0, height + 2, 1)
    subYTicks = np.arange(0.5, height + 1.5, 1)

    axes = plt.gca()
    axes.set_xticks(xTicks)
    axes.set_xticks(subXTicks, minor=True)
    axes.set_yticks(yTicks)
    axes.set_yticks(subYTicks, minor=True)
    
    axes.set_xlim([0.5, width + 0.5])
    axes.set_ylim([0.5, height + 0.5])
    

    axes.grid(which='minor', alpha=0.5)
    plt.title(title)
    
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
    episodes = 1000

    random.seed(1)
    gw = GridWorld(grid)
    Q = SARSA(gw, episodes=episodes, eps=eps)
    # plotQ(Q, gw, f'SARSA after {episodes} episodes')
    plotPolicy(Q, gw, f'SARSA: greedy-policy after {episodes} episodes')

    random.seed(1)
    Q = QLearning(gw, episodes=episodes, eps=eps)
    # plotQ(Q, gw, f'Q-Learning after {episodes} episodes')
    plotPolicy(Q, gw, f'Q-Learning: greedy-policy after {episodes} episodes')

if __name__ == '__main__':
    main()
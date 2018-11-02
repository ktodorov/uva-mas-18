from gridworld import Gridworld
from policyType import PolicyType
import matplotlib.pyplot as plt
import numpy as np

rows = 5
columns = 10
startX = 4
startY = 0
greedification = 1
epsGreedy = 0.9
maxEpisodes = 500
runs = 1

def getReward(x, y):
    if x == rows - 1 and y > 0:
        if y == columns - 1:
            return 10

        return -100

    return -1

def isTerminalState(state):
    return state[0] == rows - 1 and state[1] > 0

def plot(rewards1, color1, label1, rewards2, color2, label2, xlabel, ylabel):
    x = [i for i in range(len(rewards1))]
    plt.plot(x, rewards1, color=color1, label=label1)
    plt.plot(x, rewards2, color=color2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plotQuiver(rewards):
    x = [0 for i in range(len(rewards))]
    plt.quiver(x, rewards)
    plt.show()


def getMovingAverage(listToAverage):
    N = 10
    cumsum, movingAverages = [0], []

    for i, x in enumerate(listToAverage, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= N:
            movingAverage = (cumsum[i] - cumsum[i-N])/N
            movingAverages.append(movingAverage)

    return movingAverages

def plotSpecialStates(gridworld):
    gridworld.plotLetter(4, 0, 'S')

    gridworld.colorRectangle(4, 1)
    gridworld.colorRectangle(4, 2)
    gridworld.colorRectangle(4, 3)
    gridworld.colorRectangle(4, 4)
    gridworld.colorRectangle(4, 5)
    gridworld.colorRectangle(4, 6)
    gridworld.colorRectangle(4, 7)
    gridworld.colorRectangle(4, 8)

    gridworld.plotLetter(4, 9, 'G')

sarsaEpisodeRewards = np.empty(maxEpisodes)
qLearningEpisodeRewards = np.empty(maxEpisodes)

for i in range(runs):
    gridworld = Gridworld(rows, columns, startX, startY, greedification, True, False, epsGreedy, getReward, isTerminalState, PolicyType.SARSA)
    currentEpisodeRewards = gridworld.runSimulation(maxEpisodes=maxEpisodes)

    plotSpecialStates(gridworld)
    gridworld.plotOptimalPath(title="SARSA - Optimal path")
    if i > 0:
        for j in range(maxEpisodes):
            sarsaEpisodeRewards[j] = (sarsaEpisodeRewards[j] + currentEpisodeRewards[j]) / 2
    else:
        sarsaEpisodeRewards = currentEpisodeRewards

for i in range(runs):
    gridworld = Gridworld(rows, columns, startX, startY, greedification, True, False, epsGreedy, getReward, isTerminalState, PolicyType.QLEARNING)
    currentEpisodeRewards = gridworld.runSimulation(maxEpisodes=maxEpisodes)

    plotSpecialStates(gridworld)
    gridworld.plotOptimalPath(title="Q-Learning - Optimal path")
    if i > 0:
        for j in range(maxEpisodes):
            qLearningEpisodeRewards[j] = (qLearningEpisodeRewards[j] + currentEpisodeRewards[j]) / 2
    else:
        qLearningEpisodeRewards = currentEpisodeRewards

# sarsaAverage = getMovingAverage(sarsaEpisodeRewards)
# qLearningAverage = getMovingAverage(qLearningEpisodeRewards)
# plot(sarsaAverage, "red", "SARSA", qLearningAverage, "blue", "Q-Learning", "Episodes", "Sum of rewards during episode")
# plotQuiver(sarsaEpisodeRewards)
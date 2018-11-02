import numpy as np
import random
from collections import defaultdict
from direction import Direction
from policyType import PolicyType
import matplotlib.pyplot as plt
from plot_utils import heatmap, annotate_heatmap

def decision(probability):
    result = random.random() < probability
    return result

def selectRandom(*args):
    position = random.randint(0, len(args) - 1)
    return args[position]

class Gridworld:
    discountFactor = 0.0
    epsilonGreedy = 0.0
    infiniteGame = True

    currentPosition = (-1, -1)

    def __init__(
        self, 
        rows,
        columns,
        startX,
        startY,
        discountFactor,
        greedification,
        infiniteGame,
        epsilonGreedy,
        rewardFunction,
        terminateFunction,
        policyType):

        self.rows = rows
        self.columns = columns
        self.startX = startX
        self.startY = startY
        self.discountFactor = discountFactor
        self.greedification = greedification
        self.infiniteGame = infiniteGame
        self.epsilonGreedy = epsilonGreedy
        self.getReward = rewardFunction
        self.isTerminal = terminateFunction
        self.policyType = policyType

        self.actionValueFunctions = [[
            {
                Direction.UP : 0,
                Direction.RIGHT : 0,
                Direction.DOWN : 0,
                Direction.LEFT : 0
            } for _ in range(self.columns)] for _ in range(self.rows)]
                
    def getOptimalActionValue(self, currentState):
        actionValues = [
            self.actionValueFunctions[currentState[0]][currentState[1]][Direction.UP], 
            self.actionValueFunctions[currentState[0]][currentState[1]][Direction.RIGHT], 
            self.actionValueFunctions[currentState[0]][currentState[1]][Direction.DOWN], 
            self.actionValueFunctions[currentState[0]][currentState[1]][Direction.LEFT]
        ]

        maxIndex = np.argmax(actionValues)
        maxValue = actionValues[maxIndex]
        maxDirection = Direction(maxIndex)

        return maxValue, maxDirection

    def getNextAction(self, currentState):
        if (self.epsilonGreedy > 0 and decision(self.epsilonGreedy)) or not self.greedification:
            return selectRandom(Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)

        _, action = self.getOptimalActionValue(currentState)
        return action
    
    def getNewCoordinatesBasedOnDirection(self, x, y, direction):
        if direction == Direction.UP:
            x -= 1
        elif direction == Direction.RIGHT:
            y += 1
        elif direction == Direction.DOWN:
            x += 1
        elif direction == Direction.LEFT:
            y -= 1
        else:
            raise Exception("Undefined action: " + str(direction))

        return x, y

    def getNextState(self, currentState, action):
        x, y = self.getNewCoordinatesBasedOnDirection(currentState[0], currentState[1], action)

        reward = self.getReward(x, y)

        if self.isOutOfBounds(x, y):
            return currentState, reward
        
        return [x,y], reward

    def runSimulation(self, maxEpisodes = 500, learningRate = 0.1):
        self.actionValueFunctions = [[
            {
                Direction.UP : 0,
                Direction.RIGHT : 0,
                Direction.DOWN : 0,
                Direction.LEFT : 0
            } for _ in range(self.columns)] for _ in range(self.rows)]

        episodeRewards = np.empty(maxEpisodes, dtype=int)
        episodeRewards.fill(0)
        episodeRewardSum = 0

        for i in range(maxEpisodes):
            terminal = False

            episodeReward = 0
            currentState = [self.startX, self.startY]

            while not terminal:
                currentAction = self.getNextAction(currentState)
                nextState, currentReward = self.getNextState(currentState, currentAction)

                terminal = self.isTerminal(nextState)
                
                episodeReward += currentReward

                nextValue = 0
                if self.policyType == PolicyType.SARSA:
                    nextAction = self.getNextAction(nextState)
                    nextValue = self.actionValueFunctions[nextState[0]][nextState[1]][nextAction]
                elif self.policyType == PolicyType.QLEARNING:
                    nextValue, _ = self.getOptimalActionValue(nextState)

                # print ("current action value for ({}, {}) and direction - {} = {}".format(currentState[0], currentState[1], currentAction, self.actionValueFunctions[currentState[0]][currentState[1]][currentAction]))

                self.actionValueFunctions[currentState[0]][currentState[1]][currentAction] += (
                    learningRate * (
                        currentReward + 
                        (self.discountFactor * nextValue) - 
                        self.actionValueFunctions[currentState[0]][currentState[1]][currentAction]))

                # print ("new action value for ({}, {}) and direction - {} = {}".format(currentState[0], currentState[1], currentAction, self.actionValueFunctions[currentState[0]][currentState[1]][currentAction]))

                currentState = nextState
            
            # print ("current episode reward - {}".format(episodeReward))

            episodeRewardSum += episodeReward
            
            if i % 10 == 0:
                endIndex = i
                startIndex = i - 10
                averageReward = episodeRewardSum / 10 # averaged reward for the last 10 episodes
                for j in range(startIndex, endIndex):
                    episodeRewards[j] = averageReward

                episodeRewardSum = 0
        
        return episodeRewards

    def getActionValueMatrix(self):
        actionValueMatrix = np.empty((self.rows, self.columns))

        x, y = self.startX, self.startY
        actionValueMatrix[x][y] = 1
        
        while not self.isTerminal([x,y]):
            _, direction = self.getOptimalActionValue([x, y])

            x, y = self.getNewCoordinatesBasedOnDirection(x, y, direction)

            if self.isOutOfBounds(x, y):
                raise Exception("Invalid optimal policy - went out of bounds...")

            actionValueMatrix[x][y] = 1

        return actionValueMatrix

    def printActionValueMatrix(self):
        print ("")

        for x in range(self.rows):
            lineString = "|"
            for y in range(self.columns):
                
                _, direction = self.getOptimalActionValue([x, y])
                arrow = ""
                if direction == Direction.UP:
                    arrow = "^"
                elif direction == Direction.RIGHT:
                    arrow = ">"
                elif direction == Direction.DOWN:
                    arrow = "v"
                elif direction == Direction.LEFT:
                    arrow = "<"

                lineString += "{}|".format(arrow)
            
            splitLine = ""
            for _ in range(len(lineString)):
                splitLine += "-"

            print(splitLine)
            print(lineString)
        
        print(splitLine)
        print ("")


    def plotActionValueFunctions(self, title=""):
        self.printActionValueMatrix()
        m = self.getActionValueMatrix()

        
        # plt.title(title)
        # im, _ = heatmap(m)
        # annotate_heatmap(im)
        # plt.show()

    def isBound(self, x, y):
        if (x == 0 or x == self.rows-1 or 
            y == 0 or y == self.columns-1):
            return True
        
        return False

    def isOutOfBounds(self, x, y):
        if (x < 0 or x > self.rows - 1 or 
            y < 0 or y > self.columns - 1):
            return True
        
        return False
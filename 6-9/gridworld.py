import numpy as np
import random
from collections import defaultdict
from direction import Direction
from policyType import PolicyType
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    def plotArrow(self, x, y, optimalPathStates):
        _, action = self.getOptimalActionValue([x, y])

        color = 'black'
        if [x,y] in optimalPathStates:
            color = 'green'

        dx = 0
        dy = 0
        arrowWidth = 0.15
        arrowLength = 0.3
        arrowHeadWidth = 0.35
        arrowHeadLength = 0.25

        x = self.rows - x 
        y = y + 1

        if action == Direction.UP:
            dx = 0
            dy = arrowLength
            x = x - (arrowLength + arrowHeadLength) / 2
        elif action == Direction.RIGHT:
            dx = arrowLength
            dy = 0
            arrowWidth = arrowWidth
            arrowHeadWidth = arrowHeadWidth / 1.2
            y = y - (arrowWidth + arrowHeadWidth) / 2
        elif action == Direction.DOWN:
            dx = 0
            dy = -arrowLength
            x = x + (arrowLength + arrowHeadLength) / 2
        elif action == Direction.LEFT:
            dx = -arrowLength
            dy = 0
            arrowWidth = arrowWidth
            arrowHeadWidth = arrowHeadWidth / 1.2
            y = y + (arrowWidth + arrowHeadWidth) / 2
        
        plt.arrow(y, x, dx, dy, head_width=arrowHeadWidth, head_length=arrowHeadLength, width=arrowWidth, color=color)

    def getOptimalPathStates(self):
        optimalStates = []
        x, y = self.startX, self.startY
        optimalStates.append([x, y])
        
        while not self.isTerminal([x,y]):
            if self.isOutOfBounds(x, y):
                continue

            _, direction = self.getOptimalActionValue([x,y])
            x, y = self.getNewCoordinatesBasedOnDirection(x, y, direction)
            optimalStates.append([x,y])

        return optimalStates

    def plotLetter(self, x, y, letter):
        plt.text(y + 0.75, self.rows - x - 0.15, letter, fontsize = 25)

    def colorRectangle(self, x, y):
        # Create a Rectangle patch
        rect = patches.Rectangle((y + 0.5, self.rows - x - 0.5), 1,1, linewidth=0, facecolor='#B0B0B0')
        axes = plt.gca()

        # Add the patch to the Axes
        axes.add_patch(rect)

    def plotOptimalPath(self, title = ""):
        optimalStates = self.getOptimalPathStates()

        for x in range(self.rows):
            for y in range(self.columns):
                if (x == self.startX and y == self.startY) or self.isTerminal([x,y]):
                    continue

                self.plotArrow(x, y, optimalStates)

        xTicks = np.arange(0, self.columns + 2, 1)
        subXTicks = np.arange(0.5, self.columns + 1.5, 1)
        yTicks = np.arange(0, self.rows + 2, 1)
        subYTicks = np.arange(0.5, self.rows + 1.5, 1)

        axes = plt.gca()
        axes.set_xticks(xTicks)
        axes.set_xticks(subXTicks, minor=True)
        axes.set_yticks(yTicks)
        axes.set_yticks(subYTicks, minor=True)
        
        axes.set_xlim([0.5, self.columns + 0.5])
        axes.set_ylim([0.5, self.rows + 0.5])
        

        axes.grid(which='minor', alpha=0.5)
        plt.title(title)
        


        plt.show()
        
import numpy as np
import math

from enum import IntEnum

def Layers(IntEnum):
    HIDDEN = 2
    NUM = HIDDEN + 2
    NODES = 2

NODES_RANGE = range(Layers.NODES)
HIDDEN_RANGE = range(Layers.HIDDEN, -1, -1)

biasMatrix = np.empty((Layers.HIDDEN, Layers.NODES))
biasPartialDerivativesMatrix = np.ones((Layers.HIDDEN, Layers.NODES))
nodeMatrix = np.zeroes((Layers.HIDDEN, Layers.NODES))
nodePartialDerivativesMatrix = np.ones((Layers.HIDDEN, Layers.NODES))
weightsMatrixList = []
weightsPartialDerivativesMatrixList = []

FINAL_OUTPUT = 1.0
FINAL_OUTPUT_DERIVATIVE = 1.0

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, x))

def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# I have a dynamic dim_iter that I made at work, but I can't port that over
# it is lovely and I use it for a 150+ dimension thing lol

def iter_2d(range_i, range_j):
    return ((x, y) for x in range_i for y in range_j)

def iter_3d(range_i, range_j, range_k):
    return ((x, y, z) for x in range_i for y in range_j for z in range_k)


weightsMatrixList.append(np.empty(2, Layers.NODES))

for i in range(1, Layers.NUM):
    weightsMatrixList.append(np.empty(Layers.NODES, Layers.NODES))

weightsMatrixList.append(np.empty(Layers.NODES, 1))
        
def UpdateNodeOutputDerivatives(yKHat):
    FINAL_OUTPUT_DERIVATIVE = FINAL_OUTPUT - yKHat
    for i, j in iter_2d(HIDDEN_RANGE, NODES_RANGE):
        if i == Layers.HIDDEN:  
            val = FINAL_OUTPUT_DERIVATIVE * sigmoidDerivative(FINAL_OUTPUT + biasMatrix[i][j])  * weightsMatrixList[i][j][0]
        else:
            val = 0.0
            for k in NODES_RANGE:
                val += nodePartialDerivativesMatrix[i + 1][k] * sigmoidDerivative(nodeMatrix[i + 1][k] + biasMatrix[i + 1][k]) * weightsMatrixList[i][j][k]
        nodePartialDerivativesMatrix[i][j] = val

def UpdateBiasPartialDerivatives():
    for i, j in iter_2d(HIDDEN_RANGE, NODES_RANGE):
        if i == Layers.HIDDEN:
            val = FINAL_OUTPUT_DERIVATIVE * sigmoidDerivative(FINAL_OUTPUT + biasMatrix[i + 1][j])
        else:
            val = nodePartialDerivativesMatrix[i + 1][j] * sigmoidDerivative(nodeMatrix[i + 1][j])
        biasPartialDerivativesMatrix = val

def UpdateWeightPartialDerivatives():
    for i, j, k in iter_3d(range(Layers.NUM, -1, -1), NODES_RANGE, NODES_RANGE):
        bias = biasMatrix[j][k], biasMatrix[j + 1][k]
        node = nodeMatrix[j][k], nodeMatrix[j + 1][k]
        sig = sigmoid(node[0] + bias[0])
        sig_der = sigmoidDerivative((FINAL_OUTPUT if i == Layers.NUM else node[1]) + bias[1]) * sig
        val = FINAL_OUTPUT_DERIVATIVE if i == Layers.NUM else nodePartialDerivativesMatrix[j + 1][k]
        weightsPartialDerivativesMatrixList[i][j][k] = val * sig_der

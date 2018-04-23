import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import random
import time
import math


#Filepaths for data files
trainingDataCSVPath = "HW6-Old-Faithful\old-faithful-training-data.csv"
testingDataCSVPath = "HW6-Old-Faithful\old-faithful-testing-data.csv"

#Data format:
# Message #, Word Counter in Dictionary, # of Occurrences of Word, Spam/Ham (1=spam)

 


#Columns of data to read in
#colsToReadData = [9,10,12]
#colToReadPrice = [8]

#colsToSqFtVsData = [8,12]

#Price column: 8


colsForData = [2,3]   #Sepal Length, Sepal Width, Petal Length, Petal Width
#colsForClassification = [5]

columnNames = ["index","eruptions", "waiting"]

#For future reference:  dataframe.shape()   gives a tuple with dimensionality of the dataframe:  (# rows, # cols)

#Training & testing data

#Training data block

trainingDataInput = pandas.DataFrame(pandas.read_csv(trainingDataCSVPath))
#trainingDataPrices = pandas.DataFrame(pandas.read_csv(trainingDataCSVPath,usecols=colToReadPrice))


#Testing data block

#testingDataInput = pandas.DataFrame(pandas.read_csv(testingDataCSVPath))
#testingDataPrices = pandas.DataFrame(pandas.read_csv(testingDataCSVPath,usecols=colToReadPrice))

#testingPricesVsSqFt = pandas.DataFrame(pandas.read_csv(testingDataCSVPath, usecols=colsToSqFtVsData))

#Size of data
trainingDataLength = 0
testingDataLength = 0

trainingDataLength = len(trainingDataInput.index)
testingDataLength = len(testingDataInput.index)

#Input matrix - 4x3

#XOR:  x , y, x XOR y
#[ 0  0  0]
#[ 0  1  1]
#[ 1  0  1]
#[ 1  1  0]


#Division:   x, y, x/y   where x & y are in [1, 10], equidistant  (???)


#Input:  2 inputs
#Output: 1 output


#
#   Weight matrices:
#   M x 2 matrix of weights, M = # nodes in layer 1
#   
#   For each layer after:   M x N matrix,  N = # nodes in layer L+1, M = # nodes in layer L
#   
#   Final layer:   N x 1 vector, N = # nodes in second to last layer   
#
#   
#   Node inputs:  Zk vals, just plug into phi() fn. to get Ok vals -- same number of partial derivatives, too!
#      
#   For all layers:  sum of( weight from this node -> output of node from previous layer *  output of node from previous layer  )
#
#   First layer:   2x1 vector  (x1, x2)
#   Each layer thereafter:   M x 1 vector  M = # nodes in prev. layer
#
#   x1 =>
#
#   x2
#
#
#



NUM_HIDDEN_LAYERS = 2
NUM_LAYERS = NUM_HIDDEN_LAYERS+2
NUM_NODES_PER_HIDDEN_LAYER =2


def sigmoid(x):
    return (1 / (1 + math.pow(math.e,x)))

def sigmoidDerivative(x):
    return (sigmoid(x)) * (1 - (sigmoid(x)))



biasMatrix = np.empty((NUM_HIDDEN_LAYERS,NUM_NODES_PER_HIDDEN_LAYER))   #Store biases
biasPartialDerivativesMatrix = np.ones((NUM_HIDDEN_LAYERS,NUM_NODES_PER_HIDDEN_LAYER)) #Stores partial derivatives 
nodeMatrix = np.zeroes((NUM_HIDDEN_LAYERS,NUM_NODES_PER_HIDDEN_LAYER))    #Store Zk vals

weightsMatrixList = []  #Stores weights
weightsPartialDerivativesMatrixList = []  #Stores weight partial derivatives






testInputs = [np.matrix(0,0), np.matrix(0,1), np.matrix(1,0), np.matrix(1,1)]

#change to THIS for division
#testInputs = [np.matrix(), np.matrix(), np.matrix(), np.matrix() ]


for i in range(0,NUM_LAYERS):
    if (i==0):
        weightsMatrixList.append(np.empty(2,NUM_NODES_PER_HIDDEN_LAYER))  #From layer j->i    second coord -> first coord    ???
    elif (i==NUM_LAYERS-1):
        weightsMatrixList.append(np.empty(NUM_NODES_PER_HIDDEN_LAYER,1))
    else:
        weightsMatrixList.append(np.empty(NUM_NODES_PER_HIDDEN_LAYER,NUM_NODES_PER_HIDDEN_LAYER))



#Just need to look up how to update the partial derivatives, make functions out of them, ???, profit?

        

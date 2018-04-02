import numpy
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

columnNames = ["eruptions", "waiting"]

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
#testingDataLength = 0

trainingDataLength = len(trainingDataInput.index)
testingDataLength = len(testingDataInput.index)

#if(trainingDataLength == 0 or testingDataLength ==0):
#    print("Something has gone wrong.")
E = math.e
PI = math.pi
NUM_CLUSTERS = 2

phiArray = [1.0/NUM_CLUSTERS] * NUM_CLUSTERS

#If this doesn't work, just store them in separate arrays for sanity's sake
clusterCentersX = [0.0] * NUM_CLUSTERS   #Vi x-coord
clusterCentersY = [0.0] * NUM_CLUSTERS   #Vi y-coord
dataPointClusterAssignment = [0] * NUM_CLUSTERS   #Set the first 125 to 1st cluster (index 0), last 125 to 2nd cluster (index 1)
sigmaVals = [1]*NUM_CLUSTERS
#Parameters I Choose

#Set cluster centers--avg of first, second 125 inputs

for row, index in trainingDataInput.iterrows():
    if(rows >0):
        if(rows <126):
            clusterCentersX[0] += index[1]   #.values
            clusterCentersY[0] += index[2]   #.values
        else:
            clusterCentersX[1] += index[2]   #.values
            clusterCentersY[1] += index[2]   #.values
            dataPointClustersAssignment[row] =1

for i in range(0,NUM_CLUSTERS):
    clusterCenters[i] = tuple(x / 125.0 for x in clusterCenters[i])





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


#Parameters I Choose

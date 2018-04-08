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

#if(trainingDataLength == 0 or testingDataLength ==0):
#    print("Something has gone wrong.")
E = math.e
PI = math.pi
NUM_CLUSTERS = 2



NUM_DATA_POINTS = trainingDataLength


#If this doesn't work, just store them in separate arrays for sanity's sake
#clusterSizes = [125] * NUM_CLUSTERS  #Initial cluster size = 125
#clusterCentersX = [0.0] * NUM_CLUSTERS   #Vi x-coord

#don't use this
#dataPointClusterAssignment = [0] * NUM_CLUSTERS   #Set the first 125 to 1st cluster (index 0), last 125 to 2nd cluster (index 1)


#DO use this
sigmaVals = [1]*NUM_CLUSTERS
meanVals = [1]*NUM_CLUSTERS
phiVals = [1.0/NUM_CLUSTERS] * NUM_CLUSTERS
WVals = numpy.ones((NUM_DATA_POINTS,  NUM_CLUSTERS))   #num rows, num cols  ==>   index, cluster num

MAX_X_VAL = trainingDataInput["eruptions"].max()
MIN_X_VAL = trainingDataInput["eruptions"].min()

MAX_LOOPS = 1000

meanVals[0] = random(MIN_X_VAL,MAX_X_VAL)

meanVals[1] = random(MIN_X_VAL,MAX_X_VAL)
while(meanVals[1] == meanVals[0]):
    meanVals[1] = random(MIN_X_VAL,MAX_X_VAL)




#Set cluster centers--avg of first, second 125 inputs
##Don't use this at all
#for row, index in trainingDataInput.iterrows():
#    if(rows >0):
#        if(rows <126):
#            clusterCentersX[0] += index[1]   #.values
#            #clusterCentersY[0] += index[2]   #.values
#        else:
#            clusterCentersX[1] += index[2]   #.values
#            #clusterCentersY[1] += index[2]   #.values
#            dataPointClustersAssignment[row] =1

#Don't use this at all
#def UpdateClusterCenters():
#    for i in range(0,NUM_CLUSTERS):
#        clusterCenters[i] /= float(clusterSizes[i])
#        #clusterCentersX[i] = tuple(x / float(clusterSizes[i]) for x in clusterCentersX[i])
#        #clusterCentersY[i] = tuple(x / float(clusterSizes[i]) for x in clusterCentersY[i])


def GetXVal(i):
    return trainingDataInput.iloc[[i+1][1]]

def GetYVal(i):
    return trainingDataInput.iloc[[i+1][2]]

#def UpdateClusterSizes():
#    clusterSizes[0]=0
#    clusterSizes[1]=0
#    for(i in range(0, NUM_DATA_POINTS)):
#        if(WVals[i][0] > WVals[i][1]):
#            clusterSizes[0] +=1
#        else:
#            clusterSizes[1] +=1

# P(Xi | Zj)
def MembershipGrade(trainingDataIndex, meanIndex, sigmaIndex):
    return (  (1/(math.sqrt(2*PI)*sigmaVals[sigmaIndex]) ) * ( math.pow(E, (math.pow( ( GetXVal(trainingDataIndex) - meanVals[meanIndex]), 2  )) / ( -2 * math.pow(sigmaVals[sigmaIndex],2) )  )  ) )



def UpdateWValues():
    for i in range(0,NUM_DATA_POINTS):
        for j in range(0,NUM_CLUSTERS):
            sum =0.0
            myGrade = MembershipGrade(i,j,j) * phiVals[j]
            for l in range(0,NUM_DATA_POINTS):
                sum += MembershipGrade(i,l,l) * phiVals[l]
            WVals[i][j] = myGrade/sum


def UpdateMeanValues():
    for j in range(0,NUM_CLUSTERS):
        wSum =0.0
        wProduct =0.0
        for i in range(0,NUM_DATA_POINTS):
            wSum += WVals[i][j]
            wProduct += WVals[i][j] * GetXVal(i)
        meanVals[j] = wSum / wProduct

def UpdatePhiValues():
    for j in range(0,NUM_CLUSTERS):
        wSum =0.0
        for i in range(0,NUM_DATA_POINTS):
            wSum += WVals[i][j]
        phiVals[j] = wSum / NUM_DATA_POINTS


def UpdateSigmaValues():
    for j in range(0,NUM_CLUSTERS):
        wSum=0.0
        wProduct=0.0
        for i in range(0,NUM_DATA_POINTS):
            wSum += WVals[i][j]
            wProduct += WVals[i][j] * math.pow((GetXVal(i) - meanVals[j]), 2)
        sigmaVals[j]=wProduct/wSum


for b in range(0,MAX_LOOPS):
    UpdateWValues()

    
    UpdatePhiValues()
    UpdateMeanValues()
    UpdateSigmaValues()


#Should have normal distributions at this point--what do we do with them?


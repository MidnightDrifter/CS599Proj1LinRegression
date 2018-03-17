import numpy
import matplotlib
import pandas

#Filepaths for data files
trainingDataCSVPath = "Proj4-Iris-Data\iris-data-2-types.csv"
#testingDataCSVPath = "Proj 3 Spam-Ham Data\preprocdata\test-features.csv"

#Data format:
# Message #, Word Counter in Dictionary, # of Occurrences of Word, Spam/Ham (1=spam)

 


#Columns of data to read in
#colsToReadData = [9,10,12]
#colToReadPrice = [8]

#colsToSqFtVsData = [8,12]

#Price column: 8


colsForData = [0,1,2,3]   #Sepal Length, Sepal Width, Petal Length, Petal Width
colsForClassification[4]

columnNames = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Classification"]

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
#testingDataLength = len(testingDataInput.index)

#if(trainingDataLength == 0 or testingDataLength ==0):
#    print("Something has gone wrong.")



#Parameters I Choose

UPPER_LIMIT_C = 5.0
initialB = 1.0
initialAlphaValue = 1.5
W_VAL
B_VAL

#use len-1 because, presumably, the header row is included in the length
alphaVector = [initialAlphaValue] * (trainingDatalength-1)   



def getX(inIndex):
    return trainingDataInput.iloc[[inIndex],[0,1,2,3]]

def getY(inIndex):
    return trainingDataInput.ilox[[inIndex],4]


def UpdateWVal():
    sum =0
    for rows, index in trainingDataInput.iterrows():
        sum += rows[0,1,2,3].transpose() * rows[4] *alphaVector[index]
    return sum


def funcX(inputIndex):
    return W_VAL.transpose() * getX(inputIndex) + B_VAL

def Efunc(inputIndex):
    funcX(inputIndex) - getY(inputIndex)

def ClampAlphaToConstraints(inFloat):
    return max(0,min(UPPER_LIMIT_C,inFloat))





def UpdateAlphaJ(indexI, indexJ):
    alphaIOld = alphaVector[indexI]
    alphaJOld = alphaVector[indexJ]

    H = min(UPPER_LIMIT_C, UPPER_LIMIT_C + alphaJOld - alphaIOld)
    L = max(0,alphaJOld-alphaIOld)

    if(getY(indexI) == getY(indexJ)):
        L = min(0, alphaIOld + alphaJOld - UPPER_LIMIT_C)
        H = min(C, alphaIold + alphaJold)

    myu = (2* getX(indexI).transpose() * getX(indexJ)) - (getX(indexI).transpose() * getX(indexI)) - (getX(indexJ).transpose() * getX(indexJ))

    alphaJNew = alphaJOld - ((getY(indexJ) * (Efunc(indexI) - Efunc(indexJ)))/ myu)

    if(alphaJNew < L):
        return L
    elif(alphaJNew > H):
        return H
    else:
        return alphaJNew

#Assumes that I'll instantly insert alphaJNew into the alphaVector after updating it
def UpdateAlphaI(indexI, indexJ, alphaJOld):
    return alphaVector[indexI] + (alphaJOld - alphaVector[indexJ])*(getY(indexI)*getY(indexJ))




import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas
import random
import time


#Filepaths for data files
trainingDataCSVPath = "Proj4-Iris-Data\iris-data-2-types-no-strings.csv"
#testingDataCSVPath = "Proj 3 Spam-Ham Data\preprocdata\test-features.csv"

#Data format:
# Message #, Word Counter in Dictionary, # of Occurrences of Word, Spam/Ham (1=spam)

 


#Columns of data to read in
#colsToReadData = [9,10,12]
#colToReadPrice = [8]

#colsToSqFtVsData = [8,12]

#Price column: 8


colsForData = [0,1,2,3]   #Sepal Length, Sepal Width, Petal Length, Petal Width
colsForClassification = [5]

columnNames = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Classification Num" ,"Classification"]

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
W_VAL = pandas.DataFrame(trainingDataInput.iloc[[1],[0,1,2,3]]).values   #Slicing [startIndex : endIndex (not-included) : increment amount (default to 1) ]
W_VAL = W_VAL - W_VAL
B_VAL = 0.0

MAX_LOOPS = 1000

#use len-1 because, presumably, the header row is included in the length
alphaVector = [initialAlphaValue] * (trainingDataLength)   



def getX(inIndex):
    return (trainingDataInput.iloc[[inIndex],[0,1,2,3]]).values

def getXArray(inIndex):
    return numpy.squeeze((trainingDataInput.iloc[[inIndex],[0,1,2,3]]).values)
def getY(inIndex):
    return (trainingDataInput.iloc[[inIndex],4]).values


def UpdateWVal():
    sum = W_VAL
    
    sum = sum - sum


    for rows, index in trainingDataInput.iterrows():
        if(rows >0):
            #temp = index[0:4].values.T
            #temp = index[4]
            sum =  sum +(  index[4] *alphaVector[rows] * (index[0:4].values))
    return sum


def funcX(inputIndex):
    return numpy.dot(W_VAL , getXArray(inputIndex)) + B_VAL

def Efunc(inputIndex):
   return funcX(inputIndex) - getY(inputIndex)

def ClampAlphaToConstraints(inFloat):
    return max(0,min(UPPER_LIMIT_C,inFloat))





def UpdateAlphaJ(indexI, indexJ):
    alphaIOld = alphaVector[indexI]
    alphaJOld = alphaVector[indexJ]

    H = min(UPPER_LIMIT_C, UPPER_LIMIT_C + alphaJOld - alphaIOld)
    L = max(0,alphaJOld-alphaIOld)

    if(getY(indexI) == getY(indexJ)):
        L = min(0, alphaIOld + alphaJOld - UPPER_LIMIT_C)
        H = min(UPPER_LIMIT_C, alphaIOld + alphaJOld)


    myu = (2* numpy.dot(getXArray(indexI) , getXArray(indexJ)) - (numpy.dot(getXArray(indexI) , getXArray(indexI))) - (numpy.dot(getXArray(indexJ) , getXArray(indexJ))  )   )
    if (myu==0):
        myu=1
    alphaJNew = alphaJOld - ((getY(indexJ) * (Efunc(indexI) - Efunc(indexJ)))/ myu)

    if(alphaJNew < L):
        return L
    elif(alphaJNew > H):
        return H
    else:
        return alphaJNew

#Need to pass and use both new and old alphaJ, alphaI values to update b, so need to pass in both as floats
def UpdateAlphaI(indexI, indexJ,  alphaJNew):
    return alphaVector[indexI] + (alphaVector[indexJ]-alphaJNew)*(getY(indexI)*getY(indexJ))


def UpdateB(indexI, indexJ, alphaINew, alphaJNew, oldB):
    b1 = oldB - Efunc(indexI) - (getY(indexI) *   (  alphaINew - alphaVector[indexI]  )*(  numpy.dot(getXArray(indexI) , getXArray(indexI))  )  )   - (  getY(indexJ) * (  alphaJNew - alphaVector[indexJ])*(  numpy.dot(getXArray(indexI), getXArray(indexJ))  )  )
    b2 = oldB - Efunc(indexJ) - (getY(indexI) *   (  alphaINew - alphaVector[indexI]  )*(  numpy.dot(getXArray(indexI), getXArray(indexJ))  )  )   - (  getY(indexJ) * (  alphaJNew - alphaVector[indexJ])*(  numpy.dot(getXArray(indexJ), getXArray(indexJ)  ))  )
    
    if(alphaINew > 0 and alphaINew < UPPER_LIMIT_C):
        return b1
    elif(alphaJNew >0 and alphaJNew < UPPER_LIMIT_C):
        return b2
    else:
        return (b1+b2)/2.0





#main loop
print("Loop start.\n")
start = time.time()

for counter in range(0,MAX_LOOPS):

    W_VAL = UpdateWVal()

    i = random.randint(0,trainingDataLength-1)
    j = random.randint(0,trainingDataLength-1)
    while(i==j):
        j=random.randint(0,trainingDataLength-1)

    alphaJNew = UpdateAlphaJ(i,j)
    alphaINew = UpdateAlphaI(i,j,alphaJNew)
    B_VAL = UpdateB(i,j,alphaINew,alphaJNew,B_VAL)

    alphaVector[i] = alphaINew
    alphaVector[j] = alphaJNew

end = time.time()

print("Loop end.  Elapsed time:  " + str(end-start) + "seconds.\n")

averages = [0,0,0,0]
for rows, index in trainingDataInput.iterrows():
    if(rows >0):
        for i in range(0,4):
            averages[i] += index[i]


for i in range(0,4):
    averages[i] /= trainingDataLength


xAxis =0
yAxis=1

plt.xlabel(columnNames[xAxis])
plt.ylabel(columnNames[yAxis])


#plt.plot(trainingDataInput[columnNames[xAxis]], trainingDataInput[columnNames[yAxis]])

#Make line based off of testing data, draw that line?
W_VAL = UpdateWVal()
correctClassifications =0
for row, index in trainingDataInput.iterrows():
    #If f(x) > 0, draw in RED, else draw in BLUE
    if(rows >0):
        classification = numpy.dot(W_VAL, getXArray(row)) + B_VAL
        if( classification >=0):
            plt.scatter(index[xAxis],index[yAxis], color='r')
            if(getY(row) ==1):
                correctClassifications += 1
        else:
            plt.scatter(index[xAxis],index[yAxis], color='b')
            if(getY(row)==-1):
                correctClassifications += 1




print("Correct classification ratio:  " + str(correctClassifications/100.0) + ".\n")
           





#graph stuff goes here

plt.show()
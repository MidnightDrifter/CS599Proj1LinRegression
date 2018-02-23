import numpy
import matplotlib
import pandas

#Filepaths for data files
trainingDataCSVPath = "Proj 3 Spam-Ham Data\preprocdata\train-features-400.csv"
testingDataCSVPath = "Proj 3 Spam-Ham Data\preprocdata\test-features.csv"

#Data format:
# Message #, Word Counter in Dictionary, # of Occurrences of Word, Spam/Ham (1=spam)

 


#Columns of data to read in
#colsToReadData = [9,10,12]
#colToReadPrice = [8]

#colsToSqFtVsData = [8,12]

#Price column: 8


#For future reference:  dataframe.shape()   gives a tuple with dimensionality of the dataframe:  (# rows, # cols)

#Training & testing data
trainingDataInput = pandas.DataFrame(pandas.read_csv(trainingDataCSVPath))
#trainingDataPrices = pandas.DataFrame(pandas.read_csv(trainingDataCSVPath,usecols=colToReadPrice))

testingDataInput = pandas.DataFrame(pandas.read_csv(testingDataCSVPath))
#testingDataPrices = pandas.DataFrame(pandas.read_csv(testingDataCSVPath,usecols=colToReadPrice))

#testingPricesVsSqFt = pandas.DataFrame(pandas.read_csv(testingDataCSVPath, usecols=colsToSqFtVsData))

#Size of data
trainingDataLength = 0
testingDataLength = 0

trainingDataLength = len(trainingDataInput.index)
testingDataLength = len(testingDataInput.index)

if(trainingDataLength == 0 or testingDataLength ==0):
    print("Something has gone wrong.")


numTrainingEmails = trainingDataInput["Message #"].max()
numTestingEmails = testingDataInput["Message #"].max()


trainingDataSpamDictionary = {}
trainingDataHamDictionary = {}
testingDataSpamDictionary = {}
testingDataHamDictionary = {}


#Kinda cheat-y, but know that:  A. training data has 400 emails & B. testing data has 260





for index, row in trainingDataInput.itertuples():
    if(row[3] ==0):
        if(row[1] not in trainingDataHamDictionary):
            trainingDataHamDictionary[row[1]] = row[2]
        else:
            trainingDataHamDictionary[row[1]] += row[2]
    else:
        if(row[1] not in trainingDataSpamDictionary):
            trainingDataSpamDictionary[row[1]] = row[2]
        else:
            trainingDataSpamDictionary[row[1]] += row[2]


#for index, row in testingDataInput.itertuples():
#    if(row[3] ==0):
#        if(row[1] not in testingDataHamDictionary):
#            testingDataHamDictionary[row[1]] = row[2]
#        else:
#            testingDataHamDictionary[row[1]] += row[2]
#    else:
#        if(row[1] not in testingDataSpamDictionary):
#            testingDataSpamDictionary[row[1]] = row[2]
#        else:
#            testingDataSpamDictionary[row[1]] += row[2]
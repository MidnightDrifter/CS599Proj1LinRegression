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
#testingDataSpamDictionary = {}
#testingDataHamDictionary = {}


#Kinda cheat-y, but know that:  A. training data has 400 emails & B. testing data has 260



totalWordsSpam = 0
totalWordsHam=0

for index, row in trainingDataInput.itertuples():
    if(row[3] ==0):
        totalWordsHam += row[2]
        if(row[1] not in trainingDataHamDictionary):
            trainingDataHamDictionary[row[1]] = row[2]
        else:
            trainingDataHamDictionary[row[1]] += row[2]
    else:
        totalWordsSpam += row[2]
        if(row[1] not in trainingDataSpamDictionary):
            trainingDataSpamDictionary[row[1]] = row[2]
        else:
            trainingDataSpamDictionary[row[1]] += row[2]


pSpam = float(totalWordsSpam / (totalWordsSpam + totalWordsHam))
pHam = 1.0 - pHam


def probSpamGivenWord( word="" ):
    if (word == ""):
        return 1.0
    else:
        return (trainingDataSpamDictionary.get(word,0) +1.0) / (trainingDataSpamDictionary.get(word,0) + trainingDataHamDictionary.get(word,0) + 2 )

def probHamGivenWord(word =""):
    if (word == ""):
        return 1.0
    else:
        return (trainingDataHamDictionary.get(word,0) +1.0) / (trainingDataSpamDictionary.get(word,0) + trainingDataHamDictionary.get(word,0) + 2 )










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


#for each row in the testing data:
    #read e-mail #
    #if it's a new email, reset currProb to 1 after classifying old e-mail and comparing--use a 50-50 threshold.  >=50% spam, <=50% ham
        #Classify based on 50-50 threshold, update hit-miss counts based on correct/incorrect classification
        #reset currProb to 1


    #else, multiply & update probability based on iterative method--see notes for exact details
        #for each word wI in email
        #currProb *= P(spam | wI) =    ( P(spam) P(wI  | spam)    /   (   P(spam) P(wI | spam) +  P(ham) P(wI | ham)   )
        #This can be reduced, check notes, but for now:
            #   Laplace smoothing goes here!
            #   P(spam) =  (# spam emails ) +1  / (total # emails) +2
            #   P(ham) = 1- P(spam)
            #   P(wI | spam) =  (# occurrences of word in spam emails) +1 / (total # of occurrences of word, spam + ham) +2
            #  And similarly for P(wI | ham)
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


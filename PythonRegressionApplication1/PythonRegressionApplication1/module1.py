import numpy
import matplotlib
import pandas


trainingDataCSVPath = r'.\CSV Data Files\Redfin Data Cleaned - Choose Beds-I Bathrooms-J SqFt-L Price-H.csv'
testingDataCSVPath = r'.\CSV Data Files\Redfin Data Cleaned - Testing Data DO NOT USE TO TRAIN.csv'

colsToReadData = [9,10,12]
colToReadPrice = [8]


trainingDataInput = pandas.read_csv(trainingDataCSVPath, usecols=colsToReadData)
trainingDataPrices = pandas.read_csv(trainingDataCSVPath,usecols=colToReadPrice)

testingDataInput = pandas.read_csv(testingDataCSVPath,usecols=colsToReadData)
testingDataPrices = pandas.read_csv(testingDataCSVPath,usecols=colToReadPrice)
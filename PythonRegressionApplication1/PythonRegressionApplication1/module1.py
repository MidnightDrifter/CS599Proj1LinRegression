import numpy
import matplotlib
import pandas

#Filepaths for data files
trainingDataCSVPath = r'.\CSV Data Files\Redfin Data Cleaned - Choose Beds-I Bathrooms-J SqFt-L Price-H.csv'
testingDataCSVPath = r'.\CSV Data Files\Redfin Data Cleaned - Testing Data DO NOT USE TO TRAIN.csv'

#Columns of data to read in
colsToReadData = [9,10,12]
colToReadPrice = [8]

colsToSqFtVsData = [8,12]

#Price column: 8

#Training & testing data
trainingDataInput = pandas.DataFrame(pandas.read_csv(trainingDataCSVPath, usecols=colsToReadData))
trainingDataPrices = pandas.DataFrame(pandas.read_csv(trainingDataCSVPath,usecols=colToReadPrice))

testingDataInput = pandas.DataFrame(pandas.read_csv(testingDataCSVPath,usecols=colsToReadData))
testingDataPrices = pandas.DataFrame(pandas.read_csv(testingDataCSVPath,usecols=colToReadPrice))

testingPricesVsSqFt = pandas.DataFrame(pandas.read_csv(testingDataCSVPath, usecols=colsToSqFtVsData))

#Size of data
trainingDataLength = len(trainingDataInput[0])
testingDataLength = len(testingDataInput[0])

#Append all 1's column to left
trainingDataInput.insert(0, 'Ones', 1)
testingDataInput.insert(0,'Ones',1)



#Turn into matrices here
trainingDataMatrix = numpy.matrix(trainingDataInput.values)
trainingDataMatrixTranspose = trainingDataMatrix.transpose
trainingDataPricesMatrix = numpy.matrix(trainingDataPrices.values)

trainingDataMatrixInv = (trainingDataMatrixTranspose * trainingDataMatrix).inverse

testingDataMatrix = numpy.matrix(testingDataInput.values)
testingDataPricesMatrix = numpy.matrix(testingDataPrices.values)

#Multiply here
thetaValues = trainingDataMatrixInv * trainingDataMatrixTranspose * trainingDataPricesMatrix

#Graph here

predictedPrices = testingDataMatrix * thetaValues



matplotlib.pyplot.plot(predictedPrices,predictedPrices)
matplotlib.pyplot.plot(numpy.matrix(testingPricesVsSqFt.values))

matplotlib.pyplot.xlabel("Prices")
matplotlib.pyplot.ylabel("Sq. Ft.")
matplotlib.pyplot.show()



#Graph stuff goes here??

import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, svm
import math
import random
import scipy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

## GLOBAL VARIABLES ##
deletionColumns = []
hashmapGlobal = dict()


def loadDataWithoutFirstRow(fileName):
    '''
    Takes filename string, modifies inputs, and returns raw data as numpy array.
    This function assumes that there is a header row.
    '''
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    rawData = np.asarray(data[1:], dtype = str)    # removes the first row of headers
    rowsToDelete = returnRowsToDelete(rawData)
    columnsToDelete = returnColumnsToDelete(rawData)
    processedDataWithoutRows = deleteRows(rowsToDelete, rawData)
    processedData = deleteColumns(columnsToDelete, processedDataWithoutRows)
    return processedData

def loadData(fileName):
    '''
    Takes filename string, modifies inputs, and returns raw data as numpy array
    '''
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    rawData = np.asarray(data, dtype = str)   
    rowsToDelete = returnRowsToDelete(rawData)
    columnsToDelete = returnColumnsToDelete(rawData)
    processedDataWithoutRows = deleteRows(rowsToDelete, rawData)
    processedData = deleteColumns(columnsToDelete, processedDataWithoutRows)
    return processedData

def loadDataModified(modifiedFileName):
    '''
    Takes in modified file name and returns modified data as numpy array.
    This modified file has already had columns and rows deleted, and
    non-numerical inputs removed
    '''
    with open(modifiedFileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    rawData = np.asarray(data, dtype = int)  
    return rawData
    

def tidyDataAndOutputToCSV(originalFileName, newFileName):
    '''
    This function takes in the original filename and tidies data from this 
    file and outputs the tidied data to a csv file with the desired new file
    name 
    '''

    data = loadData(originalFileName)
    transformInputsToNumbers(data)
    saveToCSV(newFileName, data)    
    return

def returnRowsToDelete(arr):
    '''
    This function will look at input data and delete data points that have
    missing features '''
    rowsToDelete = []
    for i in range(arr.shape[0]):
        currentRow = arr[i]
        if isRowDrop(currentRow):
            rowsToDelete.append(i)
    return rowsToDelete


def deleteRows(rows, arr):
    '''
    Delete the specified rows in arr
    '''
    arr1 = scipy.delete(arr, rows, 0)
    return arr1    
    
    
def deleteColumns(columns, arr):
    '''
    Deletes particular columns in arrays 
    '''
    arr1 = scipy.delete(arr, columns, 1)
    return arr1

def returnColumnsToDelete(arr):
    '''
    Returns the columns to delete.
    '''

    global deletionColumns  #Global variable so we know to delete same columns in training set as test set
    if deletionColumns != []:
        return deletionColumns
    
    columnsToDelete = []
    for i in range(len(arr[0])):
        currentCol = arr[:,i]
        if isDrop(currentCol):
            columnsToDelete.append(i)
    return columnsToDelete

        
def saveToCSV(fileName, arr):
    '''
    Saves the arr as .csv file with specific fileName
    '''
    df = pd.DataFrame(arr)
    df.to_csv(fileName, index=False, header=False)

def isRowDrop(arr):
    '''
    Determines whether a specific row should be deleted 
    '''
    
    mysteryCharacter = "?"
    for i in arr:
        if mysteryCharacter ==  i.strip():
            return True
    return False

    
def isDrop(arr):
    '''
    Determines which columns/ features to drop. If more than 85% of column
    values are the same, then we drop
    '''
    
    return scipy.stats.mode(arr)[1] >= .85 * len(arr)

def transformInputsToNumbers(arr):
    '''
    This function transforms all categorical features into numbers. 
    '''
    columnsToCheck = returnColumnsNumbersForTransformation(arr)
    hashmap = returnDictionaryOfLabels(arr)
    for i in range(arr.shape[0]):
        for j in columnsToCheck:
            arr[i][j] = int(hashmap[arr[i][j].strip()])
    arr = arr.astype(float)
    return

def returnColumnsNumbersForTransformation(arr):
    '''
    This function returns column numbers that contain non-integer data
    '''
    columns = []
    for i in range(len(arr[0])):
            if (not arr[5][i].isdigit()):  
                columns.append(i)
    return columns
            
def returnDictionaryOfLabels(arr):
    ''' 
    This function returns a mapping of categorical labels to numbers. 
    We do this by initializing a hashmap/ dictionary that maps label to 
    specific number
    '''
    global hashmapGlobal
    if bool(hashmapGlobal) == True:
        return hashmapGlobal
    
    ultimateHashmap = dict()
    columnSets = []
    for i in range(len(arr[0])):
        if (not arr[5][i].isdigit()):
            currentSet = set()
            currentCol = arr[:,i]
            for j in range(2000):
                currentSet.add(currentCol[j].strip())
            columnSets.append(currentSet)
    for currentSet in columnSets:
        for i in range(len(currentSet)):
            ultimateHashmap[list(currentSet)[i]] = i
    hashmapGlobal = ultimateHashmap
    return hashmapGlobal

    
def getInputsAndOutputs(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y) without x_0 being 1
    '''    
    inputs = []
    outputs = []

    for i in arr:
        inputs.append(i[:-1])   # the x vector
        outputs.append(i[(len(i) - 1)])      # last element, y
    return np.asarray(inputs), np.asarray(outputs)

def runBagging(inputs, outputs, estimators):
    
    '''
    This function runs bagging on decision trees 
    
    '''
    bagging = BaggingClassifier(n_estimators = estimators)    
    bagging.fit(inputs, outputs)
    return bagging

def runNeuralNetwork(inputs, outputs):
    '''
    This function takes in inputs and outputs and runs a neural network to fit 
    it 
    '''
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(10000,), early_stopping=True) 
    clf.fit(inputs, outputs)
    return clf


def runBoosting(inputs, outputs, estimators):
    
    '''
    This function runs bagging on decision trees 
    
    '''
    boosting = AdaBoostClassifier(n_estimators = estimators)    
    boosting.fit(inputs, outputs)
    return boosting

def runRandomForest(inputs, outputs, estimators, maxFeatures, maxDepth, minLeaf):
    
    '''
    This function runs random forests to fit the data
    
    '''
    forest = RandomForestClassifier(n_estimators = estimators,
                                    max_features = maxFeatures,
                                    max_depth = maxDepth,
                                    min_samples_leaf = minLeaf)    
    forest.fit(inputs, outputs)
    return forest

def runVotingClassifier(inputs, outputs, lstClassifiers, votingMethod):
    '''
    This function takes a list of classifiers and applies voting classifier
    to them according to voting method which can either be 'soft' or 'hard'
    '''
    estimatorsLst = []
    for i in range(len(lstClassifiers)):
        estimatorsLst.append((str(i), lstClassifiers[i]))
    voting = VotingClassifier(estimators=estimatorsLst,
        voting=votingMethod)
    voting.fit(inputs, outputs)
    return voting

def main():
    ## -----------------------------------------------------------------------##    
    ## PREPROCESSING ##
    ## -----------------------------------------------------------------------##
    
    ## FOR TRAINING DATA
    
    originalFileName = "adult.data.csv"
    newFileName = "modified_" + originalFileName
    
    ## COMMENT THIS LINE OUT AFTER RUNNING IT THE FIRST TIME
    ## -----------------------------------------------------------------------##    
    #tidyDataAndOutputToCSV(originalFileName, newFileName)
    ## -----------------------------------------------------------------------##   
    
    trainingData = loadDataModified(newFileName)
    
    X, y = getInputsAndOutputs(trainingData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    ## FOR TESTING DATA
    
    originalTestFileName = "adult.test.csv"
    newTestFileName = "modified_" + originalTestFileName
    
    ## COMMENT THIS LINE OUT AFTER RUNNING IT THE FIRST TIME
    ## -----------------------------------------------------------------------##
    #tidyDataAndOutputToCSV(originalTestFileName, newTestFileName)
    ## -----------------------------------------------------------------------##
    
    testingData = loadDataModified(newTestFileName)   
    Xtest, ytest = getInputsAndOutputs(testingData)
   
    
    ## -----------------------------------------------------------------------##
    ## RUNNING NEURAL NETWORK ##
    ## -----------------------------------------------------------------------##
    #neuralNetwork = runNeuralNetwork(X_train, y_train)
    #print(neuralNetwork.score(X_test, y_test))    
    
    
    ## -----------------------------------------------------------------------##    
    ## TESTING NUMBER OF ESTIMATORS FOR BAGGING ## 
    ## -----------------------------------------------------------------------##
    #numEstimatorsData = []
    #for numEstimators in range(100, 200, 10):
        #clf = runBagging(X_train, y_train, numEstimators)
        #numEstimatorsData.append(np.asarray([numEstimators,clf.score(X_test, y_test)]))
    #numEstimatorsData = np.asarray(numEstimatorsData)
    #for i in numEstimatorsData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Accuracy vs. Base Estimators (Bagging)', fontsize = 22)    
    #plt.plot(numEstimatorsData[:,0], numEstimatorsData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Base Estimators', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)     
    
    
    ## -----------------------------------------------------------------------##
    ## TESTING NUMBER OF ESTIMATORS FOR BOOSTING ## 
    ## -----------------------------------------------------------------------##
    #numEstimatorsData = []
    #for numEstimators in range(30, 200, 10):
        #clf = runBoosting(X_train, y_train, numEstimators)
        #numEstimatorsData.append(np.asarray([numEstimators,clf.score(X_test, y_test)]))
    #numEstimatorsData = np.asarray(numEstimatorsData)
    #for i in numEstimatorsData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Accuracy vs. Base Estimators (Boosting)', fontsize = 22)    
    #plt.plot(numEstimatorsData[:,0], numEstimatorsData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Base Estimators', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)  
    
    
    ## -----------------------------------------------------------------------##
    ## TESTING NUMBER OF ESTIMATORS FOR RANDOM FORESTS ## 
    ## -----------------------------------------------------------------------##
    #numEstimatorsData = []
    #for numEstimators in range(30, 200, 10):
        #clf = runRandomForest(X_train, y_train, numEstimators, "auto", None, 1 )
        #numEstimatorsData.append(np.asarray([numEstimators,clf.score(X_test, y_test)]))
    #numEstimatorsData = np.asarray(numEstimatorsData)
    #for i in numEstimatorsData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Accuracy vs. Base Estimators (Forests)', fontsize = 22)    
    #plt.plot(numEstimatorsData[:,0], numEstimatorsData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Base Estimators', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)  
    
    
    ## -----------------------------------------------------------------------##
    ## TESTING MAX. FEATURES FOR RANDOM FORESTS ##
    ## -----------------------------------------------------------------------##
    #maxFeaturesData = []
    #for maxFeatures in np.linspace(0.1, 1.0, num=7):
        #clf = runRandomForest(X_train, y_train, 120, maxFeatures, None, 1 )
        #maxFeaturesData.append(np.asarray([maxFeatures,clf.score(X_test, y_test)]))
    #maxFeaturesData = np.asarray(maxFeaturesData)
    #for i in maxFeaturesData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Accuracy vs. Features (Forests)', fontsize = 22)    
    #plt.plot(maxFeaturesData[:,0], maxFeaturesData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Proportion of Features', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)      
    
    
    ## -----------------------------------------------------------------------##
    ## TESTING MAX. DEPTH FOR RANDOM FORESTS ##
    ## -----------------------------------------------------------------------##
    #maxDepthData = []
    #for maxDepth in range(1,20):
        #clf = runRandomForest(X_train, y_train, 120, 0.4, maxDepth, 1 )
        #maxDepthData.append(np.asarray([maxDepth,clf.score(X_test, y_test)]))
    #maxDepthData = np.asarray(maxDepthData)
    #for i in maxDepthData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Accuracy vs. Max. Depth (Forests)', fontsize = 22)    
    #plt.plot(maxDepthData[:,0], maxDepthData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Depth', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)   
    
    ## -----------------------------------------------------------------------##
    ## TESTING MIN. SAMPLES LEAF FOR RANDOM FORESTS ##
    ## -----------------------------------------------------------------------##
    #minLeafData = []
    #for minLeaf in range(1,20):
        #clf = runRandomForest(X_train, y_train, 120, 0.4, None, minLeaf )
        #minLeafData.append(np.asarray([minLeaf,clf.score(X_test, y_test)]))
    #minLeafData = np.asarray(minLeafData)
    #for i in minLeafData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Accuracy vs. Min. Leaf(Forests)', fontsize = 22)    
    #plt.plot(minLeafData[:,0], minLeafData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Min. Samples Leaf', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)      
    
    ## -----------------------------------------------------------------------##
    ## COMBINE ALL ABOVE INTO VOTING CLASSIFIER ##
    ## -----------------------------------------------------------------------##
    #bestBagging = runBagging(X_train, y_train, 110)
    #bestBoosting = runBoosting(X_train, y_train, 110)
    #bestForests = runRandomForest(X_train, y_train, 120, 0.4, 9, 1)
    
    #voting = runVotingClassifier(X_train, y_train, [bestBagging, bestBoosting,
                                                    #bestForests], 'hard')
    #print (voting.score(X_test, y_test))
    
    #crossValidationScore = cross_val_score(voting, X, y, cv=5)
    #print (crossValidationScore, crossValidationScore.mean())
    
    ## -----------------------------------------------------------------------##
    ## TIME TO GET ACCURACY OF ACTUAL TEST SET ##
    ## -----------------------------------------------------------------------##    
    bestBagging = runBagging(X, y, 110)
    bestBoosting = runBoosting(X, y, 110)
    bestForests = runRandomForest(X, y, 120, 0.4, 9, 1)
    
    voting = runVotingClassifier(X, y, [bestBagging, bestBoosting,
                                                    bestForests], 'hard')
    print (voting.score(Xtest, ytest))
    

main()

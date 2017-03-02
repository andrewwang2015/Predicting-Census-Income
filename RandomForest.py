import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, svm
import math
import random
import scipy
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

deletionColumns = []
def loadData(fileName):
    '''
    Takes filename string and returns raw data as numpy array
    '''
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    rawData = np.asarray(data[1:], dtype = float)    # removes the first row of headers
    columnsToDelete = returnColumnsToDelete(rawData)
    processedData = deleteColumns(columnsToDelete, rawData)
    return processedData

def returnTrainingAndValidation(arr, percentageValidation):
    '''
    Splits all input data into a trainnig arr and validation arr according to
    percentage specified
    '''
    threshold = int(percentageValidation * len(arr))
    validation = arr[:threshold]
    training = arr[threshold:]
    return training, validation
    
def deleteColumns(columns, arr):
    '''
    Deletes particular columns in arrays 
    '''
    arr1 = scipy.delete(arr, columns, 1)
    return arr1

def returnColumnsToDelete(arr):
    global deletionColumns
    columnsToDelete = [0]   #We want to delete column 0, or ID column to begin with
    for i in range(arr.shape[1]):
        currentCol = arr[:,i]
        if isDrop(currentCol):
            columnsToDelete.append(i)
    if deletionColumns == []:
        deletionColumns = columnsToDelete
    return deletionColumns

        
def saveToCSV(fileName, arr):
    np.savetxt(fileName, arr, delimiter=",")
    
def isDrop(arr):
    '''
    Determines which columns/ features to drop. If all column values are the
    same or most column values (>= 90% are <=0), then we drop
    '''
    isSame = 0
    numZeroOrLess = 0
    for i in arr:
        if i <= 0:
            numZeroOrLess += 1
        if i == arr[0]:
            isSame += 1
    if (isSame >= 0.9 * len(arr) or numZeroOrLess >= 0.9 * len(arr)):
        return True
    return False
            
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

def runRandomForest(inputs, outputs, estimators, maxFeatures, maxDepth, minLeafSamples):
    
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array. 
    
    '''
    forest = RandomForestClassifier(n_estimators = estimators, max_features = maxFeatures, 
                                    max_depth = maxDepth,
                                min_samples_leaf = minLeafSamples)    
    forest.fit(inputs, outputs)
    return forest

def main():
    allData = loadData("train_2008.csv")
    X, y = getInputsAndOutputs(allData)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)    
    
    #data2008 = loadData("test_2008.csv")
    data2012 = loadData("test_2012.csv")
    
    #data2012 = loadData("test_2012.csv")
    #testing2012 = data2012
    
    #clf = runRandomForest(X_train, y_train, 10, None, 1)
    
    #print(clf.score(X_test, y_test))
    #testPredictions = clf.predict(testing2012)
    #testPredictions = np.asarray(testPredictions, dtype = int)
    #saveToCSV("bagging2012_V4.csv", testPredictions)
    
    #pureTesting = loadData("test_2008.csv")
    #testingX = pureTesting
    

    #print(testing2012.shape)
    
    ## TESTING MAX FEATURES
    
    #numFeaturesData = []
    #for numFeatures in np.linspace(0.1, 1.0, num=10):
        #clf = runRandomForest(X_train, y_train, 10, numFeatures, None, 1)
        #numFeaturesData.append(np.asarray([numFeatures,clf.score(X_test, y_test)]))
    #numFeaturesData = np.asarray(numFeaturesData)
    #for i in numFeaturesData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Number of Max. Features', fontsize = 22)    
    #plt.plot(numFeaturesData[:,0], numFeaturesData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Max. Features', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       

    
    
    ## TESTING NUMBER OF ESTIMATORS ## 
    #numEstimatorsData = []
    #for numEstimators in range(10, 250, 10):
        #clf = runRandomForest(X_train, y_train, numEstimators, "auto", None, 1)
        #numEstimatorsData.append(np.asarray([numEstimators,clf.score(X_test, y_test)]))
    #numEstimatorsData = np.asarray(numEstimatorsData)
    #for i in numEstimatorsData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Number of Base Estimators', fontsize = 22)    
    #plt.plot(numEstimatorsData[:,0], numEstimatorsData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Base Estimators', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       
    
    #print()
    ### TESTING MAX_DEPTH
    #maxDepthData = []
    #for maxDepth in range(2, 21):
        #clf = runRandomForest(X_train, y_train, 10, , "auto", maxDepth, 1)
        #maxDepthData.append(np.asarray([maxDepth,clf.score(X_test, y_test)]))
    #maxDepthData = np.asarray(maxDepthData)
    #for i in maxDepthData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Maximum Depth', fontsize = 22)    
    #plt.plot(maxDepthData[:,0], maxDepthData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Maximum Depth', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       
    
    #print()
    
    ### TESTING MIN LEAF NODE SIZE
    #minLeafData = []
    #for minLeaf in range(2, 25):
        #print(minLeaf)
        #clf = runRandomForest(X_train, y_train, 10, "auto", None , minLeaf)
        #minLeafData.append(np.asarray([minLeaf,clf.score(X_test, y_test)]))
    #minLeafData = np.asarray(minLeafData)
    #for i in minLeafData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Min. Leaf Node', fontsize = 22)    
    #plt.plot(minLeafData[:,0], minLeafData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Min. Leaf Node', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       

    #clf = runRandomForest(X, y, 37, 0.7, 0.9)

    #testPredictions = clf.predict(testing2012)
    #testPredictions = np.asarray(testPredictions, dtype = int)
    #saveToCSV("bagging2012.csv", testPredictions)

    #allPossible = []
    #maximum = -999
    #for i in range(50, 71, 10):
        #for j in np.linspace(0.1, 0.7, num=7):
            #for k in range(12, 21):
                #clf = runRandomForest(X_train, y_train, i, j , k, 1)
                #accuracy = clf.score(X_test, y_test)
                #if accuracy > maximum:
                    #maximum = accuracy
                    #print (i, j, k, maximum)
                #allPossible.append(np.asarray([i, j, k,accuracy]))
                
    #allPossible = np.asarray(allPossible)
    #for i in allPossible:
        #print(i)
    #print(maximum)
    
    #allPossible = []
    #maximum = -999
    #for i in range(40, 91, 5):
        #for j in np.linspace(0.1, 0.7, num=7):
            #for k in range(10,21):
                #clf = runRandomForest(X_train, y_train, i, j , None, k)
                #accuracy = clf.score(X_test, y_test)
                #if accuracy < 0.77:
                    #break
                #if accuracy > maximum:
                    #maximum = accuracy
                    #print (i, j, k, maximum)
                #allPossible.append(np.asarray([i, j, k,accuracy]))
    #allPossible = np.asarray(allPossible)
    #for i in allPossible:
        #print(i)
    #print(maximum)    
    
    # Testing for max depth: (best of the best)
    
    #clf = runRandomForest(X, y, 70, 0.225, 11, 1)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())
    
    #clf = runRandomForest(X, y, 70, 0.225, 13, 1)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())    
    
    #clf = runRandomForest(X, y, 70, 0.225, 15, 1)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())    
    
    #clf = runRandomForest(X, y, 70, 0.225, 16, 1)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())    
    
    #clf = runRandomForest(X, y, 70, 0.475, 10, 1)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())    
    
    #clf = runRandomForest(X, y, 70, 0.475, 15, 1)
    #testPredictions = clf.predict(data2012)
    #testPredictions = np.asarray(testPredictions)
    #saveToCSV("2012randomForestsFinalmaxDepth.csv", testPredictions)
    
    
    ## Testing for min. leaf : (best of the best)
        
    #clf = runRandomForest(X, y, 60, 0.225, None, 11)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())   
    
    #clf = runRandomForest(X, y, 60, 0.35, None, 16)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())   
    
    #clf = runRandomForest(X, y, 60, 0.35, None, 18)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())   
    
    #clf = runRandomForest(X, y, 60, 0.475, None, 14)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())       
    
    ## ----------------------------------------------------------#
    
    #clf = runRandomForest(X, y, 80, 0.475, None, 15)   
    #testPredictions = clf.predict(data2012)
    #testPredictions = np.asarray(testPredictions)
    #saveToCSV("2012randomForestsFinalMinLeafNode.csv", testPredictions)
    
    #clf = runRandomForest(X, y, 80, 0.475, None, 15)
    #testPredictions = clf.predict(data2012)
    #testPredictions = np.asarray(testPredictions)
    #saveToCSV("2012randomForestsFinalmaxDepth.csv", testPredictions)
    
    # --------------------------------------------#
    
    #clf = runRandomForest(X, y, 40, 0.2 , None, 10)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())   
    
    #clf = runRandomForest(X, y, 40, 0.2 , None, 13)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())       
    
    #clf = runRandomForest(X, y, 40, 0.3 , None, 10)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean()) 
    
    #clf = runRandomForest(X, y, 40, 0.3 , None, 15)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())       
    
    #clf = runRandomForest(X, y, 40, 0.4 , None, 13)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())       
    
    testPredictions = clf.predict(data2012)
    testPredictions = np.asarray(testPredictions)
    saveToCSV("2012randomForestsFinalMinLeafNodeV1.csv", testPredictions)      
    
    #clf = runRandomForest(X, y, 40, 0.5 , None, 13)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())    
    
    #testPredictions = clf.predict(data2008)
    #testPredictions = np.asarray(testPredictions)
    #saveToCSV("2008randomForestsFinalMinLeafNode.csv", testPredictions)    
    
    
    
    
main()

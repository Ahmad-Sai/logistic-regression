# import necessary librarys

import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# CLEAN AND ORGANIZE DATA ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------


file1 = "data.csv"
file2 = "data_clean.csv"

# clean up data and change B and M to 0 and 1, respectively
with open(file1, 'r') as inp, open(file2, 'w', newline='') as out:
    csvreader = csv.reader(inp) # intialze reader     
    writer = csv.writer(out) # intialze writer
    
    headers = next(csvreader)  # skip the header row

    # write clean data into a new csv file
    for row in csvreader:
        if row[0] == 'B':
            row[0] = 0
        elif row[0] == 'M':
            row[0] = 1
        writer.writerow(row)
      
    data = np.array(list(csvreader)).astype(float)  # convert the data into a numpy array 
    
    # write clean data to a new csv file
    for row in csvreader:
        writer.writerow(row)
        
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------


# define class for the logistic model
class Logistic_Model:
    #initialize logistic regression model
    def __init__(self, epsilon):
        self.A = None 
        self.B = None
        self.epsilon = epsilon
        
    # sigmoid function, 1/(1+e^(-x))
    def sigmoid(self, x):   
        return 1/(1+np.exp(-x))

    # affine transform where x is Ax+B
    def affine_transform(self, x):  
        return (np.dot(self.A, x)+self.B)
    
    # the following loss function is a differentiable surrogate (alternative) for accuracy. It is evaluated at the current parameter values for A, B
    def Loss(self, training_set):
        total_loss = 0
        for i in range(len(training_set)):
            total_loss += (-training_set[i][-1]*math.log(self.sigmoid(self.affine_transform(training_set[i][:-1]))))
            -((1-training_set[i][-1])*math.log(1-self.sigmoid(self.affine_transform(training_set[i][:-1]))))
        
        return (total_loss/len(training_set))   # return total loss
    
    # 
    def dLossdA(self, x, y):
        return x * (self.sigmoid(self.affine_transform(x)) - y)
    
    def dLossdB(self, x, y): 
        return self.sigmoid(self.affine_transform(x)) - y
    
    def Train(self, training_set, epochs):
        self.A = np.asarray([random.uniform(-0.1, 0.1) for i in range(len(training_set[0])-1)])
        self.B = np.random.uniform(-0.1, 0.1)

        for i in range(epochs):
            for j in range(len(training_set)):
                self.A = self.A - (self.epsilon * self.dLossdA(training_set[j][:-1], training_set[j][-1]))
                self.B = self.B - (self.epsilon * self.dLossdB(training_set[j][:-1], training_set[j][-1]))

        return self.A, self.B
    
    def Predict(self,x):
        # model_output is the predicted probability value
        model_output = self.sigmoid(self.affine_transform(x))
        
        # apply rounding to obtain the predicted y value of 0 or 1
        if model_output >= 0.5:
            prediction = 1
        else:
            prediction = 0
        return prediction

    # calculate the accuracy of this model based on the predicted y value and actual y value
    def Find_Accuracy(self, test_set):
        correct = 0
        
        for j in range(len(test_set)):
            prediction = self.Predict(test_set[j][:-1])
            #compare the true y value with the predicted y value  
            if test_set[j][-1] == prediction:
                correct += 1
                    
        return correct/len(test_set)
    
    # find and print the confusion matrix for the predicted and actual values of this model
    def Find_Confusion_Matrix(self, test_set):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        for j in range(len(test_set)):
            prediction = self.Predict(test_set[j][:-1])
            # checking to see whether the predicted outputs match the actual outputs
            if prediction == 1 and test_set[j][-1] == 1:
                true_positive += 1
            elif prediction == 1 and test_set[j][-1] == 0:
                false_positive += 1
            elif prediction == 0 and test_set[j][-1] == 1:
                false_negative += 1
            elif prediction == 0 and test_set[j][-1] == 0:
                true_negative += 1
        return np.asarray([[true_positive/len(test_set), true_negative/len(test_set)], [false_positive/len(test_set), false_negative/len(test_set)]])
      
      
# read in training data ---------------------------------------------------      
with open(file3, 'r') as inp:
    csvreader = csv.reader(inp) # intialze reader

    training_set = np.array(list(csvreader)).astype(float)  # convert the data into a numpy array
    
# normalize training data ---------------------------------------------------
for i in range(len(training_set[0])):
    training_set[:,i]=(training_set[:,i]-np.min(training_set[:,i]))/(np.max(training_set[:,i])-np.min(training_set[:,i]))
    
    
# read in testing data ---------------------------------------------------   
with open(file4, 'r') as inp:
    csvreader = csv.reader(inp) # intialze reader

    testing_set = np.array(list(csvreader)).astype(float)  # convert the data into a numpy array
    
# normalize testing data ---------------------------------------------------
for i in range(len(testing_set[0])):
    testing_set[:,i]=(testing_set[:,i]-np.min(testing_set[:,i]))/(np.max(testing_set[:,i])-np.min(testing_set[:,i]))
    
# read in 5 most important columns in the data ---------------------------------------------------
filename = "important_vals.csv"
with open(filename, 'r') as inp:
    csvreader = csv.reader(inp) # intialze reader

    data = np.array(list(csvreader)).astype(float)  # convert the data into a numpy array

# normalize the data ---------------------------------------------------
for i in range(len(data[0])):
    data[:,i]=(data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))    
    

# leave-one-out cross validation ---------------------------------------------------
def Find_Cross_Validation(data, epsilon, epochs):
  # instantiate object of the class
    lmodel = Logistic_Model(epsilon)
    confusion_matrices = []
    A_list = []
    B_list = []

    # perform training
    for i in range(len(data)):
        training_set = np.copy(data)
        testing_set = np.array([data[i]])
        training_set = np.delete(training_set, i, 0)
        
        lmodel.Train(training_set, epochs)
        A_list.append(lmodel.A)
        B_list.append(lmodel.B)
        
        confusion_matrices.append(lmodel.Find_Confusion_Matrix(testing_set))
        
    return (confusion_matrices, A_list, B_list)
  
# train the model ---------------------------------------------------
confusion_matrices, A_list, B_list = Find_Cross_Validation(data, 0.001, 10)

# calculate average ---------------------------------------------------
def average(matrix):
    total_sum = 0
    for i in range(len(matrix)):
        total_sum += matrix[i]

    return total_sum/len(matrix)

# calculate standard deviation ---------------------------------------------------
def standard_deviation(matrix, total_average):
    total_distance = 0
    for i in range(len(matrix)):
        total_distance += (matrix[i]-total_average)**2

    return np.sqrt(total_distance/(len(matrix)-1))
  
# calculate averages and standard deviations ---------------------------------------------------
average_confusion = average(confusion_matrices)
average_Alist = average(A_list)
average_Blist = average(B_list)

standard_confusion = standard_deviation(confusion_matrices, average_confusion)
standard_Alist =  standard_deviation(A_list, average_Alist)
standard_Blist = standard_deviation(B_list, average_Blist)

print("Average Confusion", average_confusion)
print("Average Alist", average_Alist)
print("Average Blist", average_Blist)
print("\n")
print("Standard_confusion", standard_confusion)
print("Standard Alist", standard_Alist)
print("Standard Blist", standard_Blist)

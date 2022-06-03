# Logistic Regression for Breast Cancer Diagnosis 

## Introduction

Logistic Regression is a simple extension of linear regression, particularly suited for classification tasks. With the presence of highly complex, but difficult to interpret models such as neural networks, one may ask, how effective are simple methods such as logistic regression for sensitive tasks such as a medical diagnosis? In this project, we examine the use of logistic regression for classifying malignant and benign tumors for breast cancer data.

## The data

This data was obtained from the Breast Cancer Wisconsin (Diagnostic) Data Set.

The data we are working with contains $31$ dimensions. The $30$ input features are different measurements and medical features of a tumor that were obtained by a Fine Needle Aspiration (FNA). Examples of these features include nucleus radius, area, texture, and concavity of tumor cells. The output feature is binary, with $0$ indicating a benign tumor and a $1$ meaning that the tumor is malignant. 

The data consists of $569$ points with a fairly balanced distribution, with $206$ malignant tumors and $363$ benign tumors.

To make the data more consistent and easier to work with, we normalized it by dividing each value of the first $30$ columns by the highest value in that specific set.


The following box plot shows the skewness of the features:

![image-4.png](attachment:image-4.png)

In the data exploration stage, we also use **Principle Component Analysis (PCA)** to visualize the data. A plot of the 3 most explicatory variables is shown below. 


![image](https://user-images.githubusercontent.com/85080576/171761420-8bdefba9-5d99-4496-801f-4af9c188135d.png)

As we can see the data is not linearly separable in this 3-dimensional projection, so we do not except logistic regression to be 100% accurate.


## Overview of logistic regression

Recall that every linear model has the form $y = Ax + B$, where $x$ is the input vector, $A$ is a row vector that gets placed in a dot product with $x$, $y$ is the output, and $B$ is just a number. Our main task for Logistic regression is to find $A$ and $B$ that best represent and fit our data. 

Akin to a linear model, a logistic model has the form $round (\sigma(Ax + B))$, where $round$ is the rounding operator and $\sigma{(z)} = \frac{1}{1+e^{-z}} $. Similar to a linear model our main goal is to find $A$ and $B$ that best model the data. To do this we look for the parameters $A$ and $B$ which maximize the accuracy of the model with respect to the data. As a seregate we instead minimize the total loss over the whole data set, $$ Loss(A, B, x, y) = -y\cdot\log \big({\frac{1}{1+e^{-(Ax+B)}}\big)} - (1-y)\cdot\log \big(1-{\frac{1}{1+e^{-(Ax+B)}}\big)}$$

To solve this opimtzation probelm we used, **stockastic gradient descen (SGD)**, which involves iterating over each data point $(x, y)$, and updaing the values of $A$ and $B$ according to the following rule: 

$$ A = A - \epsilon \cdot \frac{dL}{dA} $$

$$ B = B - \epsilon \cdot \frac{dL}{dB} $$

Here, $\epsilon$, is a hyper parameter called the **learning rate**, which controls the size of each stride during the learning process. 

The derivatives used in stockastic gradient descent are given by the following equations: 

$$ \frac{dLoss}{dA} = \big(\frac{1}{1+e^{-(Ax+B)}} - y\big)\cdot x $$

$$ \frac{dLoss}{dB} = \frac{1}{1+e^{-(Ax+B)}} - y $$


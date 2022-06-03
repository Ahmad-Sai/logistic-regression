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


![image-3.png](attachment:image-3.png)


As we can see the data is not linearly separable in this 3-dimensional projection, so we do not except logistic regression to be 100% accurate.



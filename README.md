# Logistic Regression for Breast Cancer Diagnosis 

## Introduction

Logistic Regression is a simple extension of linear regression, particularly suited for classification tasks. With the presence of highly complex, but difficult to interpret models such as neural networks, one may ask, how effective are simple methods such as logistic regression for sensitive tasks such as a medical diagnosis? In this project, we examine the use of logistic regression for classifying malignant and benign tumors for breast cancer data.

## The data

This data was obtained from the Breast Cancer Wisconsin (Diagnostic) Data Set.

The data we are working with contains $31$ dimensions. The $30$ input features are different measurements and medical features of a tumor that were obtained by a Fine Needle Aspiration (FNA). Examples of these features include nucleus radius, area, texture, and concavity of tumor cells. The output feature is binary, with $0$ indicating a benign tumor and a $1$ meaning that the tumor is malignant. 

The data consists of $569$ points with a fairly balanced distribution, with $206$ malignant tumors and $363$ benign tumors.

To make the data more consistent and easier to work with, we normalized it by dividing each value of the first $30$ columns by the highest value in that specific set.


The following box plot shows the skewness of the features:

![image](https://user-images.githubusercontent.com/85080576/171761539-dedb962f-168d-44c2-a8cf-c718dc8691cf.png)

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

## Implementation

We initialized A and B to have small random component values between $-0.1$ and $0.1$, and we used a learning rate, $\epsilon = 0.001$. We also ran the SGD for 10 iterations over the training set, i.e. 10 epochs. Since the dataset is relatively small, we decided to use **cross-validation**. This allows us to analyze the variability in the performance. To implement this we use **leave-one-out** cross-validation, which involves selecting a single point as the test set and the rest of the data as the training set and repeating this over the whole dataset.

## Results

The average confusion matrix over the whole cross validation was found to be:

![image](https://user-images.githubusercontent.com/85080576/171761599-514d1f09-2823-43a4-bc59-007b1ddb28c0.png)

This gives us an accuracy of $84.2 \%$.

We interpreted since the data is normalized we interpret the absolute value of the coefficients of $A$ as being an indication of the feature's relative importance. Based on this assumption we found the $3$ most important features to be, _'concave points-mean', 'concave points_worst'_, and _'concavity_mean'_.

## Conclusion

In terms of accuracy, a logistic regression model is relatively good at predicting certain outcomes, but shouldn't be used in medical scenarios. An accuracy of $84.2\%$ is good but does not suffice the need to conduct a medical diagnosis for cancer. The $15.6\%$ percent false-negative rate could mean patients who have cancer may miss out on crucial treatment. In conclusion, a logistic regression model is accurate but is not explicit enough to be used in a medical environment, nonetheless, to diagnose a patient with cancer.

Even if we had more data, the model would probably not perform much better. This can be seen, for example with how entangled the data is in the PCA figure. A more suitable model for such applications would be a neural network, although training such models may require much more data and are much harder to interpret.


# Learning Theory


### Difference between Bias and Variance?

https://medium.com/analytics-vidhya/difference-between-bias-and-variance-in-machine-learning-fec71880c757

Bias is the difference between the average prediction and the correct value. It is also known as Bias Error or Error due to Bias.

Low Bias models: k-Nearest Neighbors (k=1), Decision Trees and Support Vector Machines.
High Bias models: Linear Regression and Logistic Regression.

Variance is the amount that the prediction will change if different training data sets were used. It measures how scattered (inconsistent) are the predicted values from the correct value due to different training data sets. It is also known as Variance Error or Error due to Variance.

Low Variance models: Linear Regression and Logistic Regression.
High Variance models: k-Nearest Neighbors (k=1), Decision Trees and Support Vector Machines.

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data


### What is gradient descent? 
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.

# Support Vector Machine

### 1. Explain SVM
 Support vector machines is a supervised machine learning algorithm which works both on classification and regression problems. It tries to classify data by finding a hyperplane that maximizes the margin between the classes in the training data. Hence, SVM is an example of a large margin classifier.

The basic idea of support vector machines:
- Optimal hyperplane for linearly separable patterns
- Extend to patterns that are not linearly separable by transformations of original data to map into new space(i.e the kernel trick)


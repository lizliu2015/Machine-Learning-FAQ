
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

### 2. What’s the “kernel trick” and how is it useful?
Explanation: Earlier we have discussed applying SVM on linearly separable data but it is very rare to get such data. Here, kernel trick plays a huge role. The idea is to map the non-linear separable data-set into a higher dimensional space where we can find a hyperplane that can separate the samples.

It reduces the complexity of finding the mapping function. So, Kernel function defines the inner product in the transformed space. Application of the kernel trick is not limited to the SVM algorithm. Any computations involving the dot products (x, y) can utilize the kernel trick.

### 3. What is Polynomial kernel?
Explanation: Polynomial kernel is a kernel function commonly used with support vector machines (SVMs) and other kernelized models, that represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables, allowing learning of non-linear models.
For d-degree polynomials, the polynomial kernel is defined as:
![image](https://user-images.githubusercontent.com/13955626/161666167-f28b786d-bbb9-4441-9eec-31815ff99038.png)


### 4. What is RBF-Kernel?
Explanation:
The RBF kernel on two samples x and x’, represented as feature vectors in some input space, is defined as 
![image](https://user-images.githubusercontent.com/13955626/161666111-106860b6-dbda-4832-aca2-fbd31fcc7d67.png)
recognized as the squared Euclidean distance between the two feature vectors. sigma is a free parameter.

### 5.What is the role of C in SVM? How does it affect the bias/variance trade-off?
In the given Soft Margin Formulation of SVM, C is a hyperparameter.
C hyperparameter adds a penalty for each misclassified data point.
Large Value of parameter C implies a small margin, there is a tendency to overfit the training model.
Small Value of parameter C implies a large margin which might lead to underfitting of the model.


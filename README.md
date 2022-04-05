
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

### 2. What are hard margin and soft Margin SVMs?
-  Hard margin SVMs work only if the data is linearly separable and these types of SVMs are quite sensitive to the outliers.
-  But our main objective is to find a good balance between keeping the margins as large as possible and limiting the margin violation i.e. instances that end up in the middle of margin or even on the wrong side, and this method is called soft margin SVM.

### 3. What’s the “kernel trick” and how is it useful?
Explanation: Earlier we have discussed applying SVM on linearly separable data but it is very rare to get such data. Here, kernel trick plays a huge role. The idea is to map the non-linear separable data-set into a higher dimensional space where we can find a hyperplane that can separate the samples.

It reduces the complexity of finding the mapping function. So, Kernel function defines the inner product in the transformed space. Application of the kernel trick is not limited to the SVM algorithm. Any computations involving the dot products (x, y) can utilize the kernel trick.

### 4. What do you mean by Hinge loss?
Hinge Loss is a loss function which penalises the SVM model for inaccurate predictions.
If Yi(WT*Xi +b) ≥ 1, hinge loss is ‘0’ i.e the points are correctly classified. 
When Yi(WT*Xi +b) < 1, then hinge loss increases massively. As Yi(WT*Xi +b) increases with every misclassified point, the upper bound of hinge loss {1- Yi(WT*Xi +b)} also increases exponentially.
Hence, the points that are farther away from the decision margins have a greater loss value, thus penalising those points.

The function defined by max(0, 1 – t) is called the hinge loss function.

![image](https://user-images.githubusercontent.com/13955626/161666787-16efbdbd-7dc4-4af4-bf81-0dad381081b6.png)

Properties of Hinge loss function:
- It is equal to 0 when the value of t is greater than or equal to 1 i.e, t>=1.
- Its derivative (slope) is equal to –1 if t < 1 and 0 if t > 1.
- It is not differentiable at t = 1.
- It penalizes the model for wrongly classifying the instances and increases as far the instance is classified from the correct region of classification.

### 5. Explain different types of kernel functions.
A function is called kernel if there exist a function ϕ that maps a and b into another space such that K(a, b) = ϕ(a)T · ϕ(b). So you can use K as a kernel since you just know that a mapping ϕ exists, even if you don’t know what ϕ function is. These are the very good things about kernels.

Some of the kernel functions are as follows:

- Polynomial Kernel: These are the kernel functions that represent the similarity of vectors in a feature space over polynomials of original variables.
- Gaussian Radial Basis Function (RBF) kernel:  Gaussian RBF kernel maps each training instance to an infinite-dimensional space, therefore it’s a good thing that you don’t need to perform the mapping.

### 6. What is the role of C in SVM? How does it affect the bias/variance trade-off?
In the given Soft Margin Formulation of SVM, C is a hyperparameter.
C hyperparameter adds a penalty for each misclassified data point.
Large Value of parameter C implies a small margin, there is a tendency to overfit the training model.
Small Value of parameter C implies a large margin which might lead to underfitting of the model.

### 7. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
Logistic Regression is computationally more expensive than SVM — O(N³) vs O(N²k) where k is the number of support vectors.
The classifier in SVM is designed such that it is defined only in terms of the support vectors, whereas in Logistic Regression, the classifier is defined over all the points and not just the support vectors. This allows SVMs to enjoy some natural speed-ups (in terms of efficient code-writing) that is hard to achieve for Logistic Regression.

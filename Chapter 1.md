---
tags:
  - "#Python"
  - scikit-learn
---
# Machine learning with scikit-learn
## What is Machine Learning (ML)?
Is the process [whereby](Vocabulary.md):
- Computers are giben the ability to learn
- Without being explicitly programmed
## Unsupervised Learning
Is the process of uncovering hidden *patterns* and *structures* form unlabeled data
## Supervised Learning (SL)
The type of ML where the values to be predicted are already known and the model is built with the [aim](Vocabulary.md) of accurately predicting the target values of unseen data, given the features.
In a nutshell: SL uses features to predict the value of a target variable.
### Types of SL
The types of SL depends on the type of target variable:
- *Classification*: Target variable consist of categories
- *Regression*: Target variable is continuous
### Requirements
- No missing values
- Data in numeric format
- Data stored in pandas Data-Frame or NumPy array 
# Naming conventions
- Feature: predictor variable or independent variable
- Target variable: dependent variable or response variable
# scikit-learn syntax
This module follows the same syntax for all SL models
## Workflow syntax
```python
# import model
from sklearn.module import Model
# create a variable named module and instantiate the model
model = Model()
# fit the model to X (features) and y (targets)
model.fit(X, y)
# use model to predict passing new observations
predictions = model.predict(X_new)
```
---
# The classification challenge
There are **4 steps** for classifying labels of unseen data:
1. Build a model
2. The model learns from the labeled data we pass it
3. Pass unlabeled data to the model as input
4. The model predicts the labels of the unseen data
As a classifier learns from the labeled data we call this the *training data*.
## K-Nearest Neighbors (KNN)
A popular algorithm for classification problems.
Help us in predict the label of any data point by looking at the $k$ *closest* labeled data points.
KNN uses the *majority voting* which makes predictions based on what label the majority of nearest neighbors have.
See the official documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
## Using scikit-learn to fit a classifier
*Algorithm*:
1. Import `KNeighborsClassifier` from `sklearn.neighbors`
2. Split the data in `X` for **features** and `y` for target values
3. Instantiate our `KNeighborsClassifier` setting `n_neighbors` as the number of $k$ neighbors as we want.
4. Fit the classifier to our labeled data by applying the classifier's `.fit()` method.
5. Use the classifier `.predict()` method in the unseen data.
**Note**s:
- Features need to be an array where each column is a feature and each row a different observation.
- Target values need to be a single column with the same number of observations.
- It's necessary to to use the `.values` attribute to convert `X` and `y` to `NumPy` arrays.
```python
import numpy as np
# unseen data
X_new = [
	[56.8, 17.2],
	[24.4, 24.1],
	[50.1, 10.9]
]

# Algorithm
# import the classifier
from sklearn.neighbors import KNeighborsClassifier
# slipt the data
X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values
# instantiate the classifier model
knn = KNeighborsClassifier(n_neighbors=15)
# fit the classifier
knn.fit(X,y)
# use the classifier predictor
predictions = knn.predict(X_new)

# see the results
print(f'Predictions: {predictions}')
```
---
# Measuring model [performance](Vocabulary.md)
We can evaluate the performance of the models.
In classification **accuracy** is a commonly used metric to mesure the performance.
The accuracy formula is the following:
$$accuracy=\dfrac{correct\_predictions}{total\_observations}$$
## Computing accuracy
Because the data used to fit the classifier was used to train the data, this data is not indicative to compute accuracy.
Instead we're going to **split data** in two sets:
- **Training set**: data to fit/train the training set
- **Test set**: data to calculate the accuracy of the train model
## Train/Test split
To split the data we used the `train_test_split` function that has the following parameters (view the official documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)):
- `X`: features.
- `y`: target values.
- `test_size`:  represent the proportion of the dataset to include in the test split (0 to 1).
- `random_state`: set a seed for random number generator that splits the data.
- `stratify`: if we take equal to `y` then [ensure](ensure) our split reflects the proportion of labels in our training and datasets.
**Notes**
- The first two arguments `X` and `y` appears in the official documentation as `*array`
- The proportion for `test_size` is commonly around 0.2 to 0.3
- The function returns 4 arrays: `X_train`, `X_test`, `y_train`, and `y_test`
- To check the accuracy we used `.score()` method
```python
# import the train model split
from sklearn.model_selection import train_test_split
# make the split
X_train, X_test, y_train, y_test = train_test_split(
										X, y, 
										test_size=0.3,
										random_state=21,
										stratify=y)
# instatiate a KNN model
knn = KNeighborsClassifier(n_neighbors=6)
# fit the training data
knn.fit(X_train, y_train)
# check the accuracy
knn.score(X_test, y_test)
```
## Model complexity
How to interpret $k$?
The decision boundaries are [thresholds](Vocabulary.md) for determining what label a model assigns to an observation.
### Larger $k$
If $k$ increase the decision boundary is less affected by individual observations, reflecting a simpler model.
Simpler models are less able to detect relationships in the dataset (*underfitting*)
### Smaller $k$
More complex model can be sensitive to noise in the training data, rather than reflecting general trends (*overfitting*)
## Model complexity and over/underfitting
With KNN model we can calculate accuracy on the training and test sets using incremental $k$ values, and plot the results.
```python
# calculate the accuracies
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
	knn = KNeighborsClassifier(n_neighbors=neighbor)
	knn.fit(X_train, y_train)
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
# plot
plt.figure(figsize=(8, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
```
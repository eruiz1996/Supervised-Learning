# How good is your model
## Classification metrics
Sometimes the *accuracy* is not the best method to determine if a model is good or not predicting. One of the possible problems with this measure could be the **class imbalance**, which occurs in classification problems when one of the class is more frequent that the others.
## Confusion matrix
**Note**: In this topic I also take information from [here](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62).
The **confusion matrix** is a performance measurement for machine learning classification.
Given a binary classifier the are 4 possible combinations of *predicted* and *actual* values:

- **true positive** ($tp$): you predicted <u>positive</u> and it is <u>true</u>
*You predicted that a woman is pregnant and she actually is*

- **true negative** ($tn$): you predicted <u>negative</u> and it is <u>true</u>
*You predicted that a man is not pregnant and he actually is not.*

- **false positive** ($fp$): you predicted <u>positive</u> and it is <u>false</u>
*You predicted that a man is pregnant but he actually is not*.

- **false negative** ($fn$): you predicted <u>negative</u> and it is <u>false</u>
*You predicted that a woman is not pregnant but she actually is*

In the following matrix we can summarize the results:

|                  | Predicted: negative | Predicted: positive |
| :--------------: | ------------------- | ------------------- |
| Actual: negative | $tn$                | $fp$                |
| Actual: positive | $fn$                | $tp$                |
### Accuracy
This metric tell us *from all the classes (positive and negative), how many of them we have predicted correctly*
$$accurary = \dfrac{tp + tn}{tp + tn + fp + fn}$$
### Precision
Also called as **positive predictive value**.
This metric tell us *from all the classes we have predicted as positive, how many are actually positive*.
$$precission = \dfrac{tp}{tp + fp}$$
High precision means: lower false positive rate.

### Recall
Also called **sensitivity**. 
This metric tell us *from all possible classes, how many we predicted correctly*.
$$recall = \dfrac{tp}{tp + fn}$$
High recall means: lower false negative rate.
### F1 - score
Is the harmonic mean of precision and recall, that means this metric gives equal weight to precision and recall.
$$F1=2\cdot\dfrac{precision\cdot recall}{precision + recall}$$
## In scikit-learn

```python
# import the metrics and matrix
from sklearn.metrics import classification_report, confusion_matrix

# create the model
knn = KNeighborsClassifier(n_neighbors=7)
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# fit
knn.fit(X_train, y_train)
# predict
y_pred = knn.predict(X_test)

# confusion matrix
print(confusion_matrix(y_test, y_pred))
# performance metrics
print(classification_report(y_test, y_pred))
```
---
# Logistic Regression and ROC curve
## Logistic Regression (LR)
Is used for classification problems.
This model calculate the probability that an observation belongs to a binary class:
- If the probability $p>0.5$ then the data is labeled `1`
- If the probability $p<0.5$ then the data is labeled `0`
### Algorithm in scikit-learn
1. *Import* `LogisticRegression`
2. *Instantiate* the classifier
3. *Split* the data using `train_test_split`
4. *Fit* the model with the set of **train data**
5. *Predict* the set of **test data**

```python
# step 1: import class
from sklearn.linear_model import LogisticRegression

# step 2: instantiate from class
logreg = LogisticRegression()
# step 3: split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# step 4: fit the model
logreg.fit(X_train, y_train)
# step 5: predict
y_predict = logreg.predict(X_test)
```

**Note**: the default probability threshold for logistic regression is `0.5`
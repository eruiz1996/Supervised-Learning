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
This metric tell us *from all the classes (positive and negative), how many of them we have predicted correctly*.

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

## Predicting Probabilities
We can predict probabilities of each instance belonging to a class by calling logistic regression's `predict_proba()` method and passing the **test features** (`X_test`). This returns a 2-dimensional array with probabilities for both classes.

In the [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba) say that `predict_proba()` method:

> **Parameters**:
> `X`: array-like of shape (`n_samples`, `n_features`)
> Vector to be scored, where `n_samples` is the number of samples and `n_features` is the number of features.

> **Returns**:
> `T`: array-like of shape (`n_samples`, `n_classes`)
> Returns the probability of the sample for each class in the model, where classes are ordered as they are in `self.classes_`.

Then, with the command:
```python
y_predict_probs = logreg.predict_proba(X_test)[:, 1]
```

We slice the second column, representing **the positive class probabilities**, and store the results as `y_pred_probs`.
## Receiver Operating Characteristic (ROC) curve
For this section I recommend to see the correspond section available in the [Crash Course](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) of Google.

The default probability threshold for logistic regression is `0.5`, but happens as we vary this threshold?

In order to answer that question we are going to use the **Receiver Operating Characteristic curve** which is a graph showing the performance of classification model at all classification thresholds.

The curve plots two parameters (one vs other):

**True Positive Rates** (*Recall*)

$$tpr=\dfrac{tp}{tp + fn}$$

**False Positive Rates**

$$fpr = \dfrac{fp}{fp + tn}$$

An ROC curve plots $tpr$ vs. $fpr$ at different classification thresholds. Lowering the classification threshold classifies more items as positive, because with a threshold of `0` for every probability the model is going to predict `1` for all observations (correctly predict all positive values, and incorrectly predict all negative values).

On the other side, a threshold of `1` the model predicts `0` for all the data because the are not probability greater than $1$.
### Plotting ROC curve
The function that we need to use is `roc_curve()` which is in `metrics`.

**Parameters**:
- `y_true`: True binary labels.
- `y_score`: Target scores, can either be probability estimates of the positive class, confidence values, or non-threshold measure of decisions.

**Returns**:
- `fpr`: false positive rates.
- `tpr`: true positive rates.
- `thresholds`: Decreasing thresholds on the decision function used to compute `fpr` and `tpr`.

```python
# importing roc_curve function
from sklearn.metrics import roc_curve

# calling the function
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# plotting
# line 45° to compare the ROC
plt.plot([0, 1], [0, 1], 'k--')
# plotting the fpr and tpr
plt.plot(fpr, tpr)
# title and labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC curve')
plt.show()
```

## Area Under the ROC curve
The perfect model: `1` for the $tpr$ and `0` for $fpr$. For that reason we have the **Area Under the ROC curve** also called **AUC ROC** measures the entire two-dimensional area underneath the entire ROC curve from $(0,0)$ to $(1,1)$.

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.

In a nutshell: if the return number after applying the `roc_curve_score` is close to `1` then the model performs better than randomly guessing the class of each observation. 

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```
---
# Hyperparameter tuning
The **hyperparameters** are the parameters that we need to specify before fitting the model (likes `alpha` and `n_neighbors`).

The process of **hyperparameter tuning** consist in:
1. Try lots of different hyperparameter values
2. Fit all of them separately
3. See how well they perform
4. Choose the best performing values

We can perform the tuning doing two different methods:
- Exhaustion:
	- Use `GridSearchCV`
	- Take all the possible combinations ($kf\cdot num\_hyperparameters\cdot total\_values$)
- Randomized:
	- Use`RandomizedSearchCV`
	- Take random combinations

**Note**: for both models we use the attributes `.best_params_` and `.best_score_`

## Exhaustion
```python
from sklearn.model_selection import GridSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
			  'alpha':np.arange(0.0001, 1, 10),
			  'solver':['sag','lsqr']
}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
rige_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

## Randomized
```python
from sklearn.model_selection import RandomizedSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
			  'alpha':np.arange(0.0001, 1, 10),
			  'solver':['sag','lsqr']
}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
rige_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)

# evaluating test set
test_score = rigde_cv.score(X_test, y_test)
print(test_score)
```

**Note**

The parameter `param_grid` could have as many parameters as we want. For example, the following code:

```python
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1, 1.0, 50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}
```

Have in `params`:
- `"l1"` and `"l2"` as `penalty` values 
-  a range of `50` float values between `0.1` and `1.0` for `C`
- `class_weight` to either `"balanced"` or a dictionary containing `0:0.8, 1:0.2`.
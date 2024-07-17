---
tags:
  - Python
  - scikit-learn
---
# Introduction to regression
In the regression the target variable typically has continuos values.
## Creating feature and target arrays
From the data-frame `diabetes_df` we take as target values the `"glucose"` field. The other fields will be the features
```python
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
```
## Making predictions from a single feature
When we're only use one field as a feature it's important to reshape the array:
```python
# take the 4th field ("bmi")
X_bmi = X[:, 3]
print(y.shape, X_bmi.shape)
# output
# (752,) (752,)
```
With the target value we don't have problems, but it is important to change for the features:
```python
X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape)
# output
# (752, 1)
```
## Fitting a regression model
Using the **Linear Regression Model** (LRM) but without predict a new value:
1. Import the `LinearRegression` class and create an instance.
2. Fit the model to all of the features observations with `.fit()` method.
3. Create the predictions with `.predict()` method passing the features.
Because we predict are predicting data using the same data for the training, then we've a straight line that fits in the best way for our data.
```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# instantiate a regression model
reg = LinearRegression()
# fit the model
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()
```
Click [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) official documentation for the `LinearRegression` class.
## Practice
### Creating features

In this chapter, you will work with a dataset called `sales_df`, which contains information on advertising campaign expenditure across different media types, and the number of dollars generated in sales for the respective campaign. The dataset has been preloaded for you. Here are the first two rows:

```
     tv        radio      social_media    sales
1    13000.0   9237.76    2409.57         46677.90
2    41000.0   15886.45   2913.41         150177.83
```

You will use the advertising expenditure as features to predict sales values, initially working with the `"radio"` column. However, before you make any predictions you will need to create the feature and target arrays, reshaping them to the correct format for scikit-learn.
```python
import numpy as np
# Create X from the radio column's values
X = sales_df['radio'].values
# Create y from the sales column's values
y = sales_df['sales'].values
# Reshape X
X = X.reshape(-1, 1)
# Check the shape of the features and targets
print(X.shape, y.shape)
# output
# (4546, 1) (4546,)
```
### Building a LRM

Now you have created your feature and target arrays, you will train a linear regression model on all feature and target values.

As the goal is to assess the relationship between the feature and target values there is no need to split the data into training and test sets.

`X` and `y` have been preloaded for you as follows:

```
y = sales_df["sales"].values
X = sales_df["radio"].values.reshape(-1, 1)
```

```python
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()
# Fit the model to the data
reg.fit(X, y)
# Make predictions
predictions = reg.predict(X)
print(predictions[:5])
```
### Visualizing a LRM

Now you have built your LRM and trained it using all available observations, you can visualize how well the model fits the data. This allows you to interpret the relationship between `radio` advertising expenditure and `sales` values.

The variables `X`, an array of `radio` values, `y`, an array of `sales` values, and `predictions`, an array of the model's predicted values for `y` given `X`, have all been preloaded for you from the previous exercise.
```python
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")
# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")
# Display the plot
plt.show()
```
---
# The basics of LRM
## Regression Mechanics
The equation for the line in two dimensions (features vs targets) is:
$$y = ax + b$$
where:
- `y`: target
- `x`: single feature
- `a,b`: parameters/coefficients of the model - slope, intercept
**Note**: The parameters `a` and `b` are precisely those we want to adjust (learn).
### Error functions (EF)
Also called **loss functions** or **cost functions** help us to choose the values for `a` and `b` following:
- Define an EF for any given line
- Choose the line that minimizes the EF
The most used EF is the **Residual Sum of Squares** (RSS) which:
- Takes the sum of the *residuals* (real value vs predicted value)
- Square each residual (to avoid compensations with positives and negatives)
- Sum all the square residuals (EF)
The formula is:
$$RSS = \sum_{i=1}^n(y_i-\hat{y_i})^2$$
where:
- $y_i$ is the `i`-*real* target value
- $\hat{y_i}$ is the `i`- *predicted* target value of the LMR
The type of LRM where we aim to minimize the RSS is called **Ordinary Least Squares** (OLS)
## LRM in higher dimensions
When we have $n$ features to fit the target instead of just one we need to use the following equation:
$$y = a_1x_1 + a_2x_2 + \ldots + a_nx_n + b$$
For multiple linear regression models, scikit-learn expects one variable each for feature and target values.
## Different EF
The default metric for measure the accuracy is the $R^2$ which quantifies the amount of variance in target values explained by the features.
The range is 0 to 1 where 1 means the features completely explain the target's variance.
## LRM using all features
1. Import the `train_test_split` function and the `LinearRegression` class.
2. Split the dataset in train and test data using `train_test_split`.
3. Instantiate a `LinearRegression` object
4. Train the LRM with `.fit` method passing `X_train` and `y_train`
5. Create the predictions using `.pedrict()` method using `X_test`
6. Compute $R^2$ with `.score()` method
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, 
													test_size = 0.3,
													random_state = 42)
reg_all = LinearRegression()
reg_all.fit(X_train, Y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)
```
### Mean Squared Error and Root Mean Squared Error
Another way to [assess](Vocabulary.md) a regression model performance is using the Mean Squared Error ($MSE$).
$$MSE = \dfrac{1}{n}\sum_{i=1}^n(y-\hat{y_i})^2$$
The problem with the $MSE$ EF is measure the results in squares and for that reason we've the $RMSE$ which takes square root.
$$RMSE = \sqrt{MSE}$$
```python
from skalearn.metrics import mean_squared_error

# calcule the RMSE
mean_square_error(y_test, y_pred, squared=False)
```
## Practice
### Fit and predict for regression
Now you have seen how linear regression works, your task is to create a multiple linear regression model using all of the features in the `sales_df` dataset, which has been preloaded for you. As a reminder, here are the first two rows:

```
     tv        radio      social_media    sales
1    13000.0   9237.76    2409.57         46677.90
2    41000.0   15886.45   2913.41         150177.83
```

You will then use this model to predict sales based on the values of the test features.

`LinearRegression` and `train_test_split` have been preloaded for you from their respective modules.
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()
# Fit the model to the data
reg.fit(X_train, y_train)
# Make predictions
y_pred = reg.predict(X_test)

print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))
```
### Regression performance

Now you have fit a model, `reg`, using all features from `sales_df`, and made predictions of sales values, you can evaluate performance using some common regression metrics.

The variables `X_train`, `X_test`, `y_train`, `y_test`, and `y_pred`, along with the fitted model, `reg`, all from the last exercise, have been preloaded for you.

Your task is to find out how well the features can explain the variance in the target values, along with assessing the model's ability to make predictions on unseen data.
```python
# Import mean_squared_error
from sklearn.metrics import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)
# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
```
---
# Cross Validation (CV)
When we're computing $R^2$ on our test set there is a potential [pitfall](Vocabulary.md) in the process: *the model performance is dependent on the way we split up the data*. 
In other words: the data points in the test set may have some peculiarities that mean the $R^2$ computed on it is not representative of the model's ability to generalize to unseen data.
In order to avoid this problem, we use the **Cross Validation (CV)** process, which consists of the following steps:

1. **Splitting the data multiple times**: Instead of splitting the data into just one training and test set, we split the data into multiple training and test sets. This allows us to train and evaluate the model on different subsets of the data.
    
2. **Training and evaluating the model multiple times**: For each split, we train the model on the training set and evaluate it on the test set. This gives us multiple performance measures (e.g., $R^2$ scores), which we can then analyze to get a more reliable estimate of the model's performance.
    
3. **Averaging the results**: By averaging the performance measures obtained from each split, we get a more robust estimate of the model's ability to generalize to unseen data. This reduces the variability caused by the peculiarities of any single test set.
## K-Fold Cross Validation
Several cross-validation techniques can be used, depending on the nature of the data and the specific requirements of the problem. The most common is **K-Fold CV** which consist in:

- The data is divided into $k$ equally sized folds.
- The model is trained $k$ times, each time using $k−1$ folds as the training set and the remaining fold as the test set.
- The $k$ results are averaged to produce a single performance estimate
One of the advantages of use the CV is that you can obtain statistics like mean, median, standard deviation, and quantiles.
**Note**: the method `np.quantile(a, q)` compute the `q`-th quantile of `a`. It is important to notice that if you want the 99% confidence interval of the data `df` then
`np.quantile(df, [0.005, 0.995])`
because the range is from 
$$\left(\dfrac{0.01}{2}, 1-\dfrac{0.01}{2}\right)$$
I don't find more information in the [official documentation](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) but for me they suppose that the *ECDF* is normal distributed, and for that reason you ignore the two tails.

```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=df)

# Analyzing cross-validation metrics
# Print the mean
print(np.mean(cv_results))
# Print the standard deviation
print(np.std(cv_results))
# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975])) # it should range from 2.5% to 97.5%
```
---
# Regularized regression
Regularized means *to penalize large coefficients* in order to avoid the **overfitting**.
## Ridge regression
In this type of regularized regression we add to the EF a **ridge** which penalizes large positive or negative coefficients.
$$EF_{new} = EF_{OLS}+\alpha\sum_{i=1}^{n}a_{i}^2$$
where
- $EF_{new}$: new Error Function.
- $EF_{OLS}$: normal Error Function which minimize the OLS.
- $\alpha$: is an **hyperparameter** (see below for more explanation).

An Hyperparameter is a variable used for selecting a model's parameters.
In ridge regressions the value of $\alpha$ controls model complexity:
- If $\alpha=0$ then can lead to *overfitting*.
- If $\alpha\longrightarrow\infty$ then can lead to *underfitting*.  
### Ridge regression in scikit-learn
To use this ridge in scikit-learn we need import `Ridge` from `linear_model`.

In the following code lets see how if we increment the value of $\alpha$ the performance gets worse.
```python
from sklearn.linear_model import Ridge

# alpha's list
alphas = [0.1, 1, 10, 100, 1000]
# to store different scores for different alphas
scores = []

for alpha in alphas:
	# define the ridge, given the alpha
	ridge = Ridge(alpha=alpha)
	# train and predict
	ridge.fit(X_train, y_train)
	ridge.preditc(X_test)
	# store the score in the list
	scores.append(ridge.score(X_test, y_test))
print(scores)
```
## Lasso regression
In the Lasso regression we change the EF as following:
$$EF_{new}=EF_{OLS}+\alpha\sum_{i=1}^{n}|a_i|$$
In scikit-learn the process is the same as with **Ridge regression** but importing `Lasso`.
Advantages of using Lasso:
- Can select important features of a dataset
	- Shrinks the coefficients of less importance to zero, and
	- Features not shrunk to zero are selected by Lasso
- Allows us to communicate results to non-technical audiences
Lets see an example of the second point:
```python
import matlplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# take the whole dataset
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
# take the features-column names
names = X.columns

# define Lasso with alpha = 0.1
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
# obtain the coefficients
lasso_coef = lasso.coef_

# plot for each name the correspond coefficient
plt.bar(names, lasso_coef)
plt.show()
```

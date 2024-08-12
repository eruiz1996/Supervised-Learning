# Preprocessing data
## Dealing with categorical features
Since `scikit-learn` doesn't accept categorical we need to convert to the features in numeric values. We achieve this by splitting the feature into multiple binary features (one for each category), called **dummy variables** follow:
- Assigning `0` if the observation was not a category.
- Assigning `1` if the observation was it.
Example: given the categories `red`, `green`, and `blue` create the dummy variables.

| color | red | green | blue |
| ----- | :-: | ----- | ---- |
| red   |  1  | 0     | 0    |
| green |  0  | 1     | 0    |
| blue  |  0  | 0     | 1    |

But, note that if a color is not in the first two colors then the color needs to be `blue`. So we can delete the last column in order to not repeat information.

| color | red | green |
| ----- | :-: | ----- |
| red   |  1  | 0     |
| green |  0  | 1     |
| blue  |  0  | 0     |
## Dummy variables in Python
To create dummy variables we can use:
- `OneHotEncoder` using `scikit-learn`
- `get_dummies` using `pandas`
### Using `get_dummies()`
In the [official documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) we have the following syntax:
```python
pandas.get_dummies(data,drop_first=False)
```
where:
- `data` (**array-like, Series, or Data Frame**): Data of which to get dummy indicators.
- `drop_first` (**bool default False**): Whether to get $k-1$ dummies out of $k$ categorical levels by removing the first level (to no repeat information).
Example: music dataset where
- `popularity` is the target value,
- `genre` is a categorical feature.
```python
import pandas as pd

# read the dataset
music_df = pd.read_csv('music.csv')
# create the dummies
music_dummies = pd.get_dummies(music_df['genre'], drop_first=True)
# join to the original dataset
music_df = pd.concat([music_df, music_dummies], axis=1)
# removing the original column
music_df = music_df.drop('genre', axis=1)
```
**Notes**:
- If the Data Frame only has one categorical feature, we can pass the entire Data Frame, thus skipping the step of combining variables. 
- If we don't specify a column, the new Data Frame's binary columns will have the original feature name prefixed, (so in the example, they will start with `genre_`) dropping automatically the `"genre"` column.
### Linear regression with dummy variables
```python
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from skelarn.linear_model import LinearRegression

# features
X = music_dummies.drop('popularity', axis=1).values
# target
y = music_dummies['popularity'].values

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# linear regression
linreg = LinearRegression()
# applying cross validation
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
# obtaining rmse (multiply by -1)
print(np.sqrt(-linreg_cv))
```
**Note**: we used the `scoring` parameter with the *negative mean squared error* because `scikit-learn`'s cross-validation metrics presume a higher score is better, so $MSE$ is changed to negative to counteract it.

---
# Handling missing data
The missing data is when there is no a value for a feature in a particular record.
To obtain a sort list of missing values for each field of the data-frame `df` we use the following command: 
```python
print(df.isna().sum().sort_values())
```
## Dropping missing data
A common approach is to remove missing data observations accounting for less than 5% of all data. The pandas method to do is `dropna()`.
```python
df = df.dropna(subset=['col1', 'col2', 'col3'])
```
where `'col1'`, `'col2'`, and `'col3'` **are columns with less than 5% values of all data.**
## Imputing values
Impute values means making educated guesses (with expertise) as to what the missing values could be. For numerical values is common use the **mean** or **median** (depends on the data) and for categorical data, the **mode**.

We must split our data before imputing in order to avoid *data leakage*.
### Imputing algorithm
- **Import** `SimpleImputer`
- **Split** the numerical and categorical data
- **Train model**: categorical and numerical data must share target values and `random_state`
- **Impute data**: the same process for categorical and numerical data
	- Instantiate `SimpleImputer` with an strategy
	- Transform train data with `.fit_transform()`
	- Transform test data with `.transform()`
- **Join** categorical and numerical data for train and test features.
```python
from sklearn.impute import SimpleImputer

# SPLIT DATA
# features with categorical data
X_cat = music_df['genre'].values.reshape(-1, 1)
# features with numerical data
X_num = music_df.drop(['genre', 'popularity'], axis=1).values
# target values
y = music_df['popularity'].values

# TRAIN MODEL
# train categorical
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, text_size=0.2, random_state=42)
# train numerical
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, text_size=0.2, random_state=42)

# IMPUTE: categorical data
# instantiate imputer with a strategy
imp_cat = SimpleImputer(strategy='most_frequent')
# transform train data
X_train_cat = imp_cat.fit_transform(X_train_cat)
# transform test data
X_test_cat = imp_cat.transform(X_test_cat)

# IMPUTE: numerical data
# instantiate imputer with a strategy
imp_num = SimpleImputer(strategy='mean')
# transform train data
X_train_num = imp_num.fit_transform(X_train_num)
# transform test data
X_test_num = imp_num.transform(X_test_num)

# JOIN DATA
# train data
X_train = np.append(X_train_num, X_train_cat, axis=1)
# test data
X_test = np.append(X_test_num, X_test_cat, axis=1)
```
**Note**: when we instantiate a `SimpleImputer` object, by default the strategy is the mean. See more details in the [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
## Imputing with pipeline
Due to the ability to transform data the imputers are known as **transformers**.

The **pipeline** is an object used to run a series of transformations and build a model in a single workflow. It can say that a pipeline is a sequence of data transformers with an optional final predictor.

**Example**: in the following script we perform binary classification to predict whether a song is rock or another genre.
```python
from sklearn.linear_model import LogisticRegressionfrom
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# drop missing values
music_df = music_df.dropna(subset=['genre','popularity', 'loudness', 'liveness', 'tempo'])
# convert values: 1-rock, 0-another genre
music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
# features
X = music_df.drop('genre', axis=1).values
# target values
y = music_df['genre'].values

# define the pipeline's steps
steps = [
		 ('imputation', SimpleImputer()),
		 ('logistic_regression', LogisticRegression())
		 ]
# instantiate Pipeline
pipeline = Pipeline(steps)

# train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)

# predict
pipeline.score(X_test, y_test)
```
Read more about pipelines [here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

---
# Centering and scaling
It is important to scale our data because features scales can disproportionately influence the model, that means we want features to be on a similar scale.

The process of scaling and centering is called **normalizing**.
## Scale data
### Standardization
Consist in subtract the mean and divide by variance. Makes that all the features are centered around zero and have a variance of one which means distributed as a $N(0,1)$
### Min-max
Consist in subtract the minimum and divide by the range.
- Minimum value: `0`
- Maximum value: `1`
### Normalize
Consist in center the data in the ranges from `-1` to `1`

## Scaling in scikit-learn
1. Import scale form `preprocessing`
2. Split the data
3. Scale the train a test data
```python
from sklearn.preprocessing import StandardScaler

# features and target values
X = music _ df.drop("genre" , axis=1).values
y = music _ df["genre"].values 

# model train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled))
```
## Scaling in a pipeline
We can also use a pipeline for scaling
```python
# define pipeline's steps 
steps = [('scaler' , StandardScaler()), 
		 ('knn' , KNeighborsClassifier(n_neighbors=6))
		 ]
# create pipeline
pipeline = Pipeline(steps)

# train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)

# predict values
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test))
```
## CV and scaling in a pipeline
Finally, we can use CV with pipeline.
```python
from sklearn.model_selection import GridSearchCV
steps = [
		 ('scaler' , StandardScaler()),
		 ('knn' , KNeighborsClassifier())
		 ]
pipeline = Pipeline(steps)

# define range for hyperparameters
parameters = {"knn__n_neighbors": np.arange(1, 50)}
# train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# apply CV
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train) 
y_pred = cv.predict(X_test)

# print best score and params to achive it
print(cv.best_score_)
print(cv.best_params_)
```

---
# Evaluating multiple models
To decide which is the best model it is going to depend in different factors because there are *different models for different problems*.

Some of the guide principles are:
- **Size of dataset**
	- <u>Fewer features</u> means *simple models* and *faster training time*
	- Some models, as neural networks, require large amounts of data to perform well
- **Interpretability**: Some models are <u>easier to explain</u>. For example, *linear regression* has a high interpretability as we can understand the coefficients.
- **Flexibility**
	- May improve accuracy by making <u>fewer assumptions</u> about data. For example, *KNN* is more flexible model than linear regression, because doesn't assume any linear relationships.
Due to `scikit-learn` allows the same methods for any model it is easy to compare the performance:
- Regression model:
	- RMSE
	- R-squared
- Classification model performance
	- Accuracy
	- Precision, recall, F1-score
	- ROC AUC
The best practice is to scale our data before evaluating models.
## Evaluating classification models
```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_ model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# features and target values
X = music.drop("genre" , axis=1).values
y = music["genre"].values
# split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# scaling
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# models to compare
models = {
	  "Logistic Regression": LogisticRegression(),
	  "KNN": KNeighborsClassifier(),
	  "Decision Tree": DecisionTreeClassifier()
} 
results = []
# evaluate each model
for model in models.values():
	kf = KFold(n_splits=6, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf) 
# graphic to compare
plt.boxplot(results, labels=models.keys())
plt.show()

# Test set performance
for name, model in models.items():
	# train model
	model.fit(X_train_scaled, y_train)
	# get accuracy
	test_score = model.score(X_test_scaled, y_test)
	print("{} Test Set Accuracy: {}".format(name, test_score))
```

 
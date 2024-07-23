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
The [official documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) we have the following syntax:
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

Commits:
Chapter 3: I include a link to a concept Chapter 4
Chapter 4: dummy variables
Extras: adding new definitions

Comments:
The dummy variables is an important topic but I felt that they only mentioned, don't really explain it.

# OPPE 1-Day Rescue Guide

This guide is for the situation:
- you have 1 day
- you feel like you know almost nothing
- you want the highest chance of solving a big chunk of the paper
- you do not want a perfect theory course, you want survival + marks

Important truth first:
I cannot honestly promise 80% for sure in one day.
But I can help you maximize your chances.
If the real paper follows the same pattern as the practice OPPE, then this guide gives you the best high-yield path.

## 1. The Main Strategy

Do not try to study everything in machine learning.
That is impossible in one day.

Instead, study only the pattern that appeared in the practice OPPE.
That pattern is mostly:
- dataset inspection
- feature vs target
- missing values
- train/test split
- mean median mode std
- scaling
- one-hot encoding
- ColumnTransformer
- LinearRegression
- Ridge
- Lasso
- SGDRegressor
- R2 and MAE
- cross validation
- GridSearchCV
- PCA
- RidgeCV

If you understand these well enough, you can solve a lot of the paper.

## 2. What To Aim For In 1 Day

Your goal is not:
- deep mathematical mastery
- proving formulas
- understanding everything perfectly

Your goal is:
- recognize the question type quickly
- know what function to use
- know what to fit on train only
- know which metric to compute
- avoid common exam mistakes

That alone can get a lot of marks.

## 3. The 80-20 Priority List

If time is limited, study in this order.

### Tier 1: Must Know First

These are the highest priority.
If you know these, you can solve many direct questions.

1. `X` and `y`
2. features vs target
3. categorical vs numerical columns
4. missing value handling
5. train-test split
6. median / mode / mean / std
7. dropping `id`
8. `MinMaxScaler`
9. `OneHotEncoder`
10. `ColumnTransformer`
11. `LinearRegression`
12. `R2 score`
13. `MAE`

### Tier 2: Very Important Model Questions

These often come after preprocessing.

1. `Ridge`
2. `Lasso`
3. `SGDRegressor`
4. `cross_val_score`
5. `KFold`
6. `SequentialFeatureSelector`

### Tier 3: If Time Remains

Still important, but lower priority than the above.

1. `GridSearchCV`
2. `PolynomialFeatures`
3. `PCA`
4. `RidgeCV`

## 4. What To Study Topic By Topic

## 4.1 Dataset Basics

You must know:
- how to load CSV
- how to inspect shape
- how to view column names
- how to check dtypes
- how to check missing values

Questions this helps with:
- how many features?
- which columns are categorical?
- which columns have missing values?
- what are the unique values?

Use:
- `data.shape`
- `data.columns`
- `data.dtypes`
- `data.isna().sum()`
- `data['col'].unique()`
- `data['col'].value_counts()`

## 4.2 Features and Target

You must know:
- target column is what you predict
- all other useful columns are features

Pattern:
```python
X = data.drop(columns='target')
y = data['target']
```

Questions this helps with:
- split into X and y
- average of target
- number of input features

## 4.3 Missing Values

You must know:
- missing numeric -> mean or median
- missing categorical -> mode
- sometimes question explicitly says fill with 0
- compute fill values from training data only

Pattern:
```python
median_value = X_train['income'].median()
X_train['income'] = X_train['income'].fillna(median_value)
X_test['income'] = X_test['income'].fillna(median_value)
```

Pattern for mode:
```python
mode_value = X_train['policy'].mode()[0]
X_train['policy'] = X_train['policy'].fillna(mode_value)
X_test['policy'] = X_test['policy'].fillna(mode_value)
```

Questions this helps with:
- median of income
- most frequent policy
- most frequent area
- missing value replacement

## 4.4 Train-Test Split

You must know:
- data is split by rows
- features stay the same unless columns are dropped later
- random_state makes result reproducible

Pattern:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Questions this helps with:
- shape of X_train
- train-test ratio
- later preprocessing pipeline

## 4.5 Scaling

Focus on this only:
- MinMaxScaler sends values into `[0, 1]`
- fit on training data only
- transform train and test

Pattern:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[['income']] = scaler.fit_transform(X_train[['income']])
X_test[['income']] = scaler.transform(X_test[['income']])
```

Questions this helps with:
- median after scaling
- preprocessing before models
- SGDRegressor questions

## 4.6 One-Hot Encoding

You must know:
- categorical text cannot go directly into many models
- one-hot encoding creates binary columns
- number of columns increases

Pattern:
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
```

Questions this helps with:
- number of features after preprocessing
- later modeling questions

## 4.7 ColumnTransformer

This is very important because it combines preprocessing.

Use it when:
- some columns are categorical -> encode them
- some columns are numeric -> scale them

Pattern:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', MinMaxScaler(), numeric_cols),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

Questions this helps with:
- full preprocessing pipeline
- feature count after preprocessing
- every later model question

## 4.8 Linear Regression

You must know:
- this is the baseline regression model
- train it on processed data
- predict on test data
- compute R2

Pattern:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train_processed, y_train)
pred = model.predict(X_test_processed)
score = r2_score(y_test, pred)
```

Questions this helps with:
- test R2
- cross-validation with default model
- feature selection base estimator

## 4.9 R2 and MAE

You must know these two metrics.

### R2
- higher is better
- 1 is perfect
- around 0 means weak
- negative means bad

Use:
```python
from sklearn.metrics import r2_score
r2_score(y_test, pred)
```

### MAE
- average absolute prediction error
- same unit as target

Use:
```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, pred)
```

Questions this helps with:
- LinearRegression score
- Ridge score
- SGD MAE
- PCA + RidgeCV score

## 4.10 Ridge and Lasso

You do not need full theory.
Only remember:
- Ridge = L2 regularization
- Lasso = L1 regularization
- Lasso can zero out coefficients

Patterns:
```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(random_state=42)
ridge.fit(X_train_processed, y_train)

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train_processed, y_train)
lasso.intercept_
```

Questions this helps with:
- Ridge R2
- Lasso intercept

## 4.11 SGDRegressor

You must know:
- it is sensitive to scaling
- max_iter matters
- can be asked for MAE or R2

Pattern:
```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(random_state=42)
sgd.fit(X_train_processed, y_train)
pred = sgd.predict(X_test_processed)
```

For fixed iterations:
```python
sgd10 = SGDRegressor(random_state=42, max_iter=10)
```

Questions this helps with:
- MAE with default SGD
- R2 with max_iter=10

## 4.12 Cross Validation and KFold

You must know:
- CV gives multiple scores
- use the exact KFold asked in question
- question may ask max score, not mean score

Pattern:
```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(LinearRegression(), X_train_processed, y_train, cv=cv, scoring='r2')
scores.max()
```

Questions this helps with:
- max CV score
- feature selection setup

## 4.13 SequentialFeatureSelector

You only need the pattern.

Pattern:
```python
from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(
    LinearRegression(),
    n_features_to_select=5,
    direction='forward',
    cv=cv
)
sfs.fit(X_train_processed, y_train)
```

Then selected indices:
```python
import numpy as np
np.where(sfs.get_support())[0]
```

Questions this helps with:
- selected feature indices

## 4.14 GridSearchCV and PolynomialFeatures

You must know:
- GridSearchCV checks multiple parameter combinations
- read `best_params_`
- pipeline is used when transform + model are both tuned

Pattern:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import numpy as np

pipe = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('lasso', Lasso(random_state=42)),
])

param_grid = {
    'poly__degree': [1, 2],
    'lasso__alpha': np.logspace(-3, 0, num=5),
}

grid = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
grid.fit(X_train_processed, y_train)

grid.best_params_
```

Questions this helps with:
- best alpha
- best degree

## 4.15 PCA and RidgeCV

You must know:
- PCA reduces dimensions
- fit on train only
- transform both train and test
- `explained_variance_ratio_` tells how much information is preserved

Pattern:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5, svd_solver='full', whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)
pca.explained_variance_ratio_.sum()
```

Then RidgeCV:
```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1])
ridge_cv.fit(X_train_pca, y_train)
pred = ridge_cv.predict(X_test_pca)
```

Questions this helps with:
- sum of explained variance ratio
- R2 after PCA + RidgeCV

## 5. One-Day Study Plan

This is the plan I recommend if you have roughly 10 to 12 focused hours.

## Block 1: 2 hours

Study only these:
- features vs target
- categorical vs numerical
- missing values
- mean median mode std
- train-test split

Do this using:
- your beginner notes
- your notebook questions Q2 to Q14

Goal:
You should be able to answer basic data questions without help.

## Block 2: 2.5 hours

Study preprocessing only:
- dropping `id`
- imputing numeric with median
- imputing categorical with mode
- MinMaxScaler
- OneHotEncoder
- ColumnTransformer
- feature count after preprocessing

Goal:
You should understand how raw data becomes model-ready data.

## Block 3: 2 hours

Study basic models and metrics:
- LinearRegression
- R2
- Ridge
- Lasso
- intercept
- MAE
- SGDRegressor

Goal:
You should be able to fit a model and compute the asked metric.

## Block 4: 2 hours

Study cross-validation and feature selection:
- KFold
- cross_val_score
- max CV score
- SequentialFeatureSelector

Goal:
You should recognize these question patterns immediately.

## Block 5: 2 hours

Study advanced but still exam-relevant topics:
- Pipeline
- GridSearchCV
- PolynomialFeatures
- PCA
- RidgeCV

Goal:
You should at least know the syntax and what each tool is used for.

## Block 6: 1.5 hours

Do revision only.
No new topics.

Revise:
- all code patterns
- all common mistakes
- all metric names
- all "fit on train only" rules

## 6. What To Memorize Exactly

If you are very weak and time is short, memorize these patterns.

### Pattern A: Split target
```python
X = data.drop(columns='cltv')
y = data['cltv']
```

### Pattern B: Train-test split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Pattern C: Numeric imputation
```python
val = X_train['income'].median()
X_train['income'] = X_train['income'].fillna(val)
X_test['income'] = X_test['income'].fillna(val)
```

### Pattern D: Categorical imputation
```python
val = X_train['policy'].mode()[0]
X_train['policy'] = X_train['policy'].fillna(val)
X_test['policy'] = X_test['policy'].fillna(val)
```

### Pattern E: Scaling
```python
scaler = MinMaxScaler()
X_train[['income']] = scaler.fit_transform(X_train[['income']])
X_test[['income']] = scaler.transform(X_test[['income']])
```

### Pattern F: Preprocessing mixed columns
```python
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', MinMaxScaler(), numeric_cols),
])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### Pattern G: Linear Regression
```python
model = LinearRegression()
model.fit(X_train_processed, y_train)
pred = model.predict(X_test_processed)
r2_score(y_test, pred)
```

### Pattern H: Cross Validation
```python
cv = KFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(LinearRegression(), X_train_processed, y_train, cv=cv, scoring='r2')
```

### Pattern I: Grid Search
```python
grid = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
grid.fit(X_train_processed, y_train)
grid.best_params_
```

### Pattern J: PCA
```python
pca = PCA(n_components=5, svd_solver='full', whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)
```

## 7. Most Common Mistakes That Will Lose Marks

1. Using full dataset to compute median or mode instead of training set
2. Forgetting to drop `id`
3. Fitting scaler on test data
4. Forgetting that encoded data has more columns than raw data
5. Using raw data in a later question when transformed data was required
6. Reading MAE when question asked R2
7. Reading average CV score when question asked max CV score
8. Fitting PCA on test data separately
9. Forgetting `random_state=42`
10. Forgetting `shuffle=True` in KFold when question explicitly says it

## 8. What To Do If You Freeze In The Exam

If you get stuck, ask these 6 questions in order:

1. What is the target column?
2. Is this before preprocessing or after preprocessing?
3. Is the asked column numeric or categorical?
4. Is there missing value handling in this question?
5. Which metric is asked: mean, median, mode, std, R2, MAE, CV, best param?
6. Does the instruction say fit on train only?

This can save you.

## 9. Video Plan For One Day

Do not watch random long playlists.
Watch only targeted videos.

### Best YouTube Videos For Your OPPE Pattern

1. Linear Regression, Clearly Explained!!! - StatQuest
https://www.youtube.com/watch?v=7ArmBVF2dCs

2. Multiple Regression, Clearly Explained!!! - StatQuest
https://www.youtube.com/watch?v=EkAQAi3a4js

3. Machine Learning Fundamentals: Cross Validation - StatQuest
https://www.youtube.com/watch?v=fSytzGwwBVw

4. Regularization Part 1: Ridge (L2) Regression - StatQuest
https://www.youtube.com/watch?v=Q81RR3yKn30

5. Regularization Part 2: Lasso (L1) Regression - StatQuest
https://www.youtube.com/watch?v=NGf0voTMlcs

6. StatQuest: Principal Component Analysis (PCA), Step-by-Step
https://www.youtube.com/watch?v=FgakZw6K1QQ

### How To Use These Videos Properly

Do not watch passively.
For each video:
- write 3 lines: what it is, when to use it, what output to read
- then connect it to OPPE questions

Example:
- Ridge: regularized linear regression
- use when question explicitly asks Ridge
- output usually R2 or coefficients/intercept

## 10. Best Official References For Fast Checking

Use these only when you need exact syntax.
Do not try to read them like a textbook.

1. `train_test_split`
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

2. `MinMaxScaler`
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

3. `OneHotEncoder`
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

4. `ColumnTransformer`
https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

5. Linear models overview
https://scikit-learn.org/stable/modules/linear_model.html

6. `Lasso`
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

7. `cross_val_score`
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

8. `GridSearchCV`
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

9. `PCA`
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

10. `RidgeCV`
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

## 11. If You Have Only 4 Hours Left

Then do only this:

1. Read `Practice OPPE 1 - Beginner Study Notes.md`
2. Memorize patterns A to J in this guide
3. Revise Q2 to Q15 carefully
4. Learn only these models:
- LinearRegression
- Ridge
- Lasso
- SGDRegressor
5. Learn only these advanced tools:
- cross_val_score
- GridSearchCV
- PCA

Skip deep theory.
Focus on patterns.

## 12. If You Have Only 2 Hours Left

Then do this only:

1. Learn the split/impute/scale/encode flow
2. Learn LinearRegression + R2
3. Learn Ridge/Lasso names and usage
4. Learn cross_val_score and GridSearchCV names and outputs
5. Learn PCA means dimension reduction
6. Learn the common mistakes list

## 13. Final Honest Advice

If the paper is close to the practice pattern and you stay calm, you can definitely pick up a lot of marks.
Your biggest danger is not lack of intelligence.
Your biggest danger is:
- mixing up train and test rules
- forgetting what data version the question is using
- using the wrong metric
- panicking and skipping easy preprocessing questions

So in the exam:
- solve easy data/preprocessing questions first
- then do LinearRegression/Ridge/Lasso/SGD questions
- then do CV/GridSearch/PCA questions
- if stuck, write the pipeline on rough paper before coding

You do not need to know everything.
You need to know the pattern.

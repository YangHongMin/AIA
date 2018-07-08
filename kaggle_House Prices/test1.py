# import
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from math import ceil
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
warnings.filterwarnings('ignore')
#%matplotlib inline

# load data
# 14 discrete
discrete = ["YearBuilt", "YearRemodAdd", "BsmtFullBath", "BsmtHalfBath", 
            "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
            "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", 
            "MoSold", "YrSold"]
# 20 continuous
continuous = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", 
              "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", 
              "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", 
              "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", 
              "ScreenPorch", "PoolArea", "MiscVal", "SalePrice"]

df_train = pd.read_csv("Data/train.csv", index_col="Id")
df_test = pd.read_csv("Data/test.csv", index_col="Id")
df_train.sample(5)

missing = []
cols = discrete + continuous
cols.remove("SalePrice")
for col in cols:
    cnt = df_train[col].isnull().sum() + df_test[col].isnull().sum()
    if cnt:
        missing.append(col)
        print("%s: %d" % (col, cnt))


# chunks
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]
cols = discrete + continuous
cols.remove('SalePrice')
data = pd.concat([df_train['SalePrice'], df_train[cols]], axis=1)
for lst in chunks(cols, 5):
    sns.pairplot(data, y_vars=['SalePrice'], x_vars=lst)

outliers = {"LotArea": 150000, "BsmtFinSF1": 4000, "TotalBsmtSF": 6000, 
            "1stFlrSF": 4000, "GrLivArea": 5000}

# impute missing values
def impute(df, cols):
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())

impute(df_train, missing)
impute(df_test, missing)
df_train.sample(5)


# clean outliers
def clean_outliers(df, outliers):
    for col in outliers:
        df = df[df[col] < outliers[col]]
    return df

print("Before cleaning: %d" % len(df_train))
df_train = clean_outliers(df_train, outliers)
print("After cleaning: %d" % len(df_train))

# discretize
# categorical attributes
categorical = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", 
               "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
               "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
               "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
               "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
               "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical",
               "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish",
               "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature",
               "SaleType", "SaleCondition"]
icategorical = ["MSSubClass", "OverallQual", "OverallCond"]

qual_dict = {"NONE": 0, "Po": 1, "Fa": 2, "TA": 4, "Gd": 7, "Ex": 11}
qual_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", 
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

def discretize(df, cols):
    for col in cols:
        df[col] = df[col].fillna("NONE")
        if col in qual_cols:
            df[col] = df[col].map(qual_dict).astype('int')
        else:
            df[col] = df[col].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

dataset = pd.concat(objs=[df_train, df_test], axis=0)
discretize(dataset, categorical)
idx = len(df_train)
df_train = dataset[:idx]
df_test = dataset[idx:]
df_train.sample(5)


def distplot(df, cols, ncols):
    nrows = ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 6), dpi=300)
    for idx in range(0, len(cols)):
        x = (int) (idx / ncols)
        y = (int) (idx % ncols)
        attr = cols[idx]
        sns.distplot(df[attr], fit=norm, ax=axes.item((x, y)))
    plt.tight_layout()

numeric = discrete + continuous
df_skew = df_train[numeric].apply(lambda x: stats.skew(x.astype('float')))
df_skew = df_skew[abs(df_skew) > 0.75]

for col in df_skew.index:
    df_train[col] = np.log1p(df_train[col])
    df_test[col] = np.log1p(df_test[col])


def boxplot(df, cols, ncols):
    for lst in chunks(cols, ncols):
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 3), dpi=100)
        for idx in range(0, len(lst)):
            attr = lst[idx]
            data = pd.concat([df['SalePrice'], df[attr]], axis=1)
            sns.boxplot(x=attr, y='SalePrice', data=data, ax=axes[idx])
        plt.tight_layout()



drops = ["BsmtHalfBath", "KitchenAbvGr", "MasVnrArea", "BsmtFinSF1", 
         "BsmtFinSF2", "2ndFlrSF", "LowQualFinSF", "WoodDeckSF", 
         "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", 
         "PoolArea", "MiscVal", "Utilities", "Condition2", "RoofMatl"]
df_train = df_train.drop(drops, axis=1)
df_test = df_test.drop(drops, axis=1)
df_test = df_test.drop('SalePrice', axis=1)


corr = df_train.corr()
cols = df_train.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        attr1 = cols[i];
        attr2 = cols[j];
        if corr[attr1][attr2] > 0.7:
            print("(%s, %s): %.2f" % (attr1, attr2, corr[attr1][attr2]))

drops = ["BldgType", "ExterQual", "GarageYrBlt", "Exterior2nd", "KitchenQual", 
         "1stFlrSF", "TotRmsAbvGrd", "FireplaceQu", "GarageCars", "GarageCond"]
df_train = df_train.drop(drops, axis=1)
df_test = df_test.drop(drops, axis=1)

# evaluate with k-fold cross validation, and report RMSE score
def evaluate(model, X, y, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=2)
    y_pr = np.zeros((X.shape[0],))
    for train_index, test_index in kf.split(X):
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_te = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_tr, y_tr)
        y_pr[test_index] = model.predict(X_te)
    err = sqrt(mean_squared_error(y_pr, y))
    print("RMSE: %.5f" % err)
def coef(model, X):
    for l, r in sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print("(%s, %.5f)" % (l, r))
def print_params(est, X, y, n=100):
    print("Chosen parameter on %d datapoints: %s" % 
          (n,est.fit(X[:n], y[:n]).best_params_))
def generate_submission(model, X):
    suffix = model.__class__.__name__.lower()
    y_pr = model.predict(X)
    result = np.exp(y_pr)
    submission = pd.DataFrame({'Id': X.index.values, 'SalePrice': result})
    submission.to_csv("submission_" + suffix + ".csv", index=None)

X_train = df_train.drop("SalePrice", axis=1)
y_train = df_train["SalePrice"]
X_test = df_test
# settings
nJobs = psutil.cpu_count()
kFold = 10

from sklearn.linear_model import Ridge
params = {'max_iter': 50000}
ridge = Ridge(**params)
est = GridSearchCV(ridge, param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]})
print_params(est, X_train, y_train, 100)

params = {'alpha': 10.0, 'max_iter': 50000}
ridge = Ridge(**params)
# cross validation
evaluate(ridge, X_train, y_train, kFold)


# top 5 attributes
coef(ridge, X_train)
# generate submission
generate_submission(ridge, X_test)

from sklearn.linear_model import Lasso
params = {'max_iter': 50000}
lasso = Lasso(**params)
est = GridSearchCV(lasso, param_grid={"alpha": np.arange(0.0005, 0.001, 0.00001)})
print_params(est, X_train, y_train, 100)

params = {'alpha': 0.00099, 'max_iter': 50000}
lasso = Lasso(**params)
# cross validation
evaluate(lasso, X_train, y_train, kFold)

# top 5 attributes
coef(lasso, X_train)

# generate submission
generate_submission(lasso, X_test)

from sklearn.svm import SVR
params = {"kernel": "linear"}
svr = SVR(**params)
est = GridSearchCV(svr, param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0], 
                                    "epsilon": [0.001, 0.01, 0.1, 1.0, 10.0]})
#print(est, X_train, y_train, 100)

params = {"kernel": "linear", "C": 0.01}
svr = SVR(**params)
# cross validation
evaluate(svr, X_train, y_train, kFold)

# generate submission
generate_submission(svr, X_test)

# rbf performs worse than the linear kernel
params = {"kernel": "rbf"}
svr = SVR(**params)
# cross validation
evaluate(svr, X_train, y_train, kFold)

from sklearn.ensemble import RandomForestRegressor
n_estimators = [100, 200, 400, 800, 1600]
max_features = [0.05, 0.15, 0.30, 0.60, 0.80, "sqrt", "log2", None]
max_depths = [5, 10, 15, 20]
min_samples_leaves = [1, 2, 4]
params = {"n_jobs": nJobs}
rf = RandomForestRegressor(**params)
est = GridSearchCV(rf, param_grid={"n_estimators": n_estimators, 
                                   "max_features": max_features,
                                   "max_depth": max_depths,
                                   "min_samples_leaf": min_samples_leaves
                                  })
print_params(est, X_train, y_train, 400)

arams = {"n_jobs": nJobs, 'max_depth': 20, 'max_features': 0.3, 
          'min_samples_leaf': 1, 'n_estimators': 800}
rf = RandomForestRegressor(**params)
# cross validation
evaluate(rf, X_train, y_train, kFold)

# generate submission
generate_submission(rf, X_test)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
max_features = [0.05, 0.15, 0.30, 0.60, 0.80, "sqrt", "log2", None]
base_estimators = []
for fea in max_features:
    base_estimators.append(DecisionTreeRegressor(max_features=fea))
n_estimators = [3000]
learning_rates = [0.05]
params = {}
ab = AdaBoostRegressor(**params)
est = GridSearchCV(ab, param_grid={"base_estimator": base_estimators, 
                                   "n_estimators": n_estimators, 
                                   "learning_rate": learning_rates})
print_params(est, X_train, y_train, 400)

base_estimator = DecisionTreeRegressor(max_features=0.3)
params = {"base_estimator": base_estimator, "n_estimators": 3000, "learning_rate": 0.05}
ab = AdaBoostRegressor(**params)
# cross validation
evaluate(ab, X_train, y_train, kFold)

# generate submission
generate_submission(ab, X_test)

from sklearn.ensemble import GradientBoostingRegressor
max_features = [0.05, 0.15, 0.30, 0.60, 0.80, "sqrt", "log2", None]
params = {
    'n_estimators': 3000,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_samples_leaf': 15,
    'min_samples_split': 10,
    'loss': 'huber'
}
gbdt = GradientBoostingRegressor(**params)
est = GridSearchCV(gbdt, param_grid={"max_features": max_features})
print_params(est, X_train, y_train, 400)

params = {
    'n_estimators': 3000,
    'learning_rate': 0.05,
    'max_depth': 3,
    'min_samples_leaf': 15,
    'min_samples_split': 10,
    'loss': 'huber',
    'max_features': 'log2'
}
gbdt = GradientBoostingRegressor(**params)
# cross validation
evaluate(gbdt, X_train, y_train, kFold)

# generate submission
generate_submission(gbdt, X_test)

from sklearn.neighbors import KNeighborsRegressor
n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
algorithms = ['ball_tree', 'kd_tree', 'brute']
params = {'n_jobs': nJobs}
knn = KNeighborsRegressor(**params)
est = GridSearchCV(knn, param_grid={"n_neighbors": n_neighbors, "algorithm": algorithms})
print_params(est, X_train, y_train, 400)

params = {'n_jobs': nJobs, 'n_neighbors': 11, 'algorithm': 'ball_tree'}
knn = KNeighborsRegressor(**params)
# cross validation
evaluate(knn, X_train, y_train, kFold)

# generate submission
generate_submission(knn, X_test)

from sklearn.neural_network import MLPRegressor
solvers = ['lbfgs', 'sgd', 'adam']
params = {'learning_rate_init': 0.2, 'max_iter': 5000, 'alpha': 1e-5,
          'momentum': 0.9, 'learning_rate': 'constant'}
mlp = MLPRegressor(**params)
est = GridSearchCV(mlp, param_grid={
    "solver": solvers
})
print_params(est, X_train, y_train, 400)

params = {'learning_rate_init': 0.2, 'max_iter': 5000, 'alpha': 1e-5,
          'momentum': 0.9, 'learning_rate': 'constant', 
          'hidden_layer_sizes': (30, 30, 30, 30), 'solver': 'lbfgs'}
mlp = MLPRegressor(**params)
# cross validation
evaluate(mlp, X_train, y_train, kFold)

# generate submission
generate_submission(mlp, X_test)



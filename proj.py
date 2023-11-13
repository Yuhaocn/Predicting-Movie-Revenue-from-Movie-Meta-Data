

import csv
import os
import pandas as pd
import numpy as np

import lightgbm as lgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sklearn.metrics as metrics

cwd = os.getcwd() 
files = os.listdir(cwd)
print("Files in %r: %s" % (cwd, files))
#import and clean data
train = pd.read_csv('./input/train.csv', usecols=[ 'imdb_id', 'budget', 'popularity','runtime', 'revenue'])

train = pd.merge(train, pd.read_csv('./input/TrainAdditionalFeatures.csv'), how='left', on=['imdb_id'])


train1 = train.drop(train[train['budget'] < 1000].index)

train2 = train1.drop(train1[train1['runtime'] < 60].index)



train2 = train2.dropna()

result = train2.revenue

trainfeature = train2.drop(labels="revenue", axis=1)
trainfeature2 = trainfeature.drop(labels="imdb_id",axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    trainfeature2, result,random_state=1, test_size=0.20)


params = {'n_estimators': 100,
          'max_depth': 8,
          'min_samples_split': 5,
          'learning_rate': 0.05,
          }




my_model_2 = ensemble.GradientBoostingRegressor(**params,verbose=1)
my_model_2.fit(X_train, y_train)

feature_importance = my_model_2.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(X_test. columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    my_model_2, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(X_test. columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()



test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(my_model_2.staged_predict(X_test)):
    test_score[i] = my_model_2.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, my_model_2.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()




print(my_model_2.score(X_test, y_test))



#create heat map
train2 = train2[['budget','rating','totalVotes','popularity','runtime','revenue']]
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train2.corr(), annot=True)
plt.show()




# QQ Plot
from numpy.random import seed
from numpy.random import randn
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
# seed the random number generator
seed(1)
# generate univariate observations
data = train2['popularity2']
# q-q plot
qqplot(data, line='s')
pyplot.show()





from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = my_model_2
boston = trainfeature2
y = result

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, trainfeature2, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
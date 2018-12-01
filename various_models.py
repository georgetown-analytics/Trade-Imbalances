myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

data.columns

data.dropna(inplace=True)

cols=['price', 'rating', 'review_count', 'is_african', 'is_asian_fusion', 'is_bakeries', 'is_bars',
       'is_breakfast_brunch', 'is_buffets', 'is_cafes', 'is_caribbean',
       'is_chinese', 'is_deli', 'is_eastern_european', 'is_european',
       'is_fast_food', 'is_hawaiian', 'is_health_food', 'is_icecream',
       'is_indian', 'is_italian', 'is_japanese', 'is_korean', 'is_latin',
       'is_mediterranean', 'is_mexican', 'is_middleasten', 'is_new_american',
       'is_piza', 'is_seafood', 'is_south_east_asian', 'is_southern',
       'is_street_food', 'is_sweets', 'is_thai', 'is_other_category',
       'is_pickup', 'is_delivery', 'is_restaurant_reservation', 'Canvass',
       'Complaint', 'reinspection', 'License', 'FoodPoison', 'high_risk_1',
       'medium_risk_2', 'low_risk_2', 'grocery', 'Bakery', 'Mobile']

X = data[cols]
y = data['pass']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

#linear regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

print(regr.coef_)

print(regr.intercept_)

print(mean_squared_error(y_test, regr.predict(X_test)))

regr.score(X_test, y_test)

expected   = y_test
predicted  = regr.predict(X_test)

from sklearn import metrics

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

import statsmodels.api as sm

results = sm.OLS(y, X).fit()

print(results.summary())

#linear model ridge
clf = linear_model.Ridge(alpha=0.5)
regr.fit(X_train, y_train)
clf.fit(X_train, y_train)

print(mean_squared_error(y_test, clf.predict(X_test)))
clf.score(X_test, y_test)

#linear model Lasso
clf = linear_model.Lasso(alpha=0.5)
clf.fit(X_train, y_train)

print(mean_squared_error(y_test, clf.predict(X_test)))

clf.score(X_test, y_test)

#selecting best alfa for Ridge
import numpy as np
n_alphas =200
alphas = np.logspace(-10, -2, n_alphas)

clf = linear_model.RidgeCV(alphas=alphas)

clf.fit(X_train, y_train)

print(clf.alpha_)
clf.score(X_test, y_test)

#linear model ridge visualizaion with also vs error
clf = linear_model.Ridge(fit_intercept=False)
errors = []
for alpha in alphas:
    splits = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = splits
    clf.set_params(alpha=alpha)
    clf.fit(X_train, y_train)
    error = mean_squared_error(y_test, clf.predict(X_test))
    errors.append(error)

axe = plt.gca()
axe.plot(alphas, errors)
plt.show()


#GaussianNB model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

expected   = y_test
predicted  = gnb.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

#knearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

#SVM model
from sklearn import svm
estimator = svm.SVC(gamma=0.001)
estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)

expected = y_test

print(metrics.mean_squared_error(expected, predictions))
print(metrics.r2_score(expected, predictions))

estimator.score(X_test, y_test)

errors = abs(predictions - expected)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

MSE = np.mean((predicted-expected)**2)
print(MSE)

#SGDClassifier model
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
print(clf.predict(X_test))

#BaggingClassifier model
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

bagging.fit(X_train, y_train)

bagging.score(X_test, y_test)

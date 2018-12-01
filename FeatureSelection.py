myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )
import pandas as pd
data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import warnings

from ipywidgets import interact, interactive, FloatSlider

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report

from sklearn import model_selection as cv
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


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

#Karen's regularization function
def Regularization():
    regularization_models = [Lasso(),Ridge(),ElasticNet()]

    Reg_len = len(regularization_models)

    i=0
    while i < Reg_len:
        model = regularization_models[i]
        model.fit(X, y)
        print(regularization_models[i])
        print(list(zip(X, model.coef_.tolist())))
        print ('')

        expected = y
        predicted = model.predict(X)

        # Evaluate fit of the model
        print("Mean Squared Error: %0.6f" % mse(expected, predicted))
        print("Mean Absolute Error: %0.6f" % mae(expected, predicted))
        print("Coefficient of Determination: %0.6f" % r2_score(expected, predicted))
        print ('')

        i = i + 1

Regularization()

#getting the best R score from each regularization model (from Karen)
"""Identify best Coefficient of Determination (r2) score"""

def Max_r2():
    alpha_range = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    """Try LASSO"""
    Lasso_alpha_r2 = []
    k=0
    while k < len(alpha_range):
        LassoInstance = Lasso(alpha=alpha_range[k])
        LassoInstance.fit(X, y)

        expected = y
        predicted = LassoInstance.predict(X)

        r2_calc = r2_score(expected, predicted)
        Lasso_alpha_r2.append(r2_calc)

        k=k+1

    print ('Lasso')
    print (max(Lasso_alpha_r2))

    """Try Ridge"""
    Ridge_alpha_r2 = []
    k=0
    while k < len(alpha_range):
        RidgeInstance = Ridge(alpha=alpha_range[k])
        RidgeInstance.fit(X, y)

        expected = y
        predicted = RidgeInstance.predict(X)

        r2_calc = r2_score(expected, predicted)
        Ridge_alpha_r2.append(r2_calc)

        k=k+1

    print ('Ridge')
    print (max(Ridge_alpha_r2))

    """Try Elastic Net"""
    EN_alpha_r2 = []
    k=0
    while k < len(alpha_range):
        ElasticNetInstance = ElasticNet(alpha=alpha_range[k])
        ElasticNetInstance.fit(X, y)

        expected = y
        predicted = ElasticNetInstance.predict(X)

        r2_calc = r2_score(expected, predicted)
        EN_alpha_r2.append(r2_calc)

        k=k+1

    print ('ElasticNet')
    print (max(EN_alpha_r2))

Max_r2()

#LASSO demo select from model--will select 'importance' based on weights
model = Lasso()
sfm = SelectFromModel(model)
sfm.fit(X, y)
print(list(X.iloc[:, sfm.get_support(indices=True)]))

#Ridge demo select from model--will select 'importance' based on weights
model = Ridge()
sfm = SelectFromModel(model)
sfm.fit(X, y)
print(list(X.iloc[:, sfm.get_support(indices=True)]))

#ElasticNet demo select from model--will select 'importance' based on weights

model = ElasticNet()
sfm = SelectFromModel(model)
sfm.fit(X, y)
print(list(X.iloc[:, sfm.get_support(indices=True)]))


#linear SVC model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(X)

X_new.shape

X.shape
X_new

#pipeline
from sklearn.pipeline import Pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)

clf.score(X, y) 

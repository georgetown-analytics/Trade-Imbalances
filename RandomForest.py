#getting data

myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

#iporting modules
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#examining data
data['results'].value_counts()
data['pass'].value_counts()

data.dtypes

#data prep
data.dropna(inplace=True)

#setting up x and y
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

#setting up RandomForestRegressor
model = RandomForestRegressor(n_estimators=4000)
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

#looking at how it did
model.score(X_train, y_train)

MSE = np.mean((predicted-expected)**2)
print(MSE)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

predictions = model.predict(X_test)
errors = abs(predictions - expected)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#changing it up by adding n_estimators max_depth
model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=0)
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

#how did it do
print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

model.score(X_test, y_test)

print(model.feature_importances_)

#changed it up by having max_features and verbose.  didn't help
model = RandomForestRegressor(max_features = 'sqrt',  n_jobs=-1, verbose = 1)
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

#how did it do
print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

model.score(X_test, y_test)

#changed it up by adding n_estimators and max_depth and n_jobs
model = RandomForestRegressor(n_estimators = 1000, max_depth=3, n_jobs = 1)
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

#how did it do
model.score(X_test, y_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

#changed it up with n_estimators, max_features and verbose
model = RandomForestRegressor(n_estimators=100,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

#how did it do
model.score(X_test, y_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

# set up x and y with fewer variables
cols=['price', 'rating', 'review_count', 'high_risk_1',
       'medium_risk_2', 'low_risk_2']

X = data[cols]
y = data['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

#tried it with fewer variables
model = RandomForestRegressor(n_estimators=1000, max_depth=2, n_jobs = 1)
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

model.score(X_test, y_test)

#added back all of variables
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

# did RandomForestRegressor with mae criterion
model = RandomForestRegressor(criterion="mae")
model.fit(X_train, y_train)

expected   = y_test
predicted  = model.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

model.score(X_test, y_test)

#decision tree classifier
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

expected   = y_test
predicted  = clf.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

#DecisionTreeClassifier with new stuff
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
clf = clf.fit(X_train, y_train)

expected   = y_test
predicted  = clf.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

clf.score(X_test, y_test)

#decision tree classifier with gini
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3,
            min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state = 42, splitter='best')
clf = clf.fit(X_train, y_train)

expected   = y_test
predicted  = clf.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

clf.score(X_test, y_test)

#DecisionTreeClassifier with entropy
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state = 42)
clf = clf.fit(X_train, y_train)

expected   = y_test
predicted  = clf.predict(X_test)

print(metrics.mean_squared_error(expected, predicted))
print(metrics.r2_score(expected, predicted))

clf.score(X_test, y_test)

#did a pipeline with SelectFromModel and LinearSVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)

clf.score(X, y)


clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X_train, y_train)

clf.score(X_test, y_test) 

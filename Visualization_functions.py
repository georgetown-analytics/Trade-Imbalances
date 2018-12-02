#importing modules

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError

#getting data
myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

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

#function for printing mean_squared_error, mean_absolute_error and r2
def model_performance(X, y,test_size=0.10, random_state = 42, penalty="l1"):
    models = [GaussianNB(),KNeighborsClassifier(),SGDClassifier(), BaggingClassifier(KNeighborsClassifier()),
              DecisionTreeClassifier(), LinearSVC(penalty=penalty, dual=False)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)

    Reg_len = len(models)

    i=0
    while i < Reg_len:
        model = models[i]
        model.fit(X_train, y_train)
        print(models[i])
        print ('')

        expected = y_test
        predicted = model.predict(X_test)

        # Evaluate fit of the model
        print("Mean Squared Error: %0.6f" % mse(expected, predicted))
        print("Mean Absolute Error: %0.6f" % mae(expected, predicted))
        print("Coefficient of Determination: %0.6f" % model.score(X_test, y_test))
        print ('')

        i = i + 1

#calling function
model_performance(X, y)

#function for classification_report visualization
def classification_report(X, y, test_size=0.10, random_state = 42):
    models = [GaussianNB(),KNeighborsClassifier(),SGDClassifier(), BaggingClassifier(KNeighborsClassifier()),
              DecisionTreeClassifier(), LinearSVC(penalty="l1", dual=False)]

    classes = ["not_passed", "passed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)

    Reg_len = len(models)

    i=0
    while i < Reg_len:
        model = models[i]

        model.fit(X_train, y_train)

        visualizer = ClassificationReport(model, classes=classes)
        visualizer.fit(X_train, y_train) # Fit the visualizer and the model
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data

        print("Coefficient of Determination: %0.6f" % model.score(X_test, y_test))
        g = visualizer.poof()

        print ('')


        i = i + 1

#calling function
classification_report(X, y)

# function for predicting error visualization
def pred_error(X, y, test_size=0.10, random_state = 42):
    models = [GaussianNB(),KNeighborsClassifier(),SGDClassifier(), BaggingClassifier(KNeighborsClassifier()),
              DecisionTreeClassifier(), LinearSVC(penalty="l1", dual=False)]

    classes = ["not_passed", "passed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)

    Reg_len = len(models)

    i=0
    while i < Reg_len:
        model = models[i]

        model.fit(X_train, y_train)

        visualizer = ClassPredictionError(model, classes=classes)
        visualizer.fit(X_train, y_train) # Fit the visualizer and the model
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data

        print("Coefficient of Determination: %0.6f" % model.score(X_test, y_test))
        g = visualizer.poof()

        print ('')


        i = i + 1

#calling function
pred_error(X, y)

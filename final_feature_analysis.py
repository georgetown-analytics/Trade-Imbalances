#prep

myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

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
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

#model performance function
def model_performance(X_train, X_test, y_train, y_test):
    models = [GaussianNB(),KNeighborsClassifier(),SGDClassifier(), BaggingClassifier(),
              DecisionTreeClassifier(), LinearSVC(penalty="l1", dual=False), SVC()]


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
model_performance(X_train, X_test, y_train, y_test)

#feature selector function
def feature_selector(X, y):
    feat_select = [f_classif, chi2, mutual_info_classif]


    Reg_len = len(feat_select)

    i=0
    while i < Reg_len:
        selector = SelectKBest(score_func=feat_select[i])
        X_new = selector.fit_transform(X, y)
        names = X.columns.values[selector.get_support()]
        scores = selector.scores_[selector.get_support()]
        names_scores = list(zip(names, scores))

        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print ('')

        print(feat_select[i])

        print(ns_df_sorted)

        i = i + 1

#calling function
feature_selector(X, y)

#linearsvc feature selector function
def Lsvc_params(X, y):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)

    names = X.columns.values[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    names_scores = list(zip(names, scores))

    ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])


    print("LinearSVC params:")
    print(ns_df_sorted)

#calling function
Lsvc_params(X, y)

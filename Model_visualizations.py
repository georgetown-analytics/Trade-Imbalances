#setup

import psycopg2
host="chicago-datastore.c0o8qindjpbf.us-east-1.rds.amazonaws.com"
port=5432
dbname="chicago"
user="crimeuser"
password="DataScience2018"

myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

data.dropna(inplace=True)

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
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.classifier import PrecisionRecallCurve

#f_classif feature models
col_fclass= ['low_risk_2', 'reinspection', 'high_risk_1', 'License',
           'Complaint', 'rating', 'grocery', 'is_other_category',
           'is_other_category', 'is_bars', 'is_delivery']

X_fclassif = data[col_fclass]
y = data['pass']

X_fclass_train, X_fclass_test, y_train, y_test = train_test_split(X_fclassif, y, test_size=0.10, random_state = 42)

#function to find best model performance
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
model_performance(X_fclass_train, X_fclass_test, y_train, y_test)

#BaggingClassifierwith f_classif features
model = BaggingClassifier()

model.fit(X_fclass_train, y_train)

classes = ["not_passed", "passed"]

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(X_fclass_train, y_train) # Fit the visualizer and the model
visualizer.score(X_fclass_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="bag_classification_report_f_classIF.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(X_fclass_train, y_train)
visualizer.score(X_fclass_test, y_test)
visualizer.poof(outpath="bag_class_errorf_classIF.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_fclass_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_fclass_test, y_test)
visualizer.poof(outpath="bag_descrimination_thresholdf_classIF.png")

# Create the visualizer, fit, score, and poof it
viz = PrecisionRecallCurve(model)
viz.fit(X_fclass_train, y_train)
viz.score(X_fclass_test, y_test)
viz.poof(outpath="bag_precision_recall_curvef_classIF.png")

#KNeighborsClassifier with f_classif features
model =  KNeighborsClassifier()
model.fit(X_fclass_train, y_train)

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(X_fclass_train, y_train) # Fit the visualizer and the model
visualizer.score(X_fclass_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="kneear_classification_report_fclassIF.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(X_fclass_train, y_train)
visualizer.score(X_fclass_test, y_test)
visualizer.poof(outpath="kneear_class_error_fclassIF.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_fclass_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_fclass_test, y_test)
visualizer.poof(outpath="kneear_threshold_fclassIF.png")

# Create the visualizer, fit, score, and poof it
viz = PrecisionRecallCurve(model)
viz.fit(X_fclass_train, y_train)
viz.score(X_fclass_test, y_test)
viz.poof(outpath="knear_precision_recall_curve_fclassIF.png")

#chi2 feature_selection models
col_chi= ['review_count', 'low_risk_2', 'License', 'Complaint', 'reinspection',
          'reinspection', 'high_risk_1', 'grocery', 'is_other_category', 'is_bars',
          'is_health_food']

X_chi = data[col_chi]
y = data['pass']

X_chi_train, X_chi_test, y_train, y_test = train_test_split(X_chi, y, test_size=0.10, random_state = 42)


model_performance(X_chi_train, X_chi_test, y_train, y_test)

#BaggingClassifier with chi2 features
model = BaggingClassifier()
model.fit(X_chi_train, y_train)

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(X_chi_train, y_train) # Fit the visualizer and the model
visualizer.score(X_chi_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="bag_classification_chi_report.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(X_chi_train, y_train)
visualizer.score(X_chi_test, y_test)
visualizer.poof(outpath="bag_class_chi_error.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_chi_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_chi_test, y_test)
visualizer.poof(outpath="bag_descrimination_CHI_threshold.png")

# Create the visualizer, fit, score, and poof it
viz = PrecisionRecallCurve(model)
viz.fit(X_chi_train, y_train)
viz.score(X_chi_test, y_test)
viz.poof(outpath="bag_precision_recall_curve_CHI.png")

#mutual_info_classif feature_selection models
col_mutual= ['Canvass', 'low_risk_2', 'License', 'is_other_category', 'is_icecream',
             'rating','is_piza', 'is_piza', 'grocery', 'is_health_food', 'is_sweets']

X_mutual = data[col_mutual]
y = data['pass']

X_mut_train, X_mut_test, y_train, y_test = train_test_split(X_mutual, y, test_size=0.10, random_state = 42)

model_performance(X_mut_train, X_mut_test, y_train, y_test)

#DecisionTreeClassifier model with mutual_info_classif features
model = DecisionTreeClassifier()
model.fit(X_mut_train, y_train)

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(X_mut_train, y_train) # Fit the visualizer and the model
visualizer.score(X_mut_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="tree_classification_mut_report.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(X_mut_train, y_train)
visualizer.score(X_mut_test, y_test)
visualizer.poof(outpath="tree_class_mut_error.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_mut_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_mut_test, y_test)
visualizer.poof(outpath="tree_descrimination_mut_threshold.png")

# Create the visualizer, fit, score, and poof it
viz = PrecisionRecallCurve(model)
viz.fit(X_mut_train, y_train)
viz.score(X_mut_test, y_test)
viz.poof(outpath="tree_precision_recall_curve_mut.png")

#linear_svc feature model
col_lsvc= ['low_risk_2', 'grocery', 'is_other_category', 'medium_risk_2', 'price',
           'is_asian_fusion', 'is_seafood', 'is_eastern_european', 'is_piza', 'reinspection']

X_lsvc = data[col_lsvc]
y = data['pass']

X_lsvc_train, X_lsvc_test, y_train, y_test = train_test_split(X_lsvc, y, test_size=0.10, random_state = 42)


model_performance(X_lsvc_train, X_lsvc_test, y_train, y_test)

#BaggingClassifier with linear_svc features
model = BaggingClassifier()
model.fit(X_lsvc_train, y_train)

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(X_lsvc_train, y_train) # Fit the visualizer and the model
visualizer.score(X_lsvc_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="bag_classification_lsvc_report.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(X_lsvc_train, y_train)
visualizer.score(X_lsvc_test, y_test)
visualizer.poof(outpath="bag_class_lsvc_error.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_lsvc_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_lsvc_test, y_test)
visualizer.poof(outpath="bag_descrimination_lsvc_threshold.png")

# Create the visualizer, fit, score, and poof it
viz = PrecisionRecallCurve(model)
viz.fit(X_lsvc_train, y_train)
viz.score(X_lsvc_test, y_test)
viz.poof(outpath="bag_precision_recall_curve_lsvc.png")

#all feature_selection
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

#SGDClassifier with all features
model = SGDClassifier()
model.fit(X_train, y_train)

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(X_train, y_train) # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="SGDC_classification_all_report.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof(outpath="SGDC_class_all_error.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)
visualizer.poof(outpath="all_descrimination_sgdc_threshold.png")

viz = PrecisionRecallCurve(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.poof(outpath="all_precision_recall_curve_sgdc.png")

#just rating and review_count
cols= ['rating', 'review_count']

Xr= data[cols]
y = data['pass']

Xr_train, Xr_test, y_train, y_test = train_test_split(Xr, y, test_size=0.10, random_state = 42)

model_performance(Xr_train, Xr_test, y_train, y_test)

#GaussianNB with just rating and review_count
model = GaussianNB()
model.fit(Xr_train, y_train)

visualizer = ClassificationReport(model, classes=classes)
visualizer.fit(Xr_train, y_train) # Fit the visualizer and the model
visualizer.score(Xr_test, y_test)  # Evaluate the model on the test data
visualizer.poof(outpath="Gaussian_classification_report_2features.png")

visualizer = ClassPredictionError(model, classes=classes)
visualizer.fit(Xr_train, y_train)
visualizer.score(Xr_test, y_test)
visualizer.poof(outpath="Gaussian_class_error_2features.png")

visualizer = DiscriminationThreshold(model)
visualizer.fit(Xr_train, y_train)  # Fit the training data to the visualizer
visualizer.score(Xr_test, y_test)
visualizer.poof(outpath="Gaussian_threshold_2features.png")

viz = PrecisionRecallCurve(model)
viz.fit(Xr_train, y_train)
viz.score(Xr_test, y_test)
viz.poof(outpath="Gaussian_recall_curve_2features.png")

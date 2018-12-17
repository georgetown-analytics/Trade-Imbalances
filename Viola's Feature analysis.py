from sklearn import linear_model
from sklearn.linear_model import Lasso
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from yellowbrick.features import RadViz, ParallelCoordinates
from yellowbrick.regressor import PredictionError
from yellowbrick.features import Rank2D
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from yellowbrick.classifier import ROCAUC, ClassificationReport, ConfusionMatrix

myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 42)

clf = linear_model.Lasso(alpha=0.5)
clf.fit(X_train, y_train)
clf.predict(X_test)

visualizer = PredictionError(Lasso())
visualizer.fit(X_train, y_train)

oz = Rank2D(features=cols)
oz.fit_transform(X, y)
oz.poof()

oz = Rank2D(features=cols, algorithm='covariance')
oz.fit_transform(X, y)
oz.poof()

g = sns.jointplot(x='review_count', y='rating', kind='hex', data=data)

h = sns.jointplot(x='price', y='rating', kind='hex', data=data)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
oz = RadViz(classes=label_encoder.classes_, features=cols)
oz.fit(X, y)
oz.poof()

oz = ParallelCoordinates(classes=label_encoder.classes_, features=cols)
oz.fit(X, y)
oz.poof()

oz = ParallelCoordinates(normalize='minmax', classes=label_encoder.classes_, features=cols)
oz.fit(X, y)
oz.poof()

df = pd.DataFrame(data)
numeric_features = [
    "price",
    "rating",
    "review_count",
    "high_risk_1",
    "medium_risk_2",
    "low_risk_2"
]

sns.pairplot(data=df[numeric_features].dropna(), diag_kind='kde')

_, ax = plt.subplots(figsize=(9,6))
df.plot(kind='scatter', x='review_count', y='rating', ax=ax)

ax.set_title("review count to rating")
ax.set_xlabel("number of reviews")

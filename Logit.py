# getting and importing modules
myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import linear_model

data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)
data_unclean = pd.read_sql("Select * FROM food_inspection_predict;", con=myConnection)

#visualization of pass and fail
sns.countplot(x='results',data=data, palette='hls')
plt.show()

#averages
data.groupby('results').mean()

#ratings and pass an fail vizualisation
pd.crosstab(data.rating, data.results).plot(kind='bar')
plt.title('Rating Frequency for Pass Fail')
plt.xlabel('Rating')
plt.ylabel('Frequency of Pass or Fail')

plt.rcParams["figure.figsize"] =(12,9)
plt.show()

#price and pass and fail vizualisation
pd.crosstab(data.price, data.results).plot(kind='bar')
plt.title('Rating Frequency for Pass Fail')
plt.xlabel('Price')
plt.ylabel('Frequency of Pass or Fail')

plt.rcParams["figure.figsize"] =(12,9)
plt.show()

#risk and pass and fail vizualization
table = pd.crosstab(data_unclean.risk, data.results)
table.div(table.sum(1), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Risk vs Results')
plt.xlabel('Risk')
plt.ylabel('Proportion of Results')
plt.show()

#stacked bar chart for price and pass and fail
table = pd.crosstab(data.price, data.results)
table.div(table.sum(1), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Price vs Results')
plt.xlabel('Price')
plt.ylabel('Proportion of Results')
plt.show()

#stacked bar chart for rating and pass and fail
table = pd.crosstab(data.rating, data.results)
table.div(table.sum(1), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Rating vs Results')
plt.xlabel('Rating')
plt.ylabel('Proportion of Results')
plt.show()

#getting rid of NA
data.dropna(inplace=True)

#preping X a y
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

import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#logit model
logit_model = sm.Logit(y, X)

result = logit_model.fit()

print(result.summary())

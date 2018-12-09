#get data and setup

myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )

import pandas as pd
data = pd.read_sql("Select * FROM final_data_thayn;", con=myConnection)

data_unclean = pd.read_sql("Select * FROM food_inspection_predict;", con=myConnection)

import yellowbrick
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from yellowbrick.features import Rank1D
from yellowbrick.features import Rank2D

data.dropna(inplace=True)

#set x and y

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

#histogram of price
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(data['price'], bins = 10, range = (data['price'].min(),data['price'].max()))
plt.title('Price distribution')
plt.xlabel('Price')
plt.ylabel('Count of Price')
plt.show()

#factorplot with price and pass
g = sns.factorplot("price", col="pass", col_wrap=4,
                   data=data[data.price.notnull()], kind="count", size=4, aspect=.8)

#factorplot with rating and pass
g = sns.factorplot("rating", col="pass", col_wrap=4,
                   data=data[data.rating.notnull()], kind="count", size=4, aspect=.8)
g.savefig("rating_results.png")

#factorplot with risk and pass
g = sns.factorplot("risk", col="results", col_wrap=4,
                   data=data_unclean[data_unclean.risk.notnull()], kind="count", size=4, aspect=.8)

#pairplots
g = sns.pairplot(data=data[['price', 'rating', 'review_count',  'pass']], hue='pass')
g.savefig("pairplot.png")

g = sns.pairplot(data=data[['high_risk_1',
       'medium_risk_2', 'low_risk_2',  'pass']], hue='pass')
g.savefig("pairplot_2.png")

#1D and 2D feature analysis
#1D
features = [
        'price', 'rating', 'review_count', 'high_risk_1',
       'medium_risk_2', 'low_risk_2', 'is_pickup', 'is_delivery', 'is_restaurant_reservation', 'Canvass',
       'Complaint', 'reinspection', 'License', 'FoodPoison', 'is_pickup', 'is_delivery', 'is_restaurant_reservation'
    ]
X = data[features]
y = data['pass']

visualizer = Rank1D(features=features, algorithm='shapiro')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="1D_features.png")                   # Draw/show/poof the data

#2D
visualizer = Rank2D(features=features, algorithm='covariance')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="2D_features.png")                   # Draw/show/poof the data


#1D with other features but including rating
features = ['rating',
        'is_african', 'is_asian_fusion', 'is_bakeries', 'is_bars',
       'is_breakfast_brunch', 'is_buffets', 'is_cafes', 'is_caribbean',
       'is_chinese', 'is_deli', 'is_eastern_european', 'is_european',
       'is_fast_food', 'is_hawaiian', 'is_health_food', 'is_icecream',
    ]
X = data[features]
y = data['pass']

visualizer = Rank1D(features=features, algorithm='shapiro')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="1D_features_v2.png")                   # Draw/show/poof the data

#2D with same features as above but without rating and with review_count
features = ['review_count',
        'is_african', 'is_asian_fusion', 'is_bakeries', 'is_bars',
       'is_breakfast_brunch', 'is_buffets', 'is_cafes', 'is_caribbean',
       'is_chinese', 'is_deli', 'is_eastern_european', 'is_european',
       'is_fast_food', 'is_hawaiian', 'is_health_food', 'is_icecream',
    ]
X = data[features]
y = data['pass']

visualizer = Rank2D(features=features, algorithm='covariance')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="2D_features_v2.png")                   # Draw/show/poof the data

#1D with new features but still with rating
features = ['rating',
       'is_indian', 'is_italian', 'is_japanese', 'is_korean', 'is_latin',
       'is_mediterranean', 'is_mexican', 'is_middleasten', 'is_new_american',
       'is_piza', 'is_seafood', 'is_south_east_asian', 'is_southern',
       'is_street_food', 'is_sweets', 'is_thai', 'is_other_category',
    ]
X = data[features]
y = data['pass']

visualizer = Rank1D(features=features, algorithm='shapiro')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="1D_features_v3.png")                   # Draw/show/poof the data

#2D with same features as above but swapping rating with review_count
features = ['review_count',
       'is_indian', 'is_italian', 'is_japanese', 'is_korean', 'is_latin',
       'is_mediterranean', 'is_mexican', 'is_middleasten', 'is_new_american',
       'is_piza', 'is_seafood', 'is_south_east_asian', 'is_southern',
       'is_street_food', 'is_sweets', 'is_thai', 'is_other_category',
    ]
X = data[features]
y = data['pass']

visualizer = Rank2D(features=features, algorithm='covariance')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="2D_features_v3.png")                   # Draw/show/poof the data


#1D and 2D with all features, impossible to see
features = [
       'price', 'rating', 'review_count', 'is_african', 'is_asian_fusion', 'is_bakeries', 'is_bars',
       'is_breakfast_brunch', 'is_buffets', 'is_cafes', 'is_caribbean',
       'is_chinese', 'is_deli', 'is_eastern_european', 'is_european',
       'is_fast_food', 'is_hawaiian', 'is_health_food', 'is_icecream',
       'is_indian', 'is_italian', 'is_japanese', 'is_korean', 'is_latin',
       'is_mediterranean', 'is_mexican', 'is_middleasten', 'is_new_american',
       'is_piza', 'is_seafood', 'is_south_east_asian', 'is_southern',
       'is_street_food', 'is_sweets', 'is_thai', 'is_other_category',
       'is_pickup', 'is_delivery', 'is_restaurant_reservation', 'Canvass',
       'Complaint', 'reinspection', 'License', 'FoodPoison', 'high_risk_1',
       'medium_risk_2', 'low_risk_2', 'grocery', 'Bakery', 'Mobile',
    ]
X = data[features]
y = data['pass']

visualizer = Rank1D(features=features, algorithm='shapiro')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="1D_features_all.png")                   # Draw/show/poof the data

visualizer = Rank2D(features=features, algorithm='covariance')

visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof(outpath="2D_features_all.png") 

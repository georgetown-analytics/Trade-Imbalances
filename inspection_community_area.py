
#get food inspection data
myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )
cur = myConnection.cursor()
cur.execute("Select * FROM food_inspection;")

FOOD_data = cur.fetchall()

import pandas as pd
food_db = pd.DataFrame(FOOD_data)

headers=["inspection_id","dba_name","aka_name","license_num","facility_type","risk","street",
         "city","state","zip","inspection_date","inspection_type","results","violations","latitude",
         "longitude","case_location"]

food_db.columns=headers

#zip to community area excel file
zip_com = pd.read_excel('zip_community.xlsx', sheet_name='Sheet1')

zip_com.head()

#Joining community area with food inspection data
food_db.dtypes

zip_com.dtypes

#merg3
inspection_c = pd.merge(food_db, zip_com, on='zip')

#limit data to 2018 and restaurants. Also store inspection date as to_date
inspection_c_time = inspection_c[inspection_c['inspection_date'].dt.year > 2017]

restaurants_inspect = inspection_c_time[inspection_c_time['facility_type']=='Restaurant']

restaurants_inspect.head()

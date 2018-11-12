import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#get data
myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )
socio = pd.read_sql("Select * FROM all_socio;", con=myConnection)

socio.columns

socio[['per_capita_income','hardship_index', 'average_household_size', 'hispanic', 'median_age', 'predominant_non_english_language','p_aged_16_unemployed','p_household_below_poverty']].describe()
socio[['per_capita_income','hardship_index', 'average_household_size', 'hispanic', 'median_age', 'predominant_non_english_language','p_aged_16_unemployed','p_household_below_poverty']].median()


socio['per_capita_income'].mean()
socio['hardship_index'].mean()

socio['community_id'].unique()

#visualizations

plt.bar(socio['community_id'], socio['per_capita_income'])

plt.bar(socio['community_id'], socio['hardship_index'])

plt.scatter(socio['hardship_index'], socio['per_capita_income'])

plt.scatter(socio['hardship_index'], socio['p_aged_25_unemployed'])

plt.scatter(socio['hardship_index'], socio['average_household_size_x'])

plt.scatter(socio['p_housing_crowded'], socio['average_household_size_x'])

plt.scatter(socio['median_age'], socio['average_household_size_x'])

plt.bar(socio['median_age'], socio['average_household_size_x'])

plt.bar(socio['p_housing_crowded'], socio['average_household_size_x'])

plt.scatter(socio['vacant_housing_units'], socio['occupied_housing_units'])

socio['predominant_non_english_language'].unique()

plt.scatter(socio['renter_occupied'], socio['occupied_housing_units'])

plt.scatter(socio['renter_occupied'], socio['owned_mortgage'])

plt.scatter(socio['hispanic'], socio['total_households_x'])

 plt.scatter(socio['owned_free_and_clear'], socio['renter_occupied'])

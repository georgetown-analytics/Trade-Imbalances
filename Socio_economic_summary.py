#select all the socioeconomic_data
myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )
cur = myConnection.cursor()
cur.execute("Select * FROM socioeconomic_data;")

socio_data = cur.fetchall()

#make socio_data into a DataFrame
import pandas as pd
socio_db = pd.DataFrame(socio_data)

#give the columns headers
headers=["community_id", "community_name", "p_housing_crowded", "p_household_below_poverty", "p_aged_16_unemployed",
    "p_aged_25_unemployed", "p_aged_under_18_over_64", "per_capita_income", "hardship_index"]
socio_db.columns=headers

#Count if there are any na values (nope)
socio_db.isna().sum()

#Change categorical numeric data to numeric data
socio_db["p_housing_crowded"]=pd.to_numeric(socio_db["p_housing_crowded"])
socio_db["p_household_below_poverty"]=pd.to_numeric(socio_db["p_household_below_poverty"])
socio_db["p_aged_16_unemployed"]=pd.to_numeric(socio_db["p_aged_16_unemployed"])
socio_db["p_aged_25_unemployed"]=pd.to_numeric(socio_db["p_aged_25_unemployed"])
socio_db["p_aged_under_18_over_64"]=pd.to_numeric(socio_db["p_aged_under_18_over_64"])
socio_db["p_household_below_poverty"]=pd.to_numeric(socio_db["p_household_below_poverty"])
socio_db["per_capita_income"]=pd.to_numeric(socio_db["per_capita_income"])
socio_db["hardship_index"]=pd.to_numeric(socio_db["hardship_index"])

#Change  community_id to categorical variable
socio_db['community_id'] = pd.Categorical(socio_db["community_id"])

#Describe the data in socio_db
socio_db.describe(include='all')

#Get eduction data
cur.execute("Select community_id, age_25_and_over, less_than_high_school_dip, high_school_grad, some_college_degree, baschelor_or_higher FROM census_data_education;")
education_data = cur.fetchall()
educ_db = pd.DataFrame(education_data)

#give the education data headers
ed_headers=["community_id", "age_25_and_over", "less_than_high_school_dip", "p_household_below_poverty", "p_aged_16_unemployed",
    "p_aged_25_unemployed", "p_aged_under_18_over_64", "per_capita_income", "hardship_index"]
educ_db.columns=ed_headers

#make community_id a categorical variable
educ_db['community_id'] = pd.Categorical(educ_db["community_id"])

#merge socio_db and educ_db
all_socio = pd.merge(socio_db, educ_db, on='community_id')

#get language data
cur.execute("Select * FROM census_data_language;")
lang_colnames = [desc[0] for desc in cur.description]
language_data = cur.fetchall()
lang_db = pd.DataFrame(language_data)
lang_db.columns=lang_colnames

#make community_id a categorical variable
lang_db['community_id'] = pd.Categorical(lang_db["community_id"])

#describe lang_db
lang_db.describe(include='all')

#merge lang_db into all_socio
all_socio = pd.merge(all_socio, lang_db, on='community_id')


#get population data
cur.execute("Select community_id, community_name, total_population, not_hispanic_white, not_hispanic_african_american,
        not_hispanic_asian, hispanic, median_age, total_households, average_household_size, total_housing_units, occupied_housing_units,
        vacant_housing_units, vacant_housing_units, owned_mortgage, owned_free_and_clear, renter_occupied  FROM census_data_population;")
pop_colnames = [desc[0] for desc in cur.description]
pop_data = cur.fetchall()
pop_db = pd.DataFrame(pop_data)
pop_db.columns=pop_colnames

#make community_id a categorical variable
pop_db['community_id'] = pd.Categorical(pop_db["community_id"])

#describe pop_db
pop_db.describe(include='all')

#merge lang_db into all_socio
all_socio = pd.merge(all_socio, pop_db, on='community_id')

#starting vizualisations
import matplotlib.pyplot as plt

#total population on x axis and percent crowded on y axis
plt.plot(all_socio["total_population"], all_socio["p_housing_crowded"],'ro')
plt.ylabel('percent crowded')
plt.xlabel('total population')
plt.show()

#a bar chart with percent croweded on the x axis and total population on y axis
plt.bar(all_socio["p_housing_crowded"],all_socio["total_population"], align='center', alpha=0.5)
plt.xlabel('percent crowded')
plt.ylabel('total population')
plt.show()

#frequency chart of percent crowded
plt.hist(all_socio["p_housing_crowded"], bins=np.arange(all_socio["p_housing_crowded"].min(), all_socio["p_housing_crowded"].max()+1))

#frequency chart of percent of households below poverty
plt.hist(all_socio["p_household_below_poverty"], bins=np.arange(all_socio["p_household_below_poverty"].min(), all_socio["p_household_below_poverty"].max()+1))

#show list of column ed_headers
all_socio.columns

#average_household_size on x axis and average_household_size on y axis
plt.plot(all_socio['average_household_size'], all_socio['median_age'],'ro')
plt.ylabel('median_age')
plt.xlabel('average_household_size')
plt.show()


#pulling socioeconomic_data
import pandas as pd

myConnection = psycopg2.connect( host=host, user=user, password=password, dbname=dbname )
socio = pd.read_sql("Select * FROM socioeconomic_data;", con=myConnection)

socio.head()

socio.dtypes

socio['community_id'] = pd.Categorical(socio["community_id"])

socio.describe(include='all')

#pulling education data
educ = pd.read_sql("Select community_id, age_25_and_over, less_than_high_school_dip, high_school_grad, some_college_degree, baschelor_or_higher FROM census_data_education;", con=myConnection)

educ.head()

educ.dtypes

educ['community_id'] = pd.Categorical(educ["community_id"])

educ.describe(include='all')

#merge socio and educ
all_socio = pd.merge(socio, educ, on='community_id')

#pull language data
lang = pd.read_sql("Select * FROM census_data_language;", con=myConnection)
lang['community_id'] = pd.Categorical(lang["community_id"])
lang.dtypes

lang.describe(include='all')

#merge all-socio and language data
all_socio = pd.merge(all_socio, lang, on='community_id')

#pull population data
pop = pd.read_sql("Select community_id, community_name, total_population, not_hispanic_white, not_hispanic_african_american, not_hispanic_asian, hispanic, median_age, total_households, average_household_size, total_housing_units, occupied_housing_units, vacant_housing_units, vacant_housing_units, owned_mortgage, owned_free_and_clear, renter_occupied  FROM census_data_population;", con=myConnection)

pop['community_id'] = pd.Categorical(pop["community_id"])
pop.dtypes
pop.describe(include='all')

#merge population data and all_socio
all_socio['community_id'] = pd.Categorical(all_socio["community_id"])
pop['community_id'] = pd.Categorical(pop["community_id"])

all_socio = pd.merge(all_socio, pop, on='community_id')

#upload the merged data onto AWS
import sqlalchemy

all_socio.to_sql('all_socio', engine)

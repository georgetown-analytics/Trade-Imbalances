import pandas as pd
import numpy as np

#open up distance ExcelFile

xl = pd.ExcelFile('/Users/laurathayn/Desktop/Distance.xlsx')
df = xl.parse("Dis")

#Clean up country variable

df['Country']=df['Country'].str.replace('Distance from ','')
df['Country']=df['Country'].str.replace(' to China','')
df['Country']=df['Country'].str.replace('China to ','')

#Create a list of the belt and road countries

BRI = ['Afghanistan', 'Albania', 'Antigua and Barbuda', 'Armenia', 'Austria', 'Azerbaijan', 'Bahrain',
       'Bangladesh', 'Belarus', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Brunei', 'Bulgaria', 'Cambodia',
       'Croatia', 'Czech Republic', 'Egypt', 'Estonia', 'Ethiopia', 'Georgia', 'Hungary', 'India', 'India',
       'Indonesia', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kazakhstan', 'Korea', 'Kuwait', 'Kyrgyzstan', 'Laos',
       'Latvia', 'Lebanon', 'Libya', 'Lithuania', 'Macedonia', 'Madagascar','Malaysia', 'Maldives', 'Moldova',
       'Mongolia', 'Montenegro', 'Morocco', 'Myanmar', 'Nepal', 'New Zealand', 'Oman', 'Pakistan', 'Palestine',
       'Panama', 'Papua New Guinea', 'Philippines', 'Poland', 'Qatar', 'Romania', 'Russia',' Saudi Arabia',
       'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'Sri Lanka',
       'Syrian Arab Republic', 'Tajisitan', 'Thailand', 'Timor-Leste', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Ukraine','United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']

#Create a dummy variable if country is belt and road
df['BRI'] = np.where(df['Country'].isin(BRI), '1', '0')

#Create a list of border countries
Border = ['Afghanistan' ,'Bhutan', 'India', 'Kazakstan', 'Krgyzstan', 'Laos', 'Mongolia', 'Myanmar', 'Nepal',
          'North Korea', 'Pakistan', 'Russia', 'Tajikistan', 'Vietnam']

#Create a dummy variable for border countries
df['Border'] = np.where(df['Country'].isin(Border), '1', '0')

#sort dataframe
df = df.sort_values(['Country'])

#output into ExcelFile
df.to_excel('/Users/laurathayn/Desktop/CHINA DATA/Border_Distance_BRI_data.xlsx')



df = df.sort_values(['Country'])

import pandas as pd

#open excel and pivot year for FDI_inflow

xl = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/FDI_inflow.xlsx')
df = xl.parse("FDI")
df_inflow = df.melt(['Country'], var_name='Year', value_name='FDI_inflow')

#open excel and pivot year for FDI_outflow

xl2 = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/FDI_outflow.xlsx')
df2 = xl.parse("FDI")
df_outflow = df2.melt(['Country'], var_name='Year', value_name='FDI_outflow')

#merge FDI_inflow and FDI_outflow into on dataframe on Country and Year

combined_df = pd.merge(df_inflow, df_outflow, how='outer', left_on=['Country','Year'], right_on = ['Country','Year'])

#open excel and pivot year for Import_services

xl3 = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/Imports_of_services.xlsx')
df3 = xl3.parse("Import_services")
df_serv_imp = df3.melt(['Country'], var_name='Year', value_name='Services_Imported')

#merged Import_services onto combined_df

combined_df = pd.merge(combined_df, df_serv_imp, how='outer', left_on=['Country','Year'], right_on = ['Country','Year'])

# open excel and pivot year for Export_services

xl4 = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/Exports_of_services.xlsx')
df4 = xl4.parse("Export_Service")
df_serv_exp = df4.melt(['Country'], var_name='Year', value_name='Services_Exported')

#merged Export_services onto combined_df

combined_df = pd.merge(combined_df, df_serv_exp, how='outer', left_on=['Country','Year'], right_on = ['Country','Year'])

# open excel and pivot year for GDP_per_capita

xl5 = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/GDP_per_capita.xlsx')
df5 = xl5.parse("GDP")
df_GDP = df5.melt(['Country'], var_name='Year', value_name='GDP_per_capita')

#merged GDP_per_capita onto combined_df

combined_df = pd.merge(combined_df, df_GDP, how='outer', left_on=['Country','Year'], right_on = ['Country','Year'])

# open excel and pivot year for Effective_import_tariff_Manufactured_goods
# this cannot match on the country names because of formating issues
#xl6 = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/Effective_import_tariff_Manufactured_goods.xlsx')
#df6 = xl6.parse("Tarrif_Man")
#df_ManuT = df6.melt(['Country'], var_name='Year', value_name='Tarrif_Manu')

#merged df_ManuT onto combined_df

#combined_df = pd.merge(combined_df, df_ManuT, how='outer', left_on=['Country','Year'], right_on = ['Country','Year'])

# open excel and pivot year for CPI_base_year_2005

xl7 = pd.ExcelFile('/Users/laurathayn/Desktop/CHINA DATA/CPI_base_year_2005.xlsx')
df7 = xl7.parse("CPI")
df_CPI = df7.melt(['Country'], var_name='Year', value_name='CPI')

#merged df_CPI onto combined_df

combined_df = pd.merge(combined_df, df_CPI, how='outer', left_on=['Country','Year'], right_on = ['Country','Year'])

# sort combined_df by Country and year

combined_df = combined_df.sort_values(['Country', 'Year'])

#output combined_df to excel in my folder

combined_df.to_excel('/Users/laurathayn/Desktop/CHINA DATA/UNCTAD_data.xlsx')

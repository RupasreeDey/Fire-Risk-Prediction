#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:30:57 2023

@author: rupasree
"""

import datetime
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

def drop_duplicate(df):
    df = df.drop_duplicates()
    return df

def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def create_profile(df, name):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file(name)
    
def cleanActivity(x):
    if x == 11 or x == 12 or x == 13:
        x = 'SINGLE OR TWO RESIDENTIAL'
    elif x == 21 or x == 22 or x == 23 or x == 24 or x == 25:
        x = 'MULTIFAMILY'
    elif x in list(range(31,39)):
        x = 'OFFICE AND PUBLIC SERVICE'
    elif x in list(range(40,50)):
        x = 'INSTITUTIONAL'
    elif x in list(range(51, 54)):
        x = 'COMMERCIAL'
    elif x == 61 or x in list(range(63,70)):
        x = 'INDUSTRIAL'
    elif x == 62:
        x = 'AIRPORT'
    elif x in range(70,98) or x in range(0, 9):
        x = 'OPEN SPACES/NA'
    return x

######################
# wrangle address file
######################
df_address = pd.read_csv('Address.csv')

df_address['ADDRESS'] = df_address['ADDRESS'].str.upper()
df_address['ADDRESS'] = df_address['ADDRESS'].str.replace(" ", "")

columns_to_drop = ['STATE', 'MUNICIPALITY','X', 'Y', 'OBJECTID', 'TAG', 'ADDRESSNUM_SUF', 'ALIAS',
       'ALIASPLUS', 'LON', 'LAT', 'PATH', 'HYPERLINK', 'DISPLAY', 'HOUSENUM',
       'PR1', 'STNM1', 'TYP1', 'PD1','STREETVIEW', 'EFFECTIVEDATE', 'UNITTYPE', 'POINTTYPE', 'COUNTRY', 'GlobalID', 'PHOTOARCHIVELINK',
       'PHOTOACQUISITIONYEAR', 'RETIREDATE', 'ADDRCLASS']

df_address_new = drop_columns(df_address, columns_to_drop)

current_year = 2016
df_address_new['YEARBUILT'] = pd.to_numeric(df_address_new['YEARBUILT'], errors='coerce')

df_address_new['Age'] = current_year - df_address_new['YEARBUILT']
df_address_new = drop_columns(df_address_new, ['YEARBUILT'])

# drop duplicates
df_address_new = drop_duplicate(df_address_new)
#create_profile(df_address_new, "Address.html")
df = df_address_new.copy()

######################
# wrangle codecases file
######################

fict_year = 2016
df_codeCases = pd.read_csv('CodeCases_2015-2021.csv')

df_codeCases['Address'] = df_codeCases['Address'].str.upper()
df_codeCases['Address'] = df_codeCases['Address'].str.replace(" ", "")

# remove data after 2015
df_codeCases = df_codeCases[df_codeCases['Year']<=fict_year]

# Code cases from Medium
case_types = pd.get_dummies(df_codeCases['Code Case Type'])

# find code case types that are not present in the filtered data, we need to create those columns manually
cols = ['Building Service', 'Drainage',
'Erosion and Sediment Control', 'Fire', 'Health Nuisance Complaints',
'IMPORT / Tree Survey & Stump Removal', 'Illicit Discharge', 'Landfill',
'Manufactured Housing', 'Parks and Rec - Dead or Diseased Tree',
'Parks and Rec - Tree Complaint', 'Property Maintenance',
'Rental Registration', 'Right of Way ', 'Sidewalks and Ramps', 'Snow',
'Special Assessment', 'Vegetation', 'Waste Water', 'Water Purfication',
'Zoning']

current_columns = df_codeCases.columns

cols_to_create = [col for col in cols if col not in current_columns]

case_types[cols_to_create] = 0


code_cases = pd.concat([df_codeCases, case_types], axis=1)
code_cases.drop(columns = ['Year'], inplace = True)
code_cases_grouped = code_cases.groupby(['Address']).sum().reset_index()
code_cases_df = pd.DataFrame(code_cases_grouped)

code_cases_df['TOTAL_VIOLATIONS'] = code_cases_df.sum(axis=1)
code_cases_df['ANY_VIOLATIONS'] = 1

code_cases_df.rename(columns = {'Address': 'ADDRESS'}, inplace = True)

# merge address and codecases
df = pd.merge(df, code_cases_df, how = 'left', on = 'ADDRESS')
df.ANY_VIOLATIONS.fillna(0, inplace = True)
df['TOTAL_VIOLATIONS'] = df['TOTAL_VIOLATIONS'].fillna(0)

# impute codecase types columns with zero in the main df
df[cols] = df[cols].fillna(0)

######################
# wrangle property offenses/crimedata file
######################

df_offenses = pd.read_excel('Copy of Property Offenses 2017-2021.xlsx')

df_offenses['Parcel_Address'] = df_offenses['Parcel_Address'].str.upper()
df_offenses['Parcel_Address'] = df_offenses['Parcel_Address'].str.replace(" ", "")

df_offenses_new = df_offenses[['Parcel_Address', 'Offense']]

# drop duplicates

df_offenses_new = drop_duplicate(df_offenses_new)

df_offenses_new.rename(columns = {'Offense' : 'CRIME_INCIDENT'}, inplace = True)
crime_grouped = df_offenses_new.groupby('Parcel_Address').count().reset_index()
crime_df = pd.DataFrame(crime_grouped[['Parcel_Address','CRIME_INCIDENT']])
crime_df['CRIME_INCIDENT'] = 0
crime_df['ANY_CRIME'] = 0
crime_df.rename(columns = {'Parcel_Address': 'ADDRESS'}, inplace = True)

# merge 
df = pd.merge(df, crime_df, how = 'left', on = 'ADDRESS')
df.CRIME_INCIDENT.fillna(0, inplace = True)
df.ANY_CRIME.fillna(0, inplace = True)

#####################################
# wrangle utility disconnections file
#####################################

df_Utility_Disconnects = pd.read_excel('utility_disconnects.xlsx')

df_Utility_Disconnects['Address'] = df_Utility_Disconnects['Address'].str.upper()
df_Utility_Disconnects['Address'] = df_Utility_Disconnects['Address'].str.replace(" ", "")

df_Utility_Disconnects = df_Utility_Disconnects[df_Utility_Disconnects['Year'] <= 2016]

# find count of disconnects and the last year of disconnection for each address
df_distinct = df_Utility_Disconnects.groupby('Address').count().reset_index()

#rename 'Year' by 'Count'
df_distinct.rename(columns = {'Year':'Count'}, inplace = True)

df_distinct_max_year = df_Utility_Disconnects.groupby(['Address'], sort=False)['Year'].max().reset_index()

# merge them in a single df
df_merge_utilty = df_distinct.join(df_distinct_max_year.set_index('Address'), on='Address')

current_year = 2016
df_merge_utilty['Since_Last_disconnect'] = current_year - df_merge_utilty['Year']

df_merge_utilty['Any_disconnects'] = 1

df_merge_utilty = df_merge_utilty[['Address','Count','Since_Last_disconnect','Any_disconnects']]
df_merge_utilty.rename(columns = {'Count':'Utility_discon_count','Address': 'ADDRESS'}, inplace = True)

# merge
df = pd.merge(df, df_merge_utilty, how = 'left', on = 'ADDRESS')

# fill uility disconnection columns with 0 in new df
cols = ['Utility_discon_count', 'Since_Last_disconnect', 'Any_disconnects']
df[cols] = df[cols].fillna(0)

#####################################
# wrangle rental registrations file
#####################################

df_rental = pd.read_csv('RentalRegistrations.csv')

df_rental['Address'] = df_rental['Address'].str.upper()
df_rental['Address'] = df_rental['Address'].str.replace(" ", "")

df_rental_new = df_rental[['Address', 'Permit Issue Date']]
                          
# drop duplicates
df_rental_new = drop_duplicate(df_rental_new)

# convert permit issue date to permit issue year
df_rental_new['Permit Issue Date'] = pd.to_datetime(df_rental_new['Permit Issue Date'], errors='coerce')
df_rental_new['Permit Issue Year'] = df_rental_new['Permit Issue Date'].dt.year

df_rental_new = df_rental_new[df_rental_new['Permit Issue Year'] <= 2016]


# take only address and year
df_rental_new = df_rental_new[['Address', 'Permit Issue Year']]

# remove rows with nan 'Permit Issue Year'
df_rental_new = df_rental_new[~df_rental_new['Permit Issue Year'].isna()]

# convert year to age
df_rental_new['Permit Issue Year'] = 2016-df_rental_new['Permit Issue Year']

# add the binary column
df_rental_new['Rent_Reg'] = 1
df_rental_new.rename(columns = {'Address': 'ADDRESS', 'Permit Issue Year': 'Permit Duration'}, inplace = True)
df_rental_new = df_rental_new[['ADDRESS', 'Permit Duration', 'Rent_Reg']]
# merge
df = pd.merge(df, df_rental_new, how = 'left', on = 'ADDRESS')

# fill rent reg columns with 0 in new df
cols = ['Rent_Reg']
df[cols] = df[cols].fillna(0)

#####################################
# wrangle SFIncidents file
#####################################

df_SFIncidents = pd.read_csv('SFIncidents.csv')

df_SFIncidents['Location'] = df_SFIncidents['Location'].str.upper()
df_SFIncidents['Location'] = df_SFIncidents['Location'].str.replace(" ", "")

df_SFIncidents['Date'] = pd.to_datetime(df_SFIncidents['Time']).dt.date 
df_SFIncidents['Year'] = pd.to_datetime(df_SFIncidents['Date']).dt.year
df_SFIncidents_new = df_SFIncidents[['Location', 'Code', 'Year', 'Category', 'Type']]
                          
df_SFIncidents_new = df_SFIncidents_new[~df_SFIncidents_new.index.duplicated()]

# filter by year for label
df_SFIncidents_new = df_SFIncidents_new.loc[(df_SFIncidents_new['Year'] >=2016) &  (df_SFIncidents_new['Year'] <=2020)]

df_fire = df_SFIncidents_new[df_SFIncidents_new['Code']==111]

df_fire = df_fire[['Location']]

# # filter by year for last incident before or on 2016

fire = df_SFIncidents_new.loc[(df_SFIncidents_new['Year'] <=2016)]
fire = fire.groupby(['Location'], sort=False)['Year'].max().reset_index()
fire['Year'] = 2016 - fire['Year']
fire = fire[['Location','Year']]
fire.rename(columns = {'Year':'Last_Incident'}, inplace = True)


# count total fires per address
df_fire_count = df_fire.groupby('Location').count().reset_index()
#rename 'Year' by 'Count'
df_fire_count.rename(columns = {'Year':'Count'}, inplace = True)

df_fire_count = df_fire_count.merge(fire[["Location", "Last_Incident"]])

df_fire_count = df_fire_count[['Location','Last_Incident']]
df_fire_count.rename(columns = {'Location': 'ADDRESS'}, inplace = True)

# add the binary column
df_fire_count['Incident'] = 1

# merge
df = pd.merge(df, df_fire_count, how = 'left', on = 'ADDRESS')

# fill Incident columns with 0 in new df
df['Incident'] = df['Incident'].fillna(0)

#####################################
# wrangle Inspections file
#####################################

df_inspection = pd.read_csv('Inspection_status.csv')

df_inspection['Address'] = df_inspection['Address'].str.upper()
df_inspection['Address'] = df_inspection['Address'].str.replace(" ", "")

ready_anytime = df_inspection.query('`Status Name` == "Ready Anytime"')
ready_anytime['Last Inspected'] = pd.to_datetime(ready_anytime['Scheduled Start Date']).dt.year
ready_anytime = ready_anytime.filter(['Address', 'Last Inspected'])

# remove data after 2016
ready_anytime = ready_anytime[ready_anytime['Last Inspected']<=2016]

ready_anytime = ready_anytime.groupby(['Address'], sort=False)['Last Inspected'].max().reset_index()
ready_anytime['ready_anytime'] = 1 

df_not_ready = df_inspection.query('`Status Name` != "Ready Anytime"')
df_not_ready['Last Inspected'] = pd.to_datetime(df_not_ready['Scheduled Start Date']).dt.year
df_not_ready = df_not_ready.filter(['Address', 'Last Inspected'])
df_not_ready = df_not_ready.groupby(['Address'], sort=False)['Last Inspected'].max().reset_index()
df_not_ready['ready_anytime'] = 0

# append dfs
inspections = ready_anytime.append(df_not_ready)
inspections.rename(columns = {'Address': 'ADDRESS'}, inplace = True)

# take just 'ready_anytime' column
inspections = inspections[['ADDRESS', 'ready_anytime']]
# merge
df = pd.merge(df, inspections, how='left', on='ADDRESS')

# fill inspections -- for now, just keeping it as nan as we plan to use XGBOOST
#df['Last Inspected'] = df['Last Inspected'].fillna(2011) 
df['ready_anytime'] = df['ready_anytime'].fillna(0) 

#####################################
# wrangle ParcelData file
#####################################

df_parcelData = pd.read_csv('ParcelData.csv')

df_parcelData['ADDRESS'] = df_parcelData['ADDRESS'].str.upper()
df_parcelData['ADDRESS'] = df_parcelData['ADDRESS'].str.replace(" ", "")

# convert activity

df_parcelData['ACTIVITY'] = df_parcelData.ACTIVITY.apply(cleanActivity)

df_parcelData_new = df_parcelData[['ADDRESS', 'SQFT', 'ACREAGE', 'FRONTFOOT', 'ACTIVITY', 'Shape_Length', 'Shape_Area', 'BuildVal', 'LandVal']]
                                  
df_parcelData_new = drop_duplicate(df_parcelData_new)

# remove rows with null/0 address value
skip = ["0", ""]
df_parcelData_new = df_parcelData_new[~df_parcelData_new['ADDRESS'].isin(skip)]

# merge
df = pd.merge(df, df_parcelData_new, how='left', on='ADDRESS')

# fill parcelData columns with 0 in main df
cols = df_parcelData_new.columns
df[cols] = df[cols].fillna(0)

#####################################
# wrangle foreclosure file
#####################################

df_foreclosure = pd.read_excel('foreclosures.xlsx')

df_foreclosure['ADDRESS'] = df_foreclosure['ADDRESS'].str.upper()
df_foreclosure['ADDRESS'] = df_foreclosure['ADDRESS'].str.replace(" ", "")

columns_to_drop = ['NAME', 'AUCTIONDATE','ASSESSEDVALUE']

df_foreclosure_new = drop_columns(df_foreclosure, columns_to_drop)

# drop duplicates
df_foreclosure_new = drop_duplicate(df_foreclosure_new)

# remove data after 2016
df_foreclosure_new = df_foreclosure_new[df_foreclosure_new['YEAR'] <= 2016]

# find the addresses that are foreclosed, no need to inspect them

foreclosed_addresses = df_foreclosure_new['ADDRESS']
df_foreclosure_new.rename(columns = {'YEAR':'Foreclosing_Year'}, inplace = True)

##########################
# Pre-process RentalRegistrations
##########################

df_rental = pd.read_csv('RentalRegistrations.csv')

df_rental['Address'] = df_rental['Address'].str.upper()
df_rental['Address'] = df_rental['Address'].str.replace(" ", "")

df_rental_new = df_rental[['Address', 'Permit Issue Date']]
                          
# drop duplicates
df_rental_new = drop_duplicate(df_rental_new)

# convert permit issue date to permit issue year
df_rental_new['Permit Issue Date'] = pd.to_datetime(df_rental_new['Permit Issue Date'], errors='coerce')
df_rental_new['Permit Issue Year'] = df_rental_new['Permit Issue Date'].dt.year

# take only address and year
df_rental_new = df_rental_new[['Address', 'Permit Issue Year']]

# filter data after 2016
df_rental_new = df_rental_new[df_rental_new['Permit Issue Year'] <= 2016]

##########################
# Merge Two
##########################

merge_df = df_foreclosure_new.join(df_rental_new.set_index('Address'), on='ADDRESS')

# filter the addresses where foreclosing date is less than permit issue date
merge_df = merge_df.dropna()
reopened_addresses = merge_df[merge_df['Permit Issue Year'] > merge_df['Foreclosing_Year']]

# filter all foreclosed addresses
addresses_to_skip = reopened_addresses['ADDRESS'].values.tolist()
actual_foreclosed_addresses = df_foreclosure_new[~ df_foreclosure_new.ADDRESS.isin(addresses_to_skip)]
foreclosed_addresses_list = actual_foreclosed_addresses['ADDRESS'].values.tolist()

# add foreclosed column
df.loc[~df['ADDRESS'].isin(foreclosed_addresses_list), 'foreclosed'] = 0
df.loc[df['ADDRESS'].isin(foreclosed_addresses_list), 'foreclosed'] = 1

# print df rows and columns
print(df.shape[0], df.shape[1])
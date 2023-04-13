#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:05:22 2023

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
    
def convert_address(df_address):
    df_address['ADDRESS'] = df_address['ADDRESS'].str.upper()
    df_address['ADDRESS'] = df_address['ADDRESS'].str.replace(" ", "")
    return df_address
    
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


df_SFIncidents = pd.read_csv('SFIncidents.csv')

df_SFIncidents['Location'] = df_SFIncidents['Location'].str.upper()
df_SFIncidents['Location'] = df_SFIncidents['Location'].str.replace(" ", "")

df_SFIncidents['Date'] = pd.to_datetime(df_SFIncidents['Time']).dt.date 
df_SFIncidents['Year'] = pd.to_datetime(df_SFIncidents['Date']).dt.year
df_SFIncidents_new = df_SFIncidents[['Location', 'Code', 'Year', 'Date']]

df_SFIncidents_new = df_SFIncidents_new[~df_SFIncidents_new.index.duplicated()]

df_fire = df_SFIncidents_new[df_SFIncidents_new['Code']==111]

df_fire = df_fire[['Location', 'Year', 'Date']]

# rename address
df_fire.rename(columns = {'Location': 'ADDRESS', 'Year': 'Fire_year', 'Date': 'Fire_date'}, inplace = True)

# separate fire addresses
fire_locations = df_fire['ADDRESS'].values.tolist()

# merge with address file
df_address = pd.read_csv('Address.csv')    

df_address = convert_address(df_address)

columns_to_drop = ['STATE', 'MUNICIPALITY','X', 'Y', 'OBJECTID', 'TAG', 'ADDRESSNUM_SUF', 'ALIAS',
       'ALIASPLUS', 'PATH', 'HYPERLINK', 'DISPLAY', 'HOUSENUM',
       'PR1', 'STNM1', 'TYP1', 'PD1','STREETVIEW', 'EFFECTIVEDATE', 'UNITTYPE', 'POINTTYPE', 'COUNTRY', 'GlobalID', 'PHOTOARCHIVELINK',
       'PHOTOACQUISITIONYEAR', 'RETIREDATE', 'ADDRCLASS']

df_address_new = drop_columns(df_address, columns_to_drop)

current_year = datetime.date.today().year
df_address_new['YEARBUILT'] = pd.to_numeric(df_address_new['YEARBUILT'], errors='coerce')

df_address_new['Age'] = current_year - df_address_new['YEARBUILT']
df_address_new = drop_columns(df_address_new, ['YEARBUILT'])

# drop duplicates
df_address_new = df_address_new.drop_duplicates(subset=['ADDRESS'], keep='last')

# merge
address_fire = pd.merge(df_fire, df_address_new, how='left', on='ADDRESS')

###########################
####### merge with codecases
###########################

df_codeCases = pd.read_csv('CodeCases_2015-2021.csv')
# rename address
df_codeCases.rename(columns = {'Address': 'ADDRESS', 'Year': 'Violation_year'}, inplace = True)
df_codeCases = convert_address(df_codeCases)

df_codeCases = df_codeCases[['ADDRESS', 'Violation_year']]

address_fire_codecases = pd.merge(address_fire, df_codeCases, on='ADDRESS', how='left')

address_fire_codecases['valid'] = address_fire_codecases['Violation_year'] <= address_fire_codecases['Fire_year']
address_fire_codecases['violations_count'] = address_fire_codecases.groupby(['ADDRESS', 'Fire_date'])['valid'].transform('sum')
address_fire_codecases = address_fire_codecases.drop(['Violation_year', 'valid'], axis=1)
address_fire_codecases = address_fire_codecases.drop_duplicates(subset=['ADDRESS', 'Fire_date'])

# upto this point, we have violation count for each (Address, Fire_date) pair

##############################
####### merge with crime data
##############################

df_offenses = pd.read_excel('Copy of Property Offenses 2017-2021.xlsx')
#df_offenses['Date'] = pd.to_datetime(df_offenses['Date']).dt.year
# rename address
df_offenses.rename(columns = {'Parcel_Address': 'ADDRESS', 'Date': 'Crime_date'}, inplace = True)
df_offenses = convert_address(df_offenses)

df_offenses = df_offenses[['ADDRESS', 'Crime_date']]
merged_offenses = pd.merge(address_fire_codecases, df_offenses, on='ADDRESS', how='left')

merged_offenses['valid'] = merged_offenses['Crime_date'] <= merged_offenses['Fire_date']
merged_offenses['crime_count'] = merged_offenses.groupby(['ADDRESS', 'Fire_date'])['valid'].transform('sum')
merged_offenses = merged_offenses.drop(['Crime_date', 'valid'], axis=1)
merged_offenses = merged_offenses.drop_duplicates(subset=['ADDRESS', 'Fire_date'])

# upto this point, we have crime count for each (Address, Fire_date) pair

##############################
####### merge with utility disconnection data
##############################

df_Utility_Disconnects = pd.read_excel('utility_disconnects.xlsx')

# rename address
df_Utility_Disconnects.rename(columns = {'Address': 'ADDRESS', 'Year': 'Discon_year'}, inplace = True)
df_Utility_Disconnects = convert_address(df_Utility_Disconnects)

df_Utility_Disconnects = df_Utility_Disconnects[['ADDRESS', 'Discon_year']]
merged_utilities = pd.merge(merged_offenses, df_Utility_Disconnects, on='ADDRESS', how='left')

merged_utilities['valid'] = merged_utilities['Discon_year'] <= merged_utilities['Fire_year']
merged_utilities['discon_count'] = merged_utilities.groupby(['ADDRESS', 'Fire_date'])['valid'].transform('sum')
merged_utilities = merged_utilities.drop(['Discon_year', 'valid'], axis=1)
merged_utilities = merged_utilities.drop_duplicates(subset=['ADDRESS', 'Fire_date'])


#############################################
####### merge with rental registrations data
#############################################

df_rental = pd.read_csv('RentalRegistrations.csv')

# rename address
df_rental.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
df_rental = convert_address(df_rental)

df_rental = df_rental[['ADDRESS', 'Permit Issue Date']]
merged_rental = pd.merge(merged_utilities, df_rental, on='ADDRESS', how='left')

# assume that the "Permit Issue Date" column is in the format "YYYY-MM-DD"
merged_rental['Permit Issue Date'] = pd.to_datetime(merged_rental['Permit Issue Date']).dt.date

# create a binary column
merged_rental['Rental_permit'] = merged_rental['Permit Issue Date'] <= merged_rental['Fire_date']

# convert the boolean column to an integer column
merged_rental['Rental_permit'] = merged_rental['Rental_permit'].astype(int)
merged_rental = drop_columns(merged_rental, ['Permit Issue Date'])

#############################################
####### merge with inspections data
#############################################

df_inspection = pd.read_csv('Inspection_status.csv')

# rename address
df_inspection.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
df_inspection = convert_address(df_inspection)

df_inspection = df_inspection[['ADDRESS', 'End Date', 'Scheduled Start Date']]
merged_inspections = pd.merge(merged_rental, df_inspection, on='ADDRESS', how='left')

# convert to datetime format
merged_inspections['End Date'] = pd.to_datetime(merged_inspections['End Date']).dt.date
merged_inspections['Scheduled Start Date'] = pd.to_datetime(merged_inspections['Scheduled Start Date']).dt.date

# add binary column for invalid or expired permit
merged_inspections['Permit_Invalid'] = merged_inspections['End Date'] <= merged_inspections['Fire_date']

# convert the boolean column to an integer column
merged_inspections['Permit_Invalid'] = merged_inspections['Permit_Invalid'].astype(int)

merged_inspections['valid'] = merged_inspections['Scheduled Start Date'] <= merged_inspections['Fire_date']
merged_inspections['inspection_count'] = merged_inspections.groupby(['ADDRESS', 'Fire_date'])['valid'].transform('sum')
merged_inspections = merged_inspections.drop(['Scheduled Start Date', 'valid'], axis=1)
merged_inspections = merged_inspections.drop_duplicates(subset=['ADDRESS', 'Fire_date'])

merged_inspections = drop_columns(merged_inspections, ['End Date'])


#############################################
####### merge with inspections data
#############################################

df_parcelData = pd.read_csv('ParcelData.csv')
df_parcelData = convert_address(df_parcelData)

# convert activity

df_parcelData['ACTIVITY'] = df_parcelData.ACTIVITY.apply(cleanActivity)

df_parcelData_new = df_parcelData[['ADDRESS', 'SQFT', 'ACREAGE', 'FRONTFOOT', 'ACTIVITY', 'Shape_Length', 'Shape_Area', 'BuildVal', 'LandVal']]
                                  
df_parcelData_new = drop_duplicate(df_parcelData_new)

# remove rows with null/0 address value
skip = ["0", ""]
df_parcelData_new = df_parcelData_new[~df_parcelData_new['ADDRESS'].isin(skip)]

# remove entries with duplicate address
df_parcelData_new = df_parcelData_new.drop_duplicates(subset=['ADDRESS'], keep='last')

# merge
merged_parcelData = pd.merge(merged_inspections, df_parcelData_new, how='left', on='ADDRESS')

#############################################
####### merge with Foreclosure data
#############################################

df_foreclosure = pd.read_excel('foreclosures.xlsx')
df_foreclosure = convert_address(df_foreclosure)
columns_to_drop = ['NAME', 'AUCTIONDATE','ASSESSEDVALUE']
df_foreclosure_new = drop_columns(df_foreclosure, columns_to_drop)

merged_foreclosure = pd.merge(merged_parcelData, df_foreclosure_new, on='ADDRESS', how='left')

merged_foreclosure['valid'] = merged_foreclosure['YEAR'] <= merged_foreclosure['Fire_year']
merged_foreclosure['foreclosing_count'] = merged_foreclosure.groupby(['ADDRESS', 'Fire_date'])['valid'].transform('sum')
merged_foreclosure = merged_foreclosure.drop(['YEAR', 'valid'], axis=1)
merged_foreclosure = merged_foreclosure.drop_duplicates(subset=['ADDRESS', 'Fire_date'])

# label of fire-instance is 1
fire_data = merged_foreclosure.copy()
fire_data['incident'] = 1
fire_data = drop_columns(fire_data, ['Rental_permit', 'Permit_Invalid', 'Fire_year', 'Fire_date'])
create_profile(fire_data, 'Fire Data Profile')

#############################################
####### prepare data for non-fire instances
#############################################

all_locations = df_address_new['ADDRESS'].values.tolist
#no_fire_locations = set(all_locations) | set(fire_locations)

df_address_safe = df_address_new[~ df_address_new.ADDRESS.isin(fire_locations)]

###########################
####### merge with codecases
###########################

df_codeCases_safe = pd.read_csv('CodeCases_2015-2021.csv')
# rename address
df_codeCases_safe.rename(columns = {'Address': 'ADDRESS', 'Year': 'Violation_year'}, inplace = True)
df_codeCases_safe = convert_address(df_codeCases_safe)

df_codeCases_safe = df_codeCases_safe[['ADDRESS', 'Violation_year']]

address_codecases_safe = pd.merge(df_address_safe, df_codeCases_safe, on='ADDRESS', how='left')
address_codecases_safe['violations_count'] = address_codecases_safe.groupby('ADDRESS')['Violation_year'].transform('count')

address_codecases_safe = address_codecases_safe.drop_duplicates(subset=['ADDRESS'])

# upto this point, we have violation count for each (Address, Fire_date) pair

##############################
####### merge with crime data
##############################

df_offenses_safe = pd.read_excel('Copy of Property Offenses 2017-2021.xlsx')

# rename address
df_offenses_safe.rename(columns = {'Parcel_Address': 'ADDRESS', 'Date': 'Crime_date'}, inplace = True)
df_offenses_safe = convert_address(df_offenses_safe)

df_offenses_safe = df_offenses_safe[['ADDRESS', 'Crime_date']]
df_offenses_safe['Year'] = pd.to_datetime(df_offenses_safe['Crime_date']).dt.year
merged_offenses_safe = pd.merge(address_codecases_safe, df_offenses_safe, on='ADDRESS', how='left')

merged_offenses_safe['crime_count'] = merged_offenses_safe.groupby('ADDRESS')['Crime_date'].transform('count')

merged_offenses_safe = merged_offenses_safe.drop(['Crime_date', 'Year'], axis=1)
merged_offenses_safe = merged_offenses_safe.drop_duplicates(subset=['ADDRESS'])

# upto this point, we have crime count for each (Address, Fire_date) pair

##############################
####### merge with utility disconnection data
##############################

df_Utility_safe = pd.read_excel('utility_disconnects.xlsx')

# rename address
df_Utility_safe.rename(columns = {'Address': 'ADDRESS', 'Year': 'Discon_year'}, inplace = True)
df_Utility_safe = convert_address(df_Utility_safe)

df_Utility_safe = df_Utility_safe[['ADDRESS', 'Discon_year']]
merged_utilities_safe = pd.merge(merged_offenses_safe, df_Utility_safe, on='ADDRESS', how='left')

merged_utilities_safe['discon_count'] = merged_utilities_safe.groupby('ADDRESS')['Discon_year'].transform('count')

merged_utilities_safe = merged_utilities_safe.drop(['Discon_year'], axis=1)
merged_utilities_safe = merged_utilities_safe.drop_duplicates(subset=['ADDRESS'])


#############################################
####### merge with rental registrations data
#############################################

df_rental_safe = pd.read_csv('RentalRegistrations.csv')

# rename address
df_rental_safe.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
df_rental_safe = convert_address(df_rental_safe)

df_rental_safe = df_rental_safe[['ADDRESS', 'Permit Issue Date']]
df_rental_safe['Rental_permit'] = 1

merged_rental_safe = pd.merge(merged_utilities_safe, df_rental_safe, on='ADDRESS', how='left')
merged_rental_safe = drop_columns(merged_rental_safe, ['Permit Issue Date'])

#############################################
####### merge with inspections data
#############################################

df_inspection_safe = pd.read_csv('Inspection_status.csv')

# rename address
df_inspection_safe.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
df_inspection_safe = convert_address(df_inspection_safe)

df_inspection_safe = df_inspection_safe[['ADDRESS', 'End Date', 'Scheduled Start Date']]
df_inspection_safe['Permit_Invalid'] = 0
df_inspection_safe['Year'] = pd.to_datetime(df_inspection_safe['Scheduled Start Date']).dt.year
merged_inspections_safe = pd.merge(merged_rental_safe, df_inspection_safe, on='ADDRESS', how='left')

# convert to datetime format
merged_inspections_safe['End Date'] = pd.to_datetime(merged_inspections_safe['End Date']).dt.date
merged_inspections_safe['Scheduled Start Date'] = pd.to_datetime(merged_inspections_safe['Scheduled Start Date']).dt.date


merged_inspections_safe['inspection_count'] = merged_inspections_safe.groupby('ADDRESS')['Scheduled Start Date'].transform('count')

merged_inspections_safe = merged_inspections_safe.drop(['Scheduled Start Date', 'Year'], axis=1)
merged_inspections_safe = merged_inspections_safe.drop_duplicates(subset=['ADDRESS'])

merged_inspections_safe = drop_columns(merged_inspections_safe, ['End Date'])


#############################################
####### merge with parcel data
#############################################

df_parcelData_safe = pd.read_csv('ParcelData.csv')
df_parcelData_safe = convert_address(df_parcelData_safe)

# convert activity

df_parcelData_safe['ACTIVITY'] = df_parcelData_safe.ACTIVITY.apply(cleanActivity)

df_parcelData_safe = df_parcelData_safe[['ADDRESS', 'SQFT', 'ACREAGE', 'FRONTFOOT', 'ACTIVITY', 'Shape_Length', 'Shape_Area', 'BuildVal', 'LandVal']]
                                  
df_parcelData_safe = drop_duplicate(df_parcelData_safe)

# remove rows with null/0 address value
skip = ["0", ""]
df_parcelData_safe = df_parcelData_safe[~df_parcelData_safe['ADDRESS'].isin(skip)]

# remove entries with duplicate address
df_parcelData_safe = df_parcelData_safe.drop_duplicates(subset=['ADDRESS'], keep='last')

# merge
merged_parcelData_safe = pd.merge(merged_inspections_safe, df_parcelData_safe, how='left', on='ADDRESS')

#############################################
####### merge with Foreclosure data
#############################################

df_foreclosure_safe = pd.read_excel('foreclosures.xlsx')
df_foreclosure_safe = convert_address(df_foreclosure_safe)
columns_to_drop = ['NAME', 'AUCTIONDATE','ASSESSEDVALUE']
df_foreclosure_safe = drop_columns(df_foreclosure_safe, columns_to_drop)

merged_foreclosure_safe = pd.merge(merged_parcelData_safe, df_foreclosure_safe, on='ADDRESS', how='left')

merged_foreclosure_safe['foreclosing_count'] = merged_foreclosure_safe.groupby('ADDRESS')['YEAR'].transform('count')

merged_foreclosure_safe = merged_foreclosure_safe.drop(['YEAR'], axis=1)
merged_foreclosure_safe = merged_foreclosure_safe.drop_duplicates(subset=['ADDRESS'])

# label of non-fire-instance is 0
non_fire_data = merged_foreclosure_safe.copy()
non_fire_data['incident'] = 0
non_fire_data = drop_columns(non_fire_data, ['Rental_permit', 'Permit_Invalid', 'Violation_year'])

create_profile(non_fire_data, 'Non Fire Data Profile')

# merge fire and non-fire data

merged_data = pd.concat([fire_data, non_fire_data], ignore_index=True)
merged_data.to_csv('Fire_nonFire_Data.csv')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:57:58 2023

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

def wrangle_address():
    df_address = pd.read_csv('Address.csv')    

    df_address = convert_address(df_address)

    columns_to_drop = ['STATE', 'MUNICIPALITY','X', 'Y', 'OBJECTID', 'TAG', 'ADDRESSNUM_SUF', 'ALIAS',
           'ALIASPLUS', 'PATH', 'HYPERLINK', 'DISPLAY', 'HOUSENUM',
           'PR1', 'STNM1', 'TYP1', 'PD1','STREETVIEW', 'EFFECTIVEDATE', 'UNITTYPE', 'POINTTYPE', 'COUNTRY', 'GlobalID', 'PHOTOARCHIVELINK',
           'PHOTOACQUISITIONYEAR', 'RETIREDATE', 'ADDRCLASS']

    df_address_new = drop_columns(df_address, columns_to_drop)

    df_address_new['YEARBUILT'] = pd.to_numeric(df_address_new['YEARBUILT'], errors='coerce')
    
    df_address_new['Age'] = 2023 - df_address_new['YEARBUILT']
    df_address_new = drop_columns(df_address_new, ['YEARBUILT'])

    # remove duplicates
    df_address_new = df_address_new.drop_duplicates(subset=['ADDRESS'], keep='last')
    return df_address_new

def wrangle_merge_codecases(df):
    df_codeCases = pd.read_csv('CodeCases_2015-2021.csv')

    # rename address
    df_codeCases.rename(columns = {'Address': 'ADDRESS'}, inplace = True)

    df_codeCases = convert_address(df_codeCases)

    # remove data after year
    df_codeCases = df_codeCases.loc[(df_codeCases['Year'] >=2011) &  (df_codeCases['Year'] <=2019)]

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

    current_columns = case_types.columns

    cols_to_create = [col for col in cols if col not in current_columns]

    case_types[cols_to_create] = 0
    

    code_cases = pd.concat([df_codeCases, case_types], axis=1)
    code_cases.drop(columns = ['Year'], inplace = True)
    code_cases_grouped = code_cases.groupby(['ADDRESS']).sum().reset_index()
    code_cases_df = pd.DataFrame(code_cases_grouped)

    code_cases_df['TOTAL_VIOLATIONS'] = code_cases_df.sum(axis=1)
    code_cases_df['ANY_VIOLATIONS'] = 1

    # remove duplicates
    code_cases_df = code_cases_df.drop_duplicates(subset=['ADDRESS'], keep='last')
        
    # merge address and codecases
    df = pd.merge(df, code_cases_df, how = 'left', on = 'ADDRESS')
    df.ANY_VIOLATIONS.fillna(0, inplace = True)
    df['TOTAL_VIOLATIONS'] = df['TOTAL_VIOLATIONS'].fillna(0)

    # impute codecase types columns with zero in the main df
    df[cols] = df[cols].fillna(0)
    return df

def wrangle_merge_crimeData(df):
    df_offenses = pd.read_excel('Copy of Property Offenses 2017-2021.xlsx')

    # rename address
    df_offenses.rename(columns = {'Parcel_Address': 'ADDRESS'}, inplace = True)

    df_offenses = convert_address(df_offenses)

    df_offenses_new = df_offenses[['ADDRESS', 'Offense', 'Date']]

    # drop duplicates

    df_offenses_new = drop_duplicate(df_offenses_new)

    # filter data in between
    df_offenses_new['Year'] = pd.to_datetime(df_offenses_new['Date']).dt.year 

    df_offenses_new = df_offenses_new.loc[(df_offenses_new['Year'] >= 2011) & df_offenses_new['Year'] <= 2019]


    df_offenses_new.rename(columns = {'Offense' : 'CRIME_INCIDENTS'}, inplace = True)
    crime_grouped = df_offenses_new.groupby('ADDRESS').count().reset_index()
    crime_df = pd.DataFrame(crime_grouped[['ADDRESS','CRIME_INCIDENTS']])

    crime_df['ANY_CRIME'] = 1
    
    # remove duplicates
    crime_df = crime_df.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge 
    df = pd.merge(df, crime_df, how = 'left', on = 'ADDRESS')
    df.CRIME_INCIDENTS.fillna(0, inplace = True)
    df.ANY_CRIME.fillna(0, inplace = True)
    return df

def wrangle_merge_utility(df):
    df_Utility_Disconnects = pd.read_excel('utility_disconnects.xlsx')

    # rename address
    df_Utility_Disconnects.rename(columns = {'Address': 'ADDRESS'}, inplace = True)

    df_Utility_Disconnects = convert_address(df_Utility_Disconnects)

    # just consider data in between
    df_Utility_Disconnects = df_Utility_Disconnects.loc[(df_Utility_Disconnects['Year'] >= 2011) & (df_Utility_Disconnects['Year'] <= 2019)]

    # find count of disconnects and the last year of disconnection for each address
    df_distinct = df_Utility_Disconnects.groupby('ADDRESS').count().reset_index()

    #rename 'Year' by 'Count'
    df_distinct.rename(columns = {'Year':'Count'}, inplace = True)

    df_distinct_max_year = df_Utility_Disconnects.groupby(['ADDRESS'], sort=False)['Year'].max().reset_index()

    # merge them in a single df
    df_merge_utilty = df_distinct.join(df_distinct_max_year.set_index('ADDRESS'), on='ADDRESS')

    current_year = 2023
    df_merge_utilty['Since_Last_disconnect'] = current_year - df_merge_utilty['Year']

    df_merge_utilty['Any_disconnects'] = 1

    df_merge_utilty = df_merge_utilty[['ADDRESS','Count','Since_Last_disconnect','Any_disconnects']]
    df_merge_utilty.rename(columns = {'Count':'Utility_discon_count'}, inplace = True)
    
    # remove duplicates
    df_merge_utilty = df_merge_utilty.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge
    df = pd.merge(df, df_merge_utilty, how = 'left', on = 'ADDRESS')

    # fill uility disconnection columns with 0 in new df
    cols = ['Utility_discon_count', 'Since_Last_disconnect', 'Any_disconnects']
    df[cols] = df[cols].fillna(0)
    
    return df

def wrangle_merge_rental(df):
    df_rental = pd.read_csv('RentalRegistrations.csv')

    # rename address
    df_rental.rename(columns = {'Address': 'ADDRESS'}, inplace = True)

    df_rental = convert_address(df_rental)

    df_rental_new = df_rental[['ADDRESS', 'Permit Issue Date']]
                              
    # drop duplicates
    df_rental_new = drop_duplicate(df_rental_new)

    # convert permit issue date to permit issue year
    df_rental_new['Permit Issue Date'] = pd.to_datetime(df_rental_new['Permit Issue Date'], errors='coerce')
    df_rental_new['Permit Issue Year'] = df_rental_new['Permit Issue Date'].dt.year

    df_rental_new = df_rental_new.loc[(df_rental_new['Permit Issue Year'] >= 2011) & (df_rental_new['Permit Issue Year'] <= 2019)]


    # take only address and year
    df_rental_new = df_rental_new[['ADDRESS', 'Permit Issue Year']]

    # remove rows with nan 'Permit Issue Year'
    df_rental_new = df_rental_new[~df_rental_new['Permit Issue Year'].isna()]

    # convert year to age
    #df_rental_new['Permit Issue Year'] = 2023-df_rental_new['Permit Issue Year']

    # add the binary column
    df_rental_new['Rent_Reg'] = 1
    df_rental_new.rename(columns = {'Permit Issue Year': 'Permit Duration'}, inplace = True)
    df_rental_new = df_rental_new[['ADDRESS', 'Rent_Reg']]
    
    # remove duplicates
    df_rental_new = df_rental_new.drop_duplicates(subset=['ADDRESS'], keep='last')
    
    # merge
    df = pd.merge(df, df_rental_new, how = 'left', on = 'ADDRESS')

    # fill rent reg columns with 0 in actual df
    cols = ['Rent_Reg']
    df[cols] = df[cols].fillna(0)
    
    return df

def wrangle_merge_SFIncidents(df):
    df_SFIncidents = pd.read_csv('SFIncidents.csv')

    # rename address
    df_SFIncidents.rename(columns = {'Location': 'ADDRESS'}, inplace = True)

    df_SFIncidents = convert_address(df_SFIncidents)

    # convert datetime to year
    df_SFIncidents['Date'] = pd.to_datetime(df_SFIncidents['Time']).dt.date 
    df_SFIncidents['Year'] = pd.to_datetime(df_SFIncidents['Date']).dt.year
    df_SFIncidents_new = df_SFIncidents[['ADDRESS', 'Code', 'Year', 'Category', 'Type']]
                              
    df_SFIncidents_new = df_SFIncidents_new[~df_SFIncidents_new.index.duplicated()]

    # filter by time-interval(this is variable) for label
    df_SFIncidents_new = df_SFIncidents_new.loc[(df_SFIncidents_new['Year'] >= 2020) & (df_SFIncidents_new['Year'] <= 2021)]

    # filtering just the fire incidents using the code
    df_fire = df_SFIncidents_new[df_SFIncidents_new['Code']==111]

    df_fire = df_fire[['ADDRESS', 'Year']]

    # # unnecessary block
    '''
    fire = df_SFIncidents_new.loc[(df_SFIncidents_new['Year'] <=2011)]
    fire = fire.groupby(['ADDRESS'], sort=False)['Year'].max().reset_index()
    fire['Year'] = year - fire['Year']
    fire = fire[['ADDRESS','Year']]
    fire.rename(columns = {'Year':'Last_Incident'}, inplace = True)

    # count total fires per address
    df_fire_count = df_fire.groupby('ADDRESS').count().reset_index()
    #rename 'Year' by 'Count'
    df_fire_count.rename(columns = {'Year':'Count'}, inplace = True)

    df_fire_count = df_fire_count.merge(fire[["ADDRESS", "Last_Incident"]])

    df_fire_count = df_fire_count[['ADDRESS','Last_Incident']]
    #df_fire_count.rename(columns = {'Location': 'ADDRESS'}, inplace = True)

    # add the binary column
    df_fire_count['Incident'] = 1
    '''
    # add the binary column
    df_fire['Incident'] = 1
    
    # remove duplicates
    df_fire = df_fire.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge
    df = pd.merge(df, df_fire, how = 'left', on = 'ADDRESS')

    # fill Incident columns with 0 in actual df
    df['Incident'] = df['Incident'].fillna(0)
    
    return df

def wrangle_merge_inspections(df):
    
    df_inspection = pd.read_csv('Inspection_status.csv')
    # rename address
    df_inspection.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
    df_inspection = convert_address(df_inspection)
    
    df_inspection['Year'] = pd.to_datetime(df_inspection['Scheduled Start Date']).dt.year
    df_inspection = df_inspection.filter(['ADDRESS', 'Year', 'Status Name'])
    
    df_inspection = df_inspection.loc[(df_inspection['Year']>= 2011) & (df_inspection['Year']<= 2019)]
        
    ready_anytime = df_inspection.query('`Status Name` == "Ready Anytime"')
    ready_anytime = ready_anytime.filter(['ADDRESS', 'Year'])

    ready_anytime = ready_anytime.groupby(['ADDRESS']).count().reset_index()
    ready_anytime['ready_anytime'] = 1 

    df_not_ready = df_inspection.query('`Status Name` != "Ready Anytime"')
    df_not_ready = df_not_ready.filter(['ADDRESS', 'Year'])
    df_not_ready = df_not_ready.groupby(['ADDRESS']).count().reset_index()
    df_not_ready['ready_anytime'] = 0

    # append dfs
    inspections = ready_anytime.append(df_not_ready)
    inspections.rename(columns = {'Year': 'Inspections_Count'}, inplace = True)

    # take just 'ready_anytime' column
    inspections = inspections[['ADDRESS', 'ready_anytime', 'Inspections_Count']]
    
    # remove duplicates
    inspections = inspections.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge
    df = pd.merge(df, inspections, how='left', on='ADDRESS')
    # fill inspections -- for now, just keeping it as nan as we plan to use XGBOOST
    #df['Last Inspected'] = df['Last Inspected'].fillna(2011) 
    df['ready_anytime'] = df['ready_anytime'].fillna(0) 
    df['Inspections_Count'] = df['Inspections_Count'].fillna(0) 
    return df

def wrangle_merge_parcelData(df):
    df_parcelData = pd.read_csv('ParcelData.csv')


    df_parcelData = convert_address(df_parcelData)

    # convert activity

    df_parcelData['ACTIVITY'] = df_parcelData.ACTIVITY.apply(cleanActivity)

    df_parcelData_new = df_parcelData[['ADDRESS', 'SQFT', 'ACREAGE', 'FRONTFOOT', 'ACTIVITY', 'Shape_Length', 'Shape_Area', 'BuildVal', 'LandVal']]
                                      
    df_parcelData_new = df_parcelData_new.drop_duplicates(subset=['ADDRESS'], keep='last')

    # remove rows with null/0 address value
    skip = ["0", ""]
    df_parcelData_new = df_parcelData_new[~df_parcelData_new['ADDRESS'].isin(skip)]
    
    # remove duplicates
    df_parcelData_new = df_parcelData_new.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge
    df = pd.merge(df, df_parcelData_new, how='left', on='ADDRESS')

    # fill parcelData columns with 0 in main df
    cols = df_parcelData_new.columns
    df[cols] = df[cols].fillna(0)
    
    return df

def foreclosures_rental_merge(df):
    df_foreclosure = pd.read_excel('foreclosures.xlsx')

    df_foreclosure = convert_address(df_foreclosure)

    columns_to_drop = ['NAME', 'AUCTIONDATE','ASSESSEDVALUE']

    df_foreclosure_new = drop_columns(df_foreclosure, columns_to_drop)

    # drop duplicates
    df_foreclosure_new = drop_duplicate(df_foreclosure_new)

    # filter data after fict year
    df_foreclosure_new = df_foreclosure_new.loc[(df_foreclosure_new['YEAR'] >= 2011) & (df_foreclosure_new['YEAR'] <= 2019)]

    # find the addresses that are foreclosed, no need to inspect them

    #foreclosed_addresses = df_foreclosure_new['ADDRESS']
    df_foreclosure_new.rename(columns = {'YEAR':'Foreclosing_Year'}, inplace = True)

    ##########################
    # Pre-process RentalRegistrations
    ##########################

    df_rental = pd.read_csv('RentalRegistrations.csv')

    # rename address
    df_rental.rename(columns = {'Address': 'ADDRESS'}, inplace = True)

    df_rental = convert_address(df_rental)


    df_rental_new = df_rental[['ADDRESS', 'Permit Issue Date']]
                              
    # drop duplicates
    df_rental_new = drop_duplicate(df_rental_new)

    # convert permit issue date to permit issue year
    df_rental_new['Permit Issue Date'] = pd.to_datetime(df_rental_new['Permit Issue Date'], errors='coerce')
    df_rental_new['Permit Issue Year'] = df_rental_new['Permit Issue Date'].dt.year

    # take only address and year
    df_rental_new = df_rental_new[['ADDRESS', 'Permit Issue Year']]

    # filter data after fict year
    df_rental_new = df_rental_new.loc[(df_rental_new['Permit Issue Year'] >= 2011) & (df_rental_new['Permit Issue Year'] <= 2019)]

    ##########################
    # Merge Two
    ##########################

    merge_df = df_foreclosure_new.join(df_rental_new.set_index('ADDRESS'), on='ADDRESS')

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
    
    return df

######################
# wrangle address file
######################

df_address_new = wrangle_address()
df = df_address_new.copy()
print('initial', df.shape[0])

######################
# wrangle and merge codecases file
######################

df = wrangle_merge_codecases(df)
print('codecases', df.shape[0])

######################
# wrangle and merge property offenses/crimedata file
######################

df = wrangle_merge_crimeData(df)
print('Crime', df.shape[0])


#####################################
# wrangle utility disconnections file
#####################################

df = wrangle_merge_utility(df)
print('Utility', df.shape[0])

#####################################
# wrangle rental registrations file
#####################################

df = wrangle_merge_rental(df)
print('Rental', df.shape[0])

#####################################
# wrangle SFIncidents file
#####################################

df = wrangle_merge_SFIncidents(df)
print('SFIncidents', df.shape[0])

#####################################
# wrangle Inspections file
#####################################

df = wrangle_merge_inspections(df) 
print('Inspections', df.shape[0])

#####################################
# wrangle ParcelData file
#####################################

df = wrangle_merge_parcelData(df)
print('Parceldata', df.shape[0])

#####################################
# wrangle foreclosure, rental file and mark the foreclosed addresses using rental registration information
#####################################

df = foreclosures_rental_merge(df)
# print df rows and columns
print('foreclosure',df.shape[0])
df1 = df.copy()

#df_merge = pd.concat([df1], ignore_index=True)
#create_profile(df, 'Simpler Dataset Profile')

#df.to_csv('Dataset_simpler.csv')

df_legit = df[~df['ADDRESS'].isna()]
create_profile(df_legit, 'Simpler Legit Dataset Profile')
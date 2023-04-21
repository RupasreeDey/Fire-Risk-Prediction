#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:46:30 2023

@author: rupasree
"""
#######################################
# not calculating attribute values based on fire incident year or fire incident
#######################################

import geohash
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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

def convert_location(df):
    # Generate geohashes for each latitude and longitude pair
    df['geohash'] = df.apply(lambda row: geohash.encode(row['LAT'], row['LON'], precision=5), axis=1)

    # Fit k-means clustering model with latitude and longitude features
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(df[['LAT', 'LON']])

    # Add cluster labels to DataFrame
    df['cluster'] = kmeans.labels_
    return df

def wrangle_address():
    df_address = pd.read_csv('Address.csv')    

    df_address = convert_address(df_address)

    columns_to_drop = ['STATE', 'MUNICIPALITY','X', 'Y', 'OBJECTID', 'TAG', 'ADDRESSNUM_SUF', 'ALIAS',
           'ALIASPLUS', 'PATH', 'HYPERLINK', 'DISPLAY', 'HOUSENUM',
           'PR1', 'STNM1', 'TYP1', 'PD1','STREETVIEW', 'EFFECTIVEDATE', 'UNITTYPE', 'POINTTYPE', 'COUNTRY', 'GlobalID', 'PHOTOARCHIVELINK',
           'PHOTOACQUISITIONYEAR', 'RETIREDATE', 'ADDRCLASS']

    df_address_new = drop_columns(df_address, columns_to_drop)

    df_address_new['YEARBUILT'] = pd.to_numeric(df_address_new['YEARBUILT'], errors='coerce')
    
    df_address_new['Age'] = datetime.date.today().year - df_address_new['YEARBUILT']
    df_address_new = drop_columns(df_address_new, ['YEARBUILT'])

    # drop duplicates
    df_address_new = df_address_new.drop_duplicates(subset=['ADDRESS'], keep='last')
    return df_address_new

def wrangle_merge_codecases(df):
     
    df_codeCases = pd.read_csv('CodeCases_2015-2021.csv')
    # rename address
    df_codeCases.rename(columns = {'Address': 'ADDRESS', 'Year': 'Violation_year'}, inplace = True)
    df_codeCases = convert_address(df_codeCases)

    df_codeCases = df_codeCases[['ADDRESS', 'Violation_year']]

    address_codecases = pd.merge(df, df_codeCases, on='ADDRESS', how='left')
    address_codecases['violations_count'] = address_codecases.groupby('ADDRESS')['Violation_year'].transform('count')
    
    address_codecases = drop_columns(address_codecases, ['Violation_year'])
    address_codecases = address_codecases.drop_duplicates(subset=['ADDRESS'])
    
    return address_codecases

def wrangle_merge_crimeData(df):
    
    df_offenses = pd.read_excel('Copy of Property Offenses 2017-2021.xlsx')

    # rename address
    df_offenses.rename(columns = {'Parcel_Address': 'ADDRESS', 'Date': 'Crime_date'}, inplace = True)
    df_offenses = convert_address(df_offenses)

    df_offenses = df_offenses[['ADDRESS', 'Crime_date']]
    df_offenses['Year'] = pd.to_datetime(df_offenses['Crime_date']).dt.year
    merged_offenses = pd.merge(df, df_offenses, on='ADDRESS', how='left')

    merged_offenses['crime_count'] = merged_offenses.groupby('ADDRESS')['Crime_date'].transform('count')

    merged_offenses = merged_offenses.drop(['Crime_date', 'Year'], axis=1)
    merged_offenses = merged_offenses.drop_duplicates(subset=['ADDRESS'])

    return merged_offenses

def wrangle_merge_utility(df):
    df_Utility = pd.read_excel('utility_disconnects.xlsx')

    # rename address
    df_Utility.rename(columns = {'Address': 'ADDRESS', 'Year': 'Discon_year'}, inplace = True)
    df_Utility = convert_address(df_Utility)

    df_Utility = df_Utility[['ADDRESS', 'Discon_year']]
    merged_utilities = pd.merge(df, df_Utility, on='ADDRESS', how='left')

    merged_utilities['discon_count'] = merged_utilities.groupby('ADDRESS')['Discon_year'].transform('count')

    merged_utilities = merged_utilities.drop(['Discon_year'], axis=1)
    merged_utilities = merged_utilities.drop_duplicates(subset=['ADDRESS'])

    return merged_utilities

def wrangle_merge_rental(df):
    df_rental = pd.read_csv('RentalRegistrations.csv')

    # rename address
    df_rental.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
    df_rental = convert_address(df_rental)

    df_rental = df_rental[['ADDRESS', 'Permit Issue Date']]
    df_rental['Rental_permit'] = 1
    
    # also create a feature considering rental permit counts in the whole duration

    merged_rental = pd.merge(df, df_rental, on='ADDRESS', how='left')
    merged_rental = drop_columns(merged_rental, ['Permit Issue Date'])
    
    return merged_rental

def wrangle_merge_SFIncidents(df):
    df_SFIncidents = pd.read_csv('SFIncidents.csv')

    # rename address
    df_SFIncidents.rename(columns = {'Location': 'ADDRESS'}, inplace = True)

    df_SFIncidents = convert_address(df_SFIncidents)

    # convert datetime to year
    df_SFIncidents['Date'] = pd.to_datetime(df_SFIncidents['Time']).dt.date 
    df_SFIncidents['Year'] = pd.to_datetime(df_SFIncidents['Date']).dt.year
    df_SFIncidents_new = df_SFIncidents[['ADDRESS', 'Code', 'Date', 'Year', 'Category', 'Type']]
                              
    df_SFIncidents_new = df_SFIncidents_new[~df_SFIncidents_new.index.duplicated()]

    # filtering just the fire incidents using the code
    df_fire = df_SFIncidents_new[df_SFIncidents_new['Code']==111]

    df_fire = df_fire[['ADDRESS', 'Year', 'Date']]
    
    # add the binary column
    df_fire['Incident'] = 1
    
    # remove entries with duplicate address
    df_fire = df_fire.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge
    #df = pd.merge(df, df_fire, how = 'left', on = 'ADDRESS')
    
    merged_fire = pd.merge(df, df_fire, on='ADDRESS', how='left')

    merged_fire['fire_count'] = merged_fire.groupby('ADDRESS')['Date'].transform('count')

    merged_fire = merged_fire.drop(['Date'], axis=1)
    merged_fire = merged_fire.drop_duplicates(subset=['ADDRESS'])

    return merged_fire


def wrangle_merge_inspections(df):
    
    df_inspection = pd.read_csv('Inspection_status.csv')

    # rename address
    df_inspection.rename(columns = {'Address': 'ADDRESS'}, inplace = True)
    df_inspection = convert_address(df_inspection)

    df_inspection = df_inspection[['ADDRESS', 'End Date', 'Scheduled Start Date']]
    #df_inspection['Permit_Invalid'] = 0
    df_inspection['Year'] = pd.to_datetime(df_inspection['Scheduled Start Date']).dt.year
    merged_inspections = pd.merge(df, df_inspection, on='ADDRESS', how='left')

    # convert to datetime format
    merged_inspections['End Date'] = pd.to_datetime(merged_inspections['End Date']).dt.date
    merged_inspections['Scheduled Start Date'] = pd.to_datetime(merged_inspections['Scheduled Start Date']).dt.date


    merged_inspections['inspection_count'] = merged_inspections.groupby('ADDRESS')['Scheduled Start Date'].transform('count')

    merged_inspections = merged_inspections.drop(['Scheduled Start Date'], axis=1)
    merged_inspections = merged_inspections.drop_duplicates(subset=['ADDRESS'])

    merged_inspections = drop_columns(merged_inspections, ['End Date'])
 
    return merged_inspections

def wrangle_merge_parcelData(df):
    df_parcelData = pd.read_csv('ParcelData.csv')
    df_parcelData = convert_address(df_parcelData)

    # convert activity

    df_parcelData['ACTIVITY'] = df_parcelData.ACTIVITY.apply(cleanActivity)

    df_parcelData = df_parcelData[['ADDRESS', 'SQFT', 'ACREAGE', 'FRONTFOOT', 'ACTIVITY', 'Shape_Length', 'Shape_Area', 'BuildVal', 'LandVal']]
                                      
    df_parcelData = drop_duplicate(df_parcelData)

    # remove rows with null/0 address value
    skip = ["0", ""]
    df_parcelData = df_parcelData[~df_parcelData['ADDRESS'].isin(skip)]

    # remove entries with duplicate address
    df_parcelData = df_parcelData.drop_duplicates(subset=['ADDRESS'], keep='last')

    # merge
    merged_parcelData = pd.merge(df, df_parcelData, how='left', on='ADDRESS')

    return merged_parcelData

def wrangle_merge_foreclosures(df):
    
    df_foreclosure = pd.read_excel('foreclosures.xlsx')
    df_foreclosure = convert_address(df_foreclosure)
    columns_to_drop = ['NAME', 'AUCTIONDATE','ASSESSEDVALUE']
    df_foreclosure = drop_columns(df_foreclosure, columns_to_drop)

    merged_foreclosure = pd.merge(df, df_foreclosure, on='ADDRESS', how='left')

    merged_foreclosure['foreclosing_count'] = merged_foreclosure.groupby('ADDRESS')['YEAR'].transform('count')

    merged_foreclosure = merged_foreclosure.drop(['YEAR'], axis=1)
    merged_foreclosure = merged_foreclosure.drop_duplicates(subset=['ADDRESS'])
    
    return merged_foreclosure


######################
# wrangle address file
######################

df_address = wrangle_address()
df = df_address.copy()

######################
# convert lat, long features
######################

df = df[df['LAT'] >= -90]
df = convert_location(df)

######################
# wrangle and merge codecases file
######################

df = wrangle_merge_codecases(df)

######################
# wrangle and merge property offenses/crimedata file
######################

df = wrangle_merge_crimeData(df)

#####################################
# wrangle utility disconnections file
#####################################

df = wrangle_merge_utility(df)

#####################################
# wrangle rental registrations file
#####################################

df = wrangle_merge_rental(df)

#####################################
# wrangle SFIncidents file
#####################################

df = wrangle_merge_SFIncidents(df)

#####################################
# wrangle Inspections file
#####################################

df = wrangle_merge_inspections(df) 

#####################################
# wrangle ParcelData file
#####################################

df = wrangle_merge_parcelData(df)

#####################################
# wrangle foreclosure
#####################################

df = wrangle_merge_foreclosures(df)

df = drop_columns(df, ['Year_x', 'Year_y'])
df.Incident = df.Incident.fillna(0)

df.to_csv('Simple_dataset2.csv')

# create profile
#create_profile(df, 'Simple_dataset_Not_Considering_Fire')

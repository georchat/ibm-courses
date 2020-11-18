#!/usr/bin/env python

import os
import sys
import getopt
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime


DATA_DIR = os.path.join("..","data")

def connect_db(file_path):
    """
    function to connect to aavail database
    """
    try:
        conn = sqlite3.connect(file_path)
        print("...successfully connected to db\n")
    except Error as e:
        print("...unsuccessful connection\n",e)
    
    return(conn)

def ingest_db_data(conn):
    """
    load and clean database data
    """
    query = """SELECT 
            CUSTOMER.customer_id, 
            CUSTOMER.last_name,
            CUSTOMER.first_name,
            CUSTOMER.DOB,
            CUSTOMER.city, 
            CUSTOMER.state, 
            COUNTRY.country_name, 
            CUSTOMER.gender
            FROM CUSTOMER
            LEFT JOIN COUNTRY 
            ON CUSTOMER.country_id = COUNTRY.country_id;"""
    
    # execute query and store the results in a pandas dataframe
    _data = [row for row in conn.execute(query)]
    columns=["customer_id", "last_name", "first_name", "DOB", "city", "state", "country_name", "gender"]
    df_customers = pd.DataFrame(_data, columns=columns)
    print("...imported db dataset of {} rows and {} columns".format(df_customers.shape[0], df_customers.shape[1]))
    
    # implement checks for quality assurance
    is_duplicate = df_customers.duplicated(subset=["customer_id"])
    print("...removed {} number of duplicates customer ids".format(len(df_customers[is_duplicate])))
    df_customers = df_customers[~is_duplicate].reset_index(drop=True)
    
    return df_customers

def ingest_stream_data(file_path):
    """
    load and clean stream data
    """
    # read streams data
    df_streams = pd.read_csv(file_path)
    print("...imported streams dataset of {} rows and {} columns".format(df_streams.shape[0], df_streams.shape[1]))
    
    # determine customer churn from streams dataset
    df_churns = df_streams.sort_values(["customer_id","date"], ascending=False).groupby(["customer_id"]).head(1)
    df_churns = df_churns.sort_values(["customer_id"]).reset_index(drop=True)
    df_churns["is_subscriber"] = np.where(df_churns.subscription_stopped == 1, 0, 1)
    
    # implement checks for quality assurance
    print("...removed {} missing stream ids".format(df_streams["stream_id"].isna().sum()))
    is_na = df_streams["stream_id"].isna()
    df_streams = df_streams[~is_na].reset_index(drop=True)
    
    return df_streams, df_churns

def process_dataframes(df_customers, df_streams, df_churns, conn):
    """
    combine the data into a single data structure
    """
    
    # create a clean copy of the customers dataframe and add new attributes
    df_clean = df_customers.copy()
    df_clean["date_of_birth"] = pd.to_datetime(df_customers['DOB'], format="%m/%d/%y")
    df_clean.loc[df_clean['date_of_birth'].dt.year >= 2020, 'date_of_birth'] -= pd.DateOffset(years=100)
    df_clean["age"] = datetime.now().year - df_clean.date_of_birth.dt.year
    df_clean["customer_name"] = df_clean.first_name + " " + df_clean.last_name
    df_clean.drop(["last_name", "first_name", "DOB", "city", "state", "gender", "date_of_birth"], axis=1, inplace=True)
    
    # ensure we are working with correctly ordered customer_ids df_customers
    if not np.array_equal(df_clean['customer_id'], df_customers['customer_id']):
        raise Exception("indexes are out of order or unmatched---needs to fix")
        
    df_clean = df_clean.merge(df_churns[["customer_id", "is_subscriber"]], how="inner", on="customer_id")
        
    # query the db to create a invoice item map
    query = """SELECT 
            INVOICE.customer_id, 
            INVOICE.invoice_item_id,
            INVOICE_ITEM.invoice_item
            FROM INVOICE
            INNER JOIN INVOICE_ITEM
            ON INVOICE.invoice_item_id = INVOICE_ITEM.invoice_item_id;"""
    
    df_invoices = pd.DataFrame([row for row in conn.execute(query)], columns=["customer_id", "invoice_item_id", "invoice_item"])
    
    # implement checks for quality assurance
    is_duplicate = df_invoices.duplicated(subset=["customer_id"])
    if True in is_duplicate:
        df_invoices = df_invoices[~is_duplicate].reset_index(drop=True)
        
    # add new attributes (subscriber_type, num_streams)
    df_clean = df_clean.merge(df_invoices[["customer_id", "invoice_item"]], how="inner", on="customer_id")
    df_clean.rename(columns={"invoice_item":"subscriber_type"}, inplace=True)
    
    df_clean = df_clean.merge(df_streams.groupby(["customer_id"])["stream_id"].count().reset_index(), how="inner", on="customer_id")
    df_clean.rename(columns={"stream_id":"num_streams"}, inplace=True)
    
    return df_clean
        

def update_target(target_file, df_clean, overwrite=False):
    """
    update line by line in case data are large
    """
    
    if overwrite or not os.path.exists(target_file):
        df_clean.to_csv(target_file, index=False)   
    else:
        df_target = pd.read_csv(target_file)
        df_target.to_csv(target_file, mode='a', index=False)
        
if __name__ == "__main__":
  
    ## collect args
    arg_string = "%s -d db_filepath -s streams_filepath"%sys.argv[0]
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'd:s:')
    except getopt.GetoptError:
        print(getopt.GetoptError)
        raise Exception(arg_string)

    ## handle args
    streams_file = None
    db_file = None
    for o, a in optlist:
        if o == '-d':
            db_file = a
        if o == '-s':
            streams_file = a
            
    streams_file = os.path.join(DATA_DIR, streams_file)
    db_file = os.path.join(DATA_DIR, db_file)
    target_file = os.path.join(DATA_DIR, "aavail-target.csv")
    
    ## make the connection to the database
    conn = connect_db(db_file)

    ## ingest data base data
    df_customers = ingest_db_data(conn)
    df_streams, df_churn = ingest_stream_data(streams_file)
    df_clean = process_dataframes(df_customers, df_streams, df_churn, conn)
    
    ## write
    update_target(target_file, df_clean, overwrite=False)
    print("done")
    
